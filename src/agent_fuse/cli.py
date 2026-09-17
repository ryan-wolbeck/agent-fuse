"""Agent Fuse CLI.

Exit codes (stable, documented in README):
  0 = success, no rule violations observed
  1 = a rule violation was detected (scan/watch)
  2 = configuration or input error
  3 = provider/discovery failure (e.g. no supported agent found)
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

import typer

from . import discovery
from .adapters.codex import CodexAdapter
from .config import ConfigError, FuseConfig, load_config
from .discovery import FileTailer, SessionFileInfo
from .engine import RuleEngine, scan_session_file
from .metrics import MetricsRegistry
from .models import EventType
from .output import (
    emit_fuse_event_json,
    make_console,
    render_doctor_line,
    render_inspect_summary,
    render_inspect_table,
    render_scan_largest_anomaly,
    render_scan_progress_summary,
    render_watch_normal,
    render_watch_warning,
)

app = typer.Typer(
    name="agent-fuse",
    help="A local circuit breaker monitor for runaway AI coding agents.",
    no_args_is_help=True,
    add_completion=False,
)

RECENT_WINDOW_SECONDS_DEFAULT = 30 * 60


def _load_config_or_exit(config_path: Path | None) -> FuseConfig:
    try:
        return load_config(config_path)
    except ConfigError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(code=2) from exc


@app.command()
def doctor() -> None:
    """Check that Agent Fuse can find and read local Codex telemetry.

    Entirely read-only: never modifies Codex state, config, or session files.
    """
    console = make_console()
    ok = True

    py_version = platform.python_version()
    py_ok = sys.version_info >= (3, 11)
    render_doctor_line(console, py_ok, f"Python {py_version}")
    ok = ok and py_ok

    codex_present = discovery.codex_installed()
    render_doctor_line(
        console,
        codex_present,
        "Codex detected" if codex_present else "Codex installation not detected",
    )
    if not codex_present:
        console.print("\nInstall/run the Codex CLI at least once, then re-run `agent-fuse doctor`.")
        raise typer.Exit(code=3)

    sdir = discovery.sessions_dir()
    sdir_ok = sdir.is_dir()
    render_doctor_line(
        console,
        sdir_ok,
        "Session directory readable" if sdir_ok else f"Session directory not found: {sdir}",
    )
    ok = ok and sdir_ok

    files = discovery.discover_session_files() if sdir_ok else []
    render_doctor_line(console, True, f"{len(files)} session files discovered")

    latest_ok = True
    if files:
        try:
            with open(files[0].path, encoding="utf-8", errors="replace") as fh:
                fh.readline()
        except OSError:
            latest_ok = False
        render_doctor_line(
            console,
            latest_ok,
            "Latest session readable" if latest_ok else "Latest session file unreadable",
        )
        ok = ok and latest_ok

    config_ok = True
    try:
        load_config(None)
    except ConfigError as exc:
        config_ok = False
        render_doctor_line(console, False, f"Configuration invalid: {exc}")
    if config_ok:
        render_doctor_line(console, True, "Configuration valid")
    ok = ok and config_ok

    if ok:
        console.print("\nReady to watch.")
    else:
        console.print("\nOne or more checks failed -- see above.")
        raise typer.Exit(code=2)


@app.command()
def inspect(
    recent_minutes: int = typer.Option(
        30, help="Window (minutes) defining an 'active/recent' session."
    ),
    limit: int = typer.Option(20, help="Max rows to show in the recent-sessions table."),
) -> None:
    """Show sanitized aggregate information about discoverable sessions.

    Never prints prompt text, tool arguments, source code, or other
    session content -- only counts and status derived from them.
    """
    console = make_console()
    if not discovery.codex_installed():
        console.print("No supported agent telemetry detected (Codex not found).")
        raise typer.Exit(code=3)

    config = _load_config_or_exit(None)
    all_files = discovery.discover_session_files()
    now_epoch = time.time()
    recent = [f for f in all_files if discovery.is_recent(f, now_epoch, recent_minutes * 60)]

    rows: list[tuple[str, str, int, int, str]] = []
    events_sampled = 0
    for info in recent[:limit]:
        result = scan_session_file(info.path, info.session_id_hint, config)
        events_sampled += result.diagnostics.events_emitted
        severity = result.severity or "normal"
        age = _format_age(now_epoch - info.mtime)
        rows.append(
            (
                result.session_id,
                age,
                result.metrics.total_responses,
                result.metrics.total_tool_calls,
                severity,
            )
        )

    render_inspect_summary(
        console,
        provider="Codex",
        sessions_discovered=len(all_files),
        active_recent=len(recent),
        events_sampled=events_sampled,
    )
    if rows:
        render_inspect_table(console, rows)


def _format_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    return f"{hours}h{minutes % 60}m"


@app.command()
def scan(
    config_path: Path | None = typer.Option(None, "--config", help="Path to .agent-fuse.yaml"),
    format: str = typer.Option("text", "--format", help="Output format: text or json"),
) -> None:
    """Run deterministic rules against existing historical Codex sessions.

    Read-only dry-run analysis -- lets you see whether your existing
    history already contains obvious runaway patterns, without waiting for
    a live one.
    """
    if format not in ("text", "json"):
        typer.echo("Invalid --format: expected 'text' or 'json'", err=True)
        raise typer.Exit(code=2)

    if not discovery.codex_installed():
        typer.echo("No supported agent telemetry detected (Codex not found).", err=True)
        raise typer.Exit(code=3)

    config = _load_config_or_exit(config_path)
    files = discovery.discover_session_files()

    console = make_console(quiet=(format == "json"))
    console.print(f"Scanning {len(files)} Codex sessions...")

    results = [scan_session_file(f.path, f.session_id_hint, config) for f in files]

    normal = sum(1 for r in results if r.severity is None)
    warning = sum(1 for r in results if r.severity == "warning")
    severe = sum(1 for r in results if r.severity == "severe")

    anomalies = [r for r in results if r.severity is not None]
    largest = None
    if anomalies:
        rank = {"severe": 1, "warning": 0}
        largest = max(
            anomalies, key=lambda r: (rank[r.severity or "warning"], r.metrics.total_responses)
        )

    if format == "json":
        import json as _json

        payload = {
            "schema_version": 1,
            "sessions_scanned": len(results),
            "normal": normal,
            "warning": warning,
            "severe": severe,
            "sessions": [
                {
                    "session_id": r.session_id,
                    "severity": r.severity,
                    "total_responses": r.metrics.total_responses,
                    "total_tool_calls": r.metrics.total_tool_calls,
                    "duration_seconds": r.metrics.duration_seconds(),
                    "cached_ratio": r.metrics.lifetime_cached_ratio(),
                    "triggered_rules": [x.rule for x in RuleEngine.triggered(r.worst_results)],
                }
                for r in results
            ],
        }
        typer.echo(_json.dumps(payload, indent=2))
    else:
        render_scan_progress_summary(
            console, total=len(results), normal=normal, warning=warning, severe=severe
        )
        if largest is not None:
            render_scan_largest_anomaly(
                console,
                session_id=largest.session_id,
                responses=largest.metrics.total_responses,
                duration_seconds=largest.metrics.duration_seconds(),
                cached_ratio=largest.metrics.lifetime_cached_ratio(),
                triggered_rules=[x.rule for x in RuleEngine.triggered(largest.worst_results)],
            )

    raise typer.Exit(code=1 if anomalies else 0)


@app.command()
def watch(
    config_path: Path | None = typer.Option(None, "--config", help="Path to .agent-fuse.yaml"),
    format: str = typer.Option("text", "--format", help="Output format: text or jsonl"),
    quiet: bool = typer.Option(
        False, "--quiet", help="Suppress normal-status lines (automation-friendly)."
    ),
    recent_minutes: int = typer.Option(
        30, help="Window (minutes) defining an 'active/recent' session to attach to."
    ),
    poll_seconds: float = typer.Option(1.0, help="How often to check tailed files for new events."),
    discover_seconds: float = typer.Option(
        5.0, help="How often to check for newly created session files."
    ),
) -> None:
    """Watch local Codex sessions live and warn on runaway patterns.

    Agent Fuse does NOT terminate the process. It observes and warns.
    """
    if format not in ("text", "jsonl"):
        typer.echo("Invalid --format: expected 'text' or 'jsonl'", err=True)
        raise typer.Exit(code=2)

    if not discovery.codex_installed():
        typer.echo("No supported agent telemetry detected (Codex not found).", err=True)
        raise typer.Exit(code=3)

    config = _load_config_or_exit(config_path)
    console = make_console(quiet=(format == "jsonl"))
    console.print("Agent Fuse")
    console.print("Watching Codex sessions...")

    rule_engine = RuleEngine(config)
    registry = MetricsRegistry(max_window_seconds=config.max_window_seconds())
    tailers: dict[Path, FileTailer] = {}
    adapters: dict[Path, CodexAdapter] = {}
    path_to_session_id: dict[Path, str] = {}
    last_triggered: dict[str, frozenset[str]] = {}
    any_triggered = False

    def attach(info: SessionFileInfo) -> None:
        tailers[info.path] = FileTailer(info.path)
        adapters[info.path] = CodexAdapter(info.session_id_hint)
        path_to_session_id[info.path] = info.session_id_hint

    now_epoch = time.time()
    for info in discovery.discover_session_files():
        if discovery.is_recent(info, now_epoch, recent_minutes * 60):
            attach(info)

    last_discover = time.monotonic()

    try:
        while True:
            for path in list(tailers):
                tailer = tailers[path]
                adapter = adapters[path]
                new_lines = tailer.read_new_lines()
                if tailer.gone:
                    session_id = path_to_session_id.get(path)
                    if session_id:
                        registry.drop(session_id)
                        last_triggered.pop(session_id, None)
                    tailer.close()
                    del tailers[path]
                    del adapters[path]
                    path_to_session_id.pop(path, None)
                    continue

                had_activity = False
                for raw_line in new_lines:
                    for event in adapter.parse_line(
                        raw_line, session_id_hint=path_to_session_id[path]
                    ):
                        had_activity = True
                        if event.event_type is EventType.SESSION_START:
                            path_to_session_id[path] = event.session_id
                        metrics = registry.record(event)
                        if event.event_type in (EventType.MODEL_RESPONSE, EventType.TOOL_CALL):
                            results = rule_engine.evaluate(metrics, event.timestamp)
                            triggered = RuleEngine.triggered(results)
                            triggered_names = frozenset(r.rule for r in triggered)
                            session_id = metrics.session_id
                            if triggered_names and triggered_names != last_triggered.get(
                                session_id
                            ):
                                any_triggered = True
                                last_triggered[session_id] = triggered_names
                                if format == "jsonl":
                                    for fuse_event in RuleEngine.to_fuse_events(
                                        "codex", session_id, event.timestamp, results
                                    ):
                                        typer.echo(emit_fuse_event_json(fuse_event))
                                else:
                                    render_watch_warning(
                                        console,
                                        session_id=session_id,
                                        session=metrics,
                                        results=results,
                                        now=event.timestamp,
                                    )
                            elif not triggered_names and last_triggered.get(session_id):
                                last_triggered.pop(session_id, None)

                if had_activity and not quiet and format == "text":
                    session_id = path_to_session_id[path]
                    if session_id not in last_triggered:
                        render_watch_normal(console, session_id)

            if time.monotonic() - last_discover >= discover_seconds:
                last_discover = time.monotonic()
                now_epoch = time.time()
                for info in discovery.discover_session_files():
                    if info.path not in tailers and discovery.is_recent(
                        info, now_epoch, recent_minutes * 60
                    ):
                        attach(info)

            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        console.print("\nStopped watching.")

    raise typer.Exit(code=1 if any_triggered else 0)


if __name__ == "__main__":
    app()
