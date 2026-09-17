"""Terminal (Rich) and machine-readable (JSON/JSONL) rendering.

Rules and the engine never call into this module -- it is purely a
presentation layer over RuleResult/FuseEvent/SessionMetrics, kept separate
so machine output can be tested for stability independently of how the
human-facing panels look.

Every value that ultimately comes from vendor session data (session IDs,
rule messages built from them, etc.) is written using `rich.text.Text`
with an explicit `style=`, never interpolated into a Rich markup string.
The console itself is also created with `markup=False`. Both are load
bearing: a session ID or tool name is attacker-influenceable in principle
(it's read from a local file), and Rich markup strings like
`[bold red]...[/]` interpreted from that content could otherwise be used
to corrupt or spoof terminal output.
"""

from __future__ import annotations

import json
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .engine import RuleEngine
from .metrics import SessionMetrics
from .models import FuseEvent, RuleResult

DISCLAIMER = "Agent Fuse does NOT terminate the process. Inspect the session before continuing."


def short_id(session_id: str, length: int = 8) -> str:
    if len(session_id) <= length:
        return session_id
    return f"{session_id[:length]}..."


def make_console(*, quiet: bool = False) -> Console:
    return Console(quiet=quiet, highlight=False, markup=False)


def emit_fuse_event_json(event: FuseEvent) -> str:
    """Stable single-line JSON representation. Safe for `--format jsonl`."""
    return event.model_dump_json()


def emit_fuse_events_json_array(events: list[FuseEvent]) -> str:
    return json.dumps([json.loads(e.model_dump_json()) for e in events], indent=2)


def render_watch_normal(console: Console, session_id: str) -> None:
    line = Text()
    line.append("✓", style="green")
    line.append(f" session {short_id(session_id)} normal")
    console.print(line)


def render_watch_warning(
    console: Console,
    *,
    session_id: str,
    session: SessionMetrics,
    results: list[RuleResult],
    now: datetime,
) -> None:
    del now
    severity = RuleEngine.session_severity(results) or "warning"
    triggered = RuleEngine.triggered(results)

    header = "SEVERE PATTERN" if severity == "severe" else "RUNAWAY PATTERN"
    icon_style = "bold red" if severity == "severe" else "yellow"

    heading = Text()
    heading.append("⚠", style=icon_style)
    heading.append(" ")
    heading.append(header, style="bold")
    console.print(heading)

    body = Text()
    body.append(f"Session: {short_id(session_id)}\n")

    for result in results:
        if result.window_seconds is not None:
            label = f"{result.rule} (last {result.window_seconds}s)"
        else:
            label = f"{result.rule} (lifetime)"
        observed_str = _format_observed(result)
        body.append(f"  {label}: {observed_str}\n")

    body.append("\n")
    body.append("Triggered rules:\n", style="bold")
    for result in triggered:
        body.append(f"  {result.rule}\n", style="bold")
        body.append(f"    {result.message}\n")
        body.append(f"    threshold: {_format_value(result.threshold)}\n")

    body.append(f"\n{DISCLAIMER}", style="italic")

    console.print(Panel(body, expand=False))


def _format_observed(result: RuleResult) -> str:
    if result.rule == "context_replay" and result.observed is not None:
        return f"{result.observed * 100:.1f}% cached"
    return _format_value(result.observed)


def _format_value(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return f"{value:,}"


def render_doctor_line(console: Console, ok: bool, message: str) -> None:
    line = Text()
    line.append("✓" if ok else "⚠", style="green" if ok else "yellow")
    line.append(f" {message}")
    console.print(line)


def render_inspect_summary(
    console: Console,
    *,
    provider: str,
    sessions_discovered: int,
    active_recent: int,
    events_sampled: int,
) -> None:
    console.print(Text("Agent Fuse", style="bold"))
    console.print(f"Provider: {provider}")
    console.print(f"Sessions discovered: {sessions_discovered}")
    console.print(f"Active/recent:       {active_recent}")
    console.print(f"Events sampled:      {events_sampled}")


def render_inspect_table(
    console: Console,
    rows: list[tuple[str, str, int, int, str]],
) -> None:
    table = Table(title="Recent sessions")
    table.add_column("SESSION")
    table.add_column("AGE")
    table.add_column("RESPONSES", justify="right")
    table.add_column("TOOLS", justify="right")
    table.add_column("STATUS")
    for session_id, age, responses, tools, status in rows:
        status_style = "green" if status == "normal" else "yellow"
        table.add_row(
            short_id(session_id),
            age,
            str(responses),
            str(tools),
            Text(status, style=status_style),
        )
    console.print(table)


def render_scan_progress_summary(
    console: Console,
    *,
    total: int,
    normal: int,
    warning: int,
    severe: int,
) -> None:
    console.print(f"Scanned {total} Codex sessions.")

    normal_line = Text()
    normal_line.append("✓", style="green")
    normal_line.append(f" {normal} normal")
    console.print(normal_line)

    if warning:
        warning_line = Text()
        warning_line.append("⚠", style="yellow")
        warning_line.append(f" {warning} warning")
        console.print(warning_line)

    if severe:
        severe_line = Text()
        severe_line.append("⚠", style="bold red")
        severe_line.append(f" {severe} severe")
        console.print(severe_line)


def render_scan_largest_anomaly(
    console: Console,
    *,
    session_id: str,
    responses: int,
    duration_seconds: float | None,
    cached_ratio: float | None,
    triggered_rules: list[str],
) -> None:
    console.print()
    console.print(Text("Largest anomaly", style="bold"))
    console.print(f"Session: {short_id(session_id)}")
    console.print(f"Responses: {responses:,}")
    if duration_seconds is not None:
        console.print(f"Duration: {_format_duration(duration_seconds)}")
    if cached_ratio is not None:
        console.print(f"Cached-input ratio: {cached_ratio * 100:.1f}%")
    if triggered_rules:
        console.print("Triggered:")
        for rule in triggered_rules:
            console.print(f"  {rule}")


def _format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes}m"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"
