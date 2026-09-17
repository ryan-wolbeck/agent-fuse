import json
import os
import time
from datetime import UTC, datetime

from typer.testing import CliRunner

from agent_fuse.cli import app
from tests.conftest import build_turn, session_meta_line, write_session_file

runner = CliRunner()
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_doctor_no_codex_installed(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "nonexistent"))
    monkeypatch.setattr("shutil.which", lambda _: None)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 3
    assert "not detected" in result.stdout


def test_doctor_success(codex_home) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Ready to watch." in result.stdout


def test_scan_no_sessions_is_clean_exit(codex_home) -> None:
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "Scanning 0 Codex sessions" in result.stdout


def test_scan_reports_normal_session(codex_home, new_session_id) -> None:
    lines = [session_meta_line(new_session_id, T0)]
    turn_lines, _ = build_turn(
        T0, "turn-1", input_tokens=100, cached_input_tokens=10, output_tokens=5
    )
    lines += turn_lines
    write_session_file(codex_home, new_session_id, lines)

    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "1 normal" in result.stdout


def test_scan_json_output_is_valid_json(codex_home, new_session_id) -> None:
    lines = [session_meta_line(new_session_id, T0)]
    turn_lines, _ = build_turn(
        T0, "turn-1", input_tokens=100, cached_input_tokens=10, output_tokens=5
    )
    lines += turn_lines
    write_session_file(codex_home, new_session_id, lines)

    result = runner.invoke(app, ["scan", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    assert payload["sessions_scanned"] == 1
    assert payload["sessions"][0]["session_id"] == new_session_id


def test_scan_detects_runaway_session_and_exits_1(codex_home, new_session_id) -> None:
    lines = [session_meta_line(new_session_id, T0)]
    t = T0
    for i in range(150):
        turn_lines, t = build_turn(
            t,
            f"turn-{i}",
            input_tokens=20_000,
            cached_input_tokens=19_800,
            output_tokens=100,
            call_id=f"call-{i}",
            command=["git", "status"],
        )
        lines += turn_lines
    write_session_file(codex_home, new_session_id, lines)

    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 1
    assert "severe" in result.stdout.lower()


def test_scan_invalid_config_exits_2(codex_home, tmp_path) -> None:
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text("not_a_real_field: true")
    result = runner.invoke(app, ["scan", "--config", str(bad_config)])
    assert result.exit_code == 2


def test_inspect_with_no_recent_sessions(codex_home, new_session_id) -> None:
    lines = [session_meta_line(new_session_id, T0)]
    path = write_session_file(codex_home, new_session_id, lines)
    # recency is judged by file mtime, not embedded event timestamps
    old = time.time() - 3600
    os.utime(path, (old, old))
    result = runner.invoke(app, ["inspect", "--recent-minutes", "1"])
    assert result.exit_code == 0
    assert "Active/recent:       0" in result.stdout


def test_inspect_includes_recent_session_in_table(codex_home, new_session_id) -> None:
    lines = [session_meta_line(new_session_id, T0)]
    write_session_file(codex_home, new_session_id, lines)
    # a huge recent window makes the fixed-in-the-past fixture "recent"
    result = runner.invoke(app, ["inspect", "--recent-minutes", str(60 * 24 * 365 * 10)])
    assert result.exit_code == 0
    assert new_session_id[:8] in result.stdout
