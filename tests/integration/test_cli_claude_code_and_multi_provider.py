import json
from datetime import UTC, datetime

from typer.testing import CliRunner

from agent_fuse.cli import app
from tests.conftest import (
    build_claude_turn,
    build_turn,
    claude_user_text_line,
    session_meta_line,
    write_claude_session_file,
    write_session_file,
)

runner = CliRunner()
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_doctor_detects_claude_code_only(claude_code_home) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Claude Code detected" in result.stdout
    assert "Codex installation not detected" in result.stdout
    assert "Ready to watch." in result.stdout


def test_doctor_detects_both_providers(codex_home, claude_code_home) -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Codex detected" in result.stdout
    assert "Claude Code detected" in result.stdout


def test_scan_finds_claude_code_session(claude_code_home, new_session_id) -> None:
    lines = [claude_user_text_line(T0, new_session_id, "hello")]
    turn_lines, _ = build_claude_turn(
        T0, new_session_id, "msg_1", input_tokens=100, cache_read_input_tokens=10, output_tokens=5
    )
    lines += turn_lines
    write_claude_session_file(claude_code_home, new_session_id, lines)

    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "1 normal" in result.stdout


def test_scan_json_labels_claude_code_sessions_by_provider(
    claude_code_home, new_session_id
) -> None:
    lines = [claude_user_text_line(T0, new_session_id, "hello")]
    turn_lines, _ = build_claude_turn(
        T0, new_session_id, "msg_1", input_tokens=100, cache_read_input_tokens=10, output_tokens=5
    )
    lines += turn_lines
    write_claude_session_file(claude_code_home, new_session_id, lines)

    result = runner.invoke(app, ["scan", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sessions"][0]["provider"] == "claude_code"
    assert payload["sessions"][0]["session_id"] == new_session_id


def test_scan_detects_runaway_claude_code_session(claude_code_home, new_session_id) -> None:
    lines = [claude_user_text_line(T0, new_session_id, "hello")]
    t = T0
    for i in range(150):
        turn_lines, t = build_claude_turn(
            t,
            new_session_id,
            f"msg_{i}",
            input_tokens=2,
            cache_creation_input_tokens=1000,
            cache_read_input_tokens=19_000,
            output_tokens=100,
            tool_name="Bash",
            tool_input={"command": "git status"},
            tool_id=f"tool_{i}",
        )
        lines += turn_lines
    write_claude_session_file(claude_code_home, new_session_id, lines)

    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 1
    assert "severe" in result.stdout.lower()


def test_scan_aggregates_across_both_providers(
    codex_home, claude_code_home, new_session_id
) -> None:
    codex_session_id = new_session_id
    claude_session_id = "22222222-2222-2222-2222-222222222222"

    codex_lines = [session_meta_line(codex_session_id, T0)]
    turn_lines, _ = build_turn(
        T0, "turn-1", input_tokens=100, cached_input_tokens=10, output_tokens=5
    )
    codex_lines += turn_lines
    write_session_file(codex_home, codex_session_id, codex_lines)

    claude_lines = [claude_user_text_line(T0, claude_session_id, "hello")]
    claude_turn_lines, _ = build_claude_turn(
        T0,
        claude_session_id,
        "msg_1",
        input_tokens=100,
        cache_read_input_tokens=10,
        output_tokens=5,
    )
    claude_lines += claude_turn_lines
    write_claude_session_file(claude_code_home, claude_session_id, claude_lines)

    result = runner.invoke(app, ["scan", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["sessions_scanned"] == 2
    providers = {s["provider"] for s in payload["sessions"]}
    assert providers == {"codex", "claude_code"}


def test_two_providers_with_the_same_session_id_do_not_merge_metrics(
    codex_home, claude_code_home
) -> None:
    """Regression test for the MetricsRegistry key: before it was keyed on
    (provider, session_id), two sessions sharing an id across providers
    would silently merge into one SessionMetrics."""
    shared_id = "33333333-3333-3333-3333-333333333333"

    codex_lines = [session_meta_line(shared_id, T0)]
    turn_lines, _ = build_turn(
        T0, "turn-1", input_tokens=100, cached_input_tokens=10, output_tokens=5
    )
    codex_lines += turn_lines
    write_session_file(codex_home, shared_id, codex_lines)

    claude_lines = [claude_user_text_line(T0, shared_id, "hello")]
    for i in range(3):
        turn_lines, _ = build_claude_turn(
            T0, shared_id, f"msg_{i}", input_tokens=10, output_tokens=1
        )
        claude_lines += turn_lines
    write_claude_session_file(claude_code_home, shared_id, claude_lines)

    result = runner.invoke(app, ["scan", "--format", "json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    by_provider = {s["provider"]: s for s in payload["sessions"]}
    assert by_provider["codex"]["total_responses"] == 1
    assert by_provider["claude_code"]["total_responses"] == 3


def test_inspect_shows_provider_column_for_claude_code(claude_code_home, new_session_id) -> None:
    lines = [claude_user_text_line(T0, new_session_id, "hello")]
    write_claude_session_file(claude_code_home, new_session_id, lines)
    result = runner.invoke(app, ["inspect", "--recent-minutes", str(60 * 24 * 365 * 10)])
    assert result.exit_code == 0
    assert "Claude Code" in result.stdout
    assert new_session_id[:8] in result.stdout


def test_inspect_never_leaks_claude_code_project_directory_name(
    claude_code_home, new_session_id
) -> None:
    """The parent directory of a Claude Code session file encodes the
    project's cwd path (sanitized). It must never appear in output."""
    lines = [claude_user_text_line(T0, new_session_id, "hello")]
    write_claude_session_file(
        claude_code_home,
        new_session_id,
        lines,
        project_dir="-home-alice-clients-acme-corp-secret-project",
    )
    result = runner.invoke(app, ["inspect", "--recent-minutes", str(60 * 24 * 365 * 10)])
    assert result.exit_code == 0
    assert "acme-corp" not in result.stdout
    assert "alice" not in result.stdout
    assert "secret-project" not in result.stdout
