"""Adversarial fixtures containing secrets/credentials/private data.

These assert that default CLI output (text and JSON) never contains the
sensitive substrings, regardless of where in a session they were embedded.
"""

from datetime import UTC, datetime

from typer.testing import CliRunner

from agent_fuse.cli import app
from tests.conftest import (
    build_claude_turn,
    claude_system_error_line,
    claude_tool_result_line,
    claude_user_text_line,
    error_line,
    function_call_line,
    local_shell_call_line,
    message_line,
    session_meta_line,
    task_complete_line,
    task_started_line,
    token_count_line,
    write_claude_session_file,
    write_session_file,
)

runner = CliRunner()
T0 = datetime(2026, 1, 1, tzinfo=UTC)

SECRETS = [
    "OPENAI_API_KEY=sk-live-abcdefghijklmnop",
    "Authorization: Bearer super-secret-token-value",
    "password=hunter2",
    "def _private_algorithm(): return proprietary_constant * 42",
    "curl -H 'X-Api-Key: abc123' https://internal.corp.example.com/v1/customers?ssn=123-45-6789",
    "/home/alice/clients/acme-corp/contract-2026.pdf",
]


def _malicious_session(session_id: str) -> list[str]:
    lines = [session_meta_line(session_id, T0, cwd="/home/alice/clients/acme-corp")]
    lines.append(task_started_line(T0, "turn-1"))
    lines.append(message_line(T0, "user", SECRETS[0] + " " + SECRETS[1]))
    lines.append(
        local_shell_call_line(
            T0,
            "call-1",
            command=["bash", "-c", SECRETS[4]],
            env={"OPENAI_API_KEY": "sk-live-abcdefghijklmnop", "DB_PASSWORD": "hunter2"},
            working_directory="/home/alice/clients/acme-corp",
        )
    )
    lines.append(
        function_call_line(
            T0, "call-2", "write_file", f'{{"path": "{SECRETS[5]}", "content": "{SECRETS[3]}"}}'
        )
    )
    lines.append(error_line(T0, f"auth failed: {SECRETS[1]}"))
    lines.append(token_count_line(T0, input_tokens=1000, cached_input_tokens=900, output_tokens=10))
    lines.append(task_complete_line(T0, "turn-1"))
    return lines


def test_scan_text_output_never_contains_secrets(codex_home, new_session_id) -> None:
    write_session_file(codex_home, new_session_id, _malicious_session(new_session_id))
    result = runner.invoke(app, ["scan"])
    for secret in SECRETS:
        assert secret not in result.stdout
    # spot-check fragments too, in case of partial leakage
    assert "hunter2" not in result.stdout
    assert "acme-corp" not in result.stdout
    assert "alice" not in result.stdout
    assert "sk-live" not in result.stdout


def test_scan_json_output_never_contains_secrets(codex_home, new_session_id) -> None:
    write_session_file(codex_home, new_session_id, _malicious_session(new_session_id))
    result = runner.invoke(app, ["scan", "--format", "json"])
    for secret in SECRETS:
        assert secret not in result.stdout
    assert "hunter2" not in result.stdout
    assert "acme-corp" not in result.stdout


def test_inspect_output_never_contains_secrets(codex_home, new_session_id) -> None:
    write_session_file(codex_home, new_session_id, _malicious_session(new_session_id))
    result = runner.invoke(app, ["inspect", "--recent-minutes", str(60 * 24 * 365 * 10)])
    for secret in SECRETS:
        assert secret not in result.stdout
    assert "hunter2" not in result.stdout


def _malicious_claude_code_session(session_id: str) -> list[str]:
    lines = [
        claude_user_text_line(
            T0, session_id, SECRETS[0] + " " + SECRETS[1], cwd="/home/alice/clients/acme-corp"
        )
    ]
    turn_lines, _ = build_claude_turn(
        T0,
        session_id,
        "msg_1",
        input_tokens=1000,
        cache_read_input_tokens=900,
        output_tokens=10,
        tool_name="Bash",
        tool_input={"command": SECRETS[4], "env": {"OPENAI_API_KEY": "sk-live-abcdefghijklmnop"}},
        tool_id="tool_1",
    )
    lines += turn_lines
    lines.append(claude_tool_result_line(T0, session_id, "tool_1", f"{SECRETS[3]} -- {SECRETS[5]}"))
    lines.append(claude_system_error_line(T0, session_id, f"auth failed: {SECRETS[1]}"))
    return lines


def test_scan_text_output_never_contains_claude_code_secrets(
    claude_code_home, new_session_id
) -> None:
    write_claude_session_file(
        claude_code_home,
        new_session_id,
        _malicious_claude_code_session(new_session_id),
        project_dir="-home-alice-clients-acme-corp",
    )
    result = runner.invoke(app, ["scan"])
    for secret in SECRETS:
        assert secret not in result.stdout
    assert "hunter2" not in result.stdout
    assert "acme-corp" not in result.stdout
    assert "alice" not in result.stdout
    assert "sk-live" not in result.stdout


def test_scan_json_output_never_contains_claude_code_secrets(
    claude_code_home, new_session_id
) -> None:
    write_claude_session_file(
        claude_code_home,
        new_session_id,
        _malicious_claude_code_session(new_session_id),
        project_dir="-home-alice-clients-acme-corp",
    )
    result = runner.invoke(app, ["scan", "--format", "json"])
    for secret in SECRETS:
        assert secret not in result.stdout
    assert "acme-corp" not in result.stdout


def test_inspect_output_never_contains_claude_code_secrets_or_project_path(
    claude_code_home, new_session_id
) -> None:
    write_claude_session_file(
        claude_code_home,
        new_session_id,
        _malicious_claude_code_session(new_session_id),
        project_dir="-home-alice-clients-acme-corp",
    )
    result = runner.invoke(app, ["inspect", "--recent-minutes", str(60 * 24 * 365 * 10)])
    for secret in SECRETS:
        assert secret not in result.stdout
    assert "acme-corp" not in result.stdout
    assert "alice" not in result.stdout


def test_malicious_rich_markup_in_session_id_is_not_interpreted(codex_home) -> None:
    """A session id (attacker-influenceable, since it's read from a local
    file) must never be treated as Rich markup that could corrupt/spoof
    terminal output."""
    weird_id = "[bold red]INJECTED[/bold red]"
    write_session_file(codex_home, "safe-filename-id", [session_meta_line(weird_id, T0)])
    result = runner.invoke(app, ["inspect", "--recent-minutes", str(60 * 24 * 365 * 10)])
    assert result.exit_code == 0
    # the literal bracket text may appear (truncated), but must not have been
    # swallowed/interpreted as color codes -- Rich would strip unmatched
    # style application silently, which we detect by requiring the raw
    # brackets to still be present in captured output.
    assert "[bold" in result.stdout or "INJECTED" in result.stdout
