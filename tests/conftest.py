from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import pytest


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def session_meta_line(
    session_id: str,
    timestamp: datetime,
    *,
    cwd: str = "/home/user/project",
    thread_source: str = "user",
) -> str:
    return json.dumps(
        {
            "timestamp": iso(timestamp),
            "type": "session_meta",
            "payload": {
                "id": session_id,
                "timestamp": iso(timestamp),
                "cwd": cwd,
                "originator": "cli",
                "cli_version": "0.150.1",
                "model_provider": "openai",
                "source": "cli",
                "thread_source": thread_source,
            },
        }
    )


def task_started_line(timestamp: datetime, turn_id: str) -> str:
    return json.dumps(
        {
            "timestamp": iso(timestamp),
            "type": "event_msg",
            "payload": {
                "type": "task_started",
                "turn_id": turn_id,
                "started_at": iso(timestamp),
                "collaboration_mode_kind": "default",
                "model_context_window": 128000,
            },
        }
    )


def token_count_line(
    timestamp: datetime,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> str:
    return json.dumps(
        {
            "timestamp": iso(timestamp),
            "type": "event_msg",
            "payload": {
                "type": "token_count",
                "info": {
                    "total_token_usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": cached_input_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_output_tokens": 0,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "last_token_usage": {
                        "input_tokens": input_tokens,
                        "cached_input_tokens": cached_input_tokens,
                        "output_tokens": output_tokens,
                        "reasoning_output_tokens": 0,
                        "total_tokens": input_tokens + output_tokens,
                    },
                    "model_context_window": 258400,
                },
                "rate_limits": None,
            },
        }
    )


def task_complete_line(timestamp: datetime, turn_id: str) -> str:
    return json.dumps(
        {
            "timestamp": iso(timestamp),
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": turn_id,
                "completed_at": iso(timestamp),
                "duration_ms": 200,
                "time_to_first_token_ms": 50,
                "last_agent_message": "done",
            },
        }
    )


def local_shell_call_line(
    timestamp: datetime,
    call_id: str,
    *,
    command: list[str] | None = None,
    env: dict[str, str] | None = None,
    working_directory: str | None = "/home/user/project",
) -> str:
    return json.dumps(
        {
            "timestamp": iso(timestamp),
            "type": "response_item",
            "payload": {
                "type": "local_shell_call",
                "call_id": call_id,
                "status": "completed",
                "action": {
                    "type": "exec",
                    "command": command or ["git", "status"],
                    "timeout_ms": 5000,
                    "working_directory": working_directory,
                    "env": env or {},
                },
            },
        }
    )


def function_call_line(timestamp: datetime, call_id: str, name: str, arguments: str) -> str:
    return json.dumps(
        {
            "timestamp": iso(timestamp),
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "id": None,
                "name": name,
                "call_id": call_id,
                "arguments": arguments,
            },
        }
    )


def function_call_output_line(timestamp: datetime, call_id: str, name: str, output: str) -> str:
    return json.dumps(
        {
            "timestamp": iso(timestamp),
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": call_id,
                "name": name,
                "output": {"body": output, "success": True},
            },
        }
    )


def error_line(timestamp: datetime, message: str) -> str:
    return json.dumps(
        {
            "timestamp": iso(timestamp),
            "type": "event_msg",
            "payload": {"type": "error", "message": message},
        }
    )


def message_line(timestamp: datetime, role: str, text: str) -> str:
    return json.dumps(
        {
            "timestamp": iso(timestamp),
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": text}],
            },
        }
    )


def build_turn(
    start: datetime,
    turn_id: str,
    *,
    input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
    call_id: str | None = None,
    command: list[str] | None = None,
    env: dict[str, str] | None = None,
) -> tuple[list[str], datetime]:
    """One full turn: task_started [+ tool call/output] token_count task_complete."""
    lines = [task_started_line(start, turn_id)]
    t = start
    if call_id is not None:
        t = t + timedelta(milliseconds=50)
        lines.append(local_shell_call_line(t, call_id, command=command, env=env))
        t = t + timedelta(milliseconds=50)
        lines.append(function_call_output_line(t, call_id, "local_shell", "ok"))
    t = t + timedelta(milliseconds=50)
    lines.append(
        token_count_line(
            t,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input_tokens,
            output_tokens=output_tokens,
        )
    )
    t = t + timedelta(milliseconds=50)
    lines.append(task_complete_line(t, turn_id))
    return lines, t


@pytest.fixture
def new_session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def isolated_agents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Neutralize both providers regardless of what's actually installed on
    the machine running the tests.

    This matters concretely: development and CI machines may have a real
    Codex and/or Claude Code install with real local session data. Without
    this, `codex_installed()`/`claude_code_installed()` fall back to
    `shutil.which(...)`, which finds the real binaries, and a test meant to
    exercise "0 sessions discovered" would instead pick up however many
    real sessions happen to exist on that machine -- flaky by construction.
    `codex_home`/`claude_code_home` build on this to turn one provider back
    on with fully controlled fixture data.
    """
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "no_codex_here"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "no_claude_here"))
    monkeypatch.setattr("shutil.which", lambda _name: None)
    return tmp_path


@pytest.fixture
def codex_home(isolated_agents: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = isolated_agents / "codex_home"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


@pytest.fixture
def claude_code_home(isolated_agents: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = isolated_agents / "claude_home"
    (home / "projects").mkdir(parents=True)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(home))
    return home


def write_session_file(codex_home: Path, session_id: str, lines: list[str]) -> Path:
    day_dir = codex_home / "sessions" / "2026" / "01" / "01"
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"rollout-2026-01-01T00-00-00-{session_id}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- Claude Code fixture builders -------------------------------------------
#
# Field shapes verified against real local Claude Code 2.1.269 transcripts
# (~/.claude/projects/<sanitized-cwd>/<uuid>.jsonl), not guessed.


def claude_usage(
    *,
    input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    output_tokens: int = 0,
) -> dict:
    return {
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "output_tokens": output_tokens,
        "output_tokens_details": {"thinking_tokens": 0},
    }


def claude_text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def claude_thinking_block(text: str) -> dict:
    return {"type": "thinking", "thinking": text}


def claude_tool_use_block(tool_id: str, name: str, tool_input: dict) -> dict:
    return {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}


def claude_assistant_line(
    timestamp: datetime,
    session_id: str,
    message_id: str,
    content: list[dict],
    *,
    usage: dict | None = None,
    uuid_: str | None = None,
    parent_uuid: str | None = None,
    is_sidechain: bool = False,
    cwd: str = "/home/user/project",
) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "timestamp": iso(timestamp),
            "sessionId": session_id,
            "uuid": uuid_ or str(uuid.uuid4()),
            "parentUuid": parent_uuid,
            "isSidechain": is_sidechain,
            "userType": "external",
            "cwd": cwd,
            "gitBranch": "main",
            "version": "2.1.269",
            "entrypoint": "cli",
            "message": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5",
                "content": content,
                "stop_reason": "end_turn",
                "usage": usage if usage is not None else {},
            },
        }
    )


def claude_user_text_line(
    timestamp: datetime,
    session_id: str,
    text: str,
    *,
    uuid_: str | None = None,
    parent_uuid: str | None = None,
    cwd: str = "/home/user/project",
) -> str:
    return json.dumps(
        {
            "type": "user",
            "timestamp": iso(timestamp),
            "sessionId": session_id,
            "uuid": uuid_ or str(uuid.uuid4()),
            "parentUuid": parent_uuid,
            "isSidechain": False,
            "userType": "external",
            "cwd": cwd,
            "message": {"role": "user", "content": text},
        }
    )


def claude_tool_result_line(
    timestamp: datetime,
    session_id: str,
    tool_use_id: str,
    output_text: str,
    *,
    is_error: bool = False,
    uuid_: str | None = None,
    parent_uuid: str | None = None,
) -> str:
    return json.dumps(
        {
            "type": "user",
            "timestamp": iso(timestamp),
            "sessionId": session_id,
            "uuid": uuid_ or str(uuid.uuid4()),
            "parentUuid": parent_uuid,
            "isSidechain": False,
            "userType": "external",
            "cwd": "/home/user/project",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": output_text,
                        "is_error": is_error,
                    }
                ],
            },
            "toolUseResult": {
                "stdout": output_text,
                "stderr": "",
                "interrupted": False,
                "isImage": False,
            },
        }
    )


def claude_system_error_line(
    timestamp: datetime,
    session_id: str,
    error: str,
    *,
    uuid_: str | None = None,
) -> str:
    return json.dumps(
        {
            "type": "system",
            "subtype": "api_error",
            "level": "error",
            "timestamp": iso(timestamp),
            "sessionId": session_id,
            "uuid": uuid_ or str(uuid.uuid4()),
            "error": error,
            "retryAttempt": 1,
            "maxRetries": 3,
            "retryInMs": 1000,
            "source": "api",
            "userType": "external",
            "cwd": "/home/user/project",
        }
    )


def build_claude_turn(
    start: datetime,
    session_id: str,
    message_id: str,
    *,
    input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    output_tokens: int = 0,
    tool_name: str | None = None,
    tool_input: dict | None = None,
    tool_id: str | None = None,
    is_sidechain: bool = False,
) -> tuple[list[str], datetime]:
    """One assistant turn: a text block, optionally followed by a tool_use
    block and its matching tool_result -- mirroring how Claude Code explodes
    one API response into multiple same-message-id JSONL lines."""
    usage = claude_usage(
        input_tokens=input_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
        output_tokens=output_tokens,
    )
    lines = []
    t = start
    text_uuid = f"{message_id}-text"
    lines.append(
        claude_assistant_line(
            t,
            session_id,
            message_id,
            [claude_text_block("ok")],
            usage=usage,
            uuid_=text_uuid,
            is_sidechain=is_sidechain,
        )
    )
    if tool_name is not None:
        t = t + timedelta(milliseconds=10)
        tool_uuid = f"{message_id}-tool"
        resolved_tool_id = tool_id or f"toolu_{message_id}"
        lines.append(
            claude_assistant_line(
                t,
                session_id,
                message_id,
                [claude_tool_use_block(resolved_tool_id, tool_name, tool_input or {})],
                usage=usage,
                uuid_=tool_uuid,
                parent_uuid=text_uuid,
                is_sidechain=is_sidechain,
            )
        )
        t = t + timedelta(milliseconds=10)
        lines.append(
            claude_tool_result_line(
                t,
                session_id,
                resolved_tool_id,
                "ok",
                uuid_=f"{message_id}-result",
                parent_uuid=tool_uuid,
            )
        )
    t = t + timedelta(milliseconds=10)
    return lines, t


def write_claude_session_file(
    claude_code_home: Path,
    session_id: str,
    lines: list[str],
    *,
    project_dir: str = "-home-user-project",
) -> Path:
    day_dir = claude_code_home / "projects" / project_dir
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{session_id}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
