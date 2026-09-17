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
def codex_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "codex_home"
    (home / "sessions").mkdir(parents=True)
    monkeypatch.setenv("CODEX_HOME", str(home))
    return home


def write_session_file(codex_home: Path, session_id: str, lines: list[str]) -> Path:
    day_dir = codex_home / "sessions" / "2026" / "01" / "01"
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"rollout-2026-01-01T00-00-00-{session_id}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
