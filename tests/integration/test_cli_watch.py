import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta

from tests.conftest import (
    build_claude_turn,
    build_turn,
    claude_user_text_line,
    session_meta_line,
    write_claude_session_file,
    write_session_file,
)

T0_WALL = datetime.now(UTC) - timedelta(minutes=2)


def test_watch_detects_runaway_session_and_clean_ctrl_c(
    tmp_path, codex_home, new_session_id
) -> None:
    lines = [session_meta_line(new_session_id, T0_WALL)]
    t = T0_WALL
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

    env = dict(os.environ)
    env["CODEX_HOME"] = str(codex_home)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_fuse",
            "watch",
            "--format",
            "jsonl",
            "--quiet",
            "--poll-seconds",
            "0.1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        saw_trigger = False
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            payload = json.loads(line)
            assert payload["event"] == "fuse.triggered"
            assert payload["schema_version"] == 1
            saw_trigger = True
            break
        assert saw_trigger, "watch did not emit a fuse.triggered event in time"
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            _, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
            raise AssertionError("watch did not exit cleanly after SIGINT") from None

    assert "Traceback" not in stderr


def test_watch_detects_runaway_claude_code_session_and_clean_ctrl_c(
    tmp_path, claude_code_home, new_session_id
) -> None:
    lines = [claude_user_text_line(T0_WALL, new_session_id, "hello")]
    t = T0_WALL
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

    env = dict(os.environ)
    env["CLAUDE_CONFIG_DIR"] = str(claude_code_home)

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "agent_fuse",
            "watch",
            "--format",
            "jsonl",
            "--quiet",
            "--poll-seconds",
            "0.1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    try:
        deadline = time.monotonic() + 15
        saw_trigger = False
        while time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            payload = json.loads(line)
            assert payload["event"] == "fuse.triggered"
            assert payload["provider"] == "claude_code"
            saw_trigger = True
            break
        assert saw_trigger, "watch did not emit a fuse.triggered event in time"
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            _, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate()
            raise AssertionError("watch did not exit cleanly after SIGINT") from None

    assert "Traceback" not in stderr


def test_watch_no_sessions_idles_without_crashing(codex_home) -> None:
    env = dict(os.environ)
    env["CODEX_HOME"] = str(codex_home)

    proc = subprocess.Popen(
        [sys.executable, "-m", "agent_fuse", "watch", "--poll-seconds", "0.1"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
    )
    time.sleep(1.5)
    proc.send_signal(signal.SIGINT)
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        raise AssertionError("watch did not exit cleanly after SIGINT") from None

    assert "Traceback" not in stderr
    assert "Stopped watching." in stdout
