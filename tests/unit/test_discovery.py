import os
import time
from pathlib import Path

from agent_fuse.discovery import (
    FileTailer,
    SessionFileInfo,
    codex_home,
    discover_session_files,
    is_recent,
    session_id_hint_from_path,
)


def test_session_id_hint_extracts_uuid_from_filename() -> None:
    name = Path("rollout-2026-05-18T00-45-02-019e399d-c8cd-7032-b023-d0e69b1872fa.jsonl")
    assert session_id_hint_from_path(name) == "019e399d-c8cd-7032-b023-d0e69b1872fa"


def test_session_id_hint_falls_back_to_stem_without_uuid() -> None:
    name = Path("weird-file-name.jsonl")
    assert session_id_hint_from_path(name) == "weird-file-name"


def test_codex_home_respects_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "custom"))
    assert codex_home() == tmp_path / "custom"


def test_discover_session_files_empty_when_dir_missing(tmp_path) -> None:
    assert discover_session_files(tmp_path / "nope") == []


def test_discover_session_files_sorted_newest_first(tmp_path) -> None:
    d = tmp_path / "sessions"
    d.mkdir()
    old = d / "rollout-old.jsonl"
    new = d / "rollout-new.jsonl"
    old.write_text("{}")
    new.write_text("{}")
    now = time.time()
    os.utime(old, (now - 1000, now - 1000))
    os.utime(new, (now, now))
    files = discover_session_files(d)
    assert files[0].path == new
    assert files[1].path == old


def test_is_recent() -> None:
    now = time.time()
    info = SessionFileInfo(path=Path("x"), session_id_hint="x", mtime=now - 100, size=0)
    assert is_recent(info, now, recent_window_seconds=200) is True
    assert is_recent(info, now, recent_window_seconds=50) is False


def test_tailer_reads_incrementally_appended_lines(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text("")
    tailer = FileTailer(path)
    assert tailer.read_new_lines() == []

    with open(path, "a") as f:
        f.write('{"a": 1}\n')
    assert tailer.read_new_lines() == ['{"a": 1}']

    with open(path, "a") as f:
        f.write('{"b": 2}\n{"c": 3}\n')
    assert tailer.read_new_lines() == ['{"b": 2}', '{"c": 3}']
    tailer.close()


def test_tailer_buffers_partial_trailing_line(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text("")
    tailer = FileTailer(path)

    with open(path, "a") as f:
        f.write('{"complete": true}\n{"partial":')
    lines = tailer.read_new_lines()
    assert lines == ['{"complete": true}']

    with open(path, "a") as f:
        f.write(' "now finished"}\n')
    lines = tailer.read_new_lines()
    assert lines == ['{"partial": "now finished"}']
    tailer.close()


def test_tailer_handles_truncation_by_reopening_from_start(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text('{"line": 1}\n{"line": 2}\n')
    tailer = FileTailer(path)
    assert tailer.read_new_lines() == ['{"line": 1}', '{"line": 2}']

    # simulate truncation (e.g. a vendor bug/rotation) with new, shorter content
    path.write_text('{"line": "restarted"}\n')
    lines = tailer.read_new_lines()
    assert lines == ['{"line": "restarted"}']
    tailer.close()


def test_tailer_marks_gone_when_file_disappears(tmp_path) -> None:
    path = tmp_path / "session.jsonl"
    path.write_text('{"line": 1}\n')
    tailer = FileTailer(path)
    tailer.read_new_lines()
    path.unlink()
    lines = tailer.read_new_lines()
    assert lines == []
    assert tailer.gone is True
