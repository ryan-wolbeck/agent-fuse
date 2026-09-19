"""Locating and efficiently reading local agent session files.

All access here is read-only. Nothing in this module ever writes to,
truncates, or deletes anything under a vendor's home directory.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path

_UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


def codex_home() -> Path:
    override = os.environ.get("CODEX_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".codex"


def sessions_dir() -> Path:
    return codex_home() / "sessions"


def codex_binary_on_path() -> bool:
    return shutil.which("codex") is not None


def codex_installed() -> bool:
    """True if we have any evidence of a local Codex CLI installation.

    This never modifies anything -- it only checks for the presence of the
    home directory or the `codex` executable.
    """
    return codex_home().is_dir() or codex_binary_on_path()


def session_id_hint_from_path(path: Path) -> str:
    """Best-effort session identifier derived from the filename alone.

    Used only until (or unless) the real id is read from inside the file;
    falls back to the filename stem for files that don't contain a
    recognizable UUID (Codex's `rollout-<timestamp>-<uuid>.jsonl`, Claude
    Code's bare `<uuid>.jsonl`).
    """
    match = _UUID_RE.search(path.name)
    if match:
        return match.group(1)
    return path.stem


def claude_home() -> Path:
    override = os.environ.get("CLAUDE_CONFIG_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".claude"


def claude_projects_dir() -> Path:
    return claude_home() / "projects"


def claude_binary_on_path() -> bool:
    return shutil.which("claude") is not None


def claude_code_installed() -> bool:
    """True if we have any evidence of a local Claude Code installation.

    Read-only check: only looks for the home directory or the `claude`
    executable, never modifies anything.
    """
    return claude_home().is_dir() or claude_binary_on_path()


@dataclass(frozen=True)
class SessionFileInfo:
    path: Path
    session_id_hint: str
    mtime: float
    size: int


def _discover_jsonl_files(root: Path) -> list[SessionFileInfo]:
    """Stat (never read) every `*.jsonl` file under `root`, newest first."""
    if not root.is_dir():
        return []

    infos: list[SessionFileInfo] = []
    for path in root.rglob("*.jsonl"):
        try:
            stat = path.stat()
        except OSError:
            continue
        infos.append(
            SessionFileInfo(
                path=path,
                session_id_hint=session_id_hint_from_path(path),
                mtime=stat.st_mtime,
                size=stat.st_size,
            )
        )
    infos.sort(key=lambda info: info.mtime, reverse=True)
    return infos


def discover_session_files(root: Path | None = None) -> list[SessionFileInfo]:
    """List all Codex rollout files, newest first.

    Streaming-friendly: this only stats file metadata, it never opens or
    reads file contents.
    """
    return _discover_jsonl_files(root if root is not None else sessions_dir())


def discover_claude_code_session_files(root: Path | None = None) -> list[SessionFileInfo]:
    """List all Claude Code project transcript files, newest first.

    Note: the *parent directory* of each file encodes the sanitized project
    working-directory path (e.g. `/home/alice/proj` -> `-home-alice-proj`);
    only `path` (needed to open the file) and the UUID-based
    `session_id_hint` are returned here, and callers must not surface the
    parent directory name in any user-facing output.
    """
    return _discover_jsonl_files(root if root is not None else claude_projects_dir())


def is_recent(info: SessionFileInfo, now_epoch: float, recent_window_seconds: int = 1800) -> bool:
    return (now_epoch - info.mtime) <= recent_window_seconds


class FileTailer:
    """Incrementally reads complete lines appended to a growing file.

    Handles rotation (file replaced/truncated) by detecting an inode change
    or a size that shrank since the last read, and reopening from the
    start in that case. Partial trailing lines (a write still in progress)
    are buffered until a newline arrives rather than being parsed early.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fh: TextIOWrapper | None = None
        self._inode: int | None = None
        self._buffer = ""
        self._gone = False

    @property
    def gone(self) -> bool:
        return self._gone

    def _open_from_start(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(OSError):
                self._fh.close()
        self._fh = open(self.path, encoding="utf-8", errors="replace")  # noqa: SIM115
        self._buffer = ""
        try:
            self._inode = os.fstat(self._fh.fileno()).st_ino
        except OSError:
            self._inode = None

    def read_new_lines(self) -> list[str]:
        try:
            stat = self.path.stat()
        except OSError:
            self._gone = True
            return []

        if self._fh is None:
            try:
                self._open_from_start()
            except OSError:
                self._gone = True
                return []
        else:
            try:
                current_inode = stat.st_ino
            except OSError:
                current_inode = None
            rotated = current_inode is not None and current_inode != self._inode
            truncated = stat.st_size < self._fh.tell()
            if rotated or truncated:
                try:
                    self._open_from_start()
                except OSError:
                    self._gone = True
                    return []

        assert self._fh is not None
        try:
            chunk = self._fh.read()
        except OSError:
            self._gone = True
            return []

        if not chunk:
            return []

        self._buffer += chunk
        lines = self._buffer.split("\n")
        self._buffer = lines[-1]
        return lines[:-1]

    def close(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(OSError):
                self._fh.close()
            self._fh = None
