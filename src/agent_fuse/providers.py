"""Registry of supported agent providers.

This is the one place that knows both "which adapters exist" and "how to
find each one's session files". Adding a new provider means adding one
`ProviderSpec` here -- `metrics.py`, `engine.py`, `rules/`, and `output.py`
never need to change, and `cli.py` only ever loops over `PROVIDERS`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import discovery
from .adapters.base import Adapter
from .adapters.claude_code import ClaudeCodeAdapter
from .adapters.codex import CodexAdapter
from .discovery import SessionFileInfo


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    display_name: str
    installed: Callable[[], bool]
    discover: Callable[[], list[SessionFileInfo]]
    make_adapter: Callable[[str], Adapter]


PROVIDERS: list[ProviderSpec] = [
    ProviderSpec(
        name="codex",
        display_name="Codex",
        installed=discovery.codex_installed,
        discover=discovery.discover_session_files,
        make_adapter=CodexAdapter,
    ),
    ProviderSpec(
        name="claude_code",
        display_name="Claude Code",
        installed=discovery.claude_code_installed,
        discover=discovery.discover_claude_code_session_files,
        make_adapter=ClaudeCodeAdapter,
    ),
]


def installed_providers() -> list[ProviderSpec]:
    return [p for p in PROVIDERS if p.installed()]
