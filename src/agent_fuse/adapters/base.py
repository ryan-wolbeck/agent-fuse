"""Adapter interface. Adapters are the only code that knows vendor schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..models import AgentEvent


@dataclass
class AdapterDiagnostics:
    """Counters exposed so `doctor`/`inspect` can report parser health
    without ever including the vendor payload that caused a problem.
    """

    lines_read: int = 0
    events_emitted: int = 0
    lines_malformed: int = 0
    events_unknown_type: int = 0
    errors_seen: int = 0
    last_error: str | None = None
    unknown_type_counts: dict[str, int] = field(default_factory=dict)

    def record_malformed(self, reason: str) -> None:
        self.lines_malformed += 1
        self.last_error = reason

    def record_unknown_type(self, type_name: str) -> None:
        self.events_unknown_type += 1
        self.unknown_type_counts[type_name] = self.unknown_type_counts.get(type_name, 0) + 1


class Adapter(Protocol):
    provider: str

    def parse_line(self, raw_line: str, *, session_id_hint: str) -> list[AgentEvent]: ...

    def flush(self, *, session_id_hint: str) -> list[AgentEvent]:
        """Return any buffered/correlated events once a session is finished."""
        ...

    @property
    def diagnostics(self) -> AdapterDiagnostics: ...
