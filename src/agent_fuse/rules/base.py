"""Rule interface. Rules are pure functions of (SessionMetrics, now) -> RuleResult.

Rules never touch the terminal, never know about Rich, and never see raw
vendor payloads -- they only see the aggregated numbers on SessionMetrics.
This keeps every rule independently unit-testable with a bare SessionMetrics
instance and a fixed `now`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ..metrics import SessionMetrics
from ..models import RuleResult


class FuseRule(Protocol):
    name: str

    def evaluate(self, session: SessionMetrics, now: datetime) -> RuleResult: ...
