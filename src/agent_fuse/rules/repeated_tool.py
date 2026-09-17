"""Flags the same normalized tool signature firing repeatedly in a short window.

The signature is an opaque hash produced by the adapter (see
adapters/codex.py `hash_signature`) -- this rule never sees raw tool
arguments, shell commands, or file contents, only how often a given hash
recurs.
"""

from __future__ import annotations

from datetime import datetime

from ..config import RepeatedToolCallConfig
from ..metrics import SessionMetrics
from ..models import RuleResult


class RepeatedToolCallRule:
    name = "repeated_tool_call"

    def __init__(self, config: RepeatedToolCallConfig) -> None:
        self.config = config

    def evaluate(self, session: SessionMetrics, now: datetime) -> RuleResult:
        window = self.config.window_seconds
        threshold = self.config.max_repetitions
        _sig, observed = session.max_repeated_tool_signature(now, window)
        triggered = observed >= threshold
        message = (
            f"a single tool signature repeated {observed} times in the last "
            f"{window}s (threshold {threshold})"
            if triggered
            else f"most-repeated tool signature seen {observed} times in the last "
            f"{window}s, below threshold {threshold}"
        )
        return RuleResult(
            rule=self.name,
            triggered=triggered,
            severity="warning",
            observed=observed,
            threshold=threshold,
            window_seconds=window,
            message=message,
        )
