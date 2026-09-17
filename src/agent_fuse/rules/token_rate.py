"""Flags excessive input token consumption within a rolling window.

Only fires when the provider actually reports token usage -- if no token
data is available for a session, this rule reports insufficient data rather
than treating unknown as zero.
"""

from __future__ import annotations

from datetime import datetime

from ..config import TokenRateConfig
from ..metrics import SessionMetrics
from ..models import RuleResult


class TokenRateRule:
    name = "token_rate"

    def __init__(self, config: TokenRateConfig) -> None:
        self.config = config

    def evaluate(self, session: SessionMetrics, now: datetime) -> RuleResult:
        window = self.config.window_seconds
        threshold = self.config.max_input_tokens

        totals = session.token_totals_in_window(now, window)
        if totals is None:
            return RuleResult(
                rule=self.name,
                triggered=False,
                severity="warning",
                observed=None,
                threshold=threshold,
                window_seconds=window,
                message="no token usage data reported by provider for this session",
            )

        input_tokens, _cached, _output = totals
        triggered = input_tokens >= threshold
        message = (
            f"{input_tokens:,} input tokens in the last {window}s (threshold {threshold:,})"
            if triggered
            else f"{input_tokens:,} input tokens in the last {window}s, "
            f"below threshold {threshold:,}"
        )
        return RuleResult(
            rule=self.name,
            triggered=triggered,
            severity="warning",
            observed=input_tokens,
            threshold=threshold,
            window_seconds=window,
            message=message,
        )
