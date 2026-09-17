"""Flags an excessive number of model responses within a rolling window."""

from __future__ import annotations

from datetime import datetime

from ..config import ResponseRateConfig
from ..metrics import SessionMetrics
from ..models import RuleResult


class ResponseRateRule:
    name = "response_rate"

    def __init__(self, config: ResponseRateConfig) -> None:
        self.config = config

    def evaluate(self, session: SessionMetrics, now: datetime) -> RuleResult:
        window = self.config.window_seconds
        threshold = self.config.max_responses
        observed = session.responses_in_window(now, window)
        triggered = observed >= threshold
        message = (
            f"{observed} responses in the last {window}s (threshold {threshold})"
            if triggered
            else f"{observed} responses in the last {window}s, below threshold {threshold}"
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
