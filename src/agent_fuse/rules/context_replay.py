"""Flags an unusual combination of scale and cache-replay ratio.

This is intentionally a *lifetime* session metric, not a rolling window:
a high cached-token ratio is normal and desirable in most agent sessions
(that's what prompt caching is for). It only becomes a signal worth a
human's attention when combined with genuinely large cumulative scale
(`minimum_input_tokens`) -- a short session that happens to be mostly
cached input is not "runaway", it's just cheap. Caching itself is never
the thing being flagged.
"""

from __future__ import annotations

from datetime import datetime

from ..config import ContextReplayConfig
from ..metrics import SessionMetrics
from ..models import RuleResult


class ContextReplayRule:
    name = "context_replay"

    def __init__(self, config: ContextReplayConfig) -> None:
        self.config = config

    def evaluate(self, session: SessionMetrics, now: datetime) -> RuleResult:
        del now  # lifetime metric; not windowed
        threshold = self.config.max_cached_ratio
        min_tokens = self.config.minimum_input_tokens

        if not session.has_token_data or session.total_input_tokens < min_tokens:
            observed = session.lifetime_cached_ratio()
            return RuleResult(
                rule=self.name,
                triggered=False,
                severity="warning",
                observed=observed,
                threshold=threshold,
                window_seconds=None,
                message=(
                    f"insufficient scale to evaluate ({session.total_input_tokens:,} input "
                    f"tokens, minimum {min_tokens:,})"
                ),
            )

        ratio = session.lifetime_cached_ratio()
        assert ratio is not None  # has_token_data guarantees this
        triggered = ratio >= threshold
        pct = ratio * 100
        threshold_pct = threshold * 100
        message = (
            f"{pct:.1f}% cached/replayed input across {session.total_input_tokens:,} "
            f"input tokens (threshold {threshold_pct:.0f}%)"
            if triggered
            else f"{pct:.1f}% cached/replayed input, below threshold {threshold_pct:.0f}%"
        )
        return RuleResult(
            rule=self.name,
            triggered=triggered,
            severity="warning",
            observed=ratio,
            threshold=threshold,
            window_seconds=None,
            message=message,
        )
