"""Builds enabled rules from config and evaluates them against a session."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .adapters.base import Adapter, AdapterDiagnostics
from .config import FuseConfig
from .metrics import SessionMetrics
from .models import EventType, FuseEvent, RuleResult
from .rules import (
    ContextReplayRule,
    FuseRule,
    RepeatedToolCallRule,
    ResponseRateRule,
    TokenRateRule,
)


class RuleEngine:
    def __init__(self, config: FuseConfig) -> None:
        self.config = config
        self.rules: list[FuseRule] = []
        if config.response_rate.enabled:
            self.rules.append(ResponseRateRule(config.response_rate))
        if config.token_rate.enabled:
            self.rules.append(TokenRateRule(config.token_rate))
        if config.context_replay.enabled:
            self.rules.append(ContextReplayRule(config.context_replay))
        if config.repeated_tool_call.enabled:
            self.rules.append(RepeatedToolCallRule(config.repeated_tool_call))

    def evaluate(self, session: SessionMetrics, now: datetime) -> list[RuleResult]:
        return [rule.evaluate(session, now) for rule in self.rules]

    @staticmethod
    def triggered(results: list[RuleResult]) -> list[RuleResult]:
        return [r for r in results if r.triggered]

    @staticmethod
    def session_severity(results: list[RuleResult]) -> str | None:
        """Two or more rules tripping at once escalates warning -> severe.

        This is a simple, deterministic, and honestly-documented heuristic
        (not a statistically validated risk score): a single crossed
        threshold is a warning worth a look; multiple independent
        thresholds crossed at the same moment is a stronger signal.
        """
        triggered = RuleEngine.triggered(results)
        if not triggered:
            return None
        return "severe" if len(triggered) >= 2 else "warning"

    @staticmethod
    def to_fuse_events(
        provider: str,
        session_id: str,
        now: datetime,
        results: list[RuleResult],
    ) -> list[FuseEvent]:
        events = []
        for result in RuleEngine.triggered(results):
            events.append(
                FuseEvent(
                    timestamp=now,
                    provider=provider,
                    session_id=session_id,
                    rule=result.rule,
                    severity=result.severity,
                    window_seconds=result.window_seconds,
                    observed=result.observed,
                    threshold=result.threshold,
                    message=result.message,
                )
            )
        return events


@dataclass
class SessionScanResult:
    """Outcome of replaying one session file front to back.

    `worst_results` captures the rule evaluation at whichever point in the
    session's own timeline had the most rules triggered simultaneously --
    this is what historical `scan` and `inspect` use to report a session's
    status, since (unlike `watch`) there is no single "now" to evaluate at.
    """

    session_id: str
    metrics: SessionMetrics
    worst_results: list[RuleResult]
    severity: str | None
    diagnostics: AdapterDiagnostics


def scan_session_file(
    path: Path,
    session_id_hint: str,
    config: FuseConfig,
    make_adapter: Callable[[str], Adapter],
) -> SessionScanResult:
    """Replay one provider's session file, evaluating rules as of each
    event's own timestamp -- i.e. reconstructing what `watch` would have
    reported had it been running at the time. This is what makes `scan`
    useful on sessions that finished days or weeks ago.

    `make_adapter` is any provider's adapter constructor (e.g.
    `CodexAdapter`, `ClaudeCodeAdapter`) -- the engine itself has no
    vendor-specific knowledge, by design.
    """
    adapter = make_adapter(session_id_hint)
    window = config.max_window_seconds()
    metrics = SessionMetrics(
        session_id=session_id_hint, provider=adapter.provider, max_window_seconds=window
    )
    rule_engine = RuleEngine(config)

    worst_results: list[RuleResult] = []
    worst_triggered_count = -1

    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for raw_line in fh:
                for event in adapter.parse_line(raw_line, session_id_hint=session_id_hint):
                    if event.event_type is EventType.SESSION_START:
                        metrics.session_id = event.session_id
                    metrics.record_event(event)
                    if event.event_type in (EventType.MODEL_RESPONSE, EventType.TOOL_CALL):
                        results = rule_engine.evaluate(metrics, event.timestamp)
                        count = len(RuleEngine.triggered(results))
                        if count > worst_triggered_count:
                            worst_triggered_count = count
                            worst_results = results
    except OSError:
        pass

    severity = RuleEngine.session_severity(worst_results) if worst_results else None
    return SessionScanResult(
        session_id=metrics.session_id,
        metrics=metrics,
        worst_results=worst_results,
        severity=severity,
        diagnostics=adapter.diagnostics,
    )
