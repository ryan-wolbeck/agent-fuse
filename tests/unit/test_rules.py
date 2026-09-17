from datetime import UTC, datetime, timedelta

import pytest

from agent_fuse.config import (
    ContextReplayConfig,
    RepeatedToolCallConfig,
    ResponseRateConfig,
    TokenRateConfig,
)
from agent_fuse.metrics import SessionMetrics
from agent_fuse.models import AgentEvent, EventType
from agent_fuse.rules.context_replay import ContextReplayRule
from agent_fuse.rules.repeated_tool import RepeatedToolCallRule
from agent_fuse.rules.response_rate import ResponseRateRule
from agent_fuse.rules.token_rate import TokenRateRule

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def make_metrics() -> SessionMetrics:
    return SessionMetrics(session_id="s1", provider="codex", max_window_seconds=3600)


def add_responses(
    metrics: SessionMetrics, n: int, *, input_tokens=None, cached=None, output=None
) -> None:
    for i in range(n):
        metrics.record_event(
            AgentEvent(
                event_id=f"r{i}",
                session_id="s1",
                timestamp=T0 + timedelta(seconds=i),
                provider="codex",
                event_type=EventType.MODEL_RESPONSE,
                input_tokens=input_tokens,
                cached_input_tokens=cached,
                output_tokens=output,
            )
        )


def add_tool_calls(metrics: SessionMetrics, n: int, signature: str = "sig") -> None:
    for i in range(n):
        metrics.record_event(
            AgentEvent(
                event_id=f"t{i}",
                session_id="s1",
                timestamp=T0 + timedelta(seconds=i),
                provider="codex",
                event_type=EventType.TOOL_CALL,
                tool_name="shell",
                tool_signature=signature,
            )
        )


class TestResponseRateRule:
    def test_below_threshold_not_triggered(self) -> None:
        rule = ResponseRateRule(ResponseRateConfig(window_seconds=600, max_responses=100))
        metrics = make_metrics()
        add_responses(metrics, 99)
        result = rule.evaluate(metrics, T0 + timedelta(seconds=99))
        assert result.triggered is False
        assert result.observed == 99

    def test_at_threshold_triggers(self) -> None:
        rule = ResponseRateRule(ResponseRateConfig(window_seconds=600, max_responses=100))
        metrics = make_metrics()
        add_responses(metrics, 100)
        result = rule.evaluate(metrics, T0 + timedelta(seconds=100))
        assert result.triggered is True
        assert result.observed == 100
        assert result.threshold == 100

    def test_above_threshold_triggers(self) -> None:
        rule = ResponseRateRule(ResponseRateConfig(window_seconds=600, max_responses=100))
        metrics = make_metrics()
        add_responses(metrics, 150)
        result = rule.evaluate(metrics, T0 + timedelta(seconds=150))
        assert result.triggered is True

    def test_severity_is_always_warning_at_rule_level(self) -> None:
        rule = ResponseRateRule(ResponseRateConfig(window_seconds=600, max_responses=1))
        metrics = make_metrics()
        add_responses(metrics, 5)
        result = rule.evaluate(metrics, T0 + timedelta(seconds=5))
        assert result.severity == "warning"


class TestRepeatedToolCallRule:
    def test_below_threshold(self) -> None:
        rule = RepeatedToolCallRule(RepeatedToolCallConfig(window_seconds=300, max_repetitions=30))
        metrics = make_metrics()
        add_tool_calls(metrics, 29)
        result = rule.evaluate(metrics, T0 + timedelta(seconds=29))
        assert result.triggered is False
        assert result.observed == 29

    def test_at_threshold(self) -> None:
        rule = RepeatedToolCallRule(RepeatedToolCallConfig(window_seconds=300, max_repetitions=30))
        metrics = make_metrics()
        add_tool_calls(metrics, 30)
        result = rule.evaluate(metrics, T0 + timedelta(seconds=30))
        assert result.triggered is True

    def test_above_threshold(self) -> None:
        rule = RepeatedToolCallRule(RepeatedToolCallConfig(window_seconds=300, max_repetitions=30))
        metrics = make_metrics()
        add_tool_calls(metrics, 40)
        result = rule.evaluate(metrics, T0 + timedelta(seconds=40))
        assert result.triggered is True

    def test_insufficient_data_no_tool_calls(self) -> None:
        rule = RepeatedToolCallRule(RepeatedToolCallConfig(window_seconds=300, max_repetitions=30))
        metrics = make_metrics()
        result = rule.evaluate(metrics, T0)
        assert result.triggered is False
        assert result.observed == 0

    def test_different_signatures_do_not_combine(self) -> None:
        rule = RepeatedToolCallRule(RepeatedToolCallConfig(window_seconds=300, max_repetitions=30))
        metrics = make_metrics()
        add_tool_calls(metrics, 20, signature="sig-a")
        add_tool_calls(metrics, 20, signature="sig-b")
        result = rule.evaluate(metrics, T0 + timedelta(seconds=20))
        assert result.triggered is False
        assert result.observed == 20


class TestTokenRateRule:
    def test_no_token_data_is_insufficient_not_zero(self) -> None:
        rule = TokenRateRule(TokenRateConfig(window_seconds=600, max_input_tokens=2_000_000))
        metrics = make_metrics()
        add_responses(metrics, 5)  # responses with no token fields at all
        result = rule.evaluate(metrics, T0 + timedelta(seconds=5))
        assert result.triggered is False
        assert result.observed is None
        assert "no token usage data" in result.message

    def test_below_threshold(self) -> None:
        rule = TokenRateRule(TokenRateConfig(window_seconds=600, max_input_tokens=1000))
        metrics = make_metrics()
        add_responses(metrics, 1, input_tokens=999, cached=0, output=1)
        result = rule.evaluate(metrics, T0)
        assert result.triggered is False

    def test_at_threshold(self) -> None:
        rule = TokenRateRule(TokenRateConfig(window_seconds=600, max_input_tokens=1000))
        metrics = make_metrics()
        add_responses(metrics, 1, input_tokens=1000, cached=0, output=1)
        result = rule.evaluate(metrics, T0)
        assert result.triggered is True

    def test_above_threshold(self) -> None:
        rule = TokenRateRule(TokenRateConfig(window_seconds=600, max_input_tokens=1000))
        metrics = make_metrics()
        add_responses(metrics, 1, input_tokens=5000, cached=0, output=1)
        result = rule.evaluate(metrics, T0)
        assert result.triggered is True


class TestContextReplayRule:
    def test_insufficient_scale_not_triggered_even_with_high_ratio(self) -> None:
        rule = ContextReplayRule(
            ContextReplayConfig(minimum_input_tokens=100_000, max_cached_ratio=0.9)
        )
        metrics = make_metrics()
        add_responses(metrics, 1, input_tokens=1000, cached=999, output=1)
        result = rule.evaluate(metrics, T0)
        assert result.triggered is False
        assert "insufficient scale" in result.message

    def test_high_scale_low_ratio_not_triggered(self) -> None:
        rule = ContextReplayRule(
            ContextReplayConfig(minimum_input_tokens=100_000, max_cached_ratio=0.9)
        )
        metrics = make_metrics()
        add_responses(metrics, 1, input_tokens=200_000, cached=100_000, output=1)
        result = rule.evaluate(metrics, T0)
        assert result.triggered is False
        assert result.observed == pytest.approx(0.5)

    def test_at_threshold_ratio_triggers(self) -> None:
        rule = ContextReplayRule(
            ContextReplayConfig(minimum_input_tokens=100_000, max_cached_ratio=0.9)
        )
        metrics = make_metrics()
        add_responses(metrics, 1, input_tokens=100_000, cached=90_000, output=1)
        result = rule.evaluate(metrics, T0)
        assert result.triggered is True

    def test_caching_alone_is_not_penalized_below_scale_floor(self) -> None:
        rule = ContextReplayRule(
            ContextReplayConfig(minimum_input_tokens=100_000, max_cached_ratio=0.9)
        )
        metrics = make_metrics()
        add_responses(
            metrics, 1, input_tokens=50_000, cached=50_000, output=1
        )  # 100% cached, small
        result = rule.evaluate(metrics, T0)
        assert result.triggered is False

    def test_is_lifetime_not_windowed(self) -> None:
        rule = ContextReplayRule(
            ContextReplayConfig(minimum_input_tokens=100_000, max_cached_ratio=0.9)
        )
        metrics = make_metrics()
        result = rule.evaluate(metrics, T0)
        assert result.window_seconds is None
