from datetime import UTC, datetime

from agent_fuse.config import FuseConfig
from agent_fuse.engine import RuleEngine
from agent_fuse.metrics import SessionMetrics
from agent_fuse.models import AgentEvent, EventType

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_disabled_rule_is_not_built() -> None:
    config = FuseConfig()
    config.response_rate.enabled = False
    engine = RuleEngine(config)
    names = {r.name for r in engine.rules}
    assert "response_rate" not in names
    assert "token_rate" in names


def test_session_severity_none_when_nothing_triggered() -> None:
    config = FuseConfig()
    engine = RuleEngine(config)
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    results = engine.evaluate(metrics, T0)
    assert RuleEngine.session_severity(results) is None


def test_session_severity_warning_for_single_trigger() -> None:
    config = FuseConfig()
    config.response_rate.max_responses = 1
    engine = RuleEngine(config)
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    metrics.record_event(
        AgentEvent(
            event_id="r0",
            session_id="s1",
            timestamp=T0,
            provider="codex",
            event_type=EventType.MODEL_RESPONSE,
        )
    )
    results = engine.evaluate(metrics, T0)
    assert RuleEngine.session_severity(results) == "warning"


def test_session_severity_severe_for_multiple_triggers() -> None:
    config = FuseConfig()
    config.response_rate.max_responses = 1
    config.token_rate.max_input_tokens = 1
    engine = RuleEngine(config)
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    metrics.record_event(
        AgentEvent(
            event_id="r0",
            session_id="s1",
            timestamp=T0,
            provider="codex",
            event_type=EventType.MODEL_RESPONSE,
            input_tokens=5,
            cached_input_tokens=0,
            output_tokens=1,
        )
    )
    results = engine.evaluate(metrics, T0)
    assert RuleEngine.session_severity(results) == "severe"


def test_to_fuse_events_only_includes_triggered() -> None:
    config = FuseConfig()
    config.response_rate.max_responses = 1
    engine = RuleEngine(config)
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    metrics.record_event(
        AgentEvent(
            event_id="r0",
            session_id="s1",
            timestamp=T0,
            provider="codex",
            event_type=EventType.MODEL_RESPONSE,
        )
    )
    results = engine.evaluate(metrics, T0)
    events = RuleEngine.to_fuse_events("codex", "s1", T0, results)
    assert len(events) == 1
    assert events[0].rule == "response_rate"
    assert events[0].schema_version == 1
