from datetime import UTC, datetime

from agent_fuse.models import SCHEMA_VERSION, AgentEvent, EventType, FuseEvent, RuleResult


def test_agent_event_missing_fields_stay_none() -> None:
    event = AgentEvent(
        event_id="e1",
        session_id="s1",
        timestamp=datetime.now(UTC),
        provider="codex",
        event_type=EventType.MODEL_RESPONSE,
    )
    assert event.input_tokens is None
    assert event.cached_input_tokens is None
    assert event.output_tokens is None
    assert event.tool_name is None


def test_agent_event_zero_is_distinct_from_none() -> None:
    event = AgentEvent(
        event_id="e1",
        session_id="s1",
        timestamp=datetime.now(UTC),
        provider="codex",
        event_type=EventType.MODEL_RESPONSE,
        output_tokens=0,
    )
    assert event.output_tokens == 0
    assert event.input_tokens is None


def test_fuse_event_schema_version_default() -> None:
    event = FuseEvent(
        timestamp=datetime.now(UTC),
        provider="codex",
        session_id="s1",
        rule="response_rate",
        severity="warning",
        window_seconds=600,
        observed=184,
        threshold=100,
        message="184 responses",
    )
    assert event.schema_version == SCHEMA_VERSION
    assert event.event == "fuse.triggered"
    dumped = event.model_dump()
    assert dumped["schema_version"] == 1
    assert set(dumped.keys()) == {
        "schema_version",
        "event",
        "timestamp",
        "provider",
        "session_id",
        "rule",
        "severity",
        "window_seconds",
        "observed",
        "threshold",
        "message",
    }


def test_rule_result_allows_none_observed_for_insufficient_data() -> None:
    result = RuleResult(
        rule="token_rate",
        triggered=False,
        severity="warning",
        observed=None,
        threshold=2_000_000,
        window_seconds=600,
        message="no data",
    )
    assert result.observed is None
    assert result.triggered is False
