from datetime import UTC, datetime, timedelta

from agent_fuse.metrics import MetricsRegistry, SessionMetrics
from agent_fuse.models import AgentEvent, EventType

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def response_event(
    ts: datetime, *, input_tokens=None, cached=None, output=None, idx=0
) -> AgentEvent:
    return AgentEvent(
        event_id=f"r{idx}",
        session_id="s1",
        timestamp=ts,
        provider="codex",
        event_type=EventType.MODEL_RESPONSE,
        input_tokens=input_tokens,
        cached_input_tokens=cached,
        output_tokens=output,
    )


def tool_call_event(ts: datetime, signature: str, idx: int = 0) -> AgentEvent:
    return AgentEvent(
        event_id=f"t{idx}",
        session_id="s1",
        timestamp=ts,
        provider="codex",
        event_type=EventType.TOOL_CALL,
        tool_name="shell",
        tool_signature=signature,
    )


def test_responses_in_window_counts_only_recent() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    for i in range(5):
        metrics.record_event(response_event(T0 + timedelta(seconds=i), idx=i))
    assert metrics.responses_in_window(T0 + timedelta(seconds=4), 600) == 5
    assert metrics.total_responses == 5


def test_old_events_are_evicted_from_window_but_not_lifetime() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=100)
    metrics.record_event(response_event(T0, idx=0))
    later = T0 + timedelta(seconds=200)
    metrics.record_event(response_event(later, idx=1))
    # first event is outside the 100s window relative to `later`
    assert metrics.responses_in_window(later, 100) == 1
    assert metrics.total_responses == 2


def test_token_totals_in_window_sum_and_expire() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    metrics.record_event(response_event(T0, input_tokens=100, cached=90, output=10, idx=0))
    metrics.record_event(
        response_event(T0 + timedelta(seconds=10), input_tokens=200, cached=150, output=20, idx=1)
    )
    totals = metrics.token_totals_in_window(T0 + timedelta(seconds=10), 600)
    assert totals == (300, 240, 30)
    assert metrics.total_input_tokens == 300
    assert metrics.total_cached_input_tokens == 240


def test_token_totals_none_when_no_token_data_reported() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    metrics.record_event(response_event(T0, idx=0))
    assert metrics.token_totals_in_window(T0, 600) is None
    assert metrics.lifetime_cached_ratio() is None


def test_lifetime_cached_ratio() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    metrics.record_event(response_event(T0, input_tokens=1000, cached=900, output=10, idx=0))
    assert metrics.lifetime_cached_ratio() == 0.9


def test_max_repeated_tool_signature_tracks_most_common_in_window() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    for i in range(10):
        metrics.record_event(tool_call_event(T0 + timedelta(seconds=i), "sig-a", idx=i))
    for i in range(3):
        metrics.record_event(tool_call_event(T0 + timedelta(seconds=i), "sig-b", idx=100 + i))
    sig, count = metrics.max_repeated_tool_signature(T0 + timedelta(seconds=9), 600)
    assert sig == "sig-a"
    assert count == 10
    assert metrics.total_tool_calls == 13


def test_repeated_tool_signature_eviction_decrements_counter() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=5)
    metrics.record_event(tool_call_event(T0, "sig-a", idx=0))
    later = T0 + timedelta(seconds=10)
    metrics.record_event(tool_call_event(later, "sig-a", idx=1))
    # the first sig-a call is now outside the 5s window
    _, count = metrics.max_repeated_tool_signature(later, 5)
    assert count == 1


def test_duration_seconds() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    metrics.record_event(response_event(T0, idx=0))
    metrics.record_event(response_event(T0 + timedelta(seconds=30), idx=1))
    assert metrics.duration_seconds() == 30.0


def test_duration_none_when_no_events() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=600)
    assert metrics.duration_seconds() is None


def test_registry_creates_and_reuses_session_metrics() -> None:
    registry = MetricsRegistry(max_window_seconds=600)
    event = response_event(T0, idx=0)
    m1 = registry.record(event)
    m2 = registry.get("s1")
    assert m1 is m2
    assert len(registry) == 1


def test_registry_drop_removes_session() -> None:
    registry = MetricsRegistry(max_window_seconds=600)
    registry.record(response_event(T0, idx=0))
    registry.drop("s1")
    assert registry.get("s1") is None
    assert len(registry) == 0
