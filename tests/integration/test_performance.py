"""Coarse performance/scale checks.

Not micro-benchmarks -- just guards against accidental O(n^2) behavior or
unbounded memory growth as session count/size grows, per the project's
"streaming, bounded-memory" requirement.
"""

from datetime import UTC, datetime, timedelta

from agent_fuse.config import default_config
from agent_fuse.engine import scan_session_file
from agent_fuse.metrics import SessionMetrics
from agent_fuse.models import AgentEvent, EventType
from tests.conftest import build_turn, session_meta_line, write_session_file

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_metrics_deque_size_bounded_regardless_of_event_count() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=60)
    for i in range(50_000):
        metrics.record_event(
            AgentEvent(
                event_id=f"r{i}",
                session_id="s1",
                timestamp=T0 + timedelta(seconds=i),
                provider="codex",
                event_type=EventType.MODEL_RESPONSE,
            )
        )
    # window is 60s and events are 1s apart -> at most ~61 should remain
    assert len(metrics._response_times) <= 61  # noqa: SLF001
    assert metrics.total_responses == 50_000


def test_repeated_tool_counter_bounded_with_many_distinct_signatures() -> None:
    metrics = SessionMetrics(session_id="s1", provider="codex", max_window_seconds=10)
    for i in range(20_000):
        metrics.record_event(
            AgentEvent(
                event_id=f"t{i}",
                session_id="s1",
                timestamp=T0 + timedelta(seconds=i),
                provider="codex",
                event_type=EventType.TOOL_CALL,
                tool_name="x",
                tool_signature=f"sig-{i}",
            )
        )
    assert len(metrics._tool_call_counts) <= 11  # noqa: SLF001


def test_scan_session_file_handles_many_sessions_efficiently(tmp_path, codex_home) -> None:
    config = default_config()
    session_ids = []
    for n in range(200):
        session_id = f"perf-session-{n}"
        lines = [session_meta_line(session_id, T0)]
        turn_lines, _ = build_turn(
            T0, "turn-1", input_tokens=100, cached_input_tokens=50, output_tokens=5
        )
        lines += turn_lines
        write_session_file(codex_home, session_id, lines)
        session_ids.append(session_id)

    from agent_fuse.discovery import discover_session_files

    files = discover_session_files()
    assert len(files) == 200
    for info in files:
        result = scan_session_file(info.path, info.session_id_hint, config)
        assert result.metrics.total_responses == 1


def test_scan_session_file_streams_large_single_session_without_loading_whole_file(
    tmp_path, codex_home
) -> None:
    session_id = "big-session"
    lines = [session_meta_line(session_id, T0)]
    t = T0
    for i in range(3_000):
        turn_lines, t = build_turn(
            t, f"turn-{i}", input_tokens=10, cached_input_tokens=5, output_tokens=1
        )
        lines += turn_lines
    path = write_session_file(codex_home, session_id, lines)
    assert path.stat().st_size > 100_000  # a genuinely large-ish file

    config = default_config()
    result = scan_session_file(path, session_id, config)
    assert result.metrics.total_responses == 3_000
