import json
from datetime import UTC, datetime, timedelta

from agent_fuse.adapters.codex import CodexAdapter, hash_signature
from agent_fuse.models import EventType
from tests.conftest import (
    build_turn,
    error_line,
    function_call_line,
    function_call_output_line,
    local_shell_call_line,
    message_line,
    session_meta_line,
    task_complete_line,
    task_started_line,
    token_count_line,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def parse_all(adapter: CodexAdapter, lines: list[str], session_id_hint: str = "hint"):
    events = []
    for line in lines:
        events.extend(adapter.parse_line(line, session_id_hint=session_id_hint))
    return events


def test_session_meta_emits_session_start_with_real_id() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(adapter, [session_meta_line("real-id-123", T0)])
    assert len(events) == 1
    assert events[0].event_type is EventType.SESSION_START
    assert events[0].session_id == "real-id-123"
    assert adapter.session_id == "real-id-123"


def test_session_meta_does_not_leak_cwd_or_git_info() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(adapter, [session_meta_line("s1", T0, cwd="/home/alice/secret-client-repo")])
    dumped = json.dumps(events[0].model_dump(mode="json"))
    assert "secret-client-repo" not in dumped
    assert "alice" not in dumped


def test_full_turn_emits_one_model_response_with_tokens() -> None:
    adapter = CodexAdapter("hint")
    lines, _ = build_turn(
        T0, "turn-1", input_tokens=1000, cached_input_tokens=900, output_tokens=50
    )
    events = parse_all(adapter, lines)
    responses = [e for e in events if e.event_type is EventType.MODEL_RESPONSE]
    assert len(responses) == 1
    assert responses[0].input_tokens == 1000
    assert responses[0].cached_input_tokens == 900
    assert responses[0].output_tokens == 50


def test_incomplete_turn_with_no_task_complete_emits_nothing() -> None:
    adapter = CodexAdapter("hint")
    lines = [
        task_started_line(T0, "turn-1"),
        token_count_line(T0, input_tokens=100, cached_input_tokens=50, output_tokens=5),
    ]
    events = parse_all(adapter, lines)
    assert events == []
    assert adapter.flush(session_id_hint="hint") == []


def test_turn_with_no_token_count_reports_none_not_zero() -> None:
    adapter = CodexAdapter("hint")
    lines = [task_started_line(T0, "turn-1"), task_complete_line(T0, "turn-1")]
    events = parse_all(adapter, lines)
    assert len(events) == 1
    assert events[0].input_tokens is None
    assert events[0].output_tokens is None


def test_stray_token_count_between_turns_does_not_leak_into_next_turn_if_overwritten() -> None:
    adapter = CodexAdapter("hint")
    lines = [
        task_started_line(T0, "turn-1"),
        token_count_line(T0, input_tokens=10, cached_input_tokens=5, output_tokens=1),
        task_complete_line(T0, "turn-1"),
        # stray token_count with no owning turn
        token_count_line(T0, input_tokens=999, cached_input_tokens=999, output_tokens=999),
        task_started_line(T0, "turn-2"),
        token_count_line(T0, input_tokens=20, cached_input_tokens=15, output_tokens=2),
        task_complete_line(T0, "turn-2"),
    ]
    events = parse_all(adapter, lines)
    responses = [e for e in events if e.event_type is EventType.MODEL_RESPONSE]
    assert len(responses) == 2
    assert responses[1].input_tokens == 20  # not clobbered permanently by the stray value


def test_repeated_identical_usage_across_turns_counts_as_two_responses() -> None:
    """A real Codex quirk: after a thread rollback, two different completed
    turns can report byte-identical last_token_usage. Counting on
    task_complete (not value equality) must still see two responses."""
    adapter = CodexAdapter("hint")
    lines = []
    lines += [
        task_started_line(T0, "turn-1"),
        token_count_line(T0, input_tokens=42, cached_input_tokens=40, output_tokens=1),
        task_complete_line(T0, "turn-1"),
        task_started_line(T0, "turn-2"),
        token_count_line(T0, input_tokens=42, cached_input_tokens=40, output_tokens=1),
        task_complete_line(T0, "turn-2"),
    ]
    events = parse_all(adapter, lines)
    responses = [e for e in events if e.event_type is EventType.MODEL_RESPONSE]
    assert len(responses) == 2


def test_function_call_and_output_link_via_parent_event_id() -> None:
    adapter = CodexAdapter("hint")
    lines = [
        function_call_line(T0, "call-1", "read_file", '{"path": "/etc/passwd"}'),
        function_call_output_line(T0, "call-1", "read_file", "file contents here"),
    ]
    events = parse_all(adapter, lines)
    call = next(e for e in events if e.event_type is EventType.TOOL_CALL)
    result = next(e for e in events if e.event_type is EventType.TOOL_RESULT)
    assert result.parent_event_id == call.event_id


def test_function_call_arguments_never_appear_in_event() -> None:
    adapter = CodexAdapter("hint")
    lines = [
        function_call_line(
            T0, "call-1", "run_shell", '{"cmd": "cat /etc/shadow --secret-token=abc123"}'
        )
    ]
    events = parse_all(adapter, lines)
    dumped = json.dumps(events[0].model_dump(mode="json"))
    assert "shadow" not in dumped
    assert "secret-token" not in dumped
    assert "abc123" not in dumped
    assert events[0].tool_signature is not None


def test_local_shell_call_never_leaks_command_env_or_cwd() -> None:
    adapter = CodexAdapter("hint")
    lines = [
        local_shell_call_line(
            T0,
            "call-1",
            command=["curl", "https://user:hunter2@internal.example.com/api"],
            env={"OPENAI_API_KEY": "sk-should-not-leak", "AWS_SECRET_ACCESS_KEY": "also-secret"},
            working_directory="/home/alice/private-repo",
        )
    ]
    events = parse_all(adapter, lines)
    dumped = json.dumps(events[0].model_dump(mode="json"))
    assert "hunter2" not in dumped
    assert "sk-should-not-leak" not in dumped
    assert "also-secret" not in dumped
    assert "alice" not in dumped
    assert "private-repo" not in dumped
    assert "internal.example.com" not in dumped


def test_same_tool_call_repeated_produces_same_signature() -> None:
    adapter = CodexAdapter("hint")
    lines = [local_shell_call_line(T0, f"call-{i}", command=["git", "status"]) for i in range(3)]
    events = parse_all(adapter, lines)
    signatures = {e.tool_signature for e in events}
    assert len(signatures) == 1


def test_different_commands_produce_different_signatures() -> None:
    adapter = CodexAdapter("hint")
    lines = [
        local_shell_call_line(T0, "call-1", command=["git", "status"]),
        local_shell_call_line(T0, "call-2", command=["rm", "-rf", "/"]),
    ]
    events = parse_all(adapter, lines)
    assert events[0].tool_signature != events[1].tool_signature


def test_error_event_does_not_retain_message_text() -> None:
    adapter = CodexAdapter("hint")
    lines = [error_line(T0, "connection failed: token=SECRETVALUE for user alice@example.com")]
    events = parse_all(adapter, lines)
    assert len(events) == 1
    assert events[0].event_type is EventType.ERROR
    dumped = json.dumps(events[0].model_dump(mode="json"))
    assert "SECRETVALUE" not in dumped
    assert "alice@example.com" not in dumped


def test_message_content_is_never_extracted() -> None:
    adapter = CodexAdapter("hint")
    lines = [message_line(T0, "user", "here is my API key: sk-abc123 and my SSN 123-45-6789")]
    events = parse_all(adapter, lines)
    assert events == []  # message content is recognized-but-ignored, never emitted


def test_malformed_json_line_is_skipped_not_fatal() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(adapter, ['{"not": "closed"', session_meta_line("s1", T0)])
    assert adapter.diagnostics.lines_malformed == 1
    assert len(events) == 1  # the well-formed line still parses


def test_empty_lines_are_skipped_silently() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(adapter, ["", "   ", session_meta_line("s1", T0)])
    assert len(events) == 1
    assert adapter.diagnostics.lines_malformed == 0


def test_non_object_json_is_malformed() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(adapter, ["42", '"just a string"', "[1, 2, 3]"])
    assert events == []
    assert adapter.diagnostics.lines_malformed == 3


def test_missing_type_or_payload_is_malformed() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(adapter, [json.dumps({"timestamp": "2026-01-01T00:00:00Z"})])
    assert events == []
    assert adapter.diagnostics.lines_malformed == 1


def test_missing_or_invalid_timestamp_is_malformed() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(
        adapter,
        [json.dumps({"type": "session_meta", "payload": {"id": "s1"}, "timestamp": "not-a-date"})],
    )
    assert events == []
    assert adapter.diagnostics.lines_malformed == 1


def test_unknown_top_level_type_is_recorded_not_fatal() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(
        adapter,
        [
            json.dumps(
                {"type": "some_future_type", "payload": {}, "timestamp": "2026-01-01T00:00:00Z"}
            )
        ],
    )
    assert events == []
    assert adapter.diagnostics.events_unknown_type == 1


def test_unknown_event_msg_subtype_is_recorded_not_fatal() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(
        adapter,
        [
            json.dumps(
                {
                    "type": "event_msg",
                    "payload": {"type": "some_new_event_kind", "foo": "bar"},
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            )
        ],
    )
    assert events == []
    assert adapter.diagnostics.events_unknown_type == 1


def test_unknown_response_item_subtype_is_recorded_not_fatal() -> None:
    adapter = CodexAdapter("hint")
    events = parse_all(
        adapter,
        [
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {"type": "some_future_call_kind", "foo": "bar"},
                    "timestamp": "2026-01-01T00:00:00Z",
                }
            )
        ],
    )
    assert events == []
    assert adapter.diagnostics.events_unknown_type == 1


def test_forward_compatible_unknown_extra_fields_are_ignored() -> None:
    adapter = CodexAdapter("hint")
    raw = json.loads(session_meta_line("s1", T0))
    raw["payload"]["a_brand_new_field_from_a_future_codex_version"] = {"nested": "value"}
    events = parse_all(adapter, [json.dumps(raw)])
    assert len(events) == 1
    assert events[0].session_id == "s1"


def test_null_usage_fields_stay_none() -> None:
    adapter = CodexAdapter("hint")
    raw = json.loads(token_count_line(T0, input_tokens=10, cached_input_tokens=5, output_tokens=1))
    raw["payload"]["info"]["last_token_usage"]["output_tokens"] = None
    lines = [
        task_started_line(T0, "turn-1"),
        json.dumps(raw),
        task_complete_line(T0, "turn-1"),
    ]
    events = parse_all(adapter, lines)
    assert events[0].output_tokens is None
    assert events[0].input_tokens == 10


def test_out_of_order_timestamps_do_not_crash_the_adapter() -> None:
    adapter = CodexAdapter("hint")
    later = T0 + timedelta(minutes=5)
    lines = [
        session_meta_line("s1", later),
        task_started_line(T0, "turn-1"),  # earlier timestamp arriving after a later one
        task_complete_line(T0, "turn-1"),
    ]
    events = parse_all(adapter, lines)
    assert len(events) == 2  # session_start + model_response; no exception raised


def test_duplicate_lines_reparsed_produce_identical_event_ids() -> None:
    """Re-scanning the same unmodified file must yield the same event
    identities, so repeated `scan` runs don't fabricate new events."""
    line = session_meta_line("s1", T0)
    adapter_a = CodexAdapter("hint")
    adapter_b = CodexAdapter("hint")
    events_a = parse_all(adapter_a, [line])
    events_b = parse_all(adapter_b, [line])
    assert events_a[0].event_id == events_b[0].event_id


def test_extremely_long_line_is_rejected_without_attempting_json_parse() -> None:
    adapter = CodexAdapter("hint")
    huge = json.dumps(
        {
            "type": "response_item",
            "payload": {"type": "message", "role": "user", "content": "x" * (9 * 1024 * 1024)},
            "timestamp": "2026-01-01T00:00:00Z",
        }
    )
    events = parse_all(adapter, [huge])
    assert events == []
    assert adapter.diagnostics.lines_malformed == 1


def test_hash_signature_is_deterministic_and_one_way_looking() -> None:
    sig1 = hash_signature("shell", "git status")
    sig2 = hash_signature("shell", "git status")
    sig3 = hash_signature("shell", "git status --porcelain")
    assert sig1 == sig2
    assert sig1 != sig3
    assert "git status" not in sig1
