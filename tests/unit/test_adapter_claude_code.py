import json
from datetime import UTC, datetime, timedelta

from agent_fuse.adapters.claude_code import ClaudeCodeAdapter, hash_signature
from agent_fuse.models import EventType
from tests.conftest import (
    build_claude_turn,
    claude_assistant_line,
    claude_system_error_line,
    claude_text_block,
    claude_thinking_block,
    claude_tool_result_line,
    claude_tool_use_block,
    claude_usage,
    claude_user_text_line,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def parse_all(adapter: ClaudeCodeAdapter, lines: list[str], session_id_hint: str = "hint"):
    events = []
    for line in lines:
        events.extend(adapter.parse_line(line, session_id_hint=session_id_hint))
    return events


def test_first_valid_line_emits_session_start_with_real_session_id() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [claude_user_text_line(T0, "real-session-id", "hello")]
    events = parse_all(adapter, lines)
    starts = [e for e in events if e.event_type is EventType.SESSION_START]
    assert len(starts) == 1
    assert starts[0].session_id == "real-session-id"
    assert adapter.session_id == "real-session-id"


def test_session_start_emitted_only_once() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        claude_user_text_line(T0, "s1", "hello"),
        claude_user_text_line(T0 + timedelta(seconds=1), "s1", "again"),
    ]
    events = parse_all(adapter, lines)
    starts = [e for e in events if e.event_type is EventType.SESSION_START]
    assert len(starts) == 1


def test_session_start_never_leaks_cwd_or_git_branch() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        claude_user_text_line(T0, "s1", "hello", cwd="/home/alice/secret-client-repo"),
    ]
    events = parse_all(adapter, lines)
    dumped = json.dumps(events[0].model_dump(mode="json"))
    assert "secret-client-repo" not in dumped
    assert "alice" not in dumped


def test_plain_string_user_prompt_is_never_extracted() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [claude_user_text_line(T0, "s1", "my api key is sk-abc123 and my password is hunter2")]
    events = parse_all(adapter, lines)
    dumped = json.dumps([e.model_dump(mode="json") for e in events])
    assert "sk-abc123" not in dumped
    assert "hunter2" not in dumped


def test_one_model_response_per_unique_message_id_despite_multiple_lines() -> None:
    """Claude Code explodes one API response into multiple JSONL lines
    (thinking/text/tool_use), all sharing message.id and usage. Must count
    as exactly one response, not one per line."""
    adapter = ClaudeCodeAdapter("hint")
    usage = claude_usage(
        input_tokens=2,
        cache_creation_input_tokens=100,
        cache_read_input_tokens=50,
        output_tokens=30,
    )
    lines = [
        claude_assistant_line(T0, "s1", "msg_1", [claude_thinking_block("hmm")], usage=usage),
        claude_assistant_line(T0, "s1", "msg_1", [claude_text_block("ok")], usage=usage),
        claude_assistant_line(
            T0,
            "s1",
            "msg_1",
            [claude_tool_use_block("tool_1", "Bash", {"command": "ls"})],
            usage=usage,
        ),
    ]
    events = parse_all(adapter, lines)
    responses = [e for e in events if e.event_type is EventType.MODEL_RESPONSE]
    tool_calls = [e for e in events if e.event_type is EventType.TOOL_CALL]
    assert len(responses) == 1
    assert len(tool_calls) == 1  # tool_use block still produces its own event


def test_model_response_token_accounting_sums_all_three_input_buckets() -> None:
    adapter = ClaudeCodeAdapter("hint")
    usage = claude_usage(
        input_tokens=2,
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=500,
        output_tokens=42,
    )
    lines = [claude_assistant_line(T0, "s1", "msg_1", [claude_text_block("ok")], usage=usage)]
    events = parse_all(adapter, lines)
    response = events[-1]
    assert response.input_tokens == 2 + 1000 + 500
    assert response.cached_input_tokens == 500  # cache_read only, not cache_creation
    assert response.output_tokens == 42


def test_missing_usage_reports_none_not_zero() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [claude_assistant_line(T0, "s1", "msg_1", [claude_text_block("ok")], usage={})]
    events = parse_all(adapter, lines)
    response = next(e for e in events if e.event_type is EventType.MODEL_RESPONSE)
    assert response.input_tokens is None
    assert response.cached_input_tokens is None
    assert response.output_tokens is None


def test_second_message_id_after_first_is_a_second_response() -> None:
    adapter = ClaudeCodeAdapter("hint")
    usage = claude_usage(input_tokens=1, output_tokens=1)
    lines = [
        claude_assistant_line(T0, "s1", "msg_1", [claude_text_block("a")], usage=usage),
        claude_assistant_line(T0, "s1", "msg_2", [claude_text_block("b")], usage=usage),
    ]
    events = parse_all(adapter, lines)
    responses = [e for e in events if e.event_type is EventType.MODEL_RESPONSE]
    assert len(responses) == 2


def test_tool_use_and_tool_result_link_via_parent_event_id() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines, _ = build_claude_turn(
        T0, "s1", "msg_1", tool_name="Bash", tool_input={"command": "ls"}, tool_id="tool_abc"
    )
    events = parse_all(adapter, lines)
    call = next(e for e in events if e.event_type is EventType.TOOL_CALL)
    result = next(e for e in events if e.event_type is EventType.TOOL_RESULT)
    assert result.parent_event_id == call.event_id


def test_tool_use_input_never_appears_in_event() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        claude_assistant_line(
            T0,
            "s1",
            "msg_1",
            [
                claude_tool_use_block(
                    "tool_1", "Bash", {"command": "cat /etc/shadow --token=sk-xyz"}
                )
            ],
        )
    ]
    events = parse_all(adapter, lines)
    dumped = json.dumps(events[-1].model_dump(mode="json"))
    assert "shadow" not in dumped
    assert "sk-xyz" not in dumped
    assert events[-1].tool_signature is not None


def test_tool_result_content_and_stdout_never_extracted() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        claude_tool_result_line(T0, "s1", "tool_1", "SECRET_OUTPUT_VALUE=abc123", is_error=False)
    ]
    events = parse_all(adapter, lines)
    dumped = json.dumps([e.model_dump(mode="json") for e in events])
    assert "SECRET_OUTPUT_VALUE" not in dumped
    assert "abc123" not in dumped


def test_same_tool_call_repeated_produces_same_signature() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        claude_assistant_line(
            T0,
            "s1",
            f"msg_{i}",
            [claude_tool_use_block(f"tool_{i}", "Bash", {"command": "git status"})],
        )
        for i in range(3)
    ]
    events = parse_all(adapter, lines)
    signatures = {e.tool_signature for e in events if e.event_type is EventType.TOOL_CALL}
    assert len(signatures) == 1


def test_different_tool_input_produces_different_signature() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        claude_assistant_line(
            T0, "s1", "msg_1", [claude_tool_use_block("tool_1", "Bash", {"command": "git status"})]
        ),
        claude_assistant_line(
            T0, "s1", "msg_2", [claude_tool_use_block("tool_2", "Bash", {"command": "rm -rf /"})]
        ),
    ]
    events = parse_all(adapter, lines)
    calls = [e for e in events if e.event_type is EventType.TOOL_CALL]
    assert calls[0].tool_signature != calls[1].tool_signature


def test_api_error_does_not_retain_error_text() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        claude_system_error_line(T0, "s1", "auth failed: token=SECRETVALUE user=alice@example.com")
    ]
    events = parse_all(adapter, lines)
    errors = [e for e in events if e.event_type is EventType.ERROR]
    assert len(errors) == 1
    dumped = json.dumps(errors[0].model_dump(mode="json"))
    assert "SECRETVALUE" not in dumped
    assert "alice@example.com" not in dumped


def test_ignored_top_level_types_produce_no_events() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        json.dumps({"type": "ai-title", "sessionId": "s1", "aiTitle": "some title"}),
        json.dumps(
            {
                "type": "queue-operation",
                "sessionId": "s1",
                "operation": "x",
                "timestamp": "2026-01-01T00:00:00Z",
            }
        ),
        json.dumps(
            {
                "type": "attachment",
                "sessionId": "s1",
                "timestamp": "2026-01-01T00:00:00Z",
                "attachment": {"secret": "value"},
            }
        ),
    ]
    events = parse_all(adapter, lines)
    assert events == []
    assert adapter.diagnostics.lines_malformed == 0
    assert adapter.diagnostics.events_unknown_type == 0


def test_sidechain_turns_fold_into_same_session_metrics() -> None:
    """A subagent (Task tool) turn is marked isSidechain=true but shares
    sessionId; it must still count toward the parent session so a runaway
    subagent still trips the circuit breaker."""
    adapter = ClaudeCodeAdapter("hint")
    usage = claude_usage(input_tokens=1, output_tokens=1)
    lines = [
        claude_assistant_line(
            T0, "s1", "msg_1", [claude_text_block("a")], usage=usage, is_sidechain=False
        ),
        claude_assistant_line(
            T0, "s1", "msg_2", [claude_text_block("b")], usage=usage, is_sidechain=True
        ),
    ]
    events = parse_all(adapter, lines)
    responses = [e for e in events if e.event_type is EventType.MODEL_RESPONSE]
    assert len(responses) == 2
    assert all(e.session_id == "s1" for e in responses)


def test_malformed_json_line_is_skipped_not_fatal() -> None:
    adapter = ClaudeCodeAdapter("hint")
    events = parse_all(adapter, ['{"not": "closed"', claude_user_text_line(T0, "s1", "hi")])
    assert adapter.diagnostics.lines_malformed == 1
    assert len(events) == 1  # session_start from the well-formed line


def test_empty_lines_are_skipped_silently() -> None:
    adapter = ClaudeCodeAdapter("hint")
    events = parse_all(adapter, ["", "   ", claude_user_text_line(T0, "s1", "hi")])
    assert len(events) == 1
    assert adapter.diagnostics.lines_malformed == 0


def test_non_object_json_is_malformed() -> None:
    adapter = ClaudeCodeAdapter("hint")
    events = parse_all(adapter, ["42", '"a string"', "[1, 2]"])
    assert events == []
    assert adapter.diagnostics.lines_malformed == 3


def test_missing_type_is_malformed() -> None:
    adapter = ClaudeCodeAdapter("hint")
    events = parse_all(
        adapter, [json.dumps({"sessionId": "s1", "timestamp": "2026-01-01T00:00:00Z"})]
    )
    assert events == []
    assert adapter.diagnostics.lines_malformed == 1


def test_missing_or_invalid_timestamp_on_meaningful_type_is_malformed() -> None:
    adapter = ClaudeCodeAdapter("hint")
    events = parse_all(
        adapter,
        [
            json.dumps(
                {"type": "user", "sessionId": "s1", "message": {"role": "user", "content": "hi"}}
            )
        ],
    )
    assert events == []
    assert adapter.diagnostics.lines_malformed == 1


def test_ignored_type_without_timestamp_is_not_malformed() -> None:
    """Some housekeeping line types (e.g. ai-title) genuinely never carry a
    timestamp in real Claude Code transcripts; that must not be flagged as
    malformed data."""
    adapter = ClaudeCodeAdapter("hint")
    events = parse_all(
        adapter, [json.dumps({"type": "ai-title", "sessionId": "s1", "aiTitle": "x"})]
    )
    assert events == []
    assert adapter.diagnostics.lines_malformed == 0


def test_unknown_top_level_type_is_recorded_not_fatal() -> None:
    adapter = ClaudeCodeAdapter("hint")
    events = parse_all(
        adapter,
        [
            json.dumps(
                {"type": "some_future_type", "sessionId": "s1", "timestamp": "2026-01-01T00:00:00Z"}
            )
        ],
    )
    assert events == []
    assert adapter.diagnostics.events_unknown_type == 1


def test_unknown_assistant_block_type_is_recorded_not_fatal() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        claude_assistant_line(T0, "s1", "msg_1", [{"type": "some_new_block_kind", "foo": "bar"}])
    ]
    events = parse_all(adapter, lines)
    tool_calls = [e for e in events if e.event_type is EventType.TOOL_CALL]
    assert tool_calls == []
    assert adapter.diagnostics.events_unknown_type == 1


def test_unknown_user_block_type_is_recorded_not_fatal() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        json.dumps(
            {
                "type": "user",
                "timestamp": "2026-01-01T00:00:00Z",
                "sessionId": "s1",
                "message": {
                    "role": "user",
                    "content": [{"type": "some_new_block_kind", "foo": "bar"}],
                },
            }
        )
    ]
    events = parse_all(adapter, lines)
    results = [e for e in events if e.event_type is EventType.TOOL_RESULT]
    assert results == []
    assert adapter.diagnostics.events_unknown_type == 1


def test_unknown_system_subtype_is_recorded_not_fatal() -> None:
    adapter = ClaudeCodeAdapter("hint")
    lines = [
        json.dumps(
            {
                "type": "system",
                "subtype": "some_future_subtype",
                "timestamp": "2026-01-01T00:00:00Z",
                "sessionId": "s1",
            }
        )
    ]
    events = parse_all(adapter, lines)
    assert events == [] or all(e.event_type is EventType.SESSION_START for e in events)
    assert adapter.diagnostics.events_unknown_type == 1


def test_forward_compatible_unknown_extra_fields_are_ignored() -> None:
    adapter = ClaudeCodeAdapter("hint")
    raw = json.loads(claude_user_text_line(T0, "s1", "hi"))
    raw["aBrandNewFieldFromAFutureClaudeCodeVersion"] = {"nested": "value"}
    events = parse_all(adapter, [json.dumps(raw)])
    assert len(events) == 1
    assert events[0].session_id == "s1"


def test_out_of_order_timestamps_do_not_crash() -> None:
    adapter = ClaudeCodeAdapter("hint")
    later = T0 + timedelta(minutes=5)
    lines = [
        claude_user_text_line(later, "s1", "hi"),
        claude_assistant_line(
            T0, "s1", "msg_1", [claude_text_block("a")]
        ),  # earlier ts, arrives later
    ]
    events = parse_all(adapter, lines)
    assert len(events) == 2  # session_start + model_response, no exception


def test_duplicate_lines_reparsed_produce_identical_event_ids() -> None:
    line = claude_user_text_line(T0, "s1", "hi")
    a = ClaudeCodeAdapter("hint")
    b = ClaudeCodeAdapter("hint")
    events_a = parse_all(a, [line])
    events_b = parse_all(b, [line])
    assert events_a[0].event_id == events_b[0].event_id


def test_extremely_long_line_is_rejected_without_json_parse() -> None:
    adapter = ClaudeCodeAdapter("hint")
    huge = json.dumps(
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:00Z",
            "sessionId": "s1",
            "message": {"role": "user", "content": "x" * (9 * 1024 * 1024)},
        }
    )
    events = parse_all(adapter, [huge])
    assert events == []
    assert adapter.diagnostics.lines_malformed == 1


def test_hash_signature_is_deterministic_and_one_way_looking() -> None:
    sig1 = hash_signature("Bash", json.dumps({"command": "git status"}, sort_keys=True))
    sig2 = hash_signature("Bash", json.dumps({"command": "git status"}, sort_keys=True))
    sig3 = hash_signature("Bash", json.dumps({"command": "git status --porcelain"}, sort_keys=True))
    assert sig1 == sig2
    assert sig1 != sig3
    assert "git status" not in sig1
