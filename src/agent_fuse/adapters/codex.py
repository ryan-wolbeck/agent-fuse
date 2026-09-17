"""Adapter for OpenAI Codex CLI local rollout telemetry.

Schema notes (verified against a real local Codex CLI 0.150.1 session file
and cross-checked against the public `openai/codex` protocol source,
`codex-rs/protocol/src/{protocol,models}.rs`, ResponseItem / EventMsg enums):

Each line of a rollout JSONL file is a JSON object with a top-level "type"
tag ("session_meta", "turn_context", "event_msg", "response_item") and a
"payload" object whose own shape depends on that tag. `event_msg` payloads
carry a second-level "type" tag (e.g. "task_started", "task_complete",
"token_count", "user_message", "agent_message", "error", ...).
`response_item` payloads likewise carry a second-level "type" tag
("message", "reasoning", "function_call", "function_call_output",
"local_shell_call", "custom_tool_call", "tool_search_call", ...).

What this adapter reads, and nothing else:
- session_meta.id            -> canonical session_id
- event_msg/task_complete    -> one MODEL_RESPONSE per completed turn
- event_msg/token_count.info.last_token_usage -> token counts attached to
  the MODEL_RESPONSE for the turn that was in flight when reported
- response_item/function_call, local_shell_call, custom_tool_call
                              -> TOOL_CALL (name + a hash of arguments, never
                                 the raw arguments/command/env themselves)
- response_item/function_call_output -> TOOL_RESULT (linked by call_id only)
- event_msg/error            -> ERROR (no message text retained)

Everything else (message text, reasoning content, prompts, tool output
bodies, cwd, git remote info, shell command text, environment variables) is
never read out of the payload at all -- not redacted after the fact, simply
never extracted.

What this adapter does NOT emit, and why:
- SESSION_END: Codex rollout files have no explicit "session ended" event;
  synthesizing one would be a fabricated fact. `discovery.py` infers
  activity/staleness from file mtime instead.
- SUBAGENT_START/SUBAGENT_END: a Codex subagent (thread) shows up as its own
  rollout file with `session_meta.thread_source == "subagent"`, not as an
  event inside a parent file. This adapter surfaces that as metadata on the
  SESSION_START event rather than fabricating start/end markers with no
  matching structural event in the source data.

Turn completion, not `agent_message`, is the "response" unit counted by the
response-rate rule: Codex emits `task_complete` exactly once per model turn
regardless of whether that turn produced user-visible text or only tool
calls, which makes it a more honest proxy for "the model was invoked and
finished" than counting visible chat messages.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from ..models import AgentEvent, EventType
from .base import AdapterDiagnostics

MAX_LINE_BYTES = 8 * 1024 * 1024

# response_item / event_msg subtypes we recognize but deliberately do not
# turn into canonical events (their content is never inspected).
_IGNORED_RESPONSE_ITEM_TYPES = {"message", "reasoning", "agent_message", "additional_tools"}
_IGNORED_EVENT_MSG_TYPES = {
    "task_started",
    "user_message",
    "agent_message",
    "thread_rolled_back",
    "warning",
    "auth_recovery_started",
    "stream_error",
    "patch_apply_begin",
    "patch_apply_updated",
}


def hash_signature(*parts: str) -> str:
    """One-way fingerprint used for repeated-tool-call detection.

    Deliberately irreversible: this lets us notice "the same call happened
    73 times" without retaining anything an attacker (or a support ticket
    screenshot) could turn back into a shell command, URL, or file path.
    """
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8", errors="replace"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class CodexAdapter:
    """Stateful per-session parser. Create one instance per session file."""

    provider = "codex"

    def __init__(self, session_id_hint: str) -> None:
        self._session_id = session_id_hint
        self._diagnostics = AdapterDiagnostics()
        self._pending_input: int | None = None
        self._pending_cached: int | None = None
        self._pending_output: int | None = None
        self._line_index = 0

    @property
    def diagnostics(self) -> AdapterDiagnostics:
        return self._diagnostics

    @property
    def session_id(self) -> str:
        return self._session_id

    def flush(self, *, session_id_hint: str) -> list[AgentEvent]:
        # An in-flight turn with no task_complete is not a confirmed
        # response; we do not fabricate one when the file simply ends.
        del session_id_hint
        return []

    def parse_line(self, raw_line: str, *, session_id_hint: str) -> list[AgentEvent]:
        self._diagnostics.lines_read += 1
        line = raw_line.strip()
        if not line:
            return []
        if len(line.encode("utf-8", errors="replace")) > MAX_LINE_BYTES:
            self._diagnostics.record_malformed("line exceeds max size")
            return []

        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            self._diagnostics.record_malformed("invalid JSON")
            return []

        if not isinstance(obj, dict):
            self._diagnostics.record_malformed("top-level JSON is not an object")
            return []

        line_type = obj.get("type")
        payload = obj.get("payload")
        timestamp = _parse_timestamp(obj.get("timestamp"))

        if not isinstance(line_type, str) or not isinstance(payload, dict):
            self._diagnostics.record_malformed("missing type/payload")
            return []
        if timestamp is None:
            self._diagnostics.record_malformed("missing/invalid timestamp")
            return []

        self._line_index += 1

        try:
            if line_type == "session_meta":
                return self._handle_session_meta(payload, timestamp)
            elif line_type == "event_msg":
                return self._handle_event_msg(payload, timestamp)
            elif line_type == "response_item":
                return self._handle_response_item(payload, timestamp)
            elif line_type == "turn_context":
                return []  # recognized, intentionally ignored
            else:
                self._diagnostics.record_unknown_type(f"line:{line_type}")
                return []
        except Exception as exc:  # noqa: BLE001 - one bad line must never stop the stream
            self._diagnostics.record_malformed(f"unexpected shape: {type(exc).__name__}")
            return []

    # -- line-type handlers --------------------------------------------------

    def _handle_session_meta(
        self, payload: dict[str, Any], timestamp: datetime
    ) -> list[AgentEvent]:
        session_id = payload.get("id")
        if isinstance(session_id, str) and session_id:
            self._session_id = session_id

        metadata: dict[str, Any] = {}
        thread_source = payload.get("thread_source")
        if isinstance(thread_source, str):
            metadata["thread_source"] = thread_source
        originator = payload.get("originator")
        if isinstance(originator, str):
            metadata["originator"] = originator
        cli_version = payload.get("cli_version")
        if isinstance(cli_version, str):
            metadata["cli_version"] = cli_version

        event = AgentEvent(
            event_id=f"codex:{self._session_id}:session_start",
            session_id=self._session_id,
            timestamp=timestamp,
            provider=self.provider,
            event_type=EventType.SESSION_START,
            metadata=metadata,
        )
        self._diagnostics.events_emitted += 1
        return [event]

    def _handle_event_msg(self, payload: dict[str, Any], timestamp: datetime) -> list[AgentEvent]:
        sub_type = payload.get("type")
        if not isinstance(sub_type, str):
            self._diagnostics.record_malformed("event_msg missing sub-type")
            return []

        if sub_type == "token_count":
            info = payload.get("info")
            if isinstance(info, dict):
                usage = info.get("last_token_usage")
                if isinstance(usage, dict):
                    self._pending_input = _safe_int(usage.get("input_tokens"))
                    self._pending_cached = _safe_int(usage.get("cached_input_tokens"))
                    self._pending_output = _safe_int(usage.get("output_tokens"))
            return []

        if sub_type == "task_complete":
            turn_id = payload.get("turn_id")
            turn_ref = turn_id if isinstance(turn_id, str) and turn_id else str(self._line_index)
            event = AgentEvent(
                event_id=f"codex:{self._session_id}:{turn_ref}:response",
                session_id=self._session_id,
                timestamp=timestamp,
                provider=self.provider,
                event_type=EventType.MODEL_RESPONSE,
                input_tokens=self._pending_input,
                cached_input_tokens=self._pending_cached,
                output_tokens=self._pending_output,
            )
            self._pending_input = None
            self._pending_cached = None
            self._pending_output = None
            self._diagnostics.events_emitted += 1
            return [event]

        if sub_type == "error":
            event = AgentEvent(
                event_id=f"codex:{self._session_id}:{self._line_index}:error",
                session_id=self._session_id,
                timestamp=timestamp,
                provider=self.provider,
                event_type=EventType.ERROR,
            )
            self._diagnostics.events_emitted += 1
            return [event]

        if sub_type in _IGNORED_EVENT_MSG_TYPES:
            return []

        self._diagnostics.record_unknown_type(f"event_msg:{sub_type}")
        return []

    def _handle_response_item(
        self, payload: dict[str, Any], timestamp: datetime
    ) -> list[AgentEvent]:
        sub_type = payload.get("type")
        if not isinstance(sub_type, str):
            self._diagnostics.record_malformed("response_item missing sub-type")
            return []

        if sub_type == "function_call":
            name = payload.get("name")
            call_id = payload.get("call_id")
            arguments = payload.get("arguments")
            tool_name = name if isinstance(name, str) else "unknown_tool"
            ref = call_id if isinstance(call_id, str) and call_id else str(self._line_index)
            signature = hash_signature(tool_name, arguments if isinstance(arguments, str) else "")
            event = AgentEvent(
                event_id=f"codex:{self._session_id}:{ref}:call",
                session_id=self._session_id,
                timestamp=timestamp,
                provider=self.provider,
                event_type=EventType.TOOL_CALL,
                tool_name=tool_name,
                tool_signature=signature,
            )
            self._diagnostics.events_emitted += 1
            return [event]

        if sub_type == "local_shell_call":
            call_id = payload.get("call_id")
            action = payload.get("action")
            action_type = None
            command = None
            has_cwd = False
            if isinstance(action, dict):
                action_type = action.get("type")
                cmd = action.get("command")
                if isinstance(cmd, list):
                    command = [str(c) for c in cmd]
                has_cwd = bool(action.get("working_directory"))
            ref = call_id if isinstance(call_id, str) and call_id else str(self._line_index)
            signature = hash_signature(
                "local_shell",
                action_type if isinstance(action_type, str) else "",
                json.dumps(command) if command is not None else "",
                "cwd" if has_cwd else "",
            )
            event = AgentEvent(
                event_id=f"codex:{self._session_id}:{ref}:call",
                session_id=self._session_id,
                timestamp=timestamp,
                provider=self.provider,
                event_type=EventType.TOOL_CALL,
                tool_name="local_shell",
                tool_signature=signature,
            )
            self._diagnostics.events_emitted += 1
            return [event]

        if sub_type == "custom_tool_call":
            name = payload.get("name")
            call_id = payload.get("call_id")
            tool_input = payload.get("input")
            tool_name = name if isinstance(name, str) else "unknown_tool"
            ref = call_id if isinstance(call_id, str) and call_id else str(self._line_index)
            signature = hash_signature(tool_name, tool_input if isinstance(tool_input, str) else "")
            event = AgentEvent(
                event_id=f"codex:{self._session_id}:{ref}:call",
                session_id=self._session_id,
                timestamp=timestamp,
                provider=self.provider,
                event_type=EventType.TOOL_CALL,
                tool_name=tool_name,
                tool_signature=signature,
            )
            self._diagnostics.events_emitted += 1
            return [event]

        if sub_type == "function_call_output":
            call_id = payload.get("call_id")
            name = payload.get("name")
            ref = call_id if isinstance(call_id, str) and call_id else str(self._line_index)
            event = AgentEvent(
                event_id=f"codex:{self._session_id}:{ref}:result",
                session_id=self._session_id,
                timestamp=timestamp,
                provider=self.provider,
                event_type=EventType.TOOL_RESULT,
                tool_name=name if isinstance(name, str) else None,
                parent_event_id=f"codex:{self._session_id}:{ref}:call",
            )
            self._diagnostics.events_emitted += 1
            return [event]

        if sub_type in _IGNORED_RESPONSE_ITEM_TYPES:
            return []

        self._diagnostics.record_unknown_type(f"response_item:{sub_type}")
        return []


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
