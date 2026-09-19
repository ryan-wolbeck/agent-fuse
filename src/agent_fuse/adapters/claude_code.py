"""Adapter for Claude Code's local project transcript telemetry.

Schema notes (verified against real local Claude Code 2.1.269 session
transcripts under `~/.claude/projects/<sanitized-cwd>/<session-uuid>.jsonl`;
cross-checked field-by-field, not guessed):

Each line is a JSON object with a top-level "type" tag. Across a real local
install the observed top-level types are: "user", "assistant", "system",
"attachment", "ai-title", "queue-operation", "atis-latch", "last-prompt",
"pr-link", "mode", "permission-mode", "file-history-snapshot",
"bridge-session". Only "user", "assistant", and "system" carry data this
adapter needs; the rest are UI/session-bookkeeping and are recognized but
ignored (their content, if any, is never read).

"assistant" lines carry one Anthropic Messages API response, but Claude
Code writes it as *multiple* JSONL lines -- one per content block
(thinking/text/tool_use) -- that all share the same `message.id` and an
identical `message.usage` snapshot (verified: three consecutive lines in a
real transcript had matching `message.id`/`requestId`/`usage` and one
content block apiece). This adapter counts exactly one MODEL_RESPONSE per
distinct `message.id`, taking usage from whichever line for that id is
seen first, and emits a TOOL_CALL for every `tool_use` content block
regardless of which line it's on.

Token accounting follows the Anthropic Messages API usage shape, which
splits input tokens into three disjoint buckets:
  - `input_tokens`: fresh (non-cached) input tokens
  - `cache_creation_input_tokens`: input tokens newly written to a cache
    for possible future reuse -- this is not itself replay
  - `cache_read_input_tokens`: input tokens actually served from a
    previously-written cache -- this is the "replay" Agent Fuse's
    context_replay rule cares about
This adapter reports canonical `input_tokens` as the sum of all three
(total input actually processed) and `cached_input_tokens` as
`cache_read_input_tokens` alone. `output_tokens` is used as-is (Anthropic
reports it as the total completion tokens; `output_tokens_details` is an
informational breakdown subset, not an additive figure).

What this adapter reads, and nothing else:
- sessionId (every line)               -> canonical session_id
- assistant message.id + message.usage -> one MODEL_RESPONSE per unique id
- assistant content block "tool_use"   -> TOOL_CALL (name + a hash of the
                                           tool input, never the raw input)
- user content block "tool_result"     -> TOOL_RESULT (linked by
                                           tool_use_id only, never the
                                           result content/stdout/stderr)
- system/api_error                     -> ERROR (no error text retained)
- entrypoint, version (top-level)      -> safe, low-cardinality SESSION_START
                                           metadata (not a path, not a
                                           branch name)

Never read: prompt/response text, thinking content, tool_use `input`
values, tool_result `content`/`toolUseResult` (stdout/stderr), `cwd`,
`gitBranch`. In particular, the *parent directory* a Claude Code session
file lives under encodes the project's working-directory path (Claude Code
sanitizes cwd into a directory name, e.g. `/home/alice/proj` becomes
`-home-alice-proj`) -- this adapter and `discovery.py` only ever use the
session UUID (the filename) as an identifier, never that directory name.

What this adapter does NOT emit, and why:
- SESSION_END: there is no explicit "session ended" line in a Claude Code
  transcript; `discovery.py` infers staleness from file mtime instead, the
  same policy as the Codex adapter.
- SUBAGENT_START/SUBAGENT_END: Claude Code interleaves subagent ("Task"
  tool) turns into the *same* transcript file as the parent session,
  distinguished only by an `isSidechain: true` flag, with no clean
  "subagent started/stopped" marker to key an event off of. Rather than
  fabricate lifecycle events the source data doesn't clearly delineate,
  this adapter folds sidechain turns into the same session's metrics --
  which also means a runaway subagent still trips its parent session's
  rules, arguably the more useful behavior for a circuit breaker anyway.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from ..models import AgentEvent, EventType
from .base import AdapterDiagnostics

MAX_LINE_BYTES = 8 * 1024 * 1024

_IGNORED_TOP_LEVEL_TYPES = {
    "attachment",
    "ai-title",
    "queue-operation",
    "atis-latch",
    "last-prompt",
    "pr-link",
    "mode",
    "permission-mode",
    "file-history-snapshot",
    "bridge-session",
}
_IGNORED_SYSTEM_SUBTYPES = {"turn_duration", "away_summary"}
_IGNORED_ASSISTANT_BLOCK_TYPES = {"text", "thinking", "redacted_thinking", "server_tool_use"}
_IGNORED_USER_BLOCK_TYPES = {"text", "image"}


def hash_signature(*parts: str) -> str:
    """One-way fingerprint used for repeated-tool-call detection.

    Same construction as the Codex adapter's `hash_signature`: deterministic
    and irreversible, so repetition can be detected without retaining a
    tool's raw input.
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


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _extract_usage(usage: Any) -> tuple[int | None, int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None, None
    fresh = _safe_int(usage.get("input_tokens"))
    creation = _safe_int(usage.get("cache_creation_input_tokens"))
    read = _safe_int(usage.get("cache_read_input_tokens"))
    output = _safe_int(usage.get("output_tokens"))
    if fresh is None and creation is None and read is None:
        total_input = None
    else:
        total_input = (fresh or 0) + (creation or 0) + (read or 0)
    return total_input, read, output


class ClaudeCodeAdapter:
    """Stateful per-session parser. Create one instance per session file."""

    provider = "claude_code"

    def __init__(self, session_id_hint: str) -> None:
        self._session_id = session_id_hint
        self._diagnostics = AdapterDiagnostics()
        self._seen_message_ids: set[str] = set()
        self._session_start_emitted = False
        self._line_index = 0

    @property
    def diagnostics(self) -> AdapterDiagnostics:
        return self._diagnostics

    @property
    def session_id(self) -> str:
        return self._session_id

    def flush(self, *, session_id_hint: str) -> list[AgentEvent]:
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
        if not isinstance(line_type, str):
            self._diagnostics.record_malformed("missing type")
            return []

        if line_type in _IGNORED_TOP_LEVEL_TYPES:
            return []

        if line_type not in ("user", "assistant", "system"):
            self._diagnostics.record_unknown_type(f"line:{line_type}")
            return []

        session_id = obj.get("sessionId")
        if isinstance(session_id, str) and session_id:
            self._session_id = session_id

        timestamp = _parse_timestamp(obj.get("timestamp"))
        if timestamp is None:
            self._diagnostics.record_malformed("missing/invalid timestamp")
            return []

        self._line_index += 1
        events: list[AgentEvent] = []
        if not self._session_start_emitted:
            events.append(self._make_session_start(obj, timestamp))
            self._session_start_emitted = True

        try:
            if line_type == "assistant":
                events.extend(self._handle_assistant(obj, timestamp))
            elif line_type == "user":
                events.extend(self._handle_user(obj, timestamp))
            elif line_type == "system":
                events.extend(self._handle_system(obj, timestamp))
        except Exception as exc:  # noqa: BLE001 - one bad line must never stop the stream
            self._diagnostics.record_malformed(f"unexpected shape: {type(exc).__name__}")

        return events

    # -- line-type handlers --------------------------------------------------

    def _make_session_start(self, obj: dict[str, Any], timestamp: datetime) -> AgentEvent:
        metadata: dict[str, Any] = {}
        entrypoint = obj.get("entrypoint")
        if isinstance(entrypoint, str):
            metadata["entrypoint"] = entrypoint
        version = obj.get("version")
        if isinstance(version, str):
            metadata["cli_version"] = version

        self._diagnostics.events_emitted += 1
        return AgentEvent(
            event_id=f"claude_code:{self._session_id}:session_start",
            session_id=self._session_id,
            timestamp=timestamp,
            provider=self.provider,
            event_type=EventType.SESSION_START,
            metadata=metadata,
        )

    def _handle_assistant(self, obj: dict[str, Any], timestamp: datetime) -> list[AgentEvent]:
        message = obj.get("message")
        if not isinstance(message, dict):
            self._diagnostics.record_malformed("assistant line missing message")
            return []

        events: list[AgentEvent] = []
        message_id = message.get("id")
        if isinstance(message_id, str) and message_id and message_id not in self._seen_message_ids:
            self._seen_message_ids.add(message_id)
            input_tokens, cached_input_tokens, output_tokens = _extract_usage(message.get("usage"))
            events.append(
                AgentEvent(
                    event_id=f"claude_code:{self._session_id}:{message_id}:response",
                    session_id=self._session_id,
                    timestamp=timestamp,
                    provider=self.provider,
                    event_type=EventType.MODEL_RESPONSE,
                    input_tokens=input_tokens,
                    cached_input_tokens=cached_input_tokens,
                    output_tokens=output_tokens,
                )
            )

        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type == "tool_use":
                    events.append(self._handle_tool_use(block, timestamp))
                elif block_type in _IGNORED_ASSISTANT_BLOCK_TYPES:
                    continue
                else:
                    self._diagnostics.record_unknown_type(f"assistant_block:{block_type}")

        return events

    def _handle_tool_use(self, block: dict[str, Any], timestamp: datetime) -> AgentEvent:
        name = block.get("name")
        tool_id = block.get("id")
        tool_input = block.get("input")
        tool_name = name if isinstance(name, str) else "unknown_tool"
        ref = tool_id if isinstance(tool_id, str) and tool_id else f"line{self._line_index}"
        input_repr = (
            json.dumps(tool_input, sort_keys=True, default=str) if tool_input is not None else ""
        )
        signature = hash_signature(tool_name, input_repr)

        self._diagnostics.events_emitted += 1
        return AgentEvent(
            event_id=f"claude_code:{self._session_id}:{ref}:call",
            session_id=self._session_id,
            timestamp=timestamp,
            provider=self.provider,
            event_type=EventType.TOOL_CALL,
            tool_name=tool_name,
            tool_signature=signature,
        )

    def _handle_user(self, obj: dict[str, Any], timestamp: datetime) -> list[AgentEvent]:
        message = obj.get("message")
        if not isinstance(message, dict):
            return []
        content = message.get("content")
        if not isinstance(content, list):
            return []  # plain-string user prompt: never read

        events: list[AgentEvent] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "tool_result":
                events.append(self._handle_tool_result(block, timestamp))
            elif block_type in _IGNORED_USER_BLOCK_TYPES:
                continue
            else:
                self._diagnostics.record_unknown_type(f"user_block:{block_type}")
        return events

    def _handle_tool_result(self, block: dict[str, Any], timestamp: datetime) -> AgentEvent:
        tool_use_id = block.get("tool_use_id")
        ref = (
            tool_use_id
            if isinstance(tool_use_id, str) and tool_use_id
            else f"line{self._line_index}"
        )

        self._diagnostics.events_emitted += 1
        return AgentEvent(
            event_id=f"claude_code:{self._session_id}:{ref}:result",
            session_id=self._session_id,
            timestamp=timestamp,
            provider=self.provider,
            event_type=EventType.TOOL_RESULT,
            parent_event_id=f"claude_code:{self._session_id}:{ref}:call",
        )

    def _handle_system(self, obj: dict[str, Any], timestamp: datetime) -> list[AgentEvent]:
        subtype = obj.get("subtype")
        if subtype == "api_error":
            uuid = obj.get("uuid")
            ref = uuid if isinstance(uuid, str) and uuid else f"line{self._line_index}"
            self._diagnostics.events_emitted += 1
            return [
                AgentEvent(
                    event_id=f"claude_code:{self._session_id}:{ref}:error",
                    session_id=self._session_id,
                    timestamp=timestamp,
                    provider=self.provider,
                    event_type=EventType.ERROR,
                )
            ]
        if subtype in _IGNORED_SYSTEM_SUBTYPES:
            return []
        self._diagnostics.record_unknown_type(f"system:{subtype}")
        return []
