"""Vendor-neutral canonical data model.

Everything above this module (metrics, rules, engine, CLI) only ever sees
these types. Adapters are the only code allowed to know about vendor-specific
JSON shapes; they translate into `AgentEvent` and nothing else crosses the
boundary.

Fields are optional almost everywhere on purpose: a field that a vendor does
not expose must stay `None`, never `0`. Zero and "unknown" are different
facts and collapsing them would make the rules lie.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class EventType(StrEnum):
    """Canonical event categories.

    Not every adapter emits every type. In particular the current Codex
    adapter does not emit SUBAGENT_START/SUBAGENT_END directly (see
    adapters/codex.py docstring) -- they exist here for forward
    compatibility with adapters that can support them.
    """

    SESSION_START = "session_start"
    MODEL_RESPONSE = "model_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_END = "subagent_end"
    ERROR = "error"
    SESSION_END = "session_end"
    UNKNOWN = "unknown"


class AgentEvent(BaseModel):
    """A single normalized event from any supported agent provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str
    session_id: str
    timestamp: datetime
    provider: str
    event_type: EventType

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None

    tool_name: str | None = None
    tool_signature: str | None = None

    parent_event_id: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


Severity = Literal["warning", "severe"]


class RuleResult(BaseModel):
    """Outcome of evaluating a single deterministic rule against a session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule: str
    triggered: bool
    severity: Severity
    observed: float | int | None
    threshold: float | int | None
    window_seconds: int | None
    message: str


class FuseEvent(BaseModel):
    """Machine-readable record of a triggered rule.

    This is a public, versioned interface -- see docs/JSON schema in the
    README before changing field names or semantics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    event: Literal["fuse.triggered"] = "fuse.triggered"
    timestamp: datetime
    provider: str
    session_id: str
    rule: str
    severity: Severity
    window_seconds: int | None
    observed: float | int | None
    threshold: float | int | None
    message: str
