"""Locks down the public fuse.triggered JSON schema referenced in the README.

If this test needs to change, `schema_version` in models.py must be bumped
in the same change -- the shape is a public interface for automation.
"""

import json
from datetime import UTC, datetime

from agent_fuse.models import FuseEvent

EXPECTED_FIELDS = {
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


def _sample_event() -> FuseEvent:
    return FuseEvent(
        timestamp=datetime(2026, 5, 18, 5, 47, 3, tzinfo=UTC),
        provider="codex",
        session_id="8f21e6b2-0000-0000-0000-000000000000",
        rule="response_rate",
        severity="warning",
        window_seconds=600,
        observed=184,
        threshold=100,
        message="184 responses in the last 600s (threshold 100)",
    )


def test_schema_version_is_1() -> None:
    assert _sample_event().schema_version == 1


def test_event_name_is_stable() -> None:
    assert _sample_event().event == "fuse.triggered"


def test_field_set_is_exactly_the_documented_set() -> None:
    dumped = json.loads(_sample_event().model_dump_json())
    assert set(dumped.keys()) == EXPECTED_FIELDS


def test_severity_is_one_of_two_documented_values() -> None:
    for severity in ("warning", "severe"):
        event = FuseEvent(
            timestamp=datetime.now(UTC),
            provider="codex",
            session_id="s1",
            rule="response_rate",
            severity=severity,
            window_seconds=600,
            observed=1,
            threshold=1,
            message="x",
        )
        assert event.severity == severity


def test_output_is_single_line_json() -> None:
    dumped = _sample_event().model_dump_json()
    assert "\n" not in dumped
    json.loads(dumped)  # must round-trip
