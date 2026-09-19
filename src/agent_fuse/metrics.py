"""Rolling, bounded-memory metrics for a single agent session.

Two kinds of state are kept, deliberately separate:

* Windowed state (deques) backs the response-rate, token-rate, and
  repeated-tool-call rules. Entries older than the largest configured
  window are evicted in O(1) amortized time -- no rule ever triggers a
  full-history rescan.
* Lifetime counters (plain ints) back the context-replay rule, which is
  defined in terms of a session's cumulative scale and cache ratio rather
  than a short rolling window (see rules/context_replay.py for why).

`record_event` is the only mutation entrypoint and is safe to call for
every canonical AgentEvent regardless of type; events that don't affect
metrics (e.g. UNKNOWN) are no-ops.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .models import AgentEvent, EventType


@dataclass
class SessionMetrics:
    session_id: str
    provider: str
    max_window_seconds: int = 600

    first_seen: datetime | None = None
    last_seen: datetime | None = None

    # Lifetime (unbounded, O(1) per-event) counters.
    total_responses: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_cached_input_tokens: int = 0
    total_output_tokens: int = 0
    has_token_data: bool = False

    # Windowed state.
    _response_times: deque[datetime] = field(default_factory=deque, repr=False)
    _tool_calls: deque[tuple[datetime, str]] = field(default_factory=deque, repr=False)
    _tool_call_counts: Counter[str] = field(default_factory=Counter, repr=False)
    _token_events: deque[tuple[datetime, int, int, int]] = field(default_factory=deque, repr=False)
    _window_input_tokens: int = field(default=0, repr=False)
    _window_cached_input_tokens: int = field(default=0, repr=False)
    _window_output_tokens: int = field(default=0, repr=False)

    def record_event(self, event: AgentEvent) -> None:
        ts = event.timestamp
        if self.first_seen is None or ts < self.first_seen:
            self.first_seen = ts
        if self.last_seen is None or ts > self.last_seen:
            self.last_seen = ts

        if event.event_type is EventType.MODEL_RESPONSE:
            self.total_responses += 1
            self._response_times.append(ts)

            has_any_token_field = (
                event.input_tokens is not None
                or event.cached_input_tokens is not None
                or event.output_tokens is not None
            )
            if has_any_token_field:
                self.has_token_data = True
                in_tok = event.input_tokens or 0
                cached_tok = event.cached_input_tokens or 0
                out_tok = event.output_tokens or 0
                self.total_input_tokens += in_tok
                self.total_cached_input_tokens += cached_tok
                self.total_output_tokens += out_tok
                self._token_events.append((ts, in_tok, cached_tok, out_tok))
                self._window_input_tokens += in_tok
                self._window_cached_input_tokens += cached_tok
                self._window_output_tokens += out_tok

        elif event.event_type is EventType.TOOL_CALL:
            self.total_tool_calls += 1
            if event.tool_signature is not None:
                self._tool_calls.append((ts, event.tool_signature))
                self._tool_call_counts[event.tool_signature] += 1

        self._evict_older_than(ts - timedelta(seconds=self.max_window_seconds))

    def _evict_older_than(self, cutoff: datetime) -> None:
        while self._response_times and self._response_times[0] < cutoff:
            self._response_times.popleft()

        while self._tool_calls and self._tool_calls[0][0] < cutoff:
            _, sig = self._tool_calls.popleft()
            self._tool_call_counts[sig] -= 1
            if self._tool_call_counts[sig] <= 0:
                del self._tool_call_counts[sig]

        while self._token_events and self._token_events[0][0] < cutoff:
            _, in_tok, cached_tok, out_tok = self._token_events.popleft()
            self._window_input_tokens -= in_tok
            self._window_cached_input_tokens -= cached_tok
            self._window_output_tokens -= out_tok

    def responses_in_window(self, now: datetime, window_seconds: int) -> int:
        cutoff = now - timedelta(seconds=window_seconds)
        return sum(1 for t in self._response_times if t >= cutoff)

    def max_repeated_tool_signature(
        self, now: datetime, window_seconds: int
    ) -> tuple[str | None, int]:
        cutoff = now - timedelta(seconds=window_seconds)
        counts: Counter[str] = Counter()
        for ts, sig in self._tool_calls:
            if ts >= cutoff:
                counts[sig] += 1
        if not counts:
            return None, 0
        sig, count = counts.most_common(1)[0]
        return sig, count

    def token_totals_in_window(
        self, now: datetime, window_seconds: int
    ) -> tuple[int, int, int] | None:
        if not self.has_token_data:
            return None
        cutoff = now - timedelta(seconds=window_seconds)
        in_tok = cached_tok = out_tok = 0
        found = False
        for ts, i, c, o in self._token_events:
            if ts >= cutoff:
                in_tok += i
                cached_tok += c
                out_tok += o
                found = True
        if not found:
            return (0, 0, 0)
        return (in_tok, cached_tok, out_tok)

    def lifetime_cached_ratio(self) -> float | None:
        if not self.has_token_data or self.total_input_tokens == 0:
            return None
        return self.total_cached_input_tokens / self.total_input_tokens

    def duration_seconds(self) -> float | None:
        if self.first_seen is None or self.last_seen is None:
            return None
        return (self.last_seen - self.first_seen).total_seconds()


class MetricsRegistry:
    """Tracks SessionMetrics per (provider, session_id), bounded by rule
    window config.

    Keyed on the pair rather than session_id alone: session IDs are
    vendor-generated UUIDs, and with more than one provider adapter now
    supported there is no guarantee two different providers' IDs can't
    collide (or, more mundanely, that test fixtures across providers won't
    reuse simple ids like "s1"). A collision here would silently merge two
    unrelated sessions' metrics.
    """

    def __init__(self, max_window_seconds: int = 600) -> None:
        self.max_window_seconds = max_window_seconds
        self._sessions: dict[tuple[str, str], SessionMetrics] = {}

    def get_or_create(self, provider: str, session_id: str) -> SessionMetrics:
        key = (provider, session_id)
        metrics = self._sessions.get(key)
        if metrics is None:
            metrics = SessionMetrics(
                session_id=session_id,
                provider=provider,
                max_window_seconds=self.max_window_seconds,
            )
            self._sessions[key] = metrics
        return metrics

    def record(self, event: AgentEvent) -> SessionMetrics:
        metrics = self.get_or_create(event.provider, event.session_id)
        metrics.record_event(event)
        return metrics

    def __iter__(self) -> Iterator[SessionMetrics]:
        return iter(self._sessions.values())

    def __len__(self) -> int:
        return len(self._sessions)

    def get(self, provider: str, session_id: str) -> SessionMetrics | None:
        return self._sessions.get((provider, session_id))

    def drop(self, provider: str, session_id: str) -> None:
        self._sessions.pop((provider, session_id), None)
