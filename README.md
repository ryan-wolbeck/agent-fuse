# Agent Fuse

A local circuit breaker monitor for runaway AI coding agents.

AI coding agents increasingly run unattended for long stretches: they spawn
subagents, replay enormous contexts, poll tools in a loop, and burn through
tokens and API budget before anyone notices. Agent Fuse watches local agent
telemetry that's already on your disk and warns, deterministically and
locally, when a session crosses conservative safety thresholds.

```
pip install agent-fuse
agent-fuse watch
```

```
Agent Fuse
Watching Codex sessions...
✓ session a912... normal
✓ session 29ca... normal
⚠ RUNAWAY PATTERN
Session: 8f21...
  response_rate (last 600s): 184
  repeated_tool_call (last 300s): 73
  context_replay (lifetime): 96.7% cached

Triggered rules:
  response_rate
    184 responses in the last 600s (threshold 100)
    threshold: 100
  repeated_tool_call
    a single tool signature repeated 73 times in the last 300s (threshold 30)
    threshold: 30
  context_replay
    96.7% cached/replayed input across 2,431,992 input tokens (threshold 90%)
    threshold: 0.9000

Agent Fuse does NOT terminate the process. Inspect the session before continuing.
```

**Agent Fuse v0.1 is read-only. It warns; it does not terminate agents.**

## Why not just use existing usage dashboards or the agent's own loop detection?

Some agent runtimes (e.g. OpenClaw) already detect tool loops internally,
and desktop widgets like [token-monitor](https://github.com/Javis603/token-monitor)
track token spend across many tools. Neither of those is what Agent Fuse
is for. Agent Fuse doesn't run inside the agent and doesn't track cost — it
reads the vendor's own local session logs from the outside, after the fact
or as they're written, and applies the same small set of deterministic
rules regardless of which agent produced them. It's a second, independent
pair of eyes, not a replacement for either.

This is a narrow tool solving a narrow problem: **local, deterministic,
vendor-neutral detection of obviously pathological agent behavior.** It is
not the first or only tool to look at agent resource usage, and it doesn't
try to be a complete observability or cost-management solution.

## Quickstart

```bash
pip install agent-fuse
agent-fuse doctor   # confirm Agent Fuse can find your local Codex telemetry
agent-fuse scan     # dry-run the rules against your existing session history
agent-fuse watch    # monitor live sessions and warn on runaway patterns
```

## Supported agents

| Agent | Status | Telemetry source |
|---|---|---|
| Codex CLI | Supported | `$CODEX_HOME/sessions/**/*.jsonl` rollout files (verified against Codex CLI 0.150.1 and the public `openai/codex` protocol source) |
| Claude Code | Not yet implemented | See [Roadmap](#roadmap) |
| OpenCode | Not yet implemented | See [Roadmap](#roadmap) |

Codex was prioritized for v0.1 because its local rollout format is the
richest, best-documented source of the telemetry these rules need (per-turn
completion events, token usage with cache breakdown, and tool call/result
records). A fragile, half-working second adapter would have cost more trust
than it was worth; a Claude Code adapter is the top roadmap item.

## Rules

All rules are deterministic and threshold-based — there is no ML/LLM
anomaly detection in Agent Fuse. **The default thresholds are initial
safety heuristics, not scientifically validated universal limits.** They
were chosen to be conservative enough not to fire on typical agent
sessions, but your workload may legitimately be larger or noisier than
what we assumed. Tune them in `.agent-fuse.yaml`.

| Rule | Default | What it means |
|---|---|---|
| `response_rate` | 100 responses / 600s | Too many completed model turns in a rolling 10-minute window. |
| `token_rate` | 2,000,000 input tokens / 600s | Too much input token volume in a rolling 10-minute window. Skipped (not fabricated) if the provider reports no token data. |
| `context_replay` | ≥90% cached input, over a session with ≥100,000 lifetime input tokens | A large, mostly-replayed context. This is a *lifetime* session metric, not a rolling window — see [rules/context_replay.py](src/agent_fuse/rules/context_replay.py) for why. **Caching itself is not the problem** — most healthy long sessions have a high cache ratio. This rule only fires on the combination of large scale *and* an extreme ratio. |
| `repeated_tool_call` | 30 repetitions of the same tool signature / 300s | The same normalized tool call firing over and over in a short window. |

A session is reported as **severe** rather than **warning** when two or
more rules trigger at the same time — a simple, documented escalation
rule, not a statistically derived risk score.

## Configuration

Agent Fuse looks for `.agent-fuse.yaml` in the current directory (walking
up to the filesystem root), then `~/.config/agent-fuse/config.yaml`, then
falls back to built-in defaults.

```yaml
version: 1
response_rate:
  enabled: true
  window_seconds: 600
  max_responses: 100
token_rate:
  enabled: true
  window_seconds: 600
  max_input_tokens: 2000000
context_replay:
  enabled: true
  minimum_input_tokens: 100000
  max_cached_ratio: 0.90
repeated_tool_call:
  enabled: true
  window_seconds: 300
  max_repetitions: 30
```

Any rule can be disabled with `enabled: false`. An invalid config file is a
hard error (exit code 2) — Agent Fuse never silently falls back to
defaults when you've explicitly configured something, since that could
mask thresholds you believe are active.

## Commands

```
agent-fuse doctor    # read-only environment/health check
agent-fuse inspect   # sanitized aggregate view of discoverable sessions
agent-fuse scan      # historical dry-run of the rules against past sessions
agent-fuse watch     # live monitoring of active/recent sessions
```

Machine-readable output:

```bash
agent-fuse scan --format json
agent-fuse watch --format jsonl
agent-fuse watch --format jsonl --quiet   # automation-friendly: fuse events only
```

### Fuse event schema

Each triggered rule produces a versioned JSON record:

```json
{
  "schema_version": 1,
  "event": "fuse.triggered",
  "timestamp": "2026-05-18T05:47:03.112000+00:00",
  "provider": "codex",
  "session_id": "8f21e6b2-...",
  "rule": "response_rate",
  "severity": "warning",
  "window_seconds": 600,
  "observed": 184,
  "threshold": 100,
  "message": "184 responses in the last 600s (threshold 100)"
}
```

Treat this shape as a public, versioned interface. `schema_version` will be
bumped on any breaking change; see [tests/unit/test_fuse_event_schema.py](tests/unit/test_fuse_event_schema.py).

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success — no rule violations observed |
| 1 | A rule violation was detected (`scan`, `watch`) |
| 2 | Configuration or input error |
| 3 | Provider/discovery failure (e.g. Codex not found) |

`Ctrl+C` during `watch` is a clean shutdown, not a crash — exit code
reflects whether anything triggered during that run.

## Privacy

Agent Fuse is built to be run against your real, potentially sensitive
agent history. By default and by design it:

- **makes no network requests** and has no telemetry of its own
- never prints or stores prompt text, model responses, tool arguments,
  shell command text, environment variable values, file contents, or raw
  local/repository paths
- identifies repeated tool calls using a one-way SHA-256 fingerprint of
  the tool name plus its arguments — never the raw arguments themselves
- sanitizes its own error output (vendor error messages are recorded as
  "an error occurred", not the underlying text)
- **never executes** anything found inside a session history
- is strictly **read-only** against vendor state: it never modifies,
  truncates, rewrites, or deletes anything under `$CODEX_HOME`

See [SECURITY.md](SECURITY.md) for the full policy and how to report a
privacy or security issue.

## Read-only guarantee

Agent Fuse v0.1 never terminates, pauses, signals, or otherwise touches an
agent process, and never modifies session files or vendor configuration.
It observes and warns — that's the entire product surface for this
release. Explicit, user-configured actions (see [Roadmap](#roadmap)) may
be added later, opt-in, and clearly separated from the default read-only
behavior.

## Architecture

```
Codex local telemetry
        │
        ▼
   CodexAdapter            (vendor schema lives ONLY here)
        │
        ▼
Canonical Agent Events      (models.py — vendor-neutral)
        │
        ▼
Rolling Session Metrics     (metrics.py — bounded deques + lifetime counters)
        │
        ▼
Deterministic Rules         (rules/ — independently testable)
        │
        ▼
   Fuse Events
        ├── terminal (Rich)
        └── JSON / JSONL
```

Adapters are a replaceable compatibility boundary: nothing above the
adapter layer knows Codex's JSON shapes exist. A future adapter (Claude
Code, OpenCode, ...) only needs to produce the same canonical `AgentEvent`
type; no other layer changes.

## Limitations

Read this section before trusting Agent Fuse for anything important.

- **Codex-only in v0.1.** No Claude Code or OpenCode adapter yet.
- **Not anomaly detection.** Every rule is a fixed threshold. It will miss
  slow-burn or subtle runaway behavior that never crosses a threshold, and
  it can false-positive on a workload that's simply large by design.
- **`SESSION_END` is never emitted** for Codex — the rollout format has no
  explicit "session ended" event, so Agent Fuse infers staleness from file
  modification time (`discovery.py`) rather than fabricating one.
  Similarly, Codex subagent threads appear as separate session files
  linked by metadata, not as `SUBAGENT_START`/`SUBAGENT_END` events.
  Both event types exist in the canonical model for forward compatibility
  with adapters that can support them, not because Codex does.
  \- **An in-flight turn with no `task_complete` yet is not counted.**
  If a session ends mid-turn, that final turn is not reflected in metrics;
  Agent Fuse does not fabricate a response event for a turn it never saw
  finish.
- **Rules are conservative by design and default thresholds are
  heuristics**, not a validated model of "normal" — see
  [Configuration](#configuration).
- **`watch`'s initial attach to a very large pre-existing session file
  does a one-time linear read** to build rolling context before it starts
  tailing new appends; it does not re-read the file on every poll.
- No desktop notifications, webhooks, or automatic kill switch in v0.1 —
  see Roadmap.

## Roadmap

Not implemented in v0.1, and not part of this release's scope:

- Claude Code adapter (top priority — pending investigation of a stable
  local telemetry source with sufficient tool/token detail)
- OpenCode adapter
- OpenTelemetry input
- Additional deterministic rules (e.g. subagent fan-out / lineage limits,
  MCP resource limits)
- Desktop notifications, webhooks, Prometheus export
- Explicit, opt-in kill/pause hooks and budget policies

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE).
