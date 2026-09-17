# Changelog

All notable changes to this project are documented in this file.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-09-17

Initial public release.

### Added

- Codex CLI adapter reading local `$CODEX_HOME/sessions/**/*.jsonl` rollout
  telemetry, verified against Codex CLI 0.150.1 and the public
  `openai/codex` protocol source.
- Vendor-neutral canonical event model (`AgentEvent`, `EventType`).
- Bounded-memory rolling session metrics (response counts, tool-call
  repetition, windowed token usage, lifetime cache ratio).
- Four deterministic rules: `response_rate`, `token_rate`,
  `context_replay`, `repeated_tool_call`.
- CLI: `agent-fuse doctor`, `agent-fuse inspect`, `agent-fuse scan`,
  `agent-fuse watch`.
- JSON (`scan --format json`) and JSONL (`watch --format jsonl`) machine
  output with a versioned `fuse.triggered` event schema.
- YAML configuration (`.agent-fuse.yaml`) with conservative defaults.
- Privacy-by-construction design: no network access, no raw prompt/tool
  content in any output path, one-way hashed tool signatures.

### Known limitations

See the README "Limitations" section — most notably, v0.1 supports Codex
only, is strictly read-only (no automatic termination), and does not
emit `SESSION_END`/`SUBAGENT_START`/`SUBAGENT_END` for Codex since the
source telemetry has no corresponding structural event for them.
