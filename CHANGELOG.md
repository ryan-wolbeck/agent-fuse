# Changelog

All notable changes to this project are documented in this file.
Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] - 2026-09-19

### Added

- Claude Code adapter reading local `$CLAUDE_CONFIG_DIR/projects/**/*.jsonl`
  transcript telemetry, verified against real local Claude Code 2.1.269
  session data (zero malformed/unrecognized lines across all real local
  transcripts sampled during development). Handles Claude Code's
  one-API-response-becomes-multiple-JSONL-lines shape (dedup by
  `message.id`), Anthropic's three-bucket token usage accounting
  (`input_tokens`/`cache_creation_input_tokens`/`cache_read_input_tokens`),
  and folds sidechain (subagent/Task-tool) turns into the parent session's
  metrics rather than fabricating lifecycle events the source data doesn't
  delineate.
- `agent-fuse doctor`/`inspect`/`scan`/`watch` now cover every installed
  provider automatically (no flag needed) via a new `providers.py`
  registry; `inspect`'s session table gained a PROVIDER column, and
  `scan --format json` gained a `"provider"` field per session.

### Changed

- `MetricsRegistry` is now keyed by `(provider, session_id)` instead of
  `session_id` alone, so two different providers' sessions can never merge
  metrics even if their (vendor-generated) IDs were ever to collide.
- `engine.scan_session_file` no longer hardcodes `CodexAdapter`; it takes
  an adapter constructor, so the engine has no vendor-specific knowledge at
  all (this was already true in spirit; it's now true in code).

### Fixed

- `scan`'s text summary line ("Scanned N Codex sessions.") was hardcoded
  to say "Codex" even when scanning multiple providers; now provider-
  agnostic.

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
