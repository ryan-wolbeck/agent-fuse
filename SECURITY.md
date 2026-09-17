# Security & Privacy Policy

Agent Fuse reads local AI agent session telemetry that may contain
proprietary source code, prompts, credentials, shell commands, URLs, and
other sensitive material. Treating that data safely is a core requirement,
not an afterthought — see the README's "Privacy" section for the full
policy.

Summary of guarantees:

- No network requests. No telemetry of Agent Fuse's own.
- No raw prompt text, model output, tool arguments, shell command text,
  environment variable values, file contents, or local/repository paths
  are ever printed or written to disk by default.
- Tool-call repetition is detected via a one-way SHA-256 fingerprint, never
  the raw arguments.
- Strictly read-only against vendor state: Codex/Claude session files,
  config, and installation are never modified, truncated, or deleted.
- Nothing found inside a session history is ever executed.

## Reporting a vulnerability or privacy issue

If you find a way to make Agent Fuse leak sensitive session content (in
terminal output, JSON/JSONL output, logs, or exceptions), or a way for it
to write to or execute something from vendor state, please report it
privately rather than opening a public issue:

- Open a [GitHub Security Advisory](https://github.com/agent-fuse/agent-fuse/security/advisories/new)
  for the repository, or
- Email the maintainers listed in the repository's GitHub profile.

Please include:

- the Agent Fuse version and Python version
- the command run and, if possible, a redacted/synthetic fixture that
  reproduces the issue (not real session data)
- what you expected to be withheld and what was actually shown

We'll acknowledge reports within a few days. There is no bug bounty
program at this time.

## Supported versions

Agent Fuse is pre-1.0. Security fixes land on the latest released version;
older 0.x releases are not separately patched.
