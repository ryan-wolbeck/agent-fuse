# Contributing to Agent Fuse

Thanks for considering a contribution. Agent Fuse is intentionally narrow
in scope — see the "Limitations" and "Roadmap" sections of the README
before proposing a large feature. Issues and PRs that expand scope
(agent launching, process management, billing dashboards, an LLM-based
detector, a web UI) are likely to be declined, not because they're bad
ideas, but because they're a different product.

## Development setup

```bash
git clone https://github.com/ryan-wolbeck/agent-fuse
cd agent-fuse
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running checks locally

```bash
ruff check .
mypy
pytest
```

## Adding a rule

Rules live in `src/agent_fuse/rules/` and implement the `FuseRule`
protocol (`src/agent_fuse/rules/base.py`): a single `evaluate(session,
now) -> RuleResult` method. Rules:

- must be deterministic (no ML/LLM calls, no randomness)
- must handle missing/insufficient data by reporting it as such, not by
  substituting zero
- must not touch the terminal or know about output formatting
- need unit tests covering the below-threshold, at-threshold, and
  above-threshold boundary, plus a disabled-rule and insufficient-data case

## Adding an adapter

Adapters live in `src/agent_fuse/adapters/` and are the *only* code
allowed to know a vendor's JSON/log schema. See `adapters/codex.py` for
the expected shape and its extensive docstring on what is and isn't read
from the source data. Before adding an adapter:

- confirm the vendor's local telemetry actually contains what the
  existing rules need (turn/response boundaries, tool call identity,
  token usage if available) — don't fabricate fields the source can't
  support
- write the privacy review into the adapter's own docstring: what's read,
  what's deliberately never read
- add malformed/partial/unknown-event fixtures under `tests/fixtures/`
  alongside representative ones

## Reporting security or privacy issues

See [SECURITY.md](SECURITY.md) — please do not open a public issue for a
privacy-sensitive finding (e.g. "Agent Fuse can be made to print X").
