# CLAUDE.md

Claude Code guidance for this repository. The shared rules for all agents are in
`AGENTS.md`. Read that file first. Nothing here overrides it.

## Before the first edit

- Read `AGENTS.md` (the four ways to do damage are at the top).
- Read the newest file in `docs/handoffs/`.

## Claude Code specifics

- **Stop hooks that ask you to commit.** This repository is often used in sessions with
  a hook that asks for a commit whenever the tree is dirty. Before committing, run
  `git status` and check that nothing under `pictures/` or
  `chapter7Experiments/chapter7_results/` is modified. If it is, a legacy suite was run
  in place: restore those paths with `git checkout -- <path>` instead of committing.
- **Running tests.** Use `pytest -m "not slow"` while iterating and
  `python scripts/run_verifiers.py` for the legacy suites. Do not invoke the
  `verify_*.py` files directly.
- **Long runs.** Experiments on the real data take hours. Start them in the background,
  write output to a path outside the repository, and commit the code before the run
  starts so the record's provenance names a commit that contains it.
- **Subagents.** When delegating a code-reading task, tell the subagent it is read-only
  and must not run repository scripts.
- **Data in context.** Never print participant-level rows. Inspect restricted files by
  shape, column names and aggregate counts only.
- **Edits to frozen files are refused by the tests.** If `tests/test_records.py` fails
  after your change, you edited a frozen file or a record. Revert it.

## Useful entry points

- `docs/ARCHITECTURE.md` for the layout and which code is authoritative.
- `docs/EXPERIMENT_REGISTRY.md` to find an experiment.
- `docs/RESEARCH_MAP.md` for the standing of each scientific claim.
- `docs/REPRODUCIBILITY.md` for commands and evidence levels.
- `docs/AGENT_HANDOFF.md` for the handoff format.
