# Agent handoff protocol

How one agent (or person) leaves work so the next can resume without the conversation
that produced it. Shared rules are in `AGENTS.md`; this file is only the handoff format.

## When to write one

- At the end of any session that changed files, ran an experiment or reached a finding.
- Before stopping with work incomplete.
- When you discover something the next agent would otherwise rediscover.

## Where

One file per session in `docs/handoffs/`, named by date and topic, as in
`docs/handoffs/2026-11-21-repository-engineering.md`. Do not edit
earlier handoffs except to add a line under "Superseded by". The newest file is the
current state; `ls docs/handoffs | tail -1` finds it.

## What must be true before you hand off

1. `pytest -m "not slow"` passes, or the failures are listed under Known limitations.
2. `python scripts/check_records.py` passes.
3. `git status` is clean, or every uncommitted path is listed with the reason.
4. No number appears in the handoff that is not in a committed record or command output
   you can point to.

## Template

Copy this block. Keep every heading; write "none" where a section is empty.

```markdown
# Handoff: <topic>

- **Date:** YYYY-MM-DD
- **Author:** <agent or person>
- **Branch and commit:** <branch> at <short hash>
- **Supersedes:** <earlier handoff file, or none>

## Task objective
What was asked, in one or two sentences, and by whom.

## Relevant files and modules
Paths, with one line each on why they matter.

## Current implementation status
What exists and works now. Distinguish done, partly done and not started.

## Decisions made
Each decision with its reason and the alternative that was rejected.

## Changes performed
What changed, grouped by commit. Say explicitly whether any frozen file, record or
v1 file was touched.

## Tests executed and their results
Exact commands and their outcome, with counts. State the evidence level for any
reproduction claim: replayed, locked, structural or unverifiable.

## Findings
Anything learned about the science or the code that is not yet in a tracked document.
Mark each as verified (how) or suspected.

## Known limitations
What is wrong, weak or unverified in the work handed over.

## Remaining work
Ordered list. For each item: what done looks like, and what it is blocked on.

## Decisions waiting on the owner
Questions only the repository owner can answer. One line each.

## Recommended next action
The single thing the next agent should do first.
```

## Communicating between agents working at the same time

- Claim a file or experiment by naming it in your handoff's "Remaining work" before you
  start, and check the newest handoff before starting anything.
- Never rely on another agent's summary for a number. Read the record.
- If you find an error in an earlier handoff, write it in your own under Findings and
  add a "Superseded by" line to the old one. Do not silently rewrite history.
