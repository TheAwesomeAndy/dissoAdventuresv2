# Handoff: repository audit and engineering pass

- **Date:** 2026-11-21
- **Author:** Claude (AI agent), working for the repository owner
- **Branch and commit:** `ccr-0221a213-rhj47r`; this file is part of the commit that
  completes the pass (baseline was `913d457`)
- **Supersedes:** none (first handoff in this format; the older, v1-era
  `defense_figures/HANDOFF_TO_NEXT_AGENT.md` covers only the defense figures)
- **Superseded by:** `docs/handoffs/2026-11-22-decisions-regeneration-merge.md`

## Task objective

The owner asked for a full audit of the dissertation repository and its conversion from
a research workspace into a maintained scientific software project that people and AI
agents can navigate, without altering any scientific result.

## Relevant files and modules

- `docs/REPOSITORY_AUDIT.md`: findings, risks, plan, and what was and was not done.
- `docs/audit/`: the four code-reading reports the audit rests on, unedited.
- `AGENTS.md`: rules for agents. The four ways to do damage are at the top.
- `docs/ARCHITECTURE.md`, `docs/RESEARCH_MAP.md`, `docs/EXPERIMENT_REGISTRY.md`,
  `docs/REPRODUCIBILITY.md`: the navigational documents.
- `docs/experiment_registry.json`: source of truth for the experiment index.
- `experiments/records.lock.json`, `scripts/check_records.py`: records and frozen code.
- `scripts/run_verifiers.py`: the only safe way to run the legacy suites.
- `tests/`: pytest suite.
- `pyproject.toml`, `requirements-lock.txt`: packaging and environment.

## Current implementation status

Done:

- Baseline recorded at `913d457`: 22 of 22 CI suites pass, 772 checks.
- Packaging, dependency consolidation, lint and test configuration.
- Record lock: 8 records tied to 18 frozen code files; 5 legacy records tracked as
  unverified.
- 94 pytest tests in 9 files.
- Isolated verifier runner with per-suite baseline counts.
- Replay check of the v2 record.
- Experiment registry (29 entries) with a generated document.
- Architecture, research map, reproducibility guide, audit, agent instructions.
- New root README; v1 README preserved at `docs/v1/README_v1.md`.
- CI job for lint, record check and pytest.

Not started (by decision, see below): any change to v1 code, v1 results, frozen v2
code, tracked data files or git history.

## Decisions made

1. **No physical restructure.** Moving v1 directories would break hashed paths in the
   v1 record, manifest paths, 51 `sys.path` lines and the paths the dissertation cites.
   Layering and indexing were used instead. Rejected: a src-layout move with an archive
   directory, which needs the v2 records regenerated on the data.
2. **Frozen files stay byte-identical**, so `arspinet_v2/` was not formatted or
   type-annotated. Rejected: reformatting now and rerunning about four hours of
   experiments purely for style.
3. **Known v1 defects are documented, not fixed.** Each fix changes an archived result.
4. **Legacy verifiers were not rewritten.** They are wrapped, counted and isolated.
5. **Records without embedded provenance were pinned from git history**, and labelled
   "reconstructed" in the lock, after checking that their producing files had not
   changed since the record's commit.
6. **The registry is JSON with a generated Markdown view**, so a test can check every
   path and command in it.
7. **No configuration directory was introduced.** The v2 runner constants live in
   frozen files; moving them belongs with the next record regeneration.

## Changes performed

- Added: `pyproject.toml`, `requirements-lock.txt`, `AGENTS.md`, `CLAUDE.md`,
  `tests/`, `scripts/check_records.py`, `scripts/run_verifiers.py`,
  `scripts/build_registry_doc.py`, `experiments/records.lock.json`,
  `experiments/v2/check_v2_replay.py`, and under `docs/`: the audit and its appendices,
  architecture, research map, registry (JSON and Markdown), reproducibility guide,
  handoff protocol, this handoff, the data-file classification.
- Moved: the root README text to `docs/v1/README_v1.md`, with a provenance note.
- Edited: `.gitignore` (packaging artefacts); `.github/workflows/verify.yml` (new
  `tests` job; the `verify` job is unchanged); `V2_UPGRADE.md` (one limitation line
  that contradicted the same file); `experiments/README.md` (a navigation note at the
  top); three unfrozen scripts for lint (`experiments/v2/make_report.py`,
  `experiments/v2/verify_v2.py`, `experiments/zscore_artifact/make_figure.py`).
- **Not touched:** every frozen file, every record, every v1 script, verifier, figure,
  result file and dissertation source. `python scripts/check_records.py` passes.

## Tests executed and their results

| Command | Result |
|---|---|
| 22 CI commands at `913d457` in a scratch clone | 22 pass, 772 checks |
| `python scripts/run_verifiers.py` after the changes | 22 of 22 suites ok; no tracked file modified |
| `pytest` | 94 passed |
| `python scripts/check_records.py` | OK: 8 records, 18 frozen files, 5 unverified legacy records |
| `ruff check .` | clean (frozen files and the v1 record are excluded by design) |
| `python experiments/v2/check_v2_replay.py` on the SHAPE data | 19 of 19 replayed values equal the record |
| GitHub Actions on this branch | green on every push through the first commit of this pass |

Evidence levels: the v2 replay is "replayed"; everything else above is "structural" or
"locked" (`docs/REPRODUCIBILITY.md`).

## Findings

All verified unless marked.

1. Both repositories are public and the clinical table is at the tip of `main` in each
   (audit S2). Seven result files and two figure data files are keyed by real
   participant ID (S1).
2. Running the legacy suites in place modifies 21 tracked files (E1).
3. The v2 record replays exactly: 19 of 19 replayed values equal the committed
   record (fading-memory sweep, selected gain, and eight representation rows with
   intervals per cohort), with reservoir encodings recomputed from scratch.
4. Five exploratory records cannot be tied to the code that wrote them (I5).
5. Several manifest entries map results to scripts that do not compute them (I2).
6. Two confident code defects in archived or illustrative experiments were confirmed
   by a second reading (I6); others are reported by the reviewers and not re-verified.
7. Sixteen reservoir definitions form four classes; the Chapter 6 dynamics results
   describe a different network from the classification reservoir (I7).
8. Suspected, not checked: a letter listed in the v1 README as submitted to IEEE Signal
   Processing Letters may rely on the variance ratio or the random-projection advantage
   that v2 withdrew. The manuscript is not in the repository.

## Known limitations

- The audit of v1 code is a reading, not a rerun. No v1 number was re-derived.
- Reviewer findings outside the eight spot checks were not individually re-verified.
- The replay covers 19 values of the v2 record and reused cached preprocessed epochs.
- `arspinet_v2/` has no type annotations and is outside the linter.
- The doc-path test checks only paths written in backticks in the nine entry-point
  documents.
- Legacy verifier quality is unchanged.

## Remaining work

1. Owner decisions listed below. Blocked on the owner.
2. Regenerate the v2 records after formatting and typing `arspinet_v2/`; move runner
   constants to configuration. Done when the replay script and `check_records.py`
   pass on the new records. Blocked on data access and owner agreement.
3. Obtain the electrode map; then build a spatially valid kappa null. Blocked on the
   data provider.
4. Tune readout regularization for the amplitude baselines (`V2_UPGRADE.md`, open item).
5. Systematic literature review for the z-scoring artefact; cluster test over the time
   course; a third dataset.
6. Replace tautological legacy checks as areas are next touched.

## Decisions waiting on the owner

1. Make the repositories private, purge the clinical table from history, and decide the
   fate of the participant-keyed result files. Inform the data provider and the IRB?
2. Fix or annotate the confirmed defects in archived experiments?
3. Correct or mark superseded the stale manifest entries and v1 documents?
4. Prepare a dissertation erratum covering the Chapter 6 echo-state evidence, the
   r = 0.82 statistic, the PCA-64 comparisons and window-specific statements?
5. Check the submitted letter against the v2 findings?
6. Recreate the `v1.0-defense` tag here and make `main` the default branch after merging
   this branch?

## Recommended next action

Resolve owner decision 1. Everything else can wait; that one cannot.
