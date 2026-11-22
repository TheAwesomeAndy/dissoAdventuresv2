# AGENTS.md

Shared instructions for any AI agent or automated tool working in this repository.
Tool-specific files (such as `CLAUDE.md`) point here and add nothing that conflicts.

## Read this first: four ways to do damage

1. **Do not run the legacy test suites in the working tree.**
   `chapter4Experiments/verify_chapter4.py` and `chapter7Experiments/verify_chapter7.py`
   overwrite 21 tracked figures and one result file with synthetic output. Use
   `python scripts/run_verifiers.py`, which runs every suite in an isolated copy. If
   `git status` shows modified files under `pictures/` or
   `chapter7Experiments/chapter7_results/` that you did not mean to change, restore
   them with `git checkout -- <path>` and never commit them.
2. **Do not edit a frozen file without regenerating its record.**
   `python scripts/check_records.py --list-frozen` prints them. Their bytes are hashed
   in committed result records. Any edit, including formatting, comments or type
   annotations, breaks the link between a result and the code that produced it. See
   "Changing frozen code" below.
3. **Do not edit, move or "fix" the v1 record on your own initiative.** The defects the
   2026-11 audit found were fixed at the owner's request and are logged in
   `docs/v1/DEFECT_FIXES.md`. If you find another, record it and ask.
4. **Never commit data.** Raw EEG files, clinical tables, spreadsheets and caches
   derived from them stay out of git. The cohort is restricted human-subjects data.
   Never print participant-level rows into a log, a commit message or a document.

## 1. What this repository is

ARSPI-Net (Affective Reservoir-Spike Processing and Inference Network) is a measurement
chain for multichannel EEG: a fixed-weight leaky integrate-and-fire reservoir turns each
electrode's signal into spikes, binned spike counts become a code, a fixed random
projection compresses it, and a linear readout classifies affective picture conditions
across participants. The repository holds a PhD dissertation's code and manuscript
source, a corrective audit of it, and a second-generation validation ("v2") that tested
its claims on the same data.

## 2. Scientific position, in six lines

- The corrected evaluation protocol is sound and leak-free.
- The reservoir code decodes condition well above chance and beats band-power features.
- Raw ERP amplitudes decode better than the reservoir code.
- The reservoir adds nothing over coarse amplitude binning; recurrence adds nothing; the
  network at its original gain has no fading memory.
- Random projection does not beat PCA once PCA is set up fairly.
- Per-epoch z-scoring, which the original preprocessing uses, fabricates early and
  pre-stimulus decoding. This replicates on public data and is the live research line.

The authority for these statements is `V2_UPGRADE.md`. The standing of every idea is in
`docs/RESEARCH_MAP.md`. Do not restate v1 numbers as current without checking there.

## 3. Layout

| Path | What | Authority |
|---|---|---|
| `arspinet_v2/` | the library: reservoir, dynamics, features, evaluation, controls, nuisance, provenance | **authoritative**, frozen bytes |
| `experiments/v2/` | v2 runners, records, generated report | **authoritative**, runners frozen |
| `experiments/zscore_artifact/` | z-scoring artefact study on ERP CORE | ongoing |
| `experiments/records.lock.json` | index of result records and their code | authoritative |
| `tests/` | pytest suite, no data needed | maintained |
| `scripts/` | tooling: isolated verifier runner, record checker, registry generator | maintained |
| `docs/` | architecture, research map, experiment registry, reproducibility, audit, handoffs | maintained |
| `experiments/confirmatory/` | v1 corrected protocol and its record | v1 record, frozen |
| `chapter4Experiments/` to `chapter7Experiments/`, other `experiments/*`, `defense_figures/`, `validation/` | v1 chapter analyses, verifiers, figures | v1 record: historical, mostly archived tier |
| `dissertation/`, `pictures/` | manuscript source and figures | v1 record: do not edit |

Start with `docs/ARCHITECTURE.md`. To find an experiment, use
`docs/EXPERIMENT_REGISTRY.md` (or its source, `docs/experiment_registry.json`).

## 4. Setup

```text
pip install -r requirements-lock.txt
pip install -e . --no-deps
pip install pytest ruff
```

Python 3.11 or newer. No data is needed for anything in section 5.

## 5. Commands

| Purpose | Command |
|---|---|
| Fast tests | `pytest -m "not slow"` |
| All tests, including the synthetic end-to-end run | `pytest` |
| Record and frozen-file check | `python scripts/check_records.py` |
| Legacy suites, isolated | `python scripts/run_verifiers.py` |
| Lint, format, types | `ruff check .`, `ruff format --check arspinet_v2 experiments/v2 experiments/zscore_artifact tests`, `mypy` |
| Compare a regenerated record with the committed one | `python scripts/compare_records.py <committed> <new>` |
| Regenerate the experiment index | `python scripts/build_registry_doc.py` |

Running an experiment needs restricted data and is described in
`docs/REPRODUCIBILITY.md`. Most agents will not have the data. Say so instead of
substituting synthetic numbers, and never write a synthetic result where a real one is
expected.

## 6. Scientific invariants

Change none of these casually. Each is enforced by a test where one is named.

| Invariant | Enforced by |
|---|---|
| Participants never appear in both training and test folds | `tests/test_evaluation.py` |
| Scaling, projection basis and readout are fitted on training participants only | code review; `tests/test_evaluation.py` (parity with v1) |
| The projection is data-independent, seed 20240517, shared across electrodes | `tests/test_features.py` |
| The LIF update rule, weight draw and binning | `tests/test_reservoir.py` (bit parity with v1, golden spike counts) |
| The fading-memory rule thresholds | `tests/test_dynamics.py` |
| Cohorts: 211 participants and 633 observations; 210 and 840. Participant 127 excluded from both, 195 from the four-category stream | `tests/test_records.py` |
| Seeds, fold counts, bootstrap and permutation counts in the runners | frozen files |
| A record is never edited by hand | `tests/test_records.py` |
| Numbers in prose equal the record | `tests/test_records.py` |
| Balanced accuracy is the metric; chance is 1/3 and 1/4 | convention |

Also: do not tune anything on held-out accuracy. The reservoir gain was chosen by a rule
that uses no labels. Keep it that way.

## 7. How to make a change

1. Read the newest file in `docs/handoffs/`.
2. Find what you are touching in `docs/EXPERIMENT_REGISTRY.md` and
   `docs/RESEARCH_MAP.md`. Note its tier and standing.
3. Decide which kind of change it is:
   - **New code, tests, docs or tooling:** proceed.
   - **A new experiment:** add a runner under `experiments/<study>/` that writes one JSON
     record with a provenance block; add a registry entry and a lock entry; add a test
     that any prose quoting the record matches it.
   - **Anything that changes a reported number, a frozen file or the v1 record:** stop
     and ask the owner.
4. Make the change in small commits. After each: `pytest -m "not slow"`,
   `python scripts/check_records.py`, `ruff check .`.
5. Before finishing: `pytest`, `python scripts/run_verifiers.py`, and `git status` to
   confirm nothing under `pictures/` or `chapter7Experiments/chapter7_results/` changed.
6. Write a handoff (`docs/AGENT_HANDOFF.md`).

### Changing frozen code

Only with the data and the owner's agreement:

1. edit the code and mark the affected records `pending_regeneration` in
   `experiments/records.lock.json` (see the entries written in 2026-11 for the form);
   commit. `scripts/check_records.py` then reports them as PENDING. That state is
   allowed on a branch and refused on `main`;
2. rerun the experiment on the authorized data, writing the new record outside the
   repository first. Use the BLAS thread setting in the record's lock entry
   (`blas_threads`, and the `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` prefix of its
   command). A different thread count can change a value by one observation, which
   `scripts/compare_records.py` reports as a difference;
3. `python scripts/compare_records.py <committed> <new>`. If the change was meant to be
   behaviour-preserving, every value must be equal;
4. copy the new record in, `python scripts/check_records.py --update <record>`, commit;
5. regenerate reports, then fix every prose number `tests/test_records.py` flags;
6. state in the commit message which numbers changed and why, or that none did.

### Reporting results

- Say which evidence level you mean: replayed, locked, structural or unverifiable
  (`docs/REPRODUCIBILITY.md`).
- Separate what you ran from what you read and from what you infer.
- Report null and unfavourable results as prominently as favourable ones. This
  repository's value is that it does.
- A claim about timing needs input that was not z-scored per epoch.

## 8. Things that look like tasks but are not yours to do

- Editing the dissertation source. Corrections go in `DISSERTATION_ERRATUM.md`.
- Rewriting v1 documents to match v2. The stale ones carry a status note at the top and
  `results_manifest.json` carries a `standing_2026_11` field per entry. Add to those;
  do not rewrite the original text.
- Repairing a v1 defect you have just found. Record it and ask (see rule 3 above).
- Deleting large or redundant-looking files. Several are the only copy of something
  that cannot be regenerated.
- Removing the participant-keyed result files or rewriting git history. The owner has
  reviewed both (audit findings S1 and S2) and decided to leave them as they are.

## 9. Communicating with other agents

Use the handoff format in `docs/AGENT_HANDOFF.md`, one file per session under
`docs/handoffs/`. Put findings in tracked files, not only in chat. If you find an error
in someone else's handoff or document, record it in yours and add a "Superseded by" line
to theirs.
