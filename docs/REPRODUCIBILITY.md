# Reproducibility

What can be reproduced, how, and what cannot. Commands here were run during the
2026-11-21 audit unless marked otherwise.

## 1. Environment

```text
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt      # exact versions behind the v2 records
pip install -e . --no-deps
pip install pytest ruff                   # for the checks below
```

`requirements-lock.txt` is the environment that produced the v2 and z-scoring records
(Python 3.13.18). CI uses Python 3.11 with the lower bounds in `pyproject.toml`. The v1
confirmatory record was produced with Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1 and
scikit-learn 1.9.0, as its provenance block states.

Some v1 scripts need packages that are not declared because they are outside the tested
path: PyTorch (deep baselines, saliency, attention prototype), PyMuPDF
(`defense_figures/build_deck.py`), ffmpeg and pillow (animations).

## 2. Checks that need no data

| Command | What it establishes | Time on 4 cores |
|---|---|---|
| `pytest -m "not slow"` | the v2 library behaves as documented; records match their code; documents quote the records | about 1 minute |
| `pytest` | the above plus an end-to-end run of the v2 pipeline on synthetic data | about 3 minutes |
| `python scripts/check_records.py` | every record and every frozen code file has the locked hash, in the tree and in git | seconds |
| `python scripts/run_verifiers.py` | the 22 legacy suites pass, in an isolated copy | about 7 minutes |
| `ruff check .` | lint, for code outside the frozen set | seconds |

**Do not run the legacy suites in place.** `chapter4Experiments/verify_chapter4.py` and
`chapter7Experiments/verify_chapter7.py` overwrite 21 tracked files with synthetic
output. `scripts/run_verifiers.py` exists to prevent that.

These checks show that the code is internally consistent. They say nothing about
whether a reported empirical number is right.

## 3. Data

| Dataset | Access | Expected layout |
|---|---|---|
| SHAPE Community, three conditions | restricted; from the data provider under the study's IRB protocol | `./batch_data/SHAPE_Community_<id>_IAPS{Neg,Neu,Pos}_BC.txt`, 636 files |
| SHAPE Community, four categories | restricted | `./categories/SHAPE_Community_<id>_IAPS{Neg,Pos}_{Threat,Mutilation,Cute,Erotic}_BC.txt`, 848 files |
| SCID-5 labels | restricted | a spreadsheet with an `ID` column, passed with `--labels` |
| ERP CORE N170 and P3 | public, doi:10.18115/D5JW4R | `./erpcore/<paradigm>/<id>/<id>_<paradigm>_ar.set` and `.fdt` (the epoched, artifact-flagged files) |
| TCRZEM | restricted download | `./external_data/tcrzem/` |

Every SHAPE file is a 1229 x 34 matrix: 1.2 s at 1024 Hz, 205 pre-stimulus samples, 34
electrodes in an order the files do not record. A check of all 1,484 files on
2026-11-12 found every one at 1229 x 34 with no missing values; the only constant
channels are in participants 127 and 195, whom the loaders exclude.

No data file may be committed. `.gitignore` excludes the raw layouts and
`tests/test_repo_hygiene.py` fails if one is tracked.

## 4. Regenerating a record

The exact command for each record is in `experiments/records.lock.json` and in
`docs/EXPERIMENT_REGISTRY.md`. The main ones:

```text
# v1 confirmatory record (about 1 hour)
python experiments/confirmatory/run_confirmatory_validation.py \
    --data3 ./batch_data --data4 ./categories --out confirmatory_results.json
python experiments/confirmatory/check_against_manifest.py confirmatory_results.json --tol 0.05

# v2 main record (about 2 h 40 min on 4 cores) and its report
python experiments/v2/run_v2_validation.py --data3 ./batch_data --data4 ./categories \
    --cache ./v2_cache --out experiments/v2/v2_results.json --n-perm 2000 --jobs 4
python experiments/v2/make_report.py

# quick check that the tree still reproduces the v2 record (about 15 minutes)
python experiments/v2/check_v2_replay.py --data3 ./batch_data --data4 ./categories --cache ./v2_cache
```

Keep `--cache` outside the repository. It holds preprocessed epochs and reservoir codes
derived from restricted data.

After regenerating a record:

1. commit the code first, so the record's provenance names a commit that contains it;
2. run the experiment and commit the record;
3. `python scripts/check_records.py --update <record>` and commit the lock;
4. regenerate any report built from it and update prose that quotes it
   (`tests/test_records.py` tells you which numbers no longer match).

## 5. Determinism

- **v2 pipeline: regenerated from the raw files and equal, given the thread count.**
  On 2026-11-21 and 22, after the library and runners were typed, formatted and given a
  shared provenance block, the v2-era records were regenerated from the delivered text
  files into a new cache and compared with the committed record by
  `scripts/compare_records.py` at tolerance zero. The table in section 5.1 gives the
  outcome per record.
- **The BLAS thread count is part of the specification.** `experiments/v2/v2_baseline_input.json` was
  first regenerated with four BLAS threads and 213 of its 1,213 values differed from the
  committed record, which had been produced with two. Regenerated with two threads, all
  1,213 values are equal. The differences are one observation in some repeats of some
  rows: at most 0.16 points per repeat and 0.24 points on any reported value, against
  confidence intervals several points wide. No conclusion changes. Two separate
  four-thread runs agreed with each other exactly, so the result is deterministic for a
  fixed thread count. The affected rows are the ones with more features than training
  rows, where the readout is fitted in the row space after a singular value
  decomposition; a different thread count changes that decomposition in the last bits
  and can move a prediction that sits on the decision boundary. One row tested directly
  gave the committed values with two threads and a different, mutually identical result
  with one and with four. Consequences:
  - each lock entry in `experiments/records.lock.json` has a `blas_threads` field, and
    the reproduction command sets `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` where the
    record depends on them;
  - every provenance block written from now on has a `threads` entry
    (`arspinet_v2/provenance.py`);
  - quote v2 accuracies to one decimal place. The second decimal is not stable across
    thread counts.
  `experiments/v2/v2_results.json` does not show the effect: its 1,256 values are equal with one
  thread and with the default on four cores.

### 5.1 Regeneration outcome

| Record | Values compared | Outcome | BLAS threads |
|---|---|---|---|
| `experiments/v2/v2_results.json` | 1,256 | equal | default on 4 cores, 4 workers |
| `experiments/v2/v2_baseline_input.json` | 1,213 | equal | 2 (213 values differ with 4) |
| `experiments/v2/v2_pca_tuned.json` | not yet known | regeneration in progress; still marked pending in the lock | 1, 2 workers |
| `experiments/v2/v2_normalization_check.json` | 342 | equal | default on 4 cores |
| `experiments/v2/v2_depression.json` | 170 | equal | 1 |
| `experiments/zscore_artifact/erpcore_results.json` | 2,683 | equal | default on 4 cores |
| `experiments/zscore_artifact/across_null.json` | 35 | equal | 1, 4 workers |

The v1 confirmatory record was not regenerated in this pass. Its headline rows are
recomputed inside `experiments/v2/v2_results.json` (64.33% and 54.64%) and match it.

### 5.2 Other sources of variation

- **v1 confirmatory pipeline: deterministic by construction** (fixed seeds, no
  unseeded calls), and its headline rows were recomputed bit-identically by the v2 code.
- **Legacy v1 scripts: not guaranteed.** Most PCA calls have no `random_state`, and a
  few scripts use the global NumPy generator unseeded
  (`docs/REPOSITORY_AUDIT.md`, finding E11). PyTorch scripts set no torch seed.
- **Across platforms.** Spike counts are integers and the reservoir update is a fixed
  sequence of floating-point operations, but matrix products go through BLAS. A
  different BLAS could in principle flip a threshold crossing at the v1 gain, where the
  network is sensitive to perturbation. `tests/test_reservoir.py` pins golden spike
  counts and will show this if it happens.

## 6. What cannot be reproduced from this repository

| Item | Why |
|---|---|
| The dissertation's reported confirmatory values (60.6%, 53.2%) | the original analysis code is not in the repository; the committed pipeline is an independent reconstruction (`COMPUTATIONAL_ERRATUM.md`) |
| The layer ablation (ABL-01) and four interpretability scripts (INT-01, INT-02) | they need a feature pickle no committed script writes |
| The Chapter 7 per-subject feature tables | no committed generator |
| The deep-baseline numbers in the Chapter 5 table | they come from a deprecated implementation; the canonical script cannot load the data as committed |
| Defense experiments A, B, D | their input pickle's writer is not in the repository |
| The novel, nuisance and external JSON records, exactly | producing code changed after they were written and the files were annotated by hand |
| The four-category extension and three-condition extension as documented | scripts read working-directory-relative paths and intermediate files; no end-to-end command exists |

Each of these is recorded with its registry entry. None was rerun in the audit.

## 7. Levels of evidence

When you state that something is reproduced, say which of these you mean.

| Level | Meaning |
|---|---|
| Replayed | recomputed on the data and compared with the record for equality |
| Locked | the record's hash and its code's hashes verify, and the code is recoverable from git; not rerun |
| Structural | tests on synthetic data show the code does what the documentation says |
| Unverifiable | generating code or input is missing |

The current assignment of every result set to a level is in section 6 of
`docs/REPOSITORY_AUDIT.md`.
