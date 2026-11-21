# Architecture

How the repository is organized, which code is authoritative, and where new work goes.
For the reasoning behind these choices and the list of known problems, see
`docs/REPOSITORY_AUDIT.md`.

## 1. Three layers

| Layer | Directories | You may | You may not |
|---|---|---|---|
| **v2 library** | `arspinet_v2/` | import it; add new modules | change the six existing modules without regenerating records |
| **v2 and ongoing experiments** | `experiments/v2/`, `experiments/zscore_artifact/` | add runners and studies | edit a runner whose record is locked |
| **v1 record** | everything else outside `tests/`, `scripts/`, `docs/` | read it, cite it, test against it | edit, move or "fix" it on your own initiative |

The v1 record is the defended dissertation's code and its corrective audit. The defects
the 2026-11 audit confirmed in it were fixed at the owner's request, with a regression
test each, and are logged in `docs/v1/DEFECT_FIXES.md` together with which outputs were
and were not regenerated. Its protocol choices (for example PCA fitted before the split
in the archived Chapter 5 analyses) are not defects and were left as they are.

## 2. The measurement chain

```text
SHAPE text file            1229 samples x 34 electrodes at 1024 Hz, baseline-corrected
        │  drop 205 pre-stimulus samples, anti-alias decimate by 4, z-score per epoch and channel
        ▼                  experiments/confirmatory/run_confirmatory_validation.py (v1, frozen)
epoch array                (observations, 256, 34)
        │  each electrode's signal drives a 256-neuron LIF reservoir
        ▼                  arspinet_v2/reservoir.py
spike trains               (observations x electrodes, 256, 256)
        │  count spikes in six 42-sample bins
        ▼
BSC6 code                  (observations, 34, 1536), bin-major
        │  one fixed sparse random projection, shared by all electrodes
        ▼                  arspinet_v2/features.py
feature matrix             (observations, 34 x 64)
        │  participant-grouped CV, standardize, L2 logistic readout, all fitted in-fold
        ▼                  arspinet_v2/evaluation.py
balanced accuracy, participant-bootstrap interval, permutation p
```

Array convention everywhere in v2: epochs are `(observations, time, channels)`; codes are
`(observations, channels, features)`; one observation is one participant-condition
average. `y` holds condition labels and `groups` participant IDs, aligned with axis 0.

## 3. The v2 library

| Module | Responsibility | Entry points |
|---|---|---|
| `arspinet_v2/reservoir.py` | The LIF operator, batched. Bit-identical to v1. Exposes recurrent gain, reservoir sharing, a non-spiking control | `LIFReservoir`, `encode` |
| `arspinet_v2/dynamics.py` | Label-free fading-memory tests and the rule that selects the gain | `fading_memory_report`, `select_rho`, `RULE` |
| `arspinet_v2/features.py` | Projection, matched non-reservoir controls, windows in milliseconds, LPP score | `make_srp`, `project`, `amplitude_bins`, `ms_window`, `lpp_mean_amplitude` |
| `arspinet_v2/evaluation.py` | Splits, readout, bootstrap, permutation test | `evaluate`, `bootstrap_ci`, `bootstrap_difference_ci`, `permutation_test` |
| `arspinet_v2/controls.py` | Destruction controls on the code | `collapse`, `shuffle_bins`, `shuffle_electrodes` |
| `arspinet_v2/nuisance.py` | Unbiased subject and condition variance components with nulls | `variance_components` |
| `arspinet_v2/provenance.py` | Provenance blocks for records. Not frozen: it shapes metadata, not results | `provenance`, `strip_provenance` |

The six analysis modules have no I/O and no global state. Everything that reads data or
writes records is in `experiments/`. All of it is typed (`mypy` runs in CI), formatted
and linted.

It is installable (`pip install -e .`), so `import arspinet_v2` works from anywhere. The
experiment scripts still insert the repository root into `sys.path`; that is harmless
and keeps them runnable without installation.

## 4. How a v2 experiment is built

```text
experiments/v2/data.py                 load both cohorts through the v1 loader; optional cache
experiments/v2/run_v2_validation.py    stages -> one JSON record with a provenance block
experiments/v2/make_report.py          JSON record -> V2_RESULTS.md (generated, never edited)
experiments/records.lock.json          record hash + hashes of the code that produced it
```

A runner computes, a record stores, a report renders, and prose (`V2_UPGRADE.md`)
interprets. Numbers flow in that direction only. `tests/test_records.py` checks that the
report regenerates byte-identically and that the numbers quoted in prose equal the
record.

## 5. Records and frozen files

A *record* is a committed result file from a run CI cannot repeat. Each record is listed
in `experiments/records.lock.json` with the hashes of the files that produced it. Those
files are *frozen*: `python scripts/check_records.py --list-frozen` prints them.

- Editing a frozen file fails `tests/test_records.py`.
- The only way to change one is to mark its records pending, rerun on the data, show
  with `scripts/compare_records.py` what changed, commit the new record, and run
  `python scripts/check_records.py --update <record>`. The steps are in `AGENTS.md`.
- A record may be `pending_regeneration` on a branch. `--strict` and a test refuse that
  state on `main`.
- Every runner writes its provenance block through `arspinet_v2/provenance.py`: commit,
  code hashes, whether the hashed code matches that commit, and the environment.
- Runner parameters (gain grid, seeds, tuning grid, the depression family) are in
  `experiments/v2/config.toml`, which is hashed into the records it shapes.

Five older records are listed as `unverified_records`: their code changed after they
were written, so only their own hash is checked.

## 6. Which reservoir is which

The word "reservoir" refers to four different networks in this repository. Only the
first is the model whose features were classified.

| Class | Weights and update | Used by | Authoritative for |
|---|---|---|---|
| **A. v1 operator** | dense uniform, recurrent matrix scaled to spectral radius 0.9; multiplicative reset, subtract theta, clamp at 0 | `chapter5Experiments/experiment_zero.py`, confirmatory, novel, external, Chapters 3 to 5; `arspinet_v2/reservoir.py` reproduces it and adds a gain parameter | classification results |
| B. dense Gaussian | Gaussian recurrent matrix scaled to 0.9 | `experiments/ch6_ch7_3class/` | that extension only |
| C. sparse Gaussian | 10% density, no spectral scaling (radius about 0.26), reset without subtraction | all of `chapter6Experiments/`, Chapter 7 experiment A, defense experiment A | the Chapter 6 dynamics results, which therefore do not describe class A |
| D. comparators | tanh RNN, random Fourier features | one Chapter 6 script | nothing |

For new work use `arspinet_v2.reservoir`. v2 measured that class A at gain 0.9 has no
fading memory and that the gain selected by the label-free rule is 0.025.

## 7. Dependencies between v1 directories

There are no packages in the v1 record. Scripts reach each other through `sys.path`.

```text
chapter5Experiments/experiment_zero.py      LIFReservoir, bsc6_encode
        ▲                     ▲
experiments/confirmatory/     experiments/novel/novel_pipeline.py
        ▲        ▲                    ▲
        │   experiments/novel/        │
        │        ▲                    │
        │   experiments/external/ ────┘
experiments/v2/data.py        (v2 loads SHAPE through the v1 confirmatory loader)
```

`experiments/ch5_4class/`, `experiments/ch6_ch7_3class/`, `experiments/ablation/`,
`chapter6Experiments/`, `chapter7Experiments/` and `defense_figures/` import nothing
from other directories. They exchange data through pickles and CSV files instead, and
several of those intermediate files have no committed producer
(`docs/REPOSITORY_AUDIT.md`, finding I1).

## 8. Tooling

| Tool | Purpose |
|---|---|
| `scripts/run_verifiers.py` | Run the 22 legacy suites in an isolated copy. Never run `chapter4Experiments/verify_chapter4.py` or `chapter7Experiments/verify_chapter7.py` in place: they overwrite tracked files |
| `scripts/check_records.py` | Verify every record against its code; update the lock after a rerun |
| `scripts/compare_records.py` | Compare two records for equality outside their provenance blocks |
| `scripts/build_registry_doc.py` | Regenerate `docs/EXPERIMENT_REGISTRY.md` from `docs/experiment_registry.json` |
| `scripts/make_release_manifest.py` | v1 release checksums (see `RELEASE.md`) |
| `tests/` | pytest suite. No data needed. `pytest -m "not slow"` skips the end-to-end smoke test |
| `.github/workflows/verify.yml` | CI: the 22 legacy suites; then lint, format check, mypy, the record check and pytest |

## 9. Where new work goes

| You are adding | Put it in | And also |
|---|---|---|
| A reusable function | a new module in `arspinet_v2/` (existing modules are frozen) | unit tests in `tests/` |
| A new experiment | `experiments/<study>/run_<name>.py`, writing one JSON record with a provenance block | a registry entry; a lock entry; a test that prose quotes the record |
| A figure | generated from a record by a script next to the record | commit the script and the figure together |
| A correction to a v1 result | a new experiment that states what it supersedes | update the registry caveats and `docs/RESEARCH_MAP.md`; do not edit the v1 file |
| Data | nowhere in git | `.gitignore` already excludes the raw files; `tests/test_repo_hygiene.py` enforces it |

Conventions for new code: type annotations and docstrings on public functions; no I/O
in library modules; seeds as explicit arguments; `RandomState`-style seeded generators,
never the global one; paths from command-line arguments, never hardcoded; parameters in
`experiments/v2/config.toml` or a file like it; a provenance block from
`arspinet_v2/provenance.py`; `ruff check .`, `ruff format --check` and `mypy` clean.
