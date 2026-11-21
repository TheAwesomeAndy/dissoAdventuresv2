# ARSPI-Net

[![verify](https://github.com/TheAwesomeAndy/dissoAdventuresv2/actions/workflows/verify.yml/badge.svg)](https://github.com/TheAwesomeAndy/dissoAdventuresv2/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/code%20license-MIT-blue.svg)](LICENSE)

ARSPI-Net (Affective Reservoir-Spike Processing and Inference Network) is a measurement
chain for multichannel EEG. A fixed-weight leaky integrate-and-fire reservoir turns each
electrode's signal into spikes. Spike counts in six time bins form a code. A fixed
random projection compresses it, and a linear readout classifies affective picture
conditions across participants.

This repository holds three things:

1. **The v1 record**: code, figures and LaTeX source of the PhD dissertation (Lane,
   2026, Stony Brook University) and its corrective audit.
2. **The v2 validation**: tests of the dissertation's claims on the same data, with
   matched controls. This is the current scientific position.
3. **An ongoing study** of a preprocessing artefact that v2 uncovered, replicated on
   public data.

The contribution is a characterized measurement chain and an honest account of what it
does and does not do. It is not a state-of-the-art classifier.

## What the data say

Balanced accuracy on held-out participants, SHAPE Community cohort (211 participants,
three conditions; 210 participants, four categories). Chance is 33.3% and 25%.

| Representation | Three conditions | Four categories |
|---|---:|---:|
| Raw ERP, 16 time bins | 69.6% | 60.5% |
| Six-bin amplitudes, no reservoir | 67.1% | 55.1% |
| v2 reservoir code, mean of 10 weight draws | 65.1% | 55.4% |
| v1 reservoir configuration | 64.3% | 54.6% |
| LPP mean amplitude, 300 to 1000 ms | 58.4% | 45.8% |
| Band power and Hjorth features | 48.6% | 36.0% |

- The evaluation protocol is leak-free, and the reservoir code is far above chance.
- The reservoir does not beat plain amplitude binning. Recurrence adds nothing
  measurable, and at its original gain the network has no fading memory.
- Random projection does not beat PCA once PCA is left unwhitened and tuned.
- The original preprocessing z-scores each epoch, which fabricates decoding at
  latencies before the brain can respond. The effect replicates on public data
  (ERP CORE): pre-stimulus decoding of 58.6% and 70.8% against a 50% chance level.

Intervals, controls and limits are in [`V2_UPGRADE.md`](V2_UPGRADE.md). The standing of
every claim, including those withdrawn, is in [`docs/RESEARCH_MAP.md`](docs/RESEARCH_MAP.md).

## Repository structure

```text
arspinet_v2/               the library: reservoir, dynamics, features, evaluation, controls, nuisance
experiments/
  v2/                      v2 runners, committed records, generated report
  zscore_artifact/         z-scoring artefact study on ERP CORE
  confirmatory/            v1 corrected protocol and its record
  records.lock.json        index of result records and the code that produced each
  (other directories)      v1 record: extensions, exploratory studies
chapter4Experiments/ ... chapter7Experiments/    v1 record: chapter analyses and verifiers
defense_figures/  dissertation/  pictures/  validation/    v1 record
tests/                     pytest suite (no data needed)
scripts/                   tooling
docs/                      architecture, registry, research map, reproducibility, audit
```

`arspinet_v2/` and `experiments/v2/` are authoritative. The v1 record is frozen and
contains known defects, listed in [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md).

## Installation

```text
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e . --no-deps
pip install pytest ruff
```

Python 3.11 or newer. `requirements-lock.txt` is the exact environment behind the v2
records; `pyproject.toml` gives the lower bounds CI uses.

## Quick start, no data required

```text
pytest -m "not slow"               # library, records and documentation checks, about 1 minute
pytest                             # adds an end-to-end run on synthetic data, about 3 minutes
python scripts/check_records.py    # every result record still matches the code that produced it
python scripts/run_verifiers.py    # the 22 legacy suites, in an isolated copy, about 7 minutes
```

Do not run the `verify_*.py` files of Chapters 4 and 7 directly. They overwrite tracked
figures with synthetic output. `scripts/run_verifiers.py` runs them safely.

Using the library:

```python
import numpy as np
from arspinet_v2 import evaluation, features, reservoir

raw = np.random.default_rng(0).standard_normal((30, 256, 34))   # (observations, time, channels)
y, groups = np.tile([0, 1, 2], 10), np.repeat(np.arange(10), 3)

code = reservoir.encode(raw, rho=0.025)                         # (30, 34, 1536)
X = features.project(code, features.make_srp(code.shape[2]))    # (30, 34 * 64)
result = evaluation.evaluate(X, y, groups)                      # participant-grouped CV
print(result["balanced_accuracy"], evaluation.bootstrap_ci(result))
```

## Data

| Dataset | Access | Used by |
|---|---|---|
| SHAPE Community EEG (three conditions, four categories) | restricted human-subjects data; not distributed | v1, v2 |
| SCID-5 diagnostic labels | restricted | one v2 analysis |
| ERP CORE N170 and P3 | public, doi:10.18115/D5JW4R | z-scoring artefact study |
| TCRZEM | restricted download | one v1 exploratory replication |

Layouts and a file-integrity description are in
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). No data file may be committed.

## Core workflows

With the SHAPE files in `./batch_data` and `./categories`:

```text
# v2 validation: fading memory, matched controls, destruction controls, permutation test
python experiments/v2/run_v2_validation.py --data3 ./batch_data --data4 ./categories \
    --cache ./v2_cache --out experiments/v2/v2_results.json --n-perm 2000 --jobs 4
python experiments/v2/make_report.py

# confirm in minutes that the tree still reproduces the committed record
python experiments/v2/check_v2_replay.py --data3 ./batch_data --data4 ./categories --cache ./v2_cache

# v1 confirmatory protocol
python experiments/confirmatory/run_confirmatory_validation.py \
    --data3 ./batch_data --data4 ./categories --out confirmatory_results.json
```

With ERP CORE in `./erpcore`:

```text
python experiments/zscore_artifact/run_erpcore.py --data ./erpcore \
    --out experiments/zscore_artifact/erpcore_results.json
```

Every experiment, with its question, command, protocol, record and known problems, is
in [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md).

## Testing and validation

| Layer | What it checks | Needs data |
|---|---|---|
| `tests/` (pytest) | library behaviour, bit parity with v1, golden reservoir dynamics, record integrity, prose-versus-record agreement, repository hygiene, synthetic end-to-end run | no |
| `scripts/check_records.py` | each committed record against the hashes of its producing code, in the tree and in git | no |
| 22 legacy suites (`scripts/run_verifiers.py`) | v1 machinery on synthetic data; manifest and provenance of the v1 confirmatory record | no |
| `experiments/v2/check_v2_replay.py` | the v2 record replays exactly | yes |

CI runs the first three on every push. None of the data-free checks establishes an
empirical value.

## Reproducibility limits

- The SHAPE data are restricted, so the empirical results cannot be reproduced by the
  public. The records are tied by hash to code recoverable from git.
- A replay on 2026-11-21 reproduced the headline v2 values exactly. The rest of the v2
  record is locked, not replayed.
- The dissertation's own reported confirmatory values (60.6% and 53.2%) differ from what
  the committed pipeline produces (64.3% and 54.6%).
  [`COMPUTATIONAL_ERRATUM.md`](COMPUTATIONAL_ERRATUM.md) reconciles them.
- Several v1 results have no generating code in the repository. They are listed in
  [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
- One cohort for everything involving the reservoir. Only the z-scoring artefact has
  been replicated on a second dataset.

## The v1 record

The dissertation source is in `dissertation/`; its figures are in `pictures/`. The v1
README is preserved at [`docs/v1/README_v1.md`](docs/v1/README_v1.md), with the v1
result index in `results_manifest.json` and `docs/REPRODUCTION_MAP.md`.

v1 reported an interpretability audit with one measurement per level. Their standing
after v2:

| Level | v1 measurement | Standing |
|---|---|---|
| 1 Temporal traceability | pre-reservoir LPP r = 0.82; post-reservoir absolute r = 0.23 | withdrawal recommended: the 0.82 statistic never involves the reservoir |
| 2 Representation geometry | subject-to-condition variance ratio 7.2 | replaced by calibrated variance components (subject 44.5%, condition 10.3%) |
| 3 Dynamical correspondence | amplitude-domain 0.82; spike-domain 0.23 | as Level 1 |
| 4 Structure-function coupling | median kappa 0.273 | archived: the null is invalid without electrode coordinates |

## Documentation

| Document | Read it for |
|---|---|
| [`V2_UPGRADE.md`](V2_UPGRADE.md) | what v2 tested and found, with numbers |
| [`docs/RESEARCH_MAP.md`](docs/RESEARCH_MAP.md) | each scientific idea: implementation, evidence, standing |
| [`docs/EXPERIMENT_REGISTRY.md`](docs/EXPERIMENT_REGISTRY.md) | every experiment: command, protocol, record, caveats |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | layout, authoritative code, where new work goes |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | environment, data, regenerating records, evidence levels |
| [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md) | the 2026-11-21 audit: findings, risks, plan |
| [`AGENTS.md`](AGENTS.md), [`CLAUDE.md`](CLAUDE.md), [`docs/AGENT_HANDOFF.md`](docs/AGENT_HANDOFF.md) | rules and handoff format for AI agents |
| [`COMPUTATIONAL_ERRATUM.md`](COMPUTATIONAL_ERRATUM.md) | dissertation-reported versus code-derived values |
| [`experiments/v2/V2_RESULTS.md`](experiments/v2/V2_RESULTS.md) | full v2 tables (generated) |
| [`experiments/zscore_artifact/RESULTS.md`](experiments/zscore_artifact/RESULTS.md) | the artefact study |

## Contributing

Read [`AGENTS.md`](AGENTS.md); it applies to people as well. In short: new reusable code
goes in a new module of `arspinet_v2/`, new experiments under `experiments/<study>/`
with a committed record and a registry entry, and nothing in the v1 record or the
frozen files changes without a rerun on the data. Leave a handoff in `docs/handoffs/`.

## Citation and licence

Cite the dissertation: Lane, A. A. (2026). *Affective Reservoir-Spike Processing and
Inference Network (ARSPI-Net): A Four-Level Interpretable Neuromorphic Framework for
Clinical EEG Analysis.* PhD dissertation, Stony Brook University. Machine-readable
citation in `CITATION.cff`.

Code is MIT-licensed (`LICENSE`). Terms for text, figures and data are in `LICENSES.md`.
ERP CORE is by Kappenman et al. (2021); only aggregate results derived from it are
stored here.
