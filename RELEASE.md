# Frozen Publication Release and DOI Archival

This repository accompanies a dissertation. A publication release must bind the
reported results, source code, dissertation source, rendered dissertation, and
figure set to one immutable commit. A moving `main` branch is not an archival
record.

Tagging the release and minting the Zenodo DOI require the repository owner's
account. Those actions are marked **[owner action]**. They occur only after the
verification and authorized-data requirements below are satisfied.

---

## 1. Establish repository consistency

Confirm that `README.md`, `docs/REPRODUCTION_MAP.md`,
`results_manifest.json`, and the files under `dissertation/` agree on:

- cohort definitions and exclusions;
- preprocessing boundaries;
- confirmatory, exploratory, and archived status; and
- the generating script for every indexed result.

The confirmatory *values* deliberately differ between the submitted
dissertation (60.6% / 53.2%) and the committed reconstruction (64.3% / 54.6%);
that divergence is reconciled in `COMPUTATIONAL_ERRATUM.md`, which is
authoritative for code-derived numbers. Confirm the erratum table matches
`results_manifest.json` (`value` vs `dissertation_reported`) rather than
expecting the dissertation source to be edited to the reconstruction values.

## 2. Obtain a green data-independent verification run

Run the complete 491-check suite, preferably through
`.github/workflows/verify.yml` on the exact candidate commit:

```text
python chapter4Experiments/verify_chapter4.py
python chapter5Experiments/verify_chapter5.py
python chapter5Experiments/verify_experiment_zero.py
python chapter5Experiments/verify_reproduce_chapter5.py
python chapter5Experiments/verify_baselines.py
python chapter6Experiments/verify_chapter6.py
python chapter6Experiments/verify_reproduce_chapter6.py
python chapter7Experiments/verify_chapter7.py
python chapter7Experiments/verify_extract_utilities.py
python experiments/ch5_4class/verify_ch5_4class.py
python experiments/ch6_ch7_3class/verify_ch6_ch7_3class.py
python experiments/ablation/verify_ablation.py
python validation/verify_validators.py
python experiments/confirmatory/verify_confirmatory.py
python experiments/novel/verify_novel.py
python experiments/novel/verify_nuisance.py
python experiments/external/verify_external.py
python experiments/confirmatory/check_against_manifest.py
python experiments/novel/check_novel_manifest.py
python validation/validate_dissertation_claims.py
```

A green run establishes code behavior on the data available to the repository.
It does not establish the empirical SHAPE results or external validity.

## 3. Regenerate the confirmatory results on authorized SHAPE data

This step is required before the frozen dissertation release. The restricted
SHAPE data are not distributed by this repository, so continuous integration
cannot perform it.

Run the committed confirmatory pipeline on the authorized data:

```text
python experiments/confirmatory/run_confirmatory_validation.py \
    --data3 ./batch_data \
    --data4 ./categories \
    --out confirmatory_results.json

python experiments/confirmatory/check_against_manifest.py \
    confirmatory_results.json --tol 0.05
```

The checker must complete without mismatches. Retain
`confirmatory_results.json` as the machine-readable execution record. Do not
state that the committed runner reproduces the headline values until this
execution has been completed on the audited cohort.

## 4. Bind the release artifacts

On the exact commit intended for tagging, generate the integrity manifest:

```text
python scripts/make_release_manifest.py \
    --pdf ARSPI-Net_Defense.pdf \
    --out RELEASE_MANIFEST.json

git add RELEASE_MANIFEST.json confirmatory_results.json
git commit -m "chore: bind v1.0.0 release artifacts"
```

The release manifest records the commit and SHA-256 checksums for the
repository-level provenance files, dissertation source, dissertation PDF, and
figure tree. Re-running it on an unchanged tree must reproduce the same
checksums. This establishes artifact integrity, not scientific validity.

## 5. Tag the frozen release **[owner action]**

After the candidate commit has a green verification run and the authorized-data
checker passes:

```text
git tag -a v1.0.0 -m "Frozen publication release: ARSPI-Net dissertation"
git push origin v1.0.0
```

Create a GitHub Release from that exact tag. The release notes should identify:

- the corrected SRP-64 participant-grouped protocol;
- the authorized-data reproduction record;
- `results_manifest.json` and `docs/REPRODUCTION_MAP.md`;
- the 491-check code-verification result; and
- the distinction between implementation verification and empirical validity.

## 6. Mint and record the Zenodo DOI **[owner action]**

1. Enable the GitHub-to-Zenodo integration for
   `TheAwesomeAndy/dissoAdventuresv2`.
2. Publish the GitHub Release. Zenodo will archive the tagged snapshot and mint
   a version DOI and concept DOI.
3. Replace the placeholder DOI, version, and release date in `CITATION.cff`.
4. Add the DOI badge to `README.md`.
5. Commit the metadata-only DOI update.

The DOI follow-up commit may occur after the frozen tag because the DOI does not
exist until the release is published. The tag remains the immutable scientific
snapshot; the follow-up commit records the identifier that resolves to it.

## 7. Subsequent revisions

Continue development on `main`. Any later scientific revision requires a new
version tag, a new release manifest, a new authorized-data execution record,
and a new Zenodo version. The Zenodo concept DOI may remain the series-level
identifier.
