# Why per-epoch z-scoring leaks late signal into early samples

Written on 2026-11-12 while the ERP CORE run (`run_erpcore.py`) was in progress and
before any of its results were seen. The prediction in section 3 is therefore a
prediction, not a fit.

## 1. Setup

One trial, one channel, samples t = 1, ..., T, of which the first T_pre are
pre-stimulus. Write the trial as

    x(t) = s_y(t) + n(t)

with s_y the condition-y evoked response (zero before stimulus onset, after baseline
correction) and n zero-mean noise. Per-epoch z-scoring replaces x(t) by

    z(t) = (x(t) - m) / sd,      m = (1/T) sum_t x(t),   sd = std_t x(t).

## 2. What a pre-stimulus sample becomes

For t < 0, s_y(t) = 0, so

    z(t) = (n(t) - m) / sd.

The epoch mean m contains the response:

    E[m | y] = (T_post / T) * mean_post(s_y),

where mean_post(s_y) is the average of the evoked response over the post-stimulus
samples. Two consequences follow.

- Every pre-stimulus sample is shifted by -E[m | y] / sd. The shift depends on the
  condition, so the pre-stimulus scalp pattern carries the post-stimulus condition
  difference, sign-flipped, at every electrode at once.
- The shift is constant over the pre-stimulus interval, and m is an average over T
  samples, so its noise is small. A linear classifier can read the shift reliably at
  any pre-stimulus time point.

Division by sd adds a second, smaller route: sd also depends on the evoked response,
so conditions with larger responses are scaled down more.

The same argument applies to every post-stimulus sample, which is why time-resolved
decoding after z-scoring cannot be read as timing: each latency carries the epoch mean.

## 3. Prediction

The leak is proportional to the difference between conditions in the epoch-averaged
response, mean_post(s_a) - mean_post(s_b), relative to the noise in m. It should be

- large when the condition difference is sustained and of one sign (P3, and the late
  positive potential in SHAPE);
- smaller when the difference is brief or biphasic (N170).

Both the full-epoch and the post-stimulus-only variants should leak. Dividing by the
pre-stimulus SD only (`baseline_z`) uses no post-stimulus sample and should not.

## 4. Relation to known artefacts

This is in the same family as the high-pass filtering artefact of van Driel, Olivers and
Fahrenfort (2021, J. Neurosci. Methods 352:109080), where a temporal filter displaces
multivariate patterns into activity-silent periods, and the preprocessing effects mapped
by Kessler, Enge and Skeide (arXiv 2410.14453). Searches run on 2026-11-12 found no paper
identifying per-epoch z-scoring as the cause. That is a search result, not proof of
novelty. A full literature review is still needed.

The fix is to estimate any scaling from data that cannot contain the response: the
pre-stimulus interval of the same trial, or statistics computed on training data only.
