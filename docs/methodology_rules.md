# ARSPI-Net Methodology Rules

> **Status note, added 2026-11-21.** This document is part of the v1 record and is kept
> as written. Rule 11 calls one clinical result "notable and real". The v2
> analysis treats every clinical association in this repository as an uncorrected
> screen, and a corrected family of depression tests found nothing
> (`experiments/v2/v2_depression.json`). Rules for current work are in `AGENTS.md`.

Established March 2026 from advisor feedback.
These rules govern all experimental design and writing in this dissertation.

## Rule 1 — Horizontal Before Vertical

When a cross-chapter claim emerges (e.g. "three operationally distinct
layers"), immediately design the single experiment that tests it directly
using features from all chapters on the same task with the same protocol.
Do not finalize chapter-level experiments without the system-level
integration test. Every cross-chapter narrative needs a cross-chapter
experiment.

## Rule 2 — Claims Require Direct Tests

Every dissociation or complementarity claim must be accompanied by a
direct ablation/combination experiment. Narrative consistency across
separate analyses is not proof. If the direct test cannot be designed,
do not make the claim. Separate chapter analyses showing different
sensitivities are suggestive but do not demonstrate non-redundancy.

## Rule 3 — No Unearned Terminology

Do not use a mathematical framework's vocabulary (e.g. "information-
theoretic decomposition") unless the quantities that framework defines
have actually been computed. Use operationally precise language
("operationally distinct response layers") when only operational
evidence exists. The Scientific Voice Directive demands mathematical
grounding — rhetorical use of technical terms violates that.

## Rule 4 — Adversarial Committee Test

Before finalizing any experimental program, explicitly list the 3 most
likely committee objections and verify each has a corresponding
EXPERIMENT, not just a narrative response. "Why not EEGNet?" and "how
do you know these layers are not redundant?" require experimental
answers. Conventional baselines (EEGNet, GRU/LSTM) are strategically
non-negotiable for positioning.

## Rule 5 — Linear Readouts for Content Comparison

When comparing information content across feature families, use linear
classifiers (linear SVM, logistic regression) as primary readout so
geometric separability is exposed, not rescued by nonlinear kernels.
RBF-SVM and other nonlinear classifiers belong in appendix sensitivity
checks only. The goal is to measure feature content, not classifier
power.

## Rule 6 — Interrogate Null Results

When a null clinical or condition result emerges, ask: is this absence
of effect or absence of measurement resolution? Test at finer resolution
(per-channel instead of channel-averaged, per-condition instead of
grand-averaged) before concluding absence. Channel-averaging destroyed
spatial structure in Ch6 clinical comparisons — the null may be a
resolution artifact.

## Rule 7 — Build Summary Tables Immediately

When a cross-condition or cross-granularity comparison is central to the
argument, build the quantitative summary table as soon as the numbers
exist. Do not distribute key comparisons across prose. The 3-class vs
4-class regime table should have been built the moment both datasets
were analyzed. A 30-minute formatting task can be one of the most
important figures in the dissertation.

## Rule 8 — Defense Figure at Milestone

When a system-level claim is experimentally complete (e.g. ablation
matrix done, three-layer thesis earned), immediately design the single
defense-quality summary figure that communicates the entire argument in
30 seconds. Rule 7 applies to tables; this extends it to figures. The
ablation matrix was done but no summary figure was built. A visualization
task is not optional — it is what makes the difference between 7.8 and
9.0 at defense.

## Rule 9 — Publication Roadmap at Completion

When the experimental program is done, immediately map results to
standalone paper-level stories with specific venues. Do not wait for
the advisor to propose this. Each paper needs one central claim, not
a chapter summary. The dissertation narrative arc connects chapters;
papers must stand alone. Four papers were identifiable from the March
2026 results but were not proposed until the advisor did.

## Rule 10 — Verify Before Delivering

Before delivering any LaTeX, run a cross-file \ref{} and \cite{} audit.
A broken reference is a trust problem that drags the entire score down.
Broken placeholders (Chapter ??) signal unfinished work regardless of
the science quality. Polish is not optional — manuscript closure was
scored 5.8/10 entirely due to preventable errors.

## Rule 11 — Clinical Claims Are Exploratory Until Replicated

Every clinical subsection must open by framing the analysis as
hypothesis-generating with within-variable comparison and permutation
testing, pending independent replication. "Dominant clinical finding" →
"strongest clinical effect observed in this dataset." "Biomarker" →
"candidate network phenotype." The SUD result (p=0.0004) is notable
and real, but the language must match the validation level, not the
enthusiasm level.

## Rule 12 — Disambiguate Conditional Results

When a headline result depends on a condition (e.g. subject-centering),
state both numbers in the same sentence everywhere: abstract, intro,
conclusion. "Cross-subject accuracy is 63.4%; subject-centering reveals
latent capacity of 79.4%." Never let the conditional number appear
alone. The 79.4% was misreadable as deployment-ready. Ambiguity in
headline numbers is a credibility problem.
