# Which encoding detects faults best?

**There is no confirmed overall winner.** The provisional validation leader for “fault present vs no labeled event” is **Label-informed AE + logistic regression**, with mean average precision **0.770**. This comparison is supported by only **PUMP-01 and PUMP-09**. In the test period, that detector recognizes **45.1%** of the annotated faulty minutes on PUMP-01 and **100.0%** on PUMP-09. Neither test period has negative examples, so test false-alarm rates and a meaningful test ranking cannot be established.

The 0–1 AP score is a ranking measure, not classification accuracy or “percentage of faults detected.” Higher is better. The recall values above describe faulty **minutes**, not independent events or advance warning.

## What was tested

All existing 32D encoders were frozen. Each available pump/encoder received identical downstream classifiers trained only on known labels in its first 60% interval:

- **Primary readout:** logistic regression, standardized using the classifier's training rows, C=1, balanced class weights, max_iter=2000, seed 42.
- **Sensitivity check:** a 15-nearest-neighbor classifier, uniform voting, with the same training-only standardization.
- **Primary target:** ANY_FAULT, whether at least one observable, detectable fault is annotated.
- **Secondary targets:** individual fault types that appear in the training vocabulary and have both positive and negative training examples.

The 60/20/20 chronological split and 120-minute heldout warmup exclusions are unchanged. Pre-detectable/unobservable overlaps and unlabeled minutes are excluded. `NO_LABELED_EVENT` is a negative for this annotation-based experiment, not a guarantee of physical health. No synthetic healthy examples were created. No different pump's labels or coordinates were pooled into a model.

For every trained target, the validation F1-maximizing threshold was saved before testing. If validation has only one class or no rows, a fixed 0.5 threshold is recorded as an unevaluated fallback. Encoder ranking uses validation AP, not thresholded F1. Every candidate sees exactly the same evaluable validation pump/target cases; target scores are averaged within pump and then equally across pumps. A durable `selection.json` was written before calculating test predictions and metrics. Training vocabularies exclude future-only faults; the full vocabulary is used only to report coverage.

In total **192 downstream classifiers** were fitted without changing any of the 67 encoders. No convergence warnings were recorded.

## Main result: any fault present

| Encoder | Validation AP |
|---|---:|
| Label-informed AE | 0.770 |
| Denoising AE | 0.745 |
| Autoencoder | 0.743 |
| Temporal AE | 0.562 |
| Kernel PCA | 0.466 |
| PCA | 0.392 |

Only PUMP-01 and PUMP-09 have both classes in training and scored validation for this task across all six encoders. The other pumps do not meet that requirement. The mean validation no-skill AP baseline, equal to mean positive prevalence on these two pumps, is 0.425.

The lead over the next candidate is small: 0.024 AP. Adjacent minutes are highly correlated and only two pumps contribute, so this is not a robust estimate of future performance. Different encoder types win on the two individual pumps; a single mean hides that variation.

| Pump | Encoder | Validation AP | Test recall | Test false-positive rate |
|---|---|---:|---:|---|
| PUMP-01 | PCA | 0.530 | 89.7% | Unavailable: no negative examples |
| PUMP-01 | Kernel PCA | 0.491 | 98.3% | Unavailable: no negative examples |
| PUMP-01 | Autoencoder | 0.492 | 34.6% | Unavailable: no negative examples |
| PUMP-01 | Denoising AE | 0.496 | 39.4% | Unavailable: no negative examples |
| PUMP-01 | Temporal AE | 0.694 | 58.6% | Unavailable: no negative examples |
| PUMP-01 | Label-informed AE | 0.547 | 45.1% | Unavailable: no negative examples |
| PUMP-09 | PCA | 0.254 | 97.7% | Unavailable: no negative examples |
| PUMP-09 | Kernel PCA | 0.442 | 100.0% | Unavailable: no negative examples |
| PUMP-09 | Autoencoder | 0.993 | 100.0% | Unavailable: no negative examples |
| PUMP-09 | Denoising AE | 0.994 | 100.0% | Unavailable: no negative examples |
| PUMP-09 | Temporal AE | 0.429 | 99.5% | Unavailable: no negative examples |
| PUMP-09 | Label-informed AE | 0.992 | 100.0% | Unavailable: no negative examples |

High recall alone cannot establish a good detector: always predicting “fault” would achieve 100% recall in these all-positive test periods. Conversely, the label-informed model's low PUMP-01 test recall is direct evidence that its validation lead does not imply reliable chronological generalization.

## The result depends on the detection question and classifier

For ANY_FAULT with the 15-neighbor classifier, the leader is **Autoencoder**, with AP 0.496.

| Encoder | Validation AP |
|---|---:|
| Autoencoder | 0.496 |
| Label-informed AE | 0.487 |
| Denoising AE | 0.481 |
| Kernel PCA | 0.470 |
| PCA | 0.448 |
| Temporal AE | 0.445 |

For individual known fault types with logistic regression, **PCA** leads with AP 0.823. That is a different target and a different subset: PUMP-01 strainer clogging, PUMP-02 sensor drift, and PUMP-06 cavitation/strainer clogging. There are four pump/type cases over three pumps, weighted equally by pump.

| Encoder | Validation AP |
|---|---:|
| PCA | 0.823 |
| Label-informed AE | 0.782 |
| Kernel PCA | 0.759 |
| Temporal AE | 0.597 |
| Denoising AE | 0.538 |
| Autoencoder | 0.497 |

With the nearest-neighbor readout on those type-specific cases, **Denoising AE** leads with AP 0.527. Therefore the data do not support calling any encoder universally best for fault detection.

## Why test data cannot confirm the ranking

Every test target for which a classifier could actually be trained has only positive labels or only negative labels, after ambiguity and warmup exclusions. We deliberately mark AP/AUROC **not evaluable for ranking** for those cases. A library can return AP=1 for all-positive labels, but that would not demonstrate the ability to separate faulty from non-faulty observations. We report recall when positives exist and false-positive rate when negatives exist, without combining them across incompatible cases into a misleading headline.

PUMP-02 does have both faulty and no-event test minutes, but its ANY_FAULT training labels are all positive, so no supervised two-class detector can be learned there under the fixed within-pump protocol. PUMP-05 has no unambiguous heldout scoring minutes. PUMP-07 has no positive training examples. PUMP-10–12 have no public labels. Several later fault types never appear in their pump's training labels. These are data-coverage limitations, not missing model runs.

## Interpretation

- **Compression:** the preceding experiment favored PCA on 11 pumps and the ordinary AE on PUMP-08.
- **Binary fault detection:** the label-informed AE is the provisional leader on the prespecified logistic validation comparison; test behavior is inconsistent and false-alarm performance is unmeasurable on those test windows.
- **Known fault recognition:** PCA leads the separate logistic validation comparison, with sparse coverage and no mixed-class test confirmation.

A defensible final winner needs later labeled periods containing both fault-free and faulty behavior, ideally multiple independent incidents/pumps. Randomly shuffling adjacent minutes into train/test would disguise the current limitations rather than resolve them. We did not change the split or fit new encoders to improve the measured ranking.

## Artifacts and boundaries

`results/fault_detection/` contains saved readout models and thresholds, per-case validation/test metrics, full heldout predictions, label coverage, input hashes, selection and verification. `fault_detection.html` provides a compact browser view. `fault_detection.py` reruns the fixed experiment; `verify_fault_detection.py` checks the saved outputs; `fault_detection_report.py` rebuilds this report and charts.

This is a diagnostic on synthetic data and minute annotations, not the official hackathon event-alert/lead-time score, real-world fault detection, or an unseen-pump benchmark. Existing neural encoder checkpoints had already been selected using validation reconstruction. These test intervals were explored earlier, so this is not a new untouched final benchmark. The detector settings and evaluation policy were fixed before obtaining classifier scores and are recorded in [the recorded protocol](../../docs/superpowers/plans/2026-11-07-fault-detection-benchmark.md).

Definitions: [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) and [logistic regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).
