# Expanded encoding results

Development-selected encoding: **expanded_task_encoder/logistic**.
Development-selected frozen reference: **frozen_denoising_ae/logistic**.

Average precision (AP) measures ranking of faulty minutes, not accuracy. The same mixed-label recordings support every row below. All-negative recordings remain included in false-positive counts.

| Candidate | Development AP | Fresh-test AP | Recall | False-positive rate |
|---|---:|---:|---:|---:|
| expanded_task_encoder/logistic | 0.890 | 0.869 | 74.6% | 15.2% |
| expanded_task_encoder/histogram | 0.868 | 0.868 | 77.5% | 23.3% |
| raw_features/logistic | 0.792 | 0.857 | 83.2% | 38.5% |
| raw_features/histogram | 0.898 | 0.833 | 71.9% | 15.0% |
| expanded_kernel_pca/logistic | 0.870 | 0.827 | 85.0% | 36.9% |
| expanded_pca/logistic | 0.798 | 0.825 | 84.0% | 37.0% |
| expanded_kernel_pca/histogram | 0.840 | 0.821 | 84.8% | 35.3% |
| expanded_autoencoder/logistic | 0.785 | 0.815 | 81.2% | 50.1% |
| expanded_denoising_ae/logistic | 0.773 | 0.807 | 77.3% | 46.9% |
| frozen_denoising_ae/histogram | 0.791 | 0.806 | 82.6% | 43.9% |
| expanded_label_informed/logistic | 0.844 | 0.804 | 84.5% | 35.3% |
| frozen_autoencoder/logistic | 0.784 | 0.800 | 80.7% | 42.9% |
| frozen_pca/logistic | 0.777 | 0.798 | 81.6% | 46.8% |
| expanded_temporal_ae/logistic | 0.788 | 0.780 | 82.7% | 42.1% |
| frozen_kernel_pca/logistic | 0.749 | 0.754 | 74.2% | 33.0% |
| frozen_denoising_ae/logistic | 0.797 | 0.754 | 77.1% | 39.7% |
| frozen_temporal_ae/logistic | 0.765 | 0.724 | 80.4% | 50.3% |

## Same encoder before and after retraining

Both sides receive classifiers fitted on the same augmented training rows. The frozen label-informed comparison has only five supported families; every other row has seven.

| Encoder | Frozen test AP | Expanded test AP | Difference | Families |
|---|---:|---:|---:|---:|
| pca | 0.798 | 0.825 | +0.027 | 7 |
| kernel_pca | 0.754 | 0.827 | +0.073 | 7 |
| autoencoder | 0.800 | 0.815 | +0.016 | 7 |
| denoising_ae | 0.754 | 0.807 | +0.053 | 7 |
| temporal_ae | 0.724 | 0.780 | +0.056 | 7 |
| label_informed | 0.815 | 0.801 | -0.014 | 5 |

## Nearest-neighbor sensitivity check

| Expanded encoder + 15 neighbors | Development AP | Fresh-test AP |
|---|---:|---:|
| expanded_label_informed/knn | 0.816 | 0.802 |
| expanded_task_encoder/knn | 0.784 | 0.782 |
| expanded_autoencoder/knn | 0.755 | 0.750 |
| expanded_temporal_ae/knn | 0.794 | 0.748 |
| expanded_denoising_ae/knn | 0.774 | 0.734 |
| expanded_pca/knn | 0.735 | 0.733 |
| expanded_kernel_pca/knn | 0.742 | 0.729 |

Test AP averages recordings within each engineering family, then weights families equally. Recall and false-positive rate use pooled minute counts at thresholds chosen on development F1. They are different summaries.

Paired seed-bootstrap 95% interval for selected-minus-reference AP: **[0.061, 0.183]** (2,000 resamples, only five independent simulator seeds).

## What changed

- Reused 108 additional training recordings and the original seven training pumps’ first 60% intervals.
- Kept entire recordings together. A family encoder is shared across simulated variants derived from that pump template; different families still have different coordinate systems.
- Refit PCA, kernel PCA, autoencoder, denoising autoencoder, temporal autoencoder and label-informed autoencoder.
- Added a 32D task-trained encoder: the same MLP compresses readings; auxiliary fault heads teach it which information to retain, while a small reconstruction loss keeps general information.
- Retrained identical classifiers on both the published frozen encoders and the expanded encoders. The comparison tests the expanded pipeline, including training-supported feature coverage and checkpoint selection; it does not isolate data volume alone.
- Kept full-feature logistic regression and boosted trees as controls for information lost during compression.

## Boundaries

Development includes the earlier synthetic holdout, because it had already been examined. The fresh test set contains 35 newly generated three-day recordings from five unused seeds, created after model and threshold freeze. No test results changed training or thresholds.

These are minute-level classification tests within known simulated engineering families. They are not the hackathon’s event-alert/lead-time score, an unseen-family test, or evidence from real pumps. Original PUMP-08–12 were not fitted in this new experiment.

The published branch’s AP of 0.770 came from a different two-pump chronological validation experiment and cannot be directly compared with this table. Here, each family has both positive and negative development coverage.

No-event means no eligible annotation, not guaranteed health. Predetectable and unobservable overlaps are excluded. Missing observations remain masked. Sensor snapshots use the generator’s existing waveform path, which does not reproduce all channel sensor corruption; this is a remaining simulation limitation.

See `results/summary.json`, `comparison.csv`, `development_metrics.csv`, `test_metrics.csv`, saved prediction Parquets and `frozen.json` for the underlying results.

## Unsupported test fault types

The model was selected for ANY_FAULT. Type-level scores only cover heads learnable from training; the following positive types lack a head in the corresponding family. They remain positive for ANY_FAULT.

| Family | Type | Eligible positive minutes |
|---|---|---:|
| PUMP-01 | DISCHARGE_RESTRICTION | 1195 |
| PUMP-01 | IMPELLER_WEAR | 1902 |
| PUMP-02 | COMMUNICATION_LOSS | 1701 |
| PUMP-03 | SENSOR_STUCK | 976 |
| PUMP-04 | SENSOR_STUCK | 734 |
| PUMP-05 | COMMUNICATION_LOSS | 994 |
| PUMP-05 | SENSOR_CALIBRATION_SHIFT | 1206 |
| PUMP-06 | BEARING_WEAR | 657 |
| PUMP-06 | IMBALANCE | 4284 |
| PUMP-06 | SENSOR_NOISE_SPIKES | 991 |

## Reconstruction

Equal-family mean error over the same train-scaled input features. Lower NMSE is better. The task encoder is judged against un-clipped targets even though its neural input is clipped. Reconstruction and fault classification answer different questions.

| Expanded encoder | NMSE | Skill versus training mean |
|---|---:|---:|
| expanded_autoencoder | 26.898 | 0.566 |
| expanded_denoising_ae | 26.650 | 0.542 |
| expanded_kernel_pca | 34.987 | 0.317 |
| expanded_label_informed | 29.244 | 0.518 |
| expanded_pca | 25.926 | 0.728 |
| expanded_task_encoder | 34.816 | 0.293 |
| expanded_temporal_ae | 33.998 | 0.433 |
