# Pump encodings — encoding-astra

This branch investigates **how to turn pump sensor data into useful 32-dimensional vectors**, then tests whether a classifier can detect faults from those vectors. It includes code, trained models, full vectors, visualizations and measured results.

**Current finding:** PCA preserves the input information best on 11 of 12 pumps. For detecting whether any fault is present, the label-informed autoencoder plus logistic regression leads the main validation comparison. **There is no test-confirmed fault-detection winner.**

## What we did

For each pump independently:

1. Turn each completed minute of sensor data into features: readings, recent changes, physical relationships, missing-data indicators and vibration summaries.
2. Fit an encoder that compresses those features into **32 numbers per minute**.
3. Plot the vectors in two dimensions and inspect exploratory clusters fitted in the full 32 dimensions.
4. Freeze the encoder. Train a classifier on the 32-number vectors and known training labels.
5. Compare candidates on validation data, then evaluate the chosen setup on later test data.

```text
sensor readings → causal features → encoder → 32 numbers → classifier → fault score
                                               ↓
                                       clusters and 2D maps
```

The classifier receives the **32 numbers**, not the 2D plot coordinates. Each pump has its own models; coordinates from different pumps are not aligned and should not be pooled as a shared space.

**Labels are used during training and evaluation. New observations need only sensor data.** The label-informed encoder learns from labels during training; it does not take labels as inputs when encoding new readings.

## Data and labels

The supplied dataset is synthetic: **12 pumps over seven days**, with different sensor configurations. We use minute aggregates, operating context and vibration snapshots. Restoring missing minute positions gives **10,080 rows per pump**, or **120,960 rows overall**. Missing readings remain distinguishable from observed values.

Public event annotations cover PUMP-01 through PUMP-09 and describe fault types, onset, detectability and end times. A minute is labeled faulty only when an annotated fault is observable and detectable. Ambiguous overlaps and unlabeled minutes are excluded from supervised comparisons. **No recorded fault does not prove physical health.** PUMP-10–12 have no public labels.

Feature preparation reads no fault labels, alarms or maintenance outcomes. Labels are joined separately. Exact features and annotation rules are in [the encoding report](experiments/latent-space/REPORT.md).

## Training, validation and test

Every pump is split **in time order**, without random shuffling:

| Period | Minutes per pump | Purpose |
|---|---:|---|
| First 60%: training | 6,048 | Fit scaling, encoders, clusters and classifiers |
| Next 20%: validation | 2,016 | Select checkpoints, compare candidates and choose classifier cutoffs |
| Last 20%: test | 2,016 | Evaluate frozen choices on later observations |

The first 120 minutes of validation and test are excluded from scoring, leaving 1,896 before label exclusions. Features use past and currently available measurements; past history may cross split boundaries.

There are **two separate selection questions**:

- **Which encoder reconstructs its input best?** Use validation reconstruction loss. Neural checkpoints also use this loss, including the label-informed model.
- **Which encoding supports fault detection best?** Freeze all fitted encoders, train identical classifiers, then compare their validation average precision. Choose each classifier's cutoff using validation F1. Test labels do not choose the model or cutoff.

A cutoff converts a score into a decision: a score of 0.7 exceeds a cutoff of 0.6, so that minute is marked faulty. These are minute-level decisions, not a complete alarm policy with persistence or incident grouping.

This is a **within-pump experiment**. The original challenge holds out entire pumps. Its `events_validation.csv` names the original pump split; early annotations for those pumps belong to training in our within-pump experiment. Our scores are not official challenge scores.

## Encoders compared

| Encoder | What it does | Fault labels train the encoder? |
|---|---|---|
| PCA | Linear compression | No |
| Approximate RBF kernel PCA | Nonlinear kernel features, then compression | No |
| Autoencoder | Neural compression trained to reconstruct inputs | No |
| Denoising autoencoder | Reconstructs inputs after adding noise/masking | No |
| Temporal GRU autoencoder | Uses a sequence of past observations | No |
| Label-informed autoencoder | Reconstruction plus supervised contrastive learning | Yes, training-period labels only |

The five unsupervised encoders were fitted on all 12 pumps: **60 models**. The label-informed encoder was fitted on seven pumps: **67 models in total**. PUMP-05/07 have only one usable training label state; PUMP-10–12 have no public labels, so label-informed fitting is unavailable there.

The initial exploration pooled pumps into 16D vectors. It was replaced by the independent 32D experiment, and the label-informed candidate was subsequently added back. [Archived reports](experiments/latent-space/archive/) describe those superseded stages.

## Results: reconstruction versus fault detection

### Reconstruction

**PCA wins validation reconstruction on 11 pumps. The ordinary autoencoder wins on PUMP-08.**

The score measures how much input information survives compression: observed squared reconstruction errors after training-based feature scaling, averaged equally across features. **Lower is better.** It reconstructs engineered features rather than raw waveforms and does not directly measure fault detection.

### Detecting whether any fault is present

We trained the same logistic-regression classifier on each encoding. A fixed 15-neighbor classifier was a secondary check. Across available pumps and learnable binary/fault-type targets, this produced **192 classifier fits**.

The primary comparison averages validation AP equally over PUMP-01 and PUMP-09, using the same examples for all six encoders:

| Encoding + logistic regression | Validation AP |
|---|---:|
| **Label-informed autoencoder** | **0.770** |
| Denoising autoencoder | 0.745 |
| Autoencoder | 0.743 |
| Temporal autoencoder | 0.562 |
| Kernel PCA | 0.466 |
| PCA | 0.392 |

**AP measures whether faulty minutes tend to receive higher scores than no-event minutes. Higher is better. A score of 0.77 is not 77% accuracy.** It summarizes precision as the score threshold changes. The decision cutoff is chosen separately, using validation F1.

The lead over the denoising model is only 0.024 AP. With the nearest-neighbor classifier, the ordinary autoencoder leads instead. For identifying individual known fault types, PCA plus logistic regression leads a different comparison involving four fault/pump cases across three pumps. There is no universally best encoding in these results.

### Why only two pumps support the main comparison

A pump needs **both faulty and no-recorded-fault examples in training AND scored validation**. Only PUMP-01 and PUMP-09 meet this requirement across all six encoders.

| Pump(s) | Why excluded from that comparison |
|---|---|
| 02, 03, 04, 08 | Training has faulty examples but no usable no-event examples |
| 05 | Training has one usable fault state; no usable validation/test labels remain |
| 06 | Training has both categories, but scored validation contains only faults |
| 07 | Training has no faulty examples; scored validation has no usable labels |
| 10–12 | No public labels |

This follows from the labels and chronological split, not unfinished model runs. We did not shuffle adjacent minutes or invent healthy examples to improve the score. Exact counts are in [label_coverage.csv](experiments/latent-space/results/fault_detection/label_coverage.csv).

### What happened on test data

| Label-informed encoder + logistic regression | Faulty test minutes detected |
|---|---:|
| PUMP-01 | 45.1% |
| PUMP-09 | 100.0% |

Both comparable test periods contain only faulty examples. **We cannot measure their false-alarm rates or confirm a detector ranking.** Always saying “fault” would get 100% recall on such a period. The low PUMP-01 recall also shows that the validation lead did not hold consistently over time.

All trainable test targets are single-class after exclusions. PUMP-02 has mixed test labels, but its training period has no no-event examples, so it cannot supply a two-class detector under this protocol. Single-class AP/AUROC are excluded from rankings; an all-positive AP of 1 would be misleading.

**The label-informed encoder is a provisional candidate, not a confirmed winner.** A stronger conclusion needs later test periods with both faulty and no-event behavior and more independent incidents. These synthetic results do not establish performance on unseen pumps or real machinery.

## Open the visualizations

From the repository root; no ML environment is needed to serve saved pages:

```sh
python3 -m http.server 8766 --bind 127.0.0.1 \
  --directory experiments/latent-space/results
```

- **Embedding explorer:** <http://127.0.0.1:8766/explorer.html> — choose a pump, encoder, projection and color scheme.
- **Detection comparison:** <http://127.0.0.1:8766/fault_detection.html> — rankings, test results and limitations.

Each plotted point represents a sampled minute. The 2D distances approximate relationships in 32D; they are not fault probabilities. Cluster numbers have no assigned fault meaning. Fault colors are annotation overlays, not classifier predictions. Full vectors contain every minute; maps show one point every ten minutes.

## Code, models and reproduction

Start with [the experiment README](experiments/latent-space/README.md) for setup, loading vectors, applying a trained encoder, verifying results and rerunning training.

- [Encoding report](experiments/latent-space/REPORT.md): architecture, labels and per-pump reconstruction results.
- [Fault-detection report](experiments/latent-space/FAULT_DETECTION_REPORT.md): classifier protocol and all comparisons.
- [Branch verification](experiments/latent-space/BRANCH_VERIFICATION.md): checks actually executed before pushing.
- [Experiment history](experiments/latent-space/EXPERIMENT_LOG.md): how the work evolved.

Included: **67 trained encoders, their full per-minute 32D arrays, row metadata, 192 trained classifiers**, predictions, metrics and visualizations. Large feature caches and duplicate vector/split tables can be regenerated and are omitted. The organizer's files remain under [`RETROFIT_AI_PARTICIPANTS_2026/`](RETROFIT_AI_PARTICIPANTS_2026/), with their original [license and provenance](RETROFIT_AI_PARTICIPANTS_2026/LICENSE_AND_PROVENANCE.md).
