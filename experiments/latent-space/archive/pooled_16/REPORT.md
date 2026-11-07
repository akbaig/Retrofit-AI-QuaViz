> Historical snapshot; superseded by the current 32D experiment. See [ARCHIVE_NOTICE.md](ARCHIVE_NOTICE.md) for omitted generated artifacts.

# What is in the data, and what the embeddings learned

The most useful fault-oriented representation in this comparison is the **label-informed contrastive encoder**. It exports 16 dimensions per completed minute and improves the fixed validation neighborhood diagnostic over all five unsupervised alternatives. Raw-observation PCA is the strongest label-free baseline under that diagnostic. Neither representation cleanly separates every fault type, and there is no evidence here of a universal optimum or field-ready diagnosis.

## The problem and data construction

The `START_HERE.html` brief asks for an operator-assist system that notices developing cooling-water pump problems, explains evidence, separates instrument problems from machine problems, and recommends human checks. This experiment stops at representations and offline diagnostics; it does not create alerts, recommendations or a production classifier.

The synthetic generator first creates operating context and evolving physical fault states. A pump/controller/system model produces flow, pressures, electrical load, thermal behavior and vibration. The instrumentation layer adds sensor coverage, sparse reporting, calibration offsets, lag, noise, quantization, dropout and sensor faults. Public tables contain observations; hidden physical states are not encoder inputs.

Observed files contain:

| Item | Actual contents |
|---|---|
| Full telemetry | **7,238,783 observed rows**, nominally 1 Hz, 12 pumps × 7 days; Parquet partitioned by pump/day |
| Minute summaries | **120,650 rows × 76 columns**; mean/std/min/max/count per channel, operating mode, running fraction, row count and quality |
| Complete analysis grid | **120,960 minutes** after inserting **310 entirely missing minutes** |
| Channels | 14 sensor measurements, including speed, electrical load, pressures, flow, temperatures, vibration and valve position |
| Vibration snapshots | **358** one-second recordings, 10,240 samples each at 10,240 Hz; sparse observations, not continuous audio |
| Training | PUMP-01–07; **27 events, 26 observable** |
| Validation | PUMP-08–09; **9 observable events** |
| Test | PUMP-10–12; **30,240 minute vectors**, labels remain unknown |
| Context | Minute-level demand, supply/ambient temperatures, schedule, weather and suction level |
| Other records | Plant alarms, sometimes mistaken maintenance notes, benign-operation intervals, asset ratings and sensor availability |

The starter data are a 10-second resampling of the same benchmark, not an independent dataset. This experiment uses the full package's minute summaries and waveforms, without reprocessing every raw second. Minute aggregation may discard fast transients that future raw-series models could use.

## Missing values and labels need special treatment

T0 pumps lack many instruments. T1 pumps add wireless bearing/vibration measurements approximately every five minutes. T2 pumps have dense instrumentation. Missing hardware, an expected interval between wireless reports, an invalid/missing sample, and an entire gateway outage are different situations.

The copied causal feature pipeline uses the hardware manifest, bounded sparse forward holds, recent trends, gap counters and observed waveform spectra. Missing measurements have explicit masks in encoder inputs; autoencoder reconstruction ignores missing targets. Full outages are represented on an independent minute grid. Raw signals are not converted into fictitious healthy zero readings.

Labels are event intervals, not one class per row. They include physical onset, first detectability, resolution, observability, class, fault type, severity and evidence channels. Detectability comes from simulator counterfactual comparisons against healthy conditions and instrument noise, so it is evaluation truth, not an online observable.

- Eight machine fault types and five sensor/data fault types exist in training.
- A fault can be physically present before it is observable.
- Several faults can overlap; machine and sensor hypotheses can both be true.
- One training bearing-wear event is unobservable on its pump.
- “No labeled event” is only a metadata statement, not a proof of health.
- Benign start/stop/load/weather windows are preserved as metadata, not given fault labels.

The final evaluation excludes **22,068 development minutes** with any pre-detectable or unobservable event. These flags are independent of the displayed fault color: a visible fault must not hide an unknown overlapping fault. This conservative row-level exclusion avoids false negative labels but sacrifices valid partial supervision. In particular, all 47 five-minute sampled training positives for air entrainment are excluded; that fault therefore has no eligible training support in the contrastive fit or neighbor reference set. Per-class uncertainty masks are a useful next improvement, with a fresh validation protocol.

## Models actually run

Every representation has 16 output dimensions. Five models are unsupervised. The sixth uses training-event labels only in a contrastive objective; labels are never inference inputs.

| Representation | Inputs / training | Validation macro AP | Running-only AP | Pump/cluster NMI |
|---|---|---:|---:|---:|
| **Label-informed contrastive, seed42** | 210 engineered features; masked views, balanced event-state sampling, reconstruction regularization | **0.435** | **0.368** | **0.436** |
| Raw-observation PCA | 44 raw-level / variability / observation features | 0.301 | 0.216 | 0.732 |
| Denoising autoencoder | 210 engineered features; MLP reconstruction | 0.289 | 0.208 | 0.658 |
| Physics + history PCA | 210 engineered features | 0.264 | 0.185 | 0.668 |
| Temporal autoencoder | 12 past-only, five-minute-spaced feature vectors; GRU | 0.260 | 0.257 | 0.676 |
| Approximate RBF kernel PCA | 210 engineered features; 256 Nyström landmarks | 0.256 | 0.197 | 0.618 |

Higher neighborhood AP is better. Pump/cluster NMI measures how strongly cluster membership follows pump identity; lower values indicate less of that confounding, not necessarily a better model in isolation.

The contrastive seed repeats scored **0.422 (seed7), 0.435 (seed42), and 0.491 (seed123)**: mean **0.449**, sample standard deviation **0.036**. Seed123 is the highest observed run; the recommended export retains the prespecified seed42 rather than choosing a lucky initialization. All three vectors and model checkpoints are available.

This is a comparison of practical pipelines, not a controlled claim that one architecture alone caused the gain. Feature sets, objectives and training budgets differ. PCA fits all training minutes; neural models use strided training observations, and contrastive training uses labels and balanced state sampling. The physical inputs include rating-normalized load, hydraulic and thermal consistency features, trailing changes, and observed vibration-spectrum summaries.

The contrastive model uses an MLP encoder 420 → 128 → 64 → 16 (210 values plus 210 weighted masks), with GELU activations. A small decoder regularizes the representation. Positive pairs share the exact observed multilabel state; two masked/noisy views of the same row are also positive. No asset ID, timestamp, event ID, deadline, maintenance conclusion, or hidden simulator state is a neural input. This is a custom tabular/time-series adaptation of [supervised contrastive learning](https://arxiv.org/abs/2004.11362), not TS2Vec or a pretrained foundation model.

## How the comparison was measured

Preprocessing, encoders, latent scaling and cluster centroids fit **only PUMP-01–07**. All validation labels are held out from fitting. The contrastive extension was motivated by validation results, so validation is development/model-selection data, not a blind estimate of final performance. Test data never choose hyperparameters, weights, clusters or model identity.

For the fixed neighborhood diagnostic, use one eligible observation every five minutes: **10,530 training reference rows** and **3,202 validation query rows**. Retrieve 15 neighbors in the 16D representation, take their mean multilabel indicators, and compute average precision for each of the nine fault types present in eligible validation data. The reported macro AP is their unweighted mean; the mean prevalence reference is **0.135**. AP is the non-interpolated ranking statistic documented by [scikit-learn](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html). It is not accuracy, a calibrated probability, event recall, early-warning performance or the official challenge score.

KMeans uses 12 clusters in 16D, fitted without labels on strided training observations. A second initialization measures stability. The report also saves fault/cluster ARI and NMI, latent silhouette, pump/tier/mode association, and projection trustworthiness. These diagnostics need not agree: for example, RBF PCA has higher validation fault-cluster ARI than the contrastive encoder, while its nearest-neighbor fault ranking is worse. No cluster-count or diagnostic threshold search was performed.

## What works, and what does not

For the recommended seed42 representation:

| Validation fault | Neighborhood AP | Interpretation |
|---|---:|---|
| Communication loss | 1.000 | Missing-stream state is easy to represent |
| Suction-strainer clogging | 0.986 | Strong transferable neighborhood structure |
| Discharge restriction | 0.979 | Strong transferable neighborhood structure |
| Bearing wear | 0.360 | Not better than its pooled prevalence reference of 0.390 |
| Misalignment | 0.276 | Only a small improvement over prevalence 0.256 |
| Sensor drift | 0.174 | At prevalence reference; no useful ranking in this probe |
| Sensor stuck | 0.076 | At prevalence reference |
| Sensor noise/spikes | 0.039 | At prevalence reference |
| Air entrainment | 0.026 | At prevalence reference; no eligible training positives |

Per-pump macro AP is **0.771 on PUMP-08** and **0.374 on PUMP-09**; each averages the fault types present on that pump, so these do not average to the pooled macro AP. Transfer to the second pump remains materially harder. The gain is concentrated in specific hydraulic and missing-data behavior, rather than being a solution to every fault category.

## Reading the visualizations

The HTML explorer can color the same map by fault type, machine/sensor class, KMeans cluster, pump, sensor tier, operating mode or benign context. Click a point to see its time, event IDs, unknown-label flags and number of telemetry rows received.

The plots show one point every ten minutes from training/validation. The exported vectors cover every minute. UMAP fits only the displayed training vectors and transforms the validation vectors; PCA projection fits all training vectors. Test data are omitted from the maps. UMAP is for visual inspection: its documentation warns that dimensional reduction can create apparent tears or islands. [UMAP clustering guidance](https://umap-learn.readthedocs.io/en/latest/clustering.html)

The plots deliberately include pump and tier views because a visually tidy map can simply reflect equipment identity. Preserve the fitted 16D representation for downstream analysis, not the 2D layout.

## Verification and limits

Ten focused tests pass, covering label boundaries/overlaps, training-frozen preprocessing, missing values, past-only features, sparse sensors, sequence boundaries, masked reconstruction and encoder reload. An independent reviewer found the overlapping-uncertainty bug and stale-cache risk; both were fixed and the experiment rerun. Provisional outputs remain in `archive/provisional_v1/`.

Artifact verification checks all eight 120,960 × 16 exports, finite values, pump/time row alignment, saved-model replay, training asset lists, source hashes, test labels, selected test replay and exclusion of ambiguous intervals. The independent review also mutated every non-training input/mask/label in a controlled contrastive fit and observed identical weights.

Remaining limitations: synthetic data only; seven training pumps and two validation pumps; nine validation events and temporally correlated minutes; no independent statistical confidence claim; untested encoder dimensions, pretrained models and raw-second architectures; retained equipment confounding; limited labeled examples; sensitivity to initialization; conservative loss of partially labeled rows; and no downstream classifier or official alert benchmark yet. Those constraints limit any claim of “best.” The deliverable is a measured, reusable starting representation and an honest view of what it currently preserves.
