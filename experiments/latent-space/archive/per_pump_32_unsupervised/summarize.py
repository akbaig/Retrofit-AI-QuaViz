"""Write a reproducible report from measured per-pump results."""
import json
import numpy as np
import pandas as pd
from data import ROOT


def main():
    out=ROOT/'results';selected=pd.read_csv(out/'selected_models.csv');all_scores=pd.read_csv(out/'model_comparison.csv')
    names={'pca':'PCA','kernel_pca':'RBF kernel PCA','autoencoder':'Autoencoder','denoising_ae':'Denoising AE','temporal_ae':'Temporal AE'}
    table=['| Pump | Features | Selected encoder | Validation NMSE | Test NMSE | Test improvement over mean |','|---|---:|---|---:|---:|---:|']
    for r in selected.itertuples():
        table.append(f'| {r.asset_id} | {r.input_features} | {names[r.model]} | {r.validation_nmse:.5g} | {r.test_nmse:.5g} | {r.test_skill:.1%} |')
    diagnostic=[]
    for r in selected.itertuples():
        dest=out/r.asset_id;features=json.loads((dest/'features.json').read_text())['included']
        metrics=json.loads((dest/f'{r.model}_metrics.json').read_text())
        for split in ['validation','test']:
            losses=np.asarray(metrics[split]['feature_nmse'],dtype=float);valid=np.isfinite(losses);total=losses[valid].sum()
            for j in np.argsort(np.nan_to_num(losses,nan=-1))[::-1][:5]:
                diagnostic.append({'asset_id':r.asset_id,'model':r.model,'split':split,'feature':features[j],'feature_nmse':losses[j],'share_of_total_feature_error':losses[j]/total,'observed_minutes':metrics[split]['feature_counts'][j]})
    pd.DataFrame(diagnostic).to_csv(out/'largest_reconstruction_errors.csv',index=False)
    med=selected.test_skill.median()
    report=f'''# Independent 32D pump representations

The active experiment uses no event labels. It fits five encoders independently on each of 12 pumps and selects each winner by validation reconstruction. **PCA wins on 11 pumps; the ordinary autoencoder wins on PUMP-08.** There are 120,960 selected minute vectors, each with 32 coordinates. Across these selected encoders, median test improvement over the training-mean predictor is {med:.1%}; the range is {selected.test_skill.min():.1%} to {selected.test_skill.max():.1%}.

This is a controlled comparison of the implemented models, with one seed and modest architectures, not evidence that PCA is the globally best representation or the best eventual fault classifier.

## The problem and the data

The hackathon asks for early, explainable warnings from industrial/marine pump telemetry, including distinguishing machine degradation from sensor failures. The supplied dataset is synthetic: 12 pumps, seven days per pump, 14 possible sensor channels and three instrumentation tiers. Legacy T0 pumps have few channels; T1 adds sparse wireless measurements every five minutes; T2 has fuller instrumentation. Missing hardware and missing observations must not be treated as measured zeros.

The dataset contains 7,238,783 one-second telemetry rows, 120,650 supplied one-minute aggregates and 358 vibration snapshots. The minute grid has 120,960 rows after explicitly restoring 310 absent minutes. Each snapshot is a one-second waveform with 10,240 samples. Original event annotations describe intervals, causes, detectability and observability on the seven training and two validation pumps; the three original test pumps are unlabeled. This new experiment reads none of those event files.

Only `assets.csv`, `features_1min.parquet`, `operating_context.csv` and `vibration_snapshots.parquet` are read. The encoder inputs are causal numeric sensor levels, dispersion, changes, trailing summaries, physics residuals, measurement-quality features and waveform-derived features. No pump identifier, split identifier, timestamp, fault label, event-derived feature, alarm or maintenance outcome is an encoder input. Operating mode is used by the causal feature builder to interpret running/stopped behavior and is available as visualization metadata. Waveforms are summarized into physical features; this experiment does not learn directly from raw waveforms.

## Previous metric: what it meant and why it changed

The previous comparison used a fixed 15-nearest-neighbor probe in the latent space. For each held-out minute, the fraction of its training neighbors with a given fault label became a score for that fault. Average precision (AP) measures the quality of that ranking across precision/recall thresholds, and macro AP averages it equally over evaluated fault types. Higher is better. AP is neither classification accuracy nor reconstruction loss, and its baseline depends on fault prevalence. See the [official average-precision definition](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html).

That probe addressed whether the geometry preserved labeled fault information. It could evaluate an unsupervised encoder, but both the probe and choosing a model by AP required fault labels. The additional contrastive branch also used labels during fitting. A label-informed encoder technically needs labels during fitting, not when encoding a new observation, but the user explicitly excludes that approach, so it is removed from the active pipeline and explorer. Historical artifacts remain only under `archive/pooled_16`.

Reconstruction was not the original primary metric because excellent reconstruction does not guarantee fault separation: an encoder can preserve normal operating variation while losing small diagnostic changes. Nevertheless, it should have been included as a direct compression check. For the requested entirely unsupervised comparison, held-out reconstruction is the primary metric now. It assesses information retention without needing fault labels. Neither metric alone establishes a universally best latent space.

## Current reconstruction metric

For each retained feature j, estimate mean μ_j and standard deviation σ_j from its observed training values. Reconstruct the standardized value u_tj = (x_tj − μ_j) / σ_j. Let O_j contain scored minutes where feature j is observed:

**NMSE = mean over j [ mean over t in O_j of (u_tj − reconstructed_u_tj)² ].**

This averages within each feature first, then across features, so sparse channels do not lose weight merely because they have fewer observations. Lower is better. It is dimensionless but not bounded by one. A loss above one is possible under distribution shift; 100 is not “100% error.” Targets and predictions are not clipped. The name NMSE here means training-variance-standardized, feature-balanced MSE, not division by held-out variance.

Missing inputs are filled with the training mean and accompanied by a mask; missing targets do not contribute to loss. Features that are absent, have fewer than two observations, or are constant in training are excluded and listed in each pump's `features.json`. Depending on instrumentation and training variability, {selected.input_features.min()}–{selected.input_features.max()} features remain. Held-out behavior in an excluded feature is therefore outside the reported reconstruction score.

For context, the training-mean predictor always emits u=0. **Skill = 1 − model NMSE / mean-predictor NMSE.** A skill of 60% means 60% less squared reconstruction error than that baseline; zero is no improvement and a negative value is worse. Inputs and baseline are identical for all models within a pump. Pump feature sets and scales differ, so comparing raw NMSE between pumps is less meaningful than comparing models within a pump or their baseline-relative improvements.

The error timeline uses observed-feature mean error within each minute. Because missingness varies, averaging that timeline is not necessarily equal to the table's feature-first primary score.

## Evaluation protocol

For each pump, in time order:

| Split | Minutes | Minutes scored | Role |
|---|---:|---:|---|
| Train | 6,048 | 6,048 | Fit features/scalers/models/decoders/clusters |
| Validation | 2,016 | 1,896 | Choose neural checkpoint and encoder |
| Test | 2,016 | 1,896 | Evaluate after model selection |

The first 120 minutes of validation and test are warmup and excluded from scoring. They remain in exported datasets with `score_eligible=False`. All histories are causal; earlier measured state may carry across a split. Snapshots can be up to six hours old and thermal state has longer memory, so this is streaming-style future evaluation, not fully independent incident-level evaluation. Multi-day behavior can span a boundary.

Feature selection and all input/latent scaling fit training only. Neural networks train for up to 50 epochs, with checkpoints checked at epoch 1 and every 5 epochs; the lowest validation reconstruction checkpoint is retained. The winning encoder is frozen in `selection.json` before test scoring. The seed is 42 throughout. Each pump has its own scaler, encoder and latent coordinate system.

This follows the requested **within-pump** split. It uses early observations even from the pumps that the original challenge calls validation/test, and therefore must not be presented as performance on the original unseen-pump benchmark. No fault-classifier performance is measured.

## Models compared

| Encoder | Implementation | Reconstruction |
|---|---|---|
| PCA | 32 components on standardized values plus 0.25-weighted observation masks | Analytic inverse PCA, observed value columns scored |
| RBF kernel PCA | Train-only Nyström approximation with 256 landmarks, training median-distance bandwidth, PCA to 32 | Separate train-only ridge decoder for each observed feature |
| Autoencoder | MLP: 2D → 128 → 64 → 32 | MLP: 32 → 64 → 128 → D |
| Denoising AE | Same MLP, with 15% input masking and 0.03 standardized-unit noise during training | Clean current-feature targets |
| Temporal AE | 12 observations five minutes apart, ending now; 96-unit GRU → 32 | Current-feature decoder, same masking/noise |

D is that pump's retained feature count; masks double the encoder input width. Neural decoders optimize masked, feature-balanced reconstruction. PCA optimizes the imputed/mask-augmented matrix, and the kernel method has a separate fitted inverse. They are evaluated against identical observed targets, but their training objectives and decoder capacities are not identical. Exported latent coordinates receive train-only scaling for clustering and visualization; the saved bundle includes the inverse scaling needed by its decoder.

## Measured results

{chr(10).join(table)}

All selected models beat the training-mean baseline on their test period. The high reconstruction losses on some pumps are real in standardized feature units and are not hidden by target clipping.

- **PUMP-03 and PUMP-05:** only 55 variable training features remain; 32 PCA coordinates retain most of their held-out variation. Low loss here does not demonstrate broad fault observability from sparse sensors.
- **PUMP-06:** validation error is much higher than test error. Waveform RMS and high-frequency summaries dominate the validation error. The validation period is not uniformly easier than the test period.
- **PUMP-08:** the ordinary AE improves validation reconstruction over PCA, but test NMSE remains high. Waveform kurtosis is the largest test error contributor. Selection does not imply good absolute reconstruction everywhere.
- **PUMP-10 and PUMP-12:** waveform-envelope and harmonic summaries dominate the largest test reconstruction errors. PUMP-01/PUMP-07 also have substantial held-out shifts. These observations identify reconstruction failures, not diagnosed fault causes.

The full five-model table is `results/model_comparison.csv`. `results/largest_reconstruction_errors.csv` identifies the five largest feature errors per selected pump and split. Large MSE can be driven by features that had very small training variance and then changed substantially. That is visible, rather than clipped away, in this score.

## Reading the maps

The explorer offers a separate map for every pump and encoder, colored by unsupervised cluster, dataset split, elapsed time, operating mode or reconstruction error. The 12-pump overview shows each validation-selected encoder. Each point represents a completed minute sampled every ten minutes for display; all minute vectors are exported.

K-means uses eight clusters fitted in the full 32D training space. Eight is an exploratory display choice, not an estimated number of faults. UMAP fits the sampled training vectors; validation/test points are transformed with that fixed map. A linear two-component PCA projection is also available. UMAP can visually exaggerate separation and transform unusual held-out points near the training manifold, so inspect the loss timeline and both projections rather than treating empty gaps as fault classes. Held-out neighborhood trustworthiness is exported to `projection_quality.csv` as a projection diagnostic; it is not used to select encoders.

The axes and cluster numbers are independent across pumps and models. Cluster 2 on PUMP-01 does not mean the same thing as cluster 2 on PUMP-02. Concatenated exported rows retain pump metadata but do not establish an aligned cross-pump latent space.

## Reuse and limits

Per-pump models, all five full 32D embedding matrices, explicit train/validation/test Parquet files, reconstruction decoders, feature schemas, source hashes and an offline explorer are saved under `results`. `encode.py` applies a compatible pump model without labels or refitting. See `README.md` for commands.

The experiment covers only seven synthetic days per pump and one optimization seed. It does not establish stability across seeds, resistance to unseen operating regimes, real-world transfer, anomaly thresholds, causal root causes, or fault-classifier performance. PCA is the strongest reconstruction baseline among these runs; the next classifier or anomaly detector will need its own evaluation. More aggressive model/feature/hyperparameter exploration would require a fresh held-out assessment rather than repeatedly choosing models using these test results.
'''
    (ROOT/'REPORT.md').write_text(report)
    print('Wrote REPORT.md and largest_reconstruction_errors.csv')
if __name__=='__main__':main()
