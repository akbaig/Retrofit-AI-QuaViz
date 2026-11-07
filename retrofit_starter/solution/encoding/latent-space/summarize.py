"""Generate the current per-pump comparison report from saved results."""
import json
import numpy as np
import pandas as pd
from data import ROOT
from label_informed import LABEL_KIND
NAMES={'pca':'PCA','kernel_pca':'RBF kernel PCA','autoencoder':'Autoencoder','denoising_ae':'Denoising AE','temporal_ae':'Temporal AE','label_informed':'Label-informed AE'}


def main():
    out=ROOT/'results';selected=pd.read_csv(out/'selected_models.csv');scores=pd.read_csv(out/'model_comparison.csv')
    availability=json.loads((out/'labels_manifest.json').read_text())['pumps']
    informed=scores[scores.model.eq(LABEL_KIND)]
    counts=selected.model.value_counts();winners=', '.join(f'{NAMES[name]} on {count} '+('pump' if count==1 else 'pumps') for name,count in counts.items())
    table=['| Pump | Selected encoder | Features | Validation NMSE | Test NMSE | Test improvement over mean |','|---|---|---:|---:|---:|---:|']
    for r in selected.itertuples():table.append(f'| {r.asset_id} | {NAMES[r.model]} | {r.input_features} | {r.validation_nmse:.5g} | {r.test_nmse:.5g} | {r.test_skill:.1%} |')
    labeltable=['| Pump | Label-informed validation | Label-informed test | Best unsupervised validation | Supervised training minutes |','|---|---:|---:|---:|---:|']
    for r in informed.itertuples():
        old=scores[scores.asset_id.eq(r.asset_id)&scores.model.ne(LABEL_KIND)].validation_nmse.min()
        status=next(x for x in availability if x['asset_id']==r.asset_id)
        labeltable.append(f'| {r.asset_id} | {r.validation_nmse:.5g} | {r.test_nmse:.5g} | {old:.5g} | {status["supervised_rows"]:,} |')
    unavailable='\n'.join(f'- **{r["asset_id"]}:** {r["reason"]}. Training-state counts: {r["training_states"] or "no public annotations"}.' for r in availability if not r['available'])
    diagnostic=[]
    for r in selected.itertuples():
        dest=out/r.asset_id;features=json.loads((dest/'features.json').read_text())['included'];metrics=json.loads((dest/f'{r.model}_metrics.json').read_text())
        for split in ['validation','test']:
            losses=np.asarray(metrics[split]['feature_nmse'],dtype=float);total=np.nansum(losses)
            for j in np.argsort(np.nan_to_num(losses,nan=-1))[::-1][:5]:
                diagnostic.append({'asset_id':r.asset_id,'model':r.model,'split':split,'feature':features[j],'feature_nmse':losses[j],'share_of_total_feature_error':losses[j]/total,'observed_minutes':metrics[split]['feature_counts'][j]})
    pd.DataFrame(diagnostic).to_csv(out/'largest_reconstruction_errors.csv',index=False)
    text=f'''# Per-pump 32D encoders, including label-informed learning

The active experiment now contains **{len(scores)} fitted candidates**: the existing 60 unsupervised models plus {len(informed)} label-informed autoencoders. All output 32-dimensional sensor-derived vectors. **Validation reconstruction selects {winners}.** The label-informed model remains available for comparison even when it is not the reconstruction winner.

The user explicitly requested restoring the label-informed approach within the new per-pump structure. The prior entirely unsupervised experiment is recorded in `archive/per_pump_32_unsupervised`. Its 60 fitted baseline models and full vector files remain unchanged; this extension trains new candidates only.

## Label-informed encoder

This is a supervised contrastive autoencoder. It uses the same MLP capacity, input feature schema and 32D bottleneck as the ordinary autoencoder. Its objective is:

**training loss = observed reconstruction loss + 0.1 × supervised contrastive loss.**

Reconstruction uses every training row. The contrastive part samples training annotation states equally, produces two views with 10% masking and 0.03 standardized-unit noise, and pulls together examples with exactly the same detectable multilabel state. Temperature is 0.2. Every training step uses 32 sampled states with four examples per state and two views. A state can contain multiple simultaneous faults. `NO_LABELED_EVENT` is a separate state, not a guarantee of health. Rows overlapping pre-detectable or unobservable events are excluded from supervised pairs even if another fault is visible.

Only each pump's first 60% of minutes can supply training labels or the training-state vocabulary. Events/label states that occur only later cannot shape fitting. Neural checkpoints still use **validation reconstruction**, not annotation accuracy; neither validation nor test labels select the winner. All candidates use seed 42 and at most 50 epochs, checking validation at epoch 1 and every five epochs. The contrastive weight is fixed in advance, not tuned against these test scores.

**At inference, labels are not inputs.** The saved encoder accepts the same standardized sensor features and masks as the other models. Its decoder reconstructs those inputs, so it is evaluated with the identical reconstruction metric. Labels are used to shape training geometry; the output is not a one-hot encoding of the label.

## Label availability

The public annotation files cover PUMP-01 through PUMP-09. There is enough unambiguous state variation to train the contrastive model on {len(informed)} pumps. The remaining models are explicitly unavailable rather than being replaced by fabricated labels or a model trained on different pumps:

{unavailable}

For the two single-state pumps, all supervised examples would be positives with no contrasting state; calling such a fit a meaningful fault-informed comparison would be misleading. Their five unsupervised candidates remain available.

Public `events_validation.csv` supplies annotations for PUMP-08/09. Those names refer to the original challenge split; this user-requested experiment instead uses chronological train/validation/test **inside each pump**, so only the early 60% of those annotations train the new models. The experiment is not an official unseen-pump benchmark evaluation.

## Label-informed reconstruction results

{chr(10).join(labeltable)}

Lower NMSE is better. Label-informed learning did not beat the best unsupervised validation reconstruction in this run. This does not establish that it is worse for future fault classification: reconstruction and label geometry reward different things. The separate downstream classifier experiment is reported in FAULT_DETECTION_REPORT.md; reconstruction scores do not measure detection. Test losses remain reports after validation selection, not model-selection inputs.

## Shared data and split

The supplied dataset is synthetic: 12 pumps, seven days each, 14 possible sensor channels, three instrumentation tiers, 7,238,783 one-second telemetry rows, 120,650 supplied one-minute aggregates and 358 one-second vibration snapshots. Restoring 310 missing minute positions yields 120,960 minute vectors across the 12 pumps. Sensor coverage varies: legacy T0 is sparse, T1 has five-minute wireless measurements, and T2 has fuller instruments.

The label-free observation builder reads `assets.csv`, `features_1min.parquet`, `operating_context.csv`, and `vibration_snapshots.parquet`. The added training/annotation module reads only the two public event files, `events_train.csv` and `events_validation.csv`; it does not read private truth, alarms or maintenance outcomes. `preparation.json` records that feature preparation does not read labels; `labels_manifest.json` separately records the extension's event sources and use.

All models reconstruct the same retained features within a pump: sensor levels and variation, causal changes/trailing summaries, physics residuals, observation-quality indicators and waveform summaries. They do not reconstruct raw one-second samples or raw waveforms. Depending on training availability/variability, {selected.input_features.min()}–{selected.input_features.max()} features remain. Pump IDs, split IDs, timestamps and event annotations are not input features. Operating mode helps the causal feature builder interpret running/stopped behavior and is also display metadata.

| Split | Minutes per pump | Scored minutes | Role |
|---|---:|---:|---|
| Train | 6,048 | 6,048 | Fit scalers, encoders, decoders, clusters; supply labels when supported |
| Validation | 2,016 | 1,896 | Select neural checkpoint and encoder by reconstruction |
| Test | 2,016 | 1,896 | Report reconstruction after selection |

The first 120 validation/test minutes are warmup and excluded from scoring, but remain in split exports with `score_eligible=False`. Histories are causal and can carry earlier measured state across boundaries. Snapshots can be six hours old and thermal state has longer memory, so this is streaming-style chronological evaluation, not incident-independent evaluation. Multi-day events can span a boundary.

## Reconstruction metric and the earlier AP metric

Each feature is standardized using its observed training mean and standard deviation. For each feature, average squared reconstruction error over its observed scored minutes; then average those feature errors equally. This **feature-balanced, training-variance-normalized MSE (NMSE)** is dimensionless and lower is better. It can exceed one under distribution shift. Targets are not clipped, and absent targets are ignored. Missing inputs use the training mean plus an observation mask. Features absent or constant in training are excluded and disclosed in each pump's `features.json`; their later behavior is outside this score.

The training-mean baseline predicts standardized value zero. **Skill = 1 − model NMSE / mean-predictor NMSE.** For example, 60% skill means 60% less squared error than that baseline. Feature sets and scales vary by pump, so compare model scores within a pump rather than interpreting raw cross-pump differences as an overall ranking. The timeline averages within each minute and can differ from the feature-first table metric when missingness varies.

The previous pooled experiment used macro average precision from a fixed 15-neighbor label probe: neighbor fault-label frequencies ranked faults, AP measured ranking quality, and macro averaging weighted fault types equally. It was not accuracy or reconstruction. That metric used labels for evaluation and model choice. It is **not restored as the selection criterion** here; restoring label-informed training does not change the current reconstruction-based evaluation.

## Selected models

{chr(10).join(table)}

Median selected-model test improvement over the training-mean baseline is {selected.test_skill.median():.1%}. Large test errors on some pumps are concentrated in particular waveform or sensor-behavior features, especially when a feature had little training variation and changes later. The largest per-feature errors are in `results/largest_reconstruction_errors.csv`. They identify information the decoder fails to retain, not diagnosed fault causes.

The baseline implementations remain: PCA32 with analytic inverse; RBF Nyström256 → PCA32 with train-only ridge decoders; ordinary MLP AE; denoising MLP AE; and a 96-unit GRU using 12 observations five minutes apart. Neural MLP encoder widths are 2D→128→64→32 and decoder widths32→64→128→D. Their training objectives differ, but reconstruction targets and evaluation are shared. These particular implementations are not a proof of global optimality.

## Visualizations and saved artifacts

`results/explorer.html` offers pump/model selection, UMAP/PCA projections, and coloring by cluster, dataset split, time, operating mode, error or fault annotation. Unavailable label-informed options are disabled with the reason displayed. The comparison table explicitly marks models that use training labels. Held-out annotations are a display overlay only; `UNLABELED` and `AMBIGUOUS_EVENT` are distinct from no labeled event.

The new `results/figures/per_pump_label_informed.png` shows the label-informed spaces and unavailable pumps. Each pump also has a six-panel candidate figure. Selected-model overview grids remain available by split and cluster.

K-means fits eight exploratory clusters in the full 32D training vectors. UMAP fits a training sample, then transforms held-out points; PCA projections also fit training only. Display points are sampled every ten minutes, while all minute vectors are exported. UMAP can exaggerate gaps or map unusual points near the training manifold; the error timeline and linear projection provide complementary views. Neighborhood trustworthiness is a projection diagnostic, not a selection criterion.

Axes and cluster numbers are local to each pump and encoder. Cluster2 on one pump is not equivalent to cluster2 on another. All models, preprocessing, reconstruction decoders, vector files, explicit split datasets, source hashes and metrics are saved. Each fitted label-informed model also has `label_informed_train.parquet`, `label_informed_validation.parquet` and `label_informed_test.parquet`. `encode.py` applies its saved sensor-only inference interface without event files. See `README.md` for commands.

## Limits

This is one seed on seven synthetic days per pump. It does not establish robustness across seeds, unseen pumps, real machinery, future operating regimes or downstream classifiers. The test intervals were already inspected in the earlier experiment; this is an exploratory extension with fixed settings, not a new untouched final benchmark. Repeated optimization against these test results would require a fresh assessment set. Per-pump label coverage is limited, and some training periods contain no contrasting unambiguous states.
'''
    (ROOT/'REPORT.md').write_text(text)
    informed.to_csv(out/'label_informed_comparison.csv',index=False)
    print('Wrote report:',len(scores),'candidates;',len(informed),'label-informed models;',winners)
if __name__=='__main__':main()
