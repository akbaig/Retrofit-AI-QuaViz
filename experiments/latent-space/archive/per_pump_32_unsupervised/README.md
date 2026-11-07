> Historical snapshot; superseded by the current 32D experiment. See [ARCHIVE_NOTICE.md](ARCHIVE_NOTICE.md) for omitted generated artifacts.

# Unsupervised, 32-dimensional embeddings per pump

Five encoders are fitted independently for every pump: PCA, approximate RBF kernel PCA, an ordinary autoencoder, a denoising autoencoder and a temporal GRU autoencoder. No event labels, alarms or maintenance outcomes are read. Each pump's winner is selected using validation reconstruction error. See [REPORT.md](REPORT.md) for measured results and interpretation.

## Open and reproduce

From this directory:

```sh
.venv/bin/python run.py
.venv/bin/python visualize.py
.venv/bin/python verify.py
.venv/bin/python -m pytest -q
.venv/bin/python -m http.server 8766 --bind 127.0.0.1 --directory results
```

Open <http://127.0.0.1:8766/explorer.html>. The HTML is self-contained and also opens offline. No service or API is needed to compute or view embeddings. The existing `.venv` has the required packages; versions are in `requirements-lock.txt`.

The runner reuses model caches only when their source/input fingerprint matches. A mismatched cache raises an error instead of silently mixing experiments; archive the previous `results` directory before a source-changing retrain. `--pumps PUMP-01` can rerun one pump; aggregate exports retain every completed pump.

## Split and score

Every pump has a seven-day minute grid with 10,080 rows:

| Split | Rows | Scored rows | Use |
|---|---:|---:|---|
| Train | 6,048 | 6,048 | Fit feature selection, scalers, encoders, decoders, clusters |
| Validation | 2,016 | 1,896 | Choose neural checkpoint and encoder |
| Test | 2,016 | 1,896 | Final reconstruction evaluation |

The first 120 validation/test minutes are warmup and carry `score_eligible=False`. Split exports retain those rows so they are explicit. Inputs may carry past measured history across boundaries, as in streaming deployment. Snapshots can be up to six hours old and thermal state has longer memory; the warmup is not a claim that all histories are disjoint. No future measurements enter a row.

The score averages observed squared reconstruction errors within each feature, then averages equally across features. Each feature is standardized using its training mean and standard deviation. Missing measurements are masked, and targets are not clipped. Lower is better. `skill = 1 - model_error / training_mean_predictor_error` gives a baseline-relative interpretation. Features absent or constant during training are excluded and listed per pump. Reconstruction is scored on the engineered input representation, not on the raw one-second waveform.

This is a **within-pump chronological experiment** requested by the user. It trains on the first 60% of every pump, including those originally assigned to challenge validation/test. It is not an official unseen-pump benchmark submission. Latent axes are independent across pumps and cannot be treated as a common semantic space.

## Files

- `results/explorer.html`: all pumps, all encoders, UMAP/PCA, split/cluster/time/mode/error coloring and error timelines.
- `results/figures/per_pump_split.png`: 12 selected-model maps colored by split.
- `results/figures/per_pump_cluster.png`: 12 selected-model maps colored by cluster.
- `results/figures/PUMP-XX_models.png`: five candidate maps for that pump.
- `results/model_comparison.csv`: all 60 candidates' train, validation and test scores.
- `results/selected_models.csv`: 12 validation-selected models.
- `results/selected_embeddings.npy`: 120,960 × 32 float32; rows align to `selected_embeddings.parquet`, which includes pump, time, split, cluster, error and selected-model metadata. Coordinates from different pumps are not aligned.
- `results/PUMP-XX/{model}_embeddings.npy` and `.parquet`: all 10,080 per-minute vectors for that model.
- `results/PUMP-XX/{train,validation,test}.parquet`: explicit selected-model split datasets, including `score_eligible`.
- `results/PUMP-XX/{model}.joblib`: fitted encoder, decoder, input and latent scalers, feature schema and provenance. Load only trusted local artifacts.
- `results/PUMP-XX/{model}_metrics.json`: observed-cell counts, per-feature losses and baseline errors per split; feature order is in `features.json`.
- `results/preparation.json`, `verification.json`, `artifact_manifest.json`: input hashes, audit results and artifact hashes.
- `archive/pooled_16`: historical pooled experiment, including the removed label-informed branch. It is not used by active code or results.

## Encode later observations

Use the correct pump's saved model, keep the same sensor schema and supply past context for causal histories:

```sh
.venv/bin/python encode.py \
  --model results/PUMP-01/pca.joblib \
  --data-dir ../../BusinessTurku_SinceAI2026_FINAL_BUILDER_PUBLISH/RETROFIT_AI_PARTICIPANTS_2026/data/full \
  --asset-id PUMP-01 \
  --until 2025-08-15T06:00:00Z \
  --output /tmp/pump01_32d.parquet
```

Replace the cutoff with a timestamp inside the asset's data window. For new recordings, extend `window_end_utc` in the supplied asset metadata to cover those recordings; the builder obeys the declared window. The command reads no labels and never refits. The model rejects a different asset ID; training a new pump-specific model requires that pump's own training data.
