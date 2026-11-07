> Historical snapshot; superseded by the current 32D experiment. See [ARCHIVE_NOTICE.md](ARCHIVE_NOTICE.md) for omitted generated artifacts.

# Pump latent-space experiment

Executed data audit, six representation families, eight fitted encoders including seed repeats, clustering, and visualization. Everything is local; original data and previous experiments are unchanged.

Start with **[the interactive explorer](results/explorer.html)** and **[the findings](REPORT.md)**. The explorer opens directly in a browser and includes Plotly locally, so it works offline.

## Delivered artifacts

| File | Purpose |
|---|---|
| `results/explorer.html` | Switch model, projection, split, and coloring; inspect individual observations |
| `results/figures/model_comparison.png` | Six-model validation comparison |
| `results/figures/selected_diagnostics.png` | Selected representation colored by fault class, pump, tier, and cluster |
| `results/figures/sensor_coverage.png` | Instrument availability and reporting frequency |
| `results/selected_embeddings.npy` | Recommended **120,960 × 16 float32** matrix |
| `results/metadata.parquet` | Row-aligned pump/time/split/labels and uncertainty flags |
| `results/supervised_contrastive_embeddings.parquet` | Recommended vectors with pump/time/split and cluster in one table |
| `results/raw_pca_embeddings.parquet` | Best label-free representation under the chosen diagnostic |
| `results/models/supervised_contrastive.joblib` | Selected encoder, preprocessing, latent scaler, feature order, weights, and provenance |
| `results/model_comparison.csv` | All eight runs, including alternate seeds |
| `results/*_metrics.json` | Per-fault and per-pump diagnostics |
| `results/data_audit.json`, `assets_audit.csv`, `events_audit.csv` | Data structure and source hashes |
| `results/verification.json` | Full export, serialization, source-integrity and split checks |

The selected representation uses **training-event labels in its contrastive loss**. Its inputs contain only observed telemetry-derived features and public static/context data. The five other model families are unsupervised. Test vectors have no inferred ground-truth labels.

## Use the vectors

From this directory:

```python
import numpy as np
import pandas as pd

z = np.load('results/selected_embeddings.npy')
meta = pd.read_parquet('results/metadata.parquet')
assert len(z) == len(meta)

train = meta['split'].eq('train').to_numpy()
validation = meta['split'].eq('validation').to_numpy()
test = meta['split'].eq('test').to_numpy()

z_train, z_validation, z_test = z[train], z[validation], z[test]
```

Vectors are standardized using training latent means and standard deviations. Keep the 16D vectors for downstream work; do not train classifiers on the 2D UMAP coordinates. Preserve the asset split. The supplied event labels are intervals and can overlap; `fault_types` is pipe-separated and `label` is only a display category. The 13-column multilabel target matrix for development rows is in `labels_development.npy`, with column order in `fault_names.json`.

For the conservative evaluation population used in this experiment:

```python
from data import evaluation_eligible
eligible = evaluation_eligible(meta)
```

This excludes any row with a pre-detectable or unobservable event, even when another event is visible. It is intentionally conservative and loses some useful partially labeled observations; see the report.

## Reproduce

Python 3.12 was used. A dedicated `.venv` already exists in this directory.

```bash
cd /Users/khizar/Projects/SinceAI/experiments/latent-space
.venv/bin/python -m pytest -q
.venv/bin/python run.py
.venv/bin/python extend_contrastive.py
.venv/bin/python visualize.py
.venv/bin/python verify.py
```

On another machine, create a Python 3.12 environment and install `requirements.txt`; `requirements-lock.txt` records the complete executed environment. Models run on CPU with four PyTorch threads. No API key, GPU, paid service, or pretrained-weight download is required. The full source dataset is located by the original project-relative path in `data.py`.

`run.py` evaluates the five unsupervised candidates and sets their provisional selection. **Run `extend_contrastive.py` after it** to restore the full comparison and recommended label-informed selection. Seeds 7 and 123 are sensitivity runs; seed42 is the prespecified exported default. Source/config fingerprints guard cached preparation and fitted encoders. If a fingerprint fails, rename the existing results directory to preserve it and rerun into a fresh `results/` directory. A Python/library upgrade also warrants a fresh run.

## Encode another compatible dataset

The inference command reads no event label files and does not refit anything:

```bash
.venv/bin/python encode.py \
  --model results/models/supervised_contrastive.joblib \
  --data-dir /path/to/public-format-dataset \
  --output /path/to/new_embeddings.parquet
```

The directory needs `assets.csv` and `features_1min.parquet`; `operating_context.csv` and `vibration_snapshots.parquet` are used when present. Optional `--asset-id PUMP-08` and `--until 2025-09-09T00:00:00Z` restrict a causal replay. Each output is available only after its minute has finished. Meaningful transfer assumes compatible channels, units, physical metadata and sampling conventions.

## Implementation

- `data.py`: audit, labels, masks, preprocessing, cache fingerprints, sequence boundaries.
- `vendor/`: frozen copies of the project's existing feature and physics functions; original hashes recorded in the audit.
- `models.py`: PCA, approximate RBF kernel PCA, denoising MLP, causal GRU sequence autoencoder, supervised contrastive encoder.
- `evaluate.py`: fixed 15-neighbor diagnostic and latent-space KMeans.
- `run.py`, `extend_contrastive.py`: reproducible experiments and exports.
- `visualize.py`: UMAP/PCA maps, static figures and offline explorer.
- `encode.py`, `verify.py`, `tests/`: reuse and verification.

`archive/provisional_v1/` contains superseded artifacts retained after an independent review found ambiguity in overlapping event eligibility. Those numbers are not the final comparison.
