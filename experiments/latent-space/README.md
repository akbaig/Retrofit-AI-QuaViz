# Use and reproduce the pump-encoding experiment

Read the [repository README](../../README.md) for the plain-language method and conclusions. This guide covers artifacts and commands. [REPORT.md](REPORT.md) covers reconstruction; [FAULT_DETECTION_REPORT.md](FAULT_DETECTION_REPORT.md) covers downstream classifiers. They use separate selection criteria.

## Set up

From the repository root, use Python 3.12:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r experiments/latent-space/requirements-lock.txt
cd experiments/latent-space
```

The default dataset is `../../RETROFIT_AI_PARTICIPANTS_2026/data/full`. For another copy, set `ENCODING_DATA_DIR` to its absolute `data/full` directory before running Python. This path configuration is the only change to training/data implementation for branch packaging. Saved models and vectors are unchanged. Original source is retained in [provenance/data_original.py.txt](provenance/data_original.py.txt).

## Load a saved 32D representation

From this directory:

```python
import numpy as np
import pandas as pd

pump = "PUMP-01"
encoder = "label_informed"  # or pca, kernel_pca, autoencoder, denoising_ae, temporal_ae
folder = f"results/{pump}"
z = np.load(f"{folder}/{encoder}_embeddings.npy")
metadata = pd.read_parquet(f"{folder}/metadata.parquet")
labels = pd.read_parquet(f"{folder}/label_metadata.parquet")

assert z.shape == (10080, 32)
assert len(z) == len(metadata) == len(labels)
train = z[metadata["split"].eq("train").to_numpy()]
validation = z[(metadata["split"].eq("validation") & metadata["score_eligible"]).to_numpy()]
test = z[(metadata["split"].eq("test") & metadata["score_eligible"]).to_numpy()]
```

Row `i` aligns across vectors and metadata. Metadata contains pump, timestamps, split and `score_eligible`; label metadata contains `fault_label` and `label_eligible`. Detector scoring also excludes ineligible labels. The vectors contain no label columns. Different pumps' latent axes are not aligned.

Label-informed files exist for PUMP-01/02/03/04/06/08/09 only; `results/label_availability.csv` explains the unavailable cases. `selected_models.csv` records **reconstruction** winners. `fault_detection/selection.json` records separate **detector** validation decisions.

## Apply a saved classifier

Continue the example:

```python
import joblib

detector = joblib.load(
    f"results/fault_detection/{pump}/{encoder}/ANY_FAULT__logistic.joblib"
)
score = detector["probe"].predict_proba(z)[:, 1]
predicted_fault = score >= detector["threshold"]
```

The pipeline includes train-fitted standardization and logistic regression. Its cutoff comes from validation. No labels are passed to `predict_proba`. The scores have not been demonstrated to be calibrated probabilities. Load joblib files only from a source you trust.

## Encode sensor observations

```sh
mkdir -p output
python encode.py \
  --model results/PUMP-01/label_informed.joblib \
  --data-dir ../../RETROFIT_AI_PARTICIPANTS_2026/data/full \
  --asset-id PUMP-01 \
  --until 2025-08-15T06:00:00Z \
  --output output/pump01_32d.parquet
```

This uses saved preprocessing and an encoder, with no refitting or event-label input. The cutoff admits only completed measurements available by that time. Pass the resulting `z00` through `z31` columns to the matching classifier. Supply earlier history and the same sensor schema. Another pump needs its own model. For a longer recording, extend that asset's declared `window_end_utc` to cover it.

## Inspect and verify the saved work

```sh
python -m pytest -q
python verify_fault_detection.py
python fault_detection_report.py
python -m http.server 8766 --bind 127.0.0.1 --directory results
```

Open <http://127.0.0.1:8766/explorer.html> or <http://127.0.0.1:8766/fault_detection.html>. The explorer is self-contained; the detector page also needs its adjacent chart.

`verify_fault_detection.py` checks classifier training indices/scaling, replays every prediction, recomputes metrics/selection, and checks all 67 encoder model/vector pairs against recorded hashes. It needs shipped artifacts, not raw sensor data. [BRANCH_VERIFICATION.md](BRANCH_VERIFICATION.md) records checks run on this branch. Logs under `results/` record the earlier full workspace run.

To refit only classifiers from the shipped vectors:

```sh
python fault_detection.py
python verify_fault_detection.py
python fault_detection_report.py
```

This overwrites detector outputs under `results/fault_detection/`; it does not refit encoders. Training labels fit readouts, validation AP compares encodings and validation F1 chooses cutoffs. Test predictions are calculated after the decision is saved. Logistic settings are fixed (`C=1`, balanced classes, `max_iter=2000`, seed 42); the secondary classifier uses 15 neighbors with uniform voting.

## Retrain encoders from scratch

Preserve the delivered results first:

```sh
mv results results-recorded
python run.py
python visualize.py
python summarize.py
python fault_detection.py
python fault_detection_report.py
python -m pytest -q
```

Use an unused backup name if `results-recorded` exists. The runner fits all pumps and supported label-informed candidates, and generates omitted caches/vector tables/split tables. Fingerprints reject stale caches, so do not train on top of historical results after relocating or changing code. Fresh training is not promised to be byte-identical across versions or hardware.

`verify.py` and `verify_fault_detection.py` assert original frozen hashes: they are **historical artifact audits**, not general verifiers for arbitrarily retrained models. `verify.py` additionally requires the original full feature cache and duplicate exports omitted here; its successful original log is included. Do not use historical hashes to certify a new run.

## Files and provenance

| Path | Purpose |
|---|---|
| `data.py`, `vendor/` | Label-free features, scaling and time split |
| `models.py`, `label_informed.py` | Five unsupervised encoders and supervised contrastive AE |
| `run.py`, `extend_label_informed.py` | Encoder training and exports |
| `evaluate.py` | Reconstruction metric and selection |
| `fault_detection.py` | Frozen-encoder classifier comparison |
| `encode.py` | Label-free inference |
| `results/PUMP-XX/{model}.joblib` | Encoder, decoder, preprocessing and provenance |
| `results/PUMP-XX/{model}_embeddings.npy` | Full 10,080 × 32 float32 vectors |
| `results/PUMP-XX/metadata.parquet`, `label_metadata.parquet` | Aligned metadata and separate labels |
| `results/fault_detection/` | 192 classifiers, predictions, metrics, coverage and decisions |
| `results/explorer.html`, `figures/` | Interactive and static visualizations |
| `archive/` | Historical reports, code and hashes; duplicate historical binaries omitted |
| `PACKAGE_MANIFEST.json` | Hashes of files actually delivered on this branch |
| `provenance/omitted_generated_files.json` | Exact omitted cache/duplicate-export list |

Historical manifests describe the complete original workspace, including some omitted files and original source hashes. `PACKAGE_MANIFEST.json` describes this delivery. Current models and full vectors are included; historical archive model binaries are not. Training/model logic is unchanged; dataset lookup was made portable, plot generators now use the platform temporary directory, and generated-report wording was clarified. Documentation now distinguishes encoder selection from downstream classifier selection.

This is one seed and seven synthetic days per pump, with correlated minutes and events spanning split boundaries. It does not measure official alert lead time, unseen-pump performance or real-plant performance.
