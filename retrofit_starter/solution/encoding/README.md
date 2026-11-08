# Encoding study (deep-learning track)

Before training a deep fault detector we asked whether compressing each minute of pump data into a short vector keeps the information needed for diagnosis. This folder holds that study. The conclusion fed the final solution: **use supervised machine learning on physical features, not a detector on the encodings.**

Needs Python packages beyond the challenge baseline: `torch`, `umap-learn`, `matplotlib`, `plotly` (see `latent-space/requirements.txt`).

## What we compared

Six encoders, each producing **32 numbers per minute**, fitted per pump:

| Encoder | Fault labels train the encoder? |
|---|---|
| PCA | No |
| Approximate RBF kernel PCA | No |
| Autoencoder | No |
| Denoising autoencoder | No |
| Temporal GRU autoencoder | No |
| Label-informed autoencoder | Yes (training period only) |

Features are causal (past and currently available measurements only). Each pump is split 60 / 20 / 20 in time. This is a within-pump experiment; scores are **not** official challenge composites.

## Findings

1. **Reconstruction.** PCA keeps the most input information on 11 of 12 pumps.
2. **Fault present (original study).** Label-informed AE + logistic regression leads validation average precision at **0.770** (supported only by PUMP-01 and PUMP-09).
3. **Fault present (expanded study).** Retraining with 108 extra simulated recordings, a task encoder that is trained to retain fault information reaches **0.869** fresh-test AP against **0.827** for kernel PCA (`encoding-expanded/REPORT.md`).
4. **Fault naming.** Classifiers on the full feature set name individual fault types better than classifiers on the 32D encodings.

**Decision for the challenge:** do not build the submission as a deep model on the encodings. Figures for the story live in `latent-space/results/figures/` and `encoding-expanded/results/`.

## Layout

- `latent-space/` — original per-pump 32D study (code, summary CSVs, figures). Large embedding and model binaries were removed; re-run `run.py` to regenerate them.
- `encoding-expanded/` — follow-up with more training recordings and a fault-trained encoder.
- Detailed write-ups: `latent-space/REPORT.md`, `latent-space/FAULT_DETECTION_REPORT.md`, `encoding-expanded/REPORT.md`.

## Reproduce (optional)

```sh
cd latent-space
python -m pip install -r requirements-lock.txt
python run.py            # fit encoders
python visualize.py      # 2D maps
# then fault-detection probe as in FAULT_DETECTION_REPORT.md
```
