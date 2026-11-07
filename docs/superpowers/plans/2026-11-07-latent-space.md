# Pump Latent Space Implementation Plan

> **For agentic workers:** Execute inline using superpowers:executing-plans. No implementer delegation is needed; request one independent final review.

**Goal:** Execute a reproducible comparison of five 16-dimensional pump representations and deliver reusable embeddings and cluster maps.

**Architecture:** Completed minute observations and causal engineered features feed train-only preprocessing and unsupervised encoders. Offline event metadata supports evaluation; separate fitted projections support visualization.

**Tech Stack:** Python, pandas, PyArrow, scikit-learn, PyTorch, UMAP, matplotlib, Plotly.

**Spec:** `experiments/latent-space/DESIGN.md`

## Global Constraints

- Fit on training assets only; labels are offline evaluation metadata.
- Never infer test labels or select using test data.
- Preserve original source files and existing experiments.
- Causal completed windows, bounded sparse holds, explicit gaps and simultaneous labels.

## Review Focus

- Missing hardware and sparse samples must not become normal zero measurements.
- Event onset, detectable time, end boundaries and overlaps must remain distinct.
- Future observations must not alter previous feature vectors.
- Sequence histories must not cross asset boundaries.
- Encoder reload must reproduce saved vectors; reported metrics must match artifacts.

### Task 1: Audit and preparation

- [x] Write failing tests for event labels, train-only scaling, sequence boundaries, and reuse existing causal-feature tests.
- [x] Implement `data.py`: audit tables, label metadata, preprocessing, chronological window indices.
- [x] Run tests; inspect actual-data counts and schemas. Save dataset audit and feature provenance.

### Task 2: Representation comparison

- [x] Implement `models.py` and `run.py` for five fixed candidates, 16-dimensional export, and fixed neighborhood/KMeans diagnostics.
- [x] Train without labels using training assets; retain training curves and settings.
- [x] Evaluate on validation; select by multilabel neighborhood AP and report per-asset limitations.
- [x] Freeze selection, then transform unlabeled test assets. Verify finite vectors and reload equivalence.

### Task 3: Visualization and delivery

- [x] Implement `visualize.py`: fitted train-only UMAP and PCA, labels/clusters/asset/tier views, interactive explorer, static comparison.
- [x] Inspect rendered figures and browser behavior. Write `README.md`, `REPORT.md`, locked dependencies and artifact manifest.
- [x] Run the experiment test suite and independent final review; resolve important findings.

Ruling: This folder is not a Git repository. A separate experiment directory and frozen vendor copies provide isolation; no commits or source-data edits are needed. Model choices were delegated by the user's explicit request, so execution proceeds without another permission round.
