# Experiment log — 2026-11-07

The user rejected label-informed encoders and requested independent 32-dimensional representations, train/validation/test data, and visualizations for every pump.

- Archived the previous pooled 16D experiment under `archive/pooled_16`; removed its supervised branch from active code and results.
- Replaced event-aware preparation/evaluation with an explicit four-file, label-free input whitelist.
- Wrote regression tests before implementing chronological 60/20/20 splitting, 120-minute held-out warmup, observed-only training scaling, unclipped targets, feature-balanced reconstruction and validation-only selection.
- Fitted PCA, approximate RBF kernel PCA + ridge reconstruction, MLP AE, denoising AE and GRU temporal AE independently for each pump. All use 32 latent coordinates and identical feature schemas within each pump.
- Added train-fitted K-means, train-fitted UMAP and PCA projections, explicit split datasets, reconstructable model bundles and per-feature diagnostics.
- Independent read-only review found a partial-rerun aggregation bug. Added a failing regression test, then changed aggregation to include all completed pump directories. The core fits and chronological protocol were judged free of event-label or future-to-training leakage in the reviewed paths.
- Review noted that model training objectives differ: PCA fits the imputed/mask-augmented input, while neural decoders optimize masked observed reconstruction. Evaluation targets and scoring are identical. Results describe these implementations, not universal model-family optimality.

Final scores and remaining scientific limitations are in `REPORT.md`. Reproducible verification output is in `results/verification.log` and `results/tests.log`.
