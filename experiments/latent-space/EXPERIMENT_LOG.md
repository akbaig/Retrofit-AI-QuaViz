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

## User-authorized label-informed extension

The user subsequently requested adding label-informed encoding to the same per-pump 32D structure. Added separate training/annotation modules, preserving all 60 baseline fitted models and vector files. Seven supervised contrastive autoencoders were fitted with training-only labels;05/07 were excluded for insufficient distinct training states and 10–12 for missing public labels. Reconstruction-based validation selection is unchanged. Label-free inference, future-label exclusion, ambiguous-overlap handling and contrastive-state sensitivity are tested. The explorer now discloses label use, offers label-informed maps and fault-annotation overlays, and disables unsupported candidates with explicit reasons. The prior unsupervised-only code/report/hashes are in `archive/per_pump_32_unsupervised`.

Final checks:15 automated tests passed;67 saved models replayed and scored consistently; all60 baseline model/vector hashes unchanged; 512-row label-informed CLI replay passed without label inputs. Independent review found no material issues. Browser checks covered label-informed selection, fault overlays, validation filtering, label-use flags and both unavailable reasons.

## Downstream fault-detection comparison

Interpreted the user's dictated “bots” as faults and tested frozen 32D vectors using identical train-only logistic and 15-neighbor classifiers. The protocol was recorded before classifier results: primary ANY_FAULT validation AP on exactly shared cases, secondary known-fault-type comparisons, validation-only thresholds, and no single-class test ranking. Fitted 192 classifiers with no warnings. The primary validation leader is label-informed AE (AP 0.7695), but only PUMP-01/09 contribute and no trained test case has both classes. Label-informed test recall is 45.1% on PUMP-01 and 100% on PUMP-09. The report explicitly rejects a confirmed overall winner; classifier and target changes produce different validation winners.

All 21 automated tests pass, including a full-run regression that mutates test vectors and labels while preserving fitted probes, thresholds and selected encoder. The artifact audit reproduced every saved prediction and metric, checked train-only scaling/indices and validation decisions, and confirmed all 67 encoder models and vector arrays match their preceding hashes (134 artifacts). Replaying float32 predictions in the original full batch avoids harmless BLAS differences seen when checking tiny row subsets. Independent review found no blocking issues; its requested orchestration regression was added. Detailed results are in `FAULT_DETECTION_REPORT.md` and `results/fault_detection.html`.

## Branch packaging: encoding-astra

Packaged this experiment from the main branch of `akbaig/Retrofit-AI-QuaViz`, separate from `astra-best`. Included current trained encoders, full per-encoder NumPy arrays, metadata, classifier models, predictions, metrics and visualizations. Omitted regenerable feature caches, duplicate vector/split tables and historical binary copies. Added a consolidated root explanation, reproduction guide, package manifest and branch verification record. Dataset discovery now uses the repository challenge directory or `ENCODING_DATA_DIR`; plotting uses the platform temporary directory. Model algorithms, fitted weights and saved vector arrays were preserved.
