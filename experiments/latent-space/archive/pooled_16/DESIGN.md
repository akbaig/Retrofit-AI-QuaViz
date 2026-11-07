# Pump latent-space experiment

The user requests an executed data audit, model comparison, reusable vectors, and cluster visualizations before developing classifiers. Model choice is delegated. This is an isolated research experiment; existing project code and source data remain unchanged.

Use the full-resolution package's completed minute aggregates, public ratings/context, and backward-joined vibration spectra. Freeze copies of the existing causal feature functions and record their hashes. Retain a complete minute grid so gateway outages are represented. Each vector represents the state at the end of a minute; temporal inputs use only previous completed minutes.

Compare raw-observation PCA, physically normalized temporal-feature PCA, approximate RBF kernel PCA, a masked denoising autoencoder, and a causal sequence autoencoder. All representations have 16 dimensions. Fit preprocessing and encoders only on PUMP-01–07 without fault labels. Use labels only for offline neighborhood probes and plots. Fit KMeans in the latent space, not in the 2D projection. Select the representation using validation multilabel average precision of a fixed 15-neighbor probe, with per-pump scores, clustering diagnostics and confounding checks. This diagnostic is not a deployed classifier or the official alert score.

Use onset, detectable time, observability and end separately; preserve simultaneous events. Do not call every unlabelled minute healthy. Test pumps remain unlabeled and never influence fitting or selection. Encode test only after selection; keep test out of exploratory plots by default. Provide 2D UMAP maps and PCA projections; UMAP islands are not proof of real fault classes.

Save fitted preprocessors/encoders, full vectors, row metadata, model metrics, input/source hashes, training logs, static figures, and a local interactive HTML explorer. Assess the best of the models actually run, not a universal optimum. The dataset is synthetic, with only seven training assets and two validation assets.

## Recorded extension after the five initial runs

The five unsupervised candidates gave validation neighborhood macro AP 0.256–0.301 after the overlap-eligibility correction, with strong asset clustering (NMI 0.62–0.73). Add a clearly separate label-informed supervised-contrastive encoder. It uses training event labels as a learning objective, never as inference inputs; validation/test remain excluded from fitting. Positives share the exact multilabel state, with class-balanced sampling and masked-view augmentation. Retain reconstruction regularization. This extension is motivated by validation development results, so validation is explicitly model-selection data, not a blind final estimate. Preserve the original unsupervised comparison and selection. Earlier provisional metrics and artifacts are archived separately and are not final results.
