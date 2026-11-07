# Historical design decisions

This file preserves the design at each stage. The current consolidated method is in the repository README; downstream detector evaluation was added later and uses validation labels for detector selection. The original exclusion of label-informed training was subsequently reversed.

# Unsupervised 32D representations per pump

User explicitly excludes label-informed encoders, requests an explanation of the old metric and reconstruction loss, and authorizes a new 32-dimensional experiment with train/validation/test from every pump. No additional design approval is needed.

Fit five separate encoders for each of 12 pumps: PCA, approximate RBF kernel PCA, ordinary autoencoder, denoising autoencoder, and temporal GRU autoencoder. All use identical observed numeric inputs from causal minute/physics/history/waveform features. No event file, label, alarm, or maintenance record is read. Input feature availability and mean/std scaling are fitted only on that pump's training interval.

Each pump has 10,080 completed-minute observations. Chronological partitions: first 6,048 train, next 2,016 validation, last 2,016 test. First 120 validation/test minutes are warmup and excluded from checkpoint selection and reconstruction scoring. Histories remain causal and may carry earlier measured state, as in streaming inference; long thermal state and old snapshots are not reset at split boundaries. This is within-pump evaluation, not the original asset-held-out challenge benchmark. Original benchmark split is retained as metadata only.

Every model reconstructs the same current-minute input feature vector. Primary metric: mean across features of observed-cell squared reconstruction error in training-standardized units (no target clipping). Also report per-feature MSE, the error of predicting the training mean, and skill relative to that baseline. Exclude features absent or constant in training and disclose them. These exclusions limit claims about sensors/conditions not seen during training. All input values remain unbounded, with missing values filled by the training mean and separately masked.

PCA uses its analytic decoder; kernel PCA uses a train-only ridge decoder; neural models use their learned decoders. Neural epochs/checkpoint selection uses validation reconstruction loss. Select each pump's model using validation only, freeze, then calculate final test losses. Test results do not choose models. Reconstruction quality is not fault detection performance.

KMeans and UMAP/PCA projections fit training embeddings only. Plot each pump separately with split, cluster, time, operating-mode, and reconstruction-error coloring. Independently fitted coordinates are not aligned across pumps. Export all minute vectors, metadata, split files, models, metrics, source hashes and an offline explorer. Archive the previous pooled 16D experiment including its rejected supervised branch; remove that branch from active code/results/UI.

## Authorized extension: label-informed comparison

The user now explicitly requests adding the earlier label-informed approach to this per-pump 32D structure. Keep the five existing candidates and their fitted artifacts. Add a supervised contrastive autoencoder per pump with public labels (01–09); pumps 10–12 have no public labels and show an explicit unavailable state. The encoder consumes only the existing sensor features and masks. Labels shape its training objective and are never inputs at inference.

Use the same MLP capacity as the ordinary AE, 32D bottleneck, preprocessing, chronological split, reconstruction targets and validation checkpoint criterion. Optimize reconstruction on all training rows plus a fixed 0.1-weighted supervised contrastive objective on state-balanced, augmented training examples. Positive pairs share the exact detectable multilabel state. Exclude pre-detectable/unobservable overlapping events from supervised pairs; no labeled event is not a guarantee of health. Temperature 0.2, two views with 10% masking and 0.03 noise, seed42, maximum50epochs. All choices are implementation defaults, not additional user requirements.

Read only the two public event CSVs. Build supervision and its vocabulary strictly from each pump's first60% interval. Validation/test labels may be shown as a visualization overlay but never select checkpoints or winners. Selection remains validation reconstruction across available models, then test scoring. Store separate label provenance and availability, and keep the original label-free feature cache unchanged. Save full 32D embeddings, split files, train-only clusters/projections, updated comparisons and offline explorer. Plot controls must handle five vs six available models without fabricating label-informed results on unlabeled pumps.

Considered using label vectors as encoder inputs or pooling training across pumps; rejected because both change the inference interface or per-pump experiment requested. The chosen approach adds labels only to the training loss and retains a decoder for the existing reconstruction comparison.
