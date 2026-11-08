# More-data encoding experiment

This extends the `encoding-astra` experiments without replacing their saved results. It tests the same six 32-number encoders on additional independently simulated recordings, adds a fault-trained encoder, and includes full-feature controls.

**Read [REPORT.md](REPORT.md) for the measured result.** The original branch's 0.770 validation AP came from two different pumps/time windows. It is not directly comparable with the new experiment. These are minute classification tests, not official event-alert or lead-time scores.

[Download the selected models and inference code](selected_model_bundle.zip). This compact bundle contains seven models, the comparison reports, and code to apply them without event labels. Unused readouts were removed and the selected predictions were verified unchanged. Full comparison caches and all candidate models remain in the local results directory.

The new encoder plus logistic regression led the fresh binary-fault test: **AP 0.869**, versus **0.827** for retrained kernel PCA, the strongest of the original six designs on this test. At their development-selected cutoffs, false-positive rate fell from **36.9% to 15.2%**, while recall fell from **85.0% to 74.6%**. That is a useful improvement in ranking and false alarms, with a recall tradeoff.

It is not the best model for every task. For identifying individual known fault types, full-feature logistic regression scored **0.632 AP**, versus **0.573** for the new encoder. PCA preserved input information better. Additional data improved five of the six original encoders' matched binary AP comparisons; the label-informed encoder declined slightly on its five comparable families.

## What the new model does

1. Read a completed minute of sensor measurements and its earlier history. Build the same physical, vibration, missing-data and change features as the original branch.
2. Standardize features using training data only. Keep missing readings separate from observed values.
3. Pass the features through layers of 128, 64 and **32** numbers.
4. During training, temporary classification heads ask whether a fault exists and which known fault types exist. Their errors teach the encoder what information to keep. A small reconstruction objective also asks it to retain general input information.
5. Save the 32-number encoder. Fit the same logistic regression and 15-neighbor classifiers used for every other encoder. The temporary training heads are not the reported classifiers.
6. Choose the encoder and decision cutoffs on development recordings. Freeze all choices, then evaluate newly generated recordings.

The task encoder uses classification loss plus 0.05 times reconstruction loss, mild training noise/masking, and a 32D bottleneck. Neural inputs for this candidate are bounded to ±10 training-standard-deviation units. It has no access to event labels at inference. Its checkpoint uses development ANY_FAULT AP; the six prior encoders keep their reconstruction-based checkpoint selection. This is an intentional change in learning objective, not just a larger autoencoder.

## Data split

| Data | Use |
|---|---|
| Original PUMP-01–07, first 60% of each | Training, preserving the branch's original time boundary |
| 108 existing additional synthetic recordings | Training |
| Earlier synthetic development (7) and previously examined holdout (21) | Development: checkpoint, cutoff and model selection |
| 35 new recordings from five unused seeds, generated after freeze | Final evaluation |

Each model belongs to one **engineering template family**. All independent recordings generated from that template share its encoder; different families still have separate coordinate systems. Modified controls and sensor tiers within a family make this harder than replaying the original pump. It is not a test on completely unseen equipment families. PUMP-08–12 are excluded from fitting and selection here.

The first 120 minutes of each recording are warmup. Fit rows are taken every five minutes; classification also excludes ambiguous annotations. Complete earlier histories remain available to temporal models. Development/test scoring uses every subsequent eligible minute, without shuffling adjacent observations across roles. A temporal sequence never crosses into another recording.

Labels follow the branch policy: observable events become positive at detectability; predetectable or unobservable overlaps are excluded. No recorded event is a negative annotation, not proof of physical health. Unsupported fault types are listed in `*_target_coverage.csv`; the binary ANY_FAULT label still includes them.

## Comparisons

- **Frozen:** apply the published per-pump encoder to the corresponding generated family, then train a new classifier on the expanded training data.
- **Expanded:** refit that encoder on the expanded training data, then train the same classifier on the same rows.
- **Task encoder:** learn which information is useful for fault classification, as described above.
- **Full features:** logistic regression and gradient-boosted trees without compression, to reveal the cost of squeezing inputs into 32 numbers.

A bounded follow-up also fits the same fixed boosted-tree classifier behind the two encoders with the best development logistic AP and behind the best frozen reference. It selects the final compressed model from the original logistic winner and the two expanded tree readouts, still using development data only. `head_extension_plan.json` and `head_selection.json` record those choices. The base six-encoder/logistic comparison remains available separately.

The frozen-versus-expanded comparison controls the classifier data. It does not isolate data volume alone: encoder checkpoint selection, recording grouping and training-supported feature coverage also change from the published within-pump experiment. Newly observed channels can be retained by the expanded preprocessor. The six expanded encoders and the task encoder share the same expanded feature schema within each family. All classifiers receive 32D vectors, except the explicitly named full-feature controls. No classifier receives 2D visualization coordinates.

Primary AP averages mixed-label recordings within family, then averages families equally. Single-class AP is unavailable, not automatically perfect. Those recordings still contribute false-positive or recall counts. Reported recall/FPR pool minute counts at development-selected cutoffs. Per-type AP is a secondary known-target comparison. A seed-level paired bootstrap describes uncertainty with only five independent simulator seeds.

## Reproduce

Use Python 3.12 and the pinned packages in `../latent-space/requirements.txt`. Generator dependencies are in `../../retrofit_starter/requirements.txt`; paths below are from the repository root. The existing local ML environment is `/Users/khizar/Projects/SinceAI/experiments/latent-space/.venv/bin/python`; generator runs used `/private/tmp/quaviz-audit/venv/bin/python`. Both use NumPy 2.5.4 and pandas 3.1.1. Substitute your own environment paths.

```sh
# In a separate clone, preserve the checked-in result before rerunning.
mv experiments/encoding-expanded/results experiments/encoding-expanded/results-recorded

# Point to the existing generated recordings used by Astra Best.
python experiments/encoding-expanded/generate_fresh.py --reserve-only
python experiments/encoding-expanded/run_expanded.py prepare \
  --synthetic-root /path/to/quaviz-learned/synthetic_data
python experiments/encoding-expanded/run_expanded.py train
python experiments/encoding-expanded/analysis/extend_readout.py
python experiments/encoding-expanded/analysis/report.py --select
python experiments/encoding-expanded/analysis/diagnostics.py --split development
python experiments/encoding-expanded/run_expanded.py freeze

# Generate test observations only after freeze.
python experiments/encoding-expanded/generate_fresh.py
python experiments/encoding-expanded/run_expanded.py test \
  --test-data experiments/encoding-expanded/synthetic_data/holdout
python experiments/encoding-expanded/analysis/diagnostics.py --split test
python experiments/encoding-expanded/analysis/report.py
python experiments/encoding-expanded/analysis/verify_results.py
python -m pytest -q experiments/latent-space/tests experiments/encoding-expanded/tests
```

For an independent rerun, use a fresh checkout/output directory. Never overwrite a frozen run. Source/data/settings changes invalidate caches. The original augmentation generator is included unchanged in `generator/synthetic.py`; its role and seed manifests describe how to regenerate the four training sets. The 108 recordings are 21 `train` at 3 days, 30 `train_balanced` at 2 days, 36 `train_crossed` at 2 days and 21 `train_sensor_crossed` at 1.5 days. Recreating historical generator outputs from a later source revision may not be byte-identical; source hashes identify the data actually evaluated.

## Apply the saved model

```sh
python experiments/encoding-expanded/infer.py \
  --model experiments/encoding-expanded/results/PUMP-01/expanded_task_encoder.joblib \
  --data-dir /path/to/observed-data --asset-id YOUR_RECORDING \
  --readout logistic --output /tmp/fault-scores.parquet
```

Use the model for the recording's `template_asset_id`. Supply public assets, minute features, context and vibration tables. Optional `--until` limits inference to completed observations at that timestamp. No events file is read. Output is per-minute uncalibrated scores and decisions, **not** the challenge's seven-column alert CSV. An incident policy and official scorer evaluation would be needed before replacing Astra Best's event detector.

## Files and audit trail

- `results/comparison.csv`, `summary.json`: readable comparison and seed uncertainty.
- `results/development_metrics.csv`, `test_metrics.csv`: every family/recording/target/readout result, including counts and threshold origins.
- `results/*_target_coverage.csv`: missing/unsupported supervised targets.
- `results/*_reconstruction.csv`: same-feature/same-scaling reconstruction comparison for the seven expanded encoders. Task-model inputs are clipped, but evaluation targets are not clipped.
- `results/PUMP-*/`: saved models, readouts, thresholds, development/test predictions and training histories.
- `results/frozen.json`: fitted model, code, comparison procedure and prepared-data hashes fixed before fresh test generation.
- `results/data_seal.json`: original data verification and prepared-output hashes.
- `provenance/training_source/`: exact initial fitting source. Review added cache/freeze/data-integrity guards afterward; learned parameters and training objectives were unchanged.
- `fresh_test_plan.json`: reserved unused seeds, before generation.

The original simulator's vibration snapshots do not reproduce all channel sensor-fault corruption. These remain synthetic-data results from one generator. Large regenerated data, feature caches, model pickles and prediction Parquets remain in this local checkout and are ignored by Git. CSV/JSON results, code and documentation are retained for review. Frozen receipts contain absolute paths for this audited local run; a fresh reproduction creates receipts for its own paths.
