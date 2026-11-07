# Fault detection comparison on frozen per-pump embeddings

User asks to test which encoder detects faults best (interpreting dictated “bots” from context). Execute the existing test/review workflow inline. This is an extension of the authorized experiment; keep all fitted encoders and reconstruction results unchanged.

**Protocol fixed before classifier results:** For each available pump/model, fit the same train-only standardized logistic regression (C=1, balanced classes, max_iter2000), and a fixed15-neighbor probe as a sensitivity check. Predict ANY_FAULT (detectable annotated fault vs no labeled event); additionally assess individual fault types learned from training vocabulary. Exclude ambiguous/unlabeled rows and heldout warmup. Do not fabricate normal periods. Fit only when training has both classes. Primary comparison is mean validation AP across exactly the same evaluable pumps for every encoder, with logistic regression; individual-fault AP and nearest-neighbor results are secondary. Select thresholds using validation F1 only. Freeze candidate/model selection before calculating test metrics. Mark AP/AUROC unavailable for ranking for single-class heldout periods; show recall/FPR only where their denominators exist. Primary binary validation comparison is possible on PUMP-01/09; their test periods have no negatives, so no test-confirmed detector ranking is possible. This limitation must stay visible.

**Scope:** Evaluate existing frozen32D representations, do not refit encoders or alter chronological splits to obtain flattering scores. Validation previously selected neural checkpoints by reconstruction, so this is an exploratory diagnostic. Adjacent minutes are correlated. Test intervals already seen in previous research; no claim of fresh final benchmark. At test time some faults are novel to training; report coverage explicitly. No official event-alert score or real-world performance claim.

- [x] Write failing tests: single-class AP is unavailable; ambiguity is excluded; train-only probes resist test mutation; thresholds use validation; rankings use common cases.
- [x] Implement fault_detection.py for target coverage, two fixed probes, frozen selection, predictions/models/metrics exports.
- [x] Run all available classifiers; verify test-case limitations and baseline encoder hashes unchanged.
- [x] Create concise report and comparison page linked from explorer, inspect results and browser.
- [x] Independent review, full test suite, reproducible audit; answer simply with strongest supported conclusion.
