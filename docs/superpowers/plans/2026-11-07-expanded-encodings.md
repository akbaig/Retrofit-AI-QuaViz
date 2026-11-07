# Expanded encoding experiment implementation plan

> For agentic workers: execute inline with the executing-plans workflow. The user authorized rerunning and improving the experiments; implementation choices are delegated.

**Goal:** Rerun the branch's six 32D encoders and classification tests using the 108 existing additional training recordings, and measure useful improvements on independent recordings.

**Architecture:** Preserve the original experiment. Add a separate experiment with one fitted encoder per engineering template family (PUMP-01 through 07), not per individual synthetic recording. Compare the original frozen encoders with retrained encoders using identical augmented-data classifiers. Add a task-trained 32D bottleneck and uncompressed classifier controls. All original templates' first 60% intervals plus additional training recordings fit models. Entire development recordings select checkpoints and thresholds; a new five-seed holdout is generated after freezing choices.

**Tech stack:** Existing Python environment, pandas, scikit-learn, PyTorch, existing causal features and simulator.

## Global constraints and rationale

- Original encoding results/code stay unchanged. New work is in experiments/encoding-expanded.
- Historical synthetic development (7 recordings) and already-examined holdout (21) are development data for this experiment, never advertised as fresh test data.
- New holdout seeds: 202611071, 202611073, 202611079, 202611083, 202611089. Seven training-template families, three days each, 35 independent recordings. Seeds may not overlap previous data.
- The original PUMP-08/09 and hidden-label PUMP-10/12 are excluded from training and selection here. This is independent-recording evaluation within known engineering families, not unseen-machine or real-world validation.
- Features use observations available at the minute end, no event/seed/time/asset identifiers. Sequence histories restart at each recording. Train-only feature scaling, 32D vectors; classifiers never see plot coordinates.
- Labels use the branch's annotation policy: detectable, observable events; ambiguous overlapping intervals excluded. No-event is a negative annotation, not proof of physical health.
- Fit rows are sampled every five minutes, excluding each recording's first 120 minutes. Validation and test metrics cover every subsequent eligible minute. Full history remains available for temporal inputs.
- Six existing encoders keep their architecture/loss, 30 maximum epochs and reconstruction-based checkpoint selection. Task-trained candidate uses the same MLP bottleneck, weighted multilabel fault classification plus a small reconstruction objective, and development classification selection.
- Primary metric: ANY_FAULT AP equally averaged across mixed-label recordings, then equally across template families. Same cases for comparisons. Secondary: per-type AP, recall, false-positive rate, F1 and reconstruction; aggregate counts retained. Single-class AP/AUROC are null.
- Thresholds selected on development F1. Final model/readout selection recorded before fresh holdout generation. All prespecified candidates evaluated once; no final-test tuning.
- Published frozen encoders with new readouts isolate representation reuse versus retraining; this is not a pure data-only ablation because checkpoint selection and data grouping differ from the original experiment.
- No automatic benchmark-win claim: these are minute classification metrics, not official event/lead-time scores.

## Approaches considered

1. Randomly split added minutes: rejected because adjacent minutes and the same incidents would leak across splits.
2. One pooled model: potentially useful for transfer, but changes the branch's independent-encoder premise substantially.
3. Per-template encoders across independent recordings: chosen to build directly on the existing experiment, repair label coverage, and retain interpretable comparisons. This was an implementation decision within the user's authorized exploration.

## Review focus

- Cross-recording sequence leakage and overlap between roles.
- Ambiguous/unobservable annotation handling and all-negative recordings.
- Sparse channels, dropped minute positions and train-only schema selection.
- Selection and threshold freeze preceding fresh test generation/evaluation.
- Fair common-case aggregation and uncertainty at recording/seed level rather than individual minutes.

## Tasks

- [x] 1. Add tested data preparation and protocol utilities. Verify split disjointness, history boundaries, target exclusions and equal-family aggregation. Reuse original features and masks; save input fingerprints and coverage.
- [x] 2. Add task-trained bottleneck with training-only supervision and serialized inference. Test supervision boundary rejection and round-trip encoding. Reuse six original fitters, fit both readouts and uncompressed controls; save models, development predictions and frozen selection.
- [x] 3. Generate fresh heldout recordings after freeze; evaluate frozen models once. Save per-recording/per-type/seed metrics, predictions and paired seed-level uncertainty. Preserve all failures and negative results.
- [x] 4. Verify artifacts, run original and new tests, independently review, and write a concise report and reproduction guide. Leave results on the requested local branch; no remote push requested this turn.

## Development-only extension, before fresh-test generation

After observing different representation rankings by family, test the same fixed 150-tree histogram classifier used by the full-feature control behind the top two expanded encoders by development logistic AP. Also test the best frozen reference encoder with this classifier to separate readout improvement from representation improvement. Select the final candidate among the original logistic winner and these two expanded histogram probes using development AP only. Freeze the added readouts, thresholds, selection and analysis code before generating any fresh test data. This is a bounded, recorded extension; no test-informed search is permitted.

## Execution and verification

- Started from fetched `encoding-astra` commit `4e18329`; original encoding code and results remain unchanged.
- Fitted 49 expanded encoders, reused 40 frozen encoders, and added full-feature controls. All retained training fits reported no convergence warnings.
- Independent review found five medium audit gaps: cache identity/integrity, fresh-test manifest validation, frozen comparison procedure, unsupported-target coverage, and reconstruction reporting. All were addressed before fresh generation. Original fitting source was preserved byte-for-byte before guard changes.
- Freeze: `2026-11-07T16:17:05.121540+00:00`. Thirty-five three-day recordings were generated afterward using five new seeds. No test-informed selection or threshold adjustment occurred.
- Selected task encoder + logistic: test AP 0.869203; kernel PCA + logistic: 0.826573. Chosen frozen reference: 0.753888. All five seed groups favor the selected model over that frozen reference; paired seed-bootstrap AP-gain interval 0.061–0.183.
- Operating-point tradeoff: selected recall 74.6%, false-positive rate 15.2%; kernel PCA recall 85.0%, false-positive rate 36.9%. Full-feature classifiers lead the separate known-fault-type comparison.
- 38 focused tests pass. Actual observation-only inference preserves a four-hour prediction prefix when a fifth hour is supplied.
- Recomputed all 11,830 test metric rows from 213 prediction files; labels/times match prepared annotations; replayed selected saved models across all seven families; every frozen hash remained unchanged.
- Remaining limits are experimental: same public simulator and known engineering families; five independent seed groups; unsupported per-family fault types; no official event-alert/lead-time evaluation. No unresolved implementation-review finding remains.
