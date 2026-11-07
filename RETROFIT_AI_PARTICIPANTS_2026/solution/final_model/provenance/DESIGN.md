# Learned fault diagnosis experiment

User authorizes building an original ML architecture and measuring a substantial improvement over published B1 (78.0125). Target: roughly ten or more composite points, with correct diagnoses and controlled alert burden. This is a target to test, not an assumed result.

Architecture: public ratings and operating context plus causal minute and waveform observations -> physical ratios and trailing temporal features -> healthy-behavior regression residuals -> supervised multi-label tree ensembles -> persistent incident states -> official alert CSV. The model learns fault identity from labels. Alerts and recommendations explain model evidence, without accessing labels during inference.

Why this design: the previous unsupervised median/MAD detector changed the trigger but left diagnosis as fragile rules. We have labels and an authorized simulator. Learn from varied synthetic faults, use physically scaled features to transfer across pumps, and use observed waveform spectra to distinguish mechanical signatures. A VAE would estimate novelty but would still require a diagnosis stage; it is not the first architecture to test here.

Implementation choices are the assistant's. The user explicitly asks to proceed with any suitable architecture, so no design approval pause is required.

## Data and evaluation

- Original training pumps01–07 and freshly randomized simulations based on their public physical specifications, not their fixed event schedules.
- Public validation08–09 is development data, as expressly permitted by BENCHMARK_RULES.md. It has already been discussed and evaluated; do not claim it is blind. Changes selected from it are recorded.
- Fresh synthetic development scenarios support model/threshold selection. A separately seeded, perturbed scenario holdout is generated/evaluated only after the final model and alert policy freeze. It remains synthetic and does not prove field generalization.
- Test telemetry/labels/predictions10–12 are not used for training or selection. No source/dataset edits.
- No asset identifiers, timestamps, seeds, scenario identifiers, labels, deadlines, work-order conclusions or simulator latent state as model inputs.
- Inference uses completed windows and backward joins; sparse samples retain age/freshness. Independent clock covers outages. Waveform availability is capture time plus sample duration, never future-filled.
- Original B1/B3 predictions and unchanged scorer are retained as published comparators. A fitted B1 comparator is also evaluated on fresh synthetic holdout using thresholds selected on development data.
- Report event matches, first-fault-type correctness, class correctness, false/excess alerts, deadline denominators, first-alert evidence, and all selected settings. Do not market wrong sensor diagnoses as early machine detection.

## Files and execution

1. `features.py`, `tests/test_causality.py`: complete grid, context/snapshot joins, observation ages, physical features, past-only histories. Check prefix invariance and missing/sparse inputs first.
2. `synthetic.py` (independent agent): new training/development scenarios, perturbations, observed telemetry, labels and snapshots; provenance manifest. Keep fresh holdout untouched until freeze.
3. `physics_features.py` (independent agent): rating-scaled observations/physical residuals and observed waveform spectra. No hidden-state input.
4. `learning.py`: multi-label target construction, balanced training samples, healthy regressors and tree classifiers; save fitted model/feature order.
5. `alerts.py`: confidence persistence, separate fault states, recovery/rearming, >5-minute arbitration, evidence/recommendation CSV. No labels or ground-truth event times.
6. `run.py`: reproducible data loading, training/model selection, official comparisons, freeze, fresh holdout and artifacts.
7. Independent audit, actual-data causal replay, readable explanation with measured wins/failures, packaged code/model/results.

All attempts are retained or logged. The implementation may evolve from training/development evidence; the frozen final holdout cannot be used for another tuning pass while still called a holdout.
