# Development record

The original public validation set is familiar development data. The challenge expressly permits using it for model selection. At the start, fresh synthetic holdout seed 1013137 was reserved for generation after freeze; the pre-freeze amendment expanded this to three seeds. No test-asset data or predictions were used for model or policy selection. The final frozen model later produced unscored test predictions for delivery.

## Initial supervised architecture

Training sources: original train, randomized simulator train (21 trajectories), and balanced supplemental simulator train (30 trajectories). The supplement adds rare mechanical signatures and crosses sensor drift/calibration shifts with current, temperature and suction channels. It includes healthy trajectories and simultaneous faults. Total input: 58 trajectories, 247,680 minute slots.

The initial fit failed because this installed sklearn histogram implementation cannot bin an entirely missing numeric predictor. Reproduced with a small regression test. Fixed by explicitly encoding missing model inputs with a reserved numeric value while preserving missingness/freshness in the observation pipeline; missing target measurements are never learned as healthy values. No model was scored from that failed fit.

Independent review also found and corrected stale aggregate statistics entering the physics branch and double event weighting during positive resampling. The corresponding missingness tests now pass. The final learner balances event sampling once, excludes nonsteady machine targets, and handles an absent class explicitly.

- `hist_01`: healthy conditional regressors, physical/temporal/waveform inputs, 13 supervised histogram-gradient-boosting classifiers.
- `extra_direct_01`: same causal physical/temporal/waveform inputs, 13 supervised ExtraTrees classifiers, without the learned healthy-regression stage. This tests whether the extra regression stage adds value.

Both classifier families are supervised fault models, not Isolation Forest. Complete settings and code hashes are saved per run. Alert policies are selected using equal weight for familiar original validation and independently generated synthetic development, retaining the full selection grid. Model comparisons and later changes will be added as results arrive.

## First results and revision

The initial global-policy selection gave `hist_01` 82.9814 on public validation (7/9 events, all seven first diagnoses correct, no false/excess episodes) and 51.5207 on synthetic development. `extra_direct_01` gave 51.039 and 59.176 respectively. Neither met the objective. The boosted model missed validation current drift and air entrainment. The direct randomized-tree model relied strongly on control/profile/missing-sensor features: e.g. static metadata explained 64% of its cavitation split importance; power sample age alone explained 41% for impeller wear. Useful measured harmonic signatures were underused. These are diagnostic findings, not evidence that a feature causes a fault.

`routed_hist_02` therefore uses physically selected input families for each learned machine classifier. Static ratings/control/profile and unrelated age/freshness are excluded from those classifier inputs; ratings still normalize measurements and the healthy regressors still condition on operating context. Mechanical specialists receive measured waveform harmonics/envelopes, vibration and thermal features. Hydraulic specialists receive curve, pressure, flow, power, suction margin and fluctuation evidence. The classifier learns boundaries and diagnoses from training labels; routing itself is not a list of hand-written alarm thresholds.

A shared channel-anonymous sensor expert is being fitted separately: fault labels align with `event.sensor_tag`, and the same learned model is applied to each eligible sensor's normalized temporal/noise/coherence features. This targets transfer between sensor tags and pre-existing current drift. A new crossed training cohort gives every machine class the same control/profile/instrumentation strata to remove class/domain correlations. Public validation remains permitted familiar development data throughout.

## Final development selection

Correction: the PUMP-09 validation current drift is not pre-existing. Its onset is six hours after the asset window starts. Early working notes called it pre-existing incorrectly. No target dates or labels were changed; the models use the real event timestamps for supervision and evaluation.

The crossed machine cohort adds 36 two-day recordings with matched static specifications across fault classes. The sensor-crossed cohort adds 21 36-hour recordings with current/power drift and current/power/acceleration/crest noise, plus matched healthy controls. All are training data. Sensor expert 03 uses all 115 training recordings; earlier core views retain their declared training subsets and common healthy regressors.

Independent per-fault 32→16 tanh neural networks were also fitted. Their bounded policy-selected scores were 79.7799 public validation and 66.0545 synthetic development, below the selected ensemble. They are retained as a tested alternative rather than claimed as a win.

An equal-weight score ensemble of hist01, routed02, crossed03 and direct-physics04, with all four sensor outputs replaced by sensor expert 03, selected 95.0213516730 /72.7851591414 (equal mean 83.9033) over 314 unique policy trials. It detects 9/9 validation events with 8/9 correct first fault types,9/9 correct classes,0 false episodes and1 excess repeat. Air entrainment remains first diagnosed as impeller wear. Selection uses development data only. Final policy: persistence15min, recovery240min, reminder1440min, default score threshold.95; overrides cavitation.01, discharge restriction.6, imbalance.6, impeller wear.15, air entrainment.08, misalignment.08.

The final artifact `runs/final_ensemble` removes the 16 overridden sensor classifiers and shares identical healthy-regressor objects. Actual recomputation over all 50,400 original-validation/development rows matched the selected ensemble probabilities exactly (maximum difference 0), and official scores reproduced exactly. It contains 14 healthy regressors and 40 active binary classifiers, about 8.7MiB. Both ensembles also passed full prediction-prefix/future-mutation audits. The full current test suite passed 35 tests.

All model/threshold choices are complete before the fresh holdout is generated. The holdout has 21 three-day recordings with predeclared seeds 1013137,1013141,1013153. Its results will be appended separately; they cannot justify another tuning pass while still being called holdout results.


## Frozen holdout result and reporting continuation

The final frozen model scored 63.508594 on 21 new synthetic recordings (78 scored events), versus published B1 41.724035, development-selected B1 40.924288 and validation-selected B1 43.529169. It detected 55/78 events and correctly named 37/78 first faults, with 14 false and 12 excess-repeat episodes. It outperformed each fixed B1 setting in all three seed groups. No model or policy changes followed the result.

The original frozen evaluator failed after aggregate scoring and hash re-verification while formatting per-asset results: the untouched official scorer cannot compare empty event windows to alert timestamps for an eventless healthy asset. Initial aggregate report and predictions were preserved. A separate reporting continuation reused those exact predictions, reproduced the aggregate report exactly, recomputed the same three fixed B1 comparators and reported per-asset composites as unavailable on eventless assets. Aggregate and per-seed scores retain every healthy asset and use the unmodified official scorer. All freeze and generated-input hashes still verify. The final complete source test suite passed 40 tests. Full outcomes and limits appear in RESULTS.txt.
