QUAVIZ LEARNED FAULT ENSEMBLE

This is a supervised fault detector and diagnostic model. It uses no Isolation
Forest. It identifies developing evidence in observations available now; it does
not predict a future failure date or detect a fault before any evidence exists.

MEASURED PUBLIC-VALIDATION RESULT

Official composite score: 95.0214 / 100.
Published threshold baseline B1: 78.0125 (+17.0088 points).
Best B1 setting within our declared 20-setting validation grid: 80.2035
(+14.8178 points). Published Isolation Forest B3: 69.1625 (+25.8589 points).

The final model detects all 9 scored validation events; the first diagnosis is
correct for 8/9, and machine-versus-sensor class is correct for 9/9. It produces
0 false-alert episodes and 1 excess-repeat episode. Air entrainment is detected
but its first diagnosis is impeller wear. The official scorer counts that as a
detection and a wrong fault type; the report keeps those metrics separate.

These two validation pumps are familiar development data. They were used to
select models and alert thresholds, as the challenge rules permit. The selected
score is not a blind estimate of generalization. The independent frozen
synthetic holdout scored 63.5086 versus 43.5292 for the strongest of three
preselected B1 settings, a 19.9794-point gain. Full results, remaining misses
and the reporting-error recovery are in RESULTS.txt and results/holdout/.

WHAT THE MODEL DOES, START TO FINISH

1. Read completed one-minute measurements, operating demand, public pump
   nameplate data, and vibration snapshots. Reconstruct an independent minute
   clock for gateway gaps. Keep sparse sensor age/count information; absent
   readings do not become healthy zeros. Ignore labels and event deadlines.

2. Convert measurements into comparable evidence: flow/head/power relative to
   the pump's ratings, departures from pump-curve and suction relations, and
   trailing changes over 10 and 60 minutes. Extract rotational harmonics and
   bearing-envelope features from the measured vibration arrays. A snapshot
   becomes available only after its acquisition finishes and is never backfilled.

3. Fourteen regressors, fitted only on healthy training observations, estimate
   expected channel values from operating context. Their errors become additional
   features. No new pump's entire first day is assumed healthy for calibration.
   Shared sensor models also use robust past-only histories, so a new noise
   episode can be compared with the same sensor's earlier behavior.

4. Four supervised boosted-tree views average their scores equally for each machine fault and
   gateway loss: broad context/residual inputs; physically routed inputs; routed
   inputs with extra crossed training domains; and direct physical inputs without
   regression residuals. Mechanical specialists see vibration/thermal evidence;
   hydraulic specialists see pressure, flow, power, suction and curve evidence.
   The models learn their decision boundaries and fault identities from labels.

5. Four shared sensor specialists examine each available channel for drift,
   freezing, calibration shift and noise. They use channel-normalized temporal,
   variance and consistency features, with labels assigned to the actually
   affected sensor tag. Their scores replace the four sensor outputs of the
   ensemble. All-channel maximum scores are thresholded after pooling. Scores
   are model scores, not calibrated probabilities of a real-world fault.

6. A per-fault incident state requires 15 consecutive minutes above its selected
   score threshold. It rearms after 240 minutes below the recovery threshold and
   permits a reminder after 24 hours. Startup/stopped readings are inhibited for
   nongateway diagnoses. Independently supported alerts are separated by six
   minutes to respect the scorer's burst definition. See policy.json for all
   class thresholds; thresholds are learned-selection settings, not physical
   failure probabilities.

7. Emit asset, time, machine/sensor class, suspected fault, score, measured
   residual evidence and an inspection recommendation. Fault diagnosis is
   learned. Recommendation text is still an explicit fault-to-action lookup;
   it does not automatically stop equipment or order maintenance.

TRAINING AND SELECTION

The training union is 7 original training pumps plus 108 generated training
recordings: 396,720 minute slots across 115 recordings. Earlier ensemble views
use subsets, recorded in their training metadata. The source simulator is
explicitly allowed by the challenge. Generated examples vary fault onset,
severity, overlap, context and instrument setup, and include healthy recordings.
Crossed cohorts give each machine fault the same control/instrumentation strata
to reduce accidental correlations with the installed sensors or pump metadata.

Training targets use fault labels and affected sensor tags. Inference inputs
exclude event labels, event deadlines, asset identifiers, timestamps, seeds,
scenario identifiers, work-order conclusions and simulator latent states.
Identifiers only group observations; timestamps only maintain causal ordering.
Static public specifications are permitted physical/context inputs.

The alert policy was selected by equal mean official score on original public
validation and separately generated synthetic development: 95.0214 and 72.7852,
respectively. The final search evaluated 314 unique policies. Earlier variants,
including an ExtraTrees model and a small neural network, are retained in the
experiment record. The neural model scored 79.7799/66.0545 after a bounded
policy search and was not selected. No variational autoencoder was required:
the available fault labels support learning diagnoses directly.

The final compact artifact is about 8.7 MiB: 14 healthy regressors and 40 active
binary boosted-tree classifiers. Overridden sensor classifiers were removed;
this was checked to leave all 50,400 development probability rows exactly equal.

RUN THE DELIVERED MODEL

Use Python 3.12 and the versions in requirements.txt, especially scikit-learn for
loading the serialized model. From the extracted solution directory:

  python -m pip install -r requirements.txt
  python infer.py --model model.joblib --policy policy.json \
    --data /path/to/RETROFIT_AI_PARTICIPANTS_2026/data/full \
    --split validation --output predictions_validation.csv

Use --split test for test predictions without labels or tuning. For streaming or
partially truncated input, also supply --until with the actual UTC availability
time. Without --until, inputs must be already completed minute aggregates; the
clock stops at the latest supplied observation/context record. With an explicit
clock, a terminal gateway outage can be detected without a new telemetry row.
The declared asset window end alone does not advance an unknown live clock.

REPRODUCING TRAINING

Set QUAVIZ_CHALLENGE_DIR to the unmodified challenge bundle directory. The
generator and training/scoring scripts then work from any extracted location.
Full reproduction commands are in REPRODUCE.txt. Raw source data and generated
training caches are not duplicated in this ZIP; provenance/seed manifests and
the generation code are included. The saved final model is ready for inference.

EVALUATION LIMITS

The public validation split has only 9 events. The untouched holdout uses the same
public simulator with new seeds and perturbed settings; it does not demonstrate
performance on a real plant or on the organizers' private test configuration.
Some generated observable sensor labels have no usable target-channel readings;
these remain in the unchanged official scoring denominator. No misses are removed.

The source simulator generates snapshots separately from telemetry corruption;
this is a property of the supplied observations. We use the provided arrays,
not unexported truth. Waveforms and physics distinguish several otherwise
confusable mechanical faults, but fault identity can still be uncertain.

The published B1 comparison retains its original first-day normalization, which
can look ahead for first-day alerts. Its fixed setting, development-selected
setting, and validation-selected setting are reported separately. The official
scorer and baseline code are unchanged.

Deadline lead means event deadline minus first alert, not time before a fault
starts. Only 2 original-validation events have deadlines: one is detected before
its deadline and one about 2 minutes after. The median lead is 9.1899 hours over
those two detected events. That small denominator is why lead time is not the
headline claim here.
