# Retrofit AI QuaViz — Astra Best

A trained pump-fault detector for the Since AI challenge. It reads sensor data, looks for developing problems, and names the suspected fault.

**The main idea: learn what healthy measurements should look like, measure the differences, then learn which combinations of evidence match each fault.**

This branch is called `astra-best` because Git branch names cannot contain spaces. The solution lives in [`solutions/astra_best/`](solutions/astra_best/). The organizer's files stay in [`RETROFIT_AI_PARTICIPANTS_2026/`](RETROFIT_AI_PARTICIPANTS_2026/).

## 1. How the solution works

Every minute, it goes through these steps:

1. **Read the pump's measurements and operating conditions.** Pressure, flow, power, temperature, vibration, speed and demand help describe what the pump is doing. Only completed measurements and information available at that time are used.
2. **Estimate healthy readings.** Small regression models learn what each measurement normally looks like at the current operating conditions. For example, high vibration may be normal at one load but unusual at another. The difference between the estimate and the actual reading becomes evidence.
3. **Look at patterns across readings.** The system checks pressure–flow–power relationships, changes over the past 10 and 60 minutes, and patterns in vibration frequencies. Missing readings stay missing; they do not become healthy zeros.
4. **Learn which fault explains the pattern.** Gradient-boosted trees are trained on labeled healthy and faulty examples. Four groups of these models use different inputs or training examples. Their scores are averaged for each machine fault and communication loss. Multiple faults can be considered at once.
5. **Check the sensors themselves.** Four shared models look at each sensor for drift, stuck readings, calibration shifts and noise. These supply the sensor-fault scores.
6. **Wait for sustained evidence.** The selected settings require 15 consecutive minutes above the relevant fault's score threshold. Startup and stopped periods are handled separately. The system also limits repeated alerts.
7. **Write an alert.** The output contains the pump, time, suspected fault, score, supporting measurements and an inspection recommendation. The models learn the diagnosis. Recommendation wording is a fixed lookup.

The healthy models estimate what a reading should be **now**. They do not predict a future failure date. The system can flag a developing problem when evidence appears in the measurements. Whether that happens before a traditional alarm depends on the fault and alarm settings; these results do not establish a general lead over traditional alarms. It cannot detect a fault before any evidence exists.

### What was trained?

- **14 regression models** learn healthy measurements from operating conditions.
- **Four groups of boosted-tree models** estimate machine-fault and communication-loss scores. One uses a broad set of inputs; one uses inputs chosen for each fault; one adds more varied training examples; one uses direct physical measurements without the healthy-model differences.
- **Four sensor models** are reused across channels to identify sensor problems.
- The saved model contains **40 active fault classifiers** in total and is about **8.7 MiB**.

The training data combine the seven supplied training pumps with 108 generated recordings: 396,720 minute slots across 115 recordings. The challenge simulator provides known faults for training. Generated examples vary operating conditions, sensor setups, timing and fault combinations.

This is supervised fault detection and diagnosis. It uses no Isolation Forest. A small neural-network alternative was also tested, but it scored lower than the selected tree ensemble.

## 2. What it scored

These are the **unchanged official challenge composite scores, out of 100**. They combine detection, timing, diagnosis and alert burden. They are not accuracy percentages.

| Approach | Public validation | Fresh synthetic holdout |
|---|---:|---:|
| Published Isolation Forest B3 | 69.2 | Not run |
| Published threshold baseline B1 | 78.0 | 41.7 |
| Best public-validation B1 setting in our 20-setting search | 80.2 | 43.5 |
| **Astra Best** | **95.0** | **63.5** |

The best tested B1 setting used the same deviation threshold as the published baseline, but required five unusual minutes instead of fifteen. “Best tested” means the best in that specific search, not the best threshold system anyone could build.

**Public validation:** 9/9 events detected, 8/9 first fault names correct, no false-alert episodes and one excess-repeat episode. Air entrainment was first called impeller wear.

**Fresh holdout:** 21 new recordings generated after the model and alert settings were frozen. Astra detected 55/78 events, versus 39/78 for the B1 setting above. First fault names were correct for 37/78 versus 18/78. Astra produced 14 false-alert episodes and 12 excess-repeat episodes. It scored higher than all three preselected B1 settings in each of the three seed groups.

Public validation **was used for model and threshold selection**. The fresh holdout was not. Both use the challenge simulator, so these results do not establish real-plant or organizers' private-test performance. Cavitation and sensor calibration shifts remain weaknesses.

Detailed results: [full explanation](solutions/astra_best/RESULTS.txt), [validation report](solutions/astra_best/results/original_validation_report.json), [holdout comparison](solutions/astra_best/results/holdout/comparison.csv), [fault-by-fault holdout results](solutions/astra_best/results/holdout/per_type.csv).

## 3. Run the trained model

These commands are for macOS/Linux with **Python 3.12**. Use the pinned dependency versions to load the saved model consistently. Start in the repository root.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r solutions/astra_best/requirements.txt

export QUAVIZ_CHALLENGE_DIR="$PWD/RETROFIT_AI_PARTICIPANTS_2026"
export LOKY_MAX_CPU_COUNT=3
export OMP_NUM_THREADS=3
export PYTHONDONTWRITEBYTECODE=1

cd solutions/astra_best
```

Generate alerts for the two public validation pumps:

```sh
python infer.py --model model.joblib --policy policy.json \
  --data "$QUAVIZ_CHALLENGE_DIR/data/full" \
  --split validation --output output/predictions_validation.csv
```

The saved model produces **17 alert rows**. The official scorer groups these into events and repeat episodes; 17 rows does not mean 17 faults.

Score those alerts with the organizer's unchanged scorer:

```sh
PYTHONPATH="$QUAVIZ_CHALLENGE_DIR/src" python -m retrofit_sim score \
  --predictions output/predictions_validation.csv \
  --events "$QUAVIZ_CHALLENGE_DIR/data/full/events_validation.csv" \
  --assets "$QUAVIZ_CHALLENGE_DIR/data/full/assets.csv" \
  --benign "$QUAVIZ_CHALLENGE_DIR/data/full/benign_windows_train_validation.csv" \
  --split validation --out output/validation_score.json
```

Expected composite: **95.0213516730**. The supplied [reference predictions](solutions/astra_best/results/original_validation_predictions.csv) and report record the original result.

Generate public-test predictions using the same frozen model:

```sh
python infer.py --model model.joblib --policy policy.json \
  --data "$QUAVIZ_CHALLENGE_DIR/data/full" \
  --split test --output output/predictions_test.csv
```

This produces **43 alert rows over three pumps**. These are unscored submission predictions. Test observations were not used to choose the model or thresholds, and no hidden test labels were used.

Inference reads observations, pump information, operating context and vibration snapshots. It does not read event labels. When replaying partially received data, provide `--until` with the actual UTC availability time. Without it, supplied minute aggregates must already be complete.

## 4. Retrain it yourself

Follow [the step-by-step training guide](solutions/astra_best/TRAINING.md). It explains how to:

1. Generate labeled training examples using the supplied simulator.
2. Fit the four groups of fault models and the shared sensor models.
3. Combine them into one saved model.
4. Select alert settings on public validation and synthetic development data.
5. Run and score the resulting model.

Training writes to ignored `synthetic_data/`, `cache/` and `runs/` directories inside the solution folder. It preserves the shipped `model.joblib` and `policy.json`. The saved artifact reproduces the reported predictions; the training commands reproduce the architecture, but a fresh refit is not promised to be byte-identical to the historical model.

## 5. Check the reported holdout score

From `solutions/astra_best/`, after the environment setup above:

```sh
mkdir -p output
PYTHONPATH="$QUAVIZ_CHALLENGE_DIR/src" python -m retrofit_sim score \
  --predictions results/holdout/Quaviz_frozen_predictions.csv \
  --events provenance/synthetic_data/holdout/events.csv \
  --assets provenance/synthetic_data/holdout/assets.csv \
  --split holdout --out output/holdout_score.json
```

Expected composite: **63.5085943586**. This checks the saved holdout predictions against their labels. It does not generate a new holdout or rerun inference on regenerated observations.

The original freeze, seeds, input hashes and reporting history are in [`provenance/`](solutions/astra_best/provenance/). The historical freeze contains paths from the original workspace. Do not treat it as a portable new-run configuration. A new unbiased holdout claim requires a new freeze and fresh seeds before examining the result.

## 6. Tests and file guide

Run the portable tests from `solutions/astra_best/`:

```sh
python -m pytest -q --ignore=tests/test_review.py
```

The original workspace passed 40 tests, including real-data replay checks. The portable subset contains 29 tests. The omitted file includes replay checks that depend on the original workspace layout and intermediate trained models. The tests check measurement availability, missing data, sensor features, alert behavior and that changing future readings cannot alter earlier predictions.

The [branch verification record](solutions/astra_best/BRANCH_VERIFICATION.md) lists what was rerun before this branch was published: 29 portable tests, validation and test inference, and official scoring of validation and saved holdout predictions.

| File or folder | Purpose |
|---|---|
| `model.joblib` and `policy.json` | Trained model and frozen alert settings |
| `infer.py` | Read observations and write alerts |
| `features.py` and `physics_features.py` | Prepare measurements, past changes and physical relationships |
| `learning.py` | Healthy models and learned fault classifiers |
| `sensor_expert.py` | Shared sensor-fault models |
| `composite.py` | Average model scores and apply sensor scores |
| `alerts.py` | Decide when evidence becomes an alert |
| `synthetic.py` | Generate labeled simulator examples |
| `run.py`, `train_sensor.py`, `make_ensemble.py`, `tune_policy.py` | Train, combine and choose alert settings |
| `results/` | Predictions, scores and selection history |
| `provenance/` | Training settings, original hashes and audit records |
| `PACKAGE_MANIFEST.json` | Hashes of the original delivered solution files |

No runtime language-model call, cloud service or equipment connection is required. The organizer's code and data retain their original notices in [`LICENSE_AND_PROVENANCE.md`](RETROFIT_AI_PARTICIPANTS_2026/LICENSE_AND_PROVENANCE.md).
