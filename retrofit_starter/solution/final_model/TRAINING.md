# Retrain the Astra Best architecture

The saved model is ready to run. Use this guide when you want to learn the models again from the challenge data.

Use the Python 3.12 environment and pinned requirements from the [main README](../../README.md). Set `QUAVIZ_CHALLENGE_DIR` to the complete organizer folder, then work from `solutions/astra_best/`. Training needs the organizer's `src/`, `configs/fleet_public.yaml`, and `data/full/` files, including the training and validation labels.

These commands write new files under `synthetic_data/`, `cache/` and `runs/`. Start in a clean checkout with those directories absent. Existing runs are preserved; a repeated experiment should use a separate checkout or consistently renamed runs. Do not delete a run just to hide an earlier result.

The commands reproduce the architecture and data-generation plan. The frozen `model.joblib` reproduces the reported inference results. A fresh fit is not guaranteed to reproduce the exact historical model bytes or scores: some training metadata records earlier script versions, and numerical-library or platform changes can also affect fitting. The original hashes remain in `provenance/experiments/`.

## Step 1: Generate examples with known faults

The simulator creates sensor observations and separate labels that tell training which faults were present. It also creates healthy examples. Inference never receives these labels.

```sh
python synthetic.py --role train --days 3
python synthetic.py --role development --days 3
python synthetic.py --role train_balanced --days 2
python synthetic.py --role train_crossed --days 2
python synthetic.py --role train_sensor_crossed --days 1.5
```

The roles mean:

- `train`: varied training examples.
- `development`: separate examples for choosing model and alert settings.
- `train_balanced`: extra examples of faults that need more coverage.
- `train_crossed`: each machine fault appears across matched operating and sensor setups, reducing accidental links between a fault and one setup.
- `train_sensor_crossed`: sensor faults appear on different channels, with matched healthy examples.

The scripts use recorded seeds. These training roles add 108 recordings to the seven supplied training pumps. The development examples are not used to fit the trees.

## Step 2: Train the four groups of machine-fault models

The first group also learns the healthy expected readings. The next two reuse those healthy models. The fourth learns directly from physical measurements and patterns.

```sh
python run.py train --name hist_01 --kind hist

python run.py train --name routed_hist_02 --kind hist \
  --reference-from hist_01 --routed

python run.py train --name crossed_hist_03 --kind hist \
  --reference-from hist_01 --routed \
  --datasets original_train train train_balanced train_crossed

python run.py train --name crossed_physics_04 --kind hist \
  --no-residuals --routed \
  --datasets original_train train train_balanced train_crossed
```

`--routed` means that each fault model receives a chosen set of relevant inputs. For example, a mechanical model examines vibration and heat, while a hydraulic model examines pressure, flow and power. The trees still learn how to combine those inputs and where to draw the boundaries.

`--no-residuals` means that the fourth group does not use the differences from the healthy regression estimates.

## Step 3: Train the shared sensor models

These models examine each channel for drift, stuck values, calibration shifts and noise. Training labels are attached to the sensor actually affected by each event.

```sh
python train_sensor.py --name sensor_expert_03 --reference hist_01 \
  --datasets original_train train train_balanced train_crossed train_sensor_crossed \
  --evaluate development original_validation --seed 20261107 --max-iter 140 \
  --max-positive-per-type-channel 1800 --max-negative-per-channel 3500
```

The `--evaluate` inputs produce development diagnostics. They are separate from the `--datasets` inputs used to fit the sensor models.

## Step 4: Combine the models

Average the four groups' machine-fault and communication-loss scores. Use the shared sensor models for the four sensor faults. Remove the unused sensor classifiers from the four groups to keep the saved model smaller.

```sh
python make_ensemble.py \
  --views hist_01 routed_hist_02 crossed_hist_03 crossed_physics_04 \
  --sensor sensor_expert_03 --name final_ensemble \
  --sensor-types SENSOR_DRIFT SENSOR_STUCK SENSOR_CALIBRATION_SHIFT SENSOR_NOISE_SPIKES \
  --prune-unused
```

The new model is saved at `runs/final_ensemble/model.joblib`.

## Step 5: Choose when the scores should produce alerts

Generate scores on public validation and synthetic development, then search alert thresholds and timing settings:

```sh
python run.py predict --name final_ensemble
python tune_policy.py --name final_ensemble --max-passes 2
```

Selection gives equal weight to the two development datasets' official scores. This step uses validation labels for model selection, so the resulting validation score must be described as a development result. Do not use test data or a previously inspected holdout to choose these settings.

The selected settings are saved at `runs/final_ensemble/policy_tuning/policy.json`. The search also records every evaluated candidate. The historical selected run evaluated 314 unique policies; a new fit can take a different search path.

## Step 6: Run and score your new model

```sh
python infer.py --model runs/final_ensemble/model.joblib \
  --policy runs/final_ensemble/policy_tuning/policy.json \
  --data "$QUAVIZ_CHALLENGE_DIR/data/full" \
  --split validation --output output/retrained_validation.csv

PYTHONPATH="$QUAVIZ_CHALLENGE_DIR/src" python -m retrofit_sim score \
  --predictions output/retrained_validation.csv \
  --events "$QUAVIZ_CHALLENGE_DIR/data/full/events_validation.csv" \
  --assets "$QUAVIZ_CHALLENGE_DIR/data/full/assets.csv" \
  --benign "$QUAVIZ_CHALLENGE_DIR/data/full/benign_windows_train_validation.csv" \
  --split validation --out output/retrained_validation_score.json
```

For submission predictions, use the same new model and settings with `--split test`. Do not tune against hidden labels or shipped test predictions.

## Tests and the historical holdout

```sh
python -m pytest -q --ignore=tests/test_review.py
```

This runs the 29 portable tests. The original workspace also ran data-dependent replay checks, for a total of 40 passing tests. See the main README for the original-layout limitation.

The original holdout was generated after a recorded model/policy freeze. Its labels and scored predictions are included so the aggregate score can be checked using the main README command. Its raw observation caches are omitted. The original freeze and reporting-recovery script contain historical workspace paths; running them unchanged from a new checkout is not a supported new holdout experiment.

To make a new unbiased holdout claim, first finish and freeze the new model, code and alert settings; record new seeds and the input hashes; then generate and evaluate fresh observations once. Once those outcomes influence another model change, they become development evidence.
