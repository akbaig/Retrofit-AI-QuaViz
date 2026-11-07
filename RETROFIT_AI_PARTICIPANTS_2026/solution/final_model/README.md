# Final model: supervised fault ensemble (Astra Best)

This is the **selected submission**. It is a supervised boosted-tree ensemble with healthy-response regressors, physics features, a per-channel sensor branch and a frozen alert policy. No Isolation Forest.

The story that puts this next to the encoding study and the fleet detector is in [`../../SOLUTION.md`](../../SOLUTION.md).

## Frozen artifact

| File | Role |
|---|---|
| `model.joblib` | Trained ensemble (do not overwrite) |
| `policy.json` | Alert thresholds and persistence |
| `results/original_validation_predictions.csv` | Validation alerts (score 95.02) |
| `results/holdout/` | Frozen synthetic holdout (score 63.51) |
| `results/test_predictions.csv` | Unscored public-test alerts |
| `provenance/FREEZE.json` / `FREEZE.sha256` | Freeze record before holdout generation |

Retraining writes under `runs/` and does **not** replace the two delivered files.

## Measured scores (official scorer)

| Set | This model | Published B1 |
|---|---:|---:|
| Public validation (PUMP-08/09) | **95.02** | 78.01 |
| Frozen synthetic holdout (78 events) | **63.51** | 41.72 |

Validation was used for model and policy selection. The holdout was generated after a freeze. Per-type tables in `results/holdout/` show the remaining weaknesses: cavitation and calibration shift stay weak.

## How it works

1. Build causal minute features from telemetry, operating context, nameplate ratings and vibration snapshots (pump-curve and suction departures, trailing changes, harmonics and bearing envelopes).
2. Fourteen healthy-response regressors, fitted only on healthy training minutes, produce expected channel values and residuals.
3. Four equally weighted boosted-tree views score machine faults and gateway loss. Four shared sensor specialists score drift, stuck, calibration shift and noise per channel.
4. The alert policy needs 15 consecutive minutes above the selected threshold, inhibits startup/stopped readings for non-gateway diagnoses, rearms after 240 minutes below the recovery threshold and allows one reminder after 24 hours.

## Run inference

From this directory, with the challenge venv and `QUAVIZ_CHALLENGE_DIR` pointing at `RETROFIT_AI_PARTICIPANTS_2026`:

```sh
export QUAVIZ_CHALLENGE_DIR="$(cd ../.. && pwd)"
python infer.py --model model.joblib --policy policy.json \
  --data "$QUAVIZ_CHALLENGE_DIR/data/full" --split validation \
  --output /tmp/val_alerts.csv
PYTHONPATH="$QUAVIZ_CHALLENGE_DIR/src" python -m retrofit_sim score \
  --predictions /tmp/val_alerts.csv \
  --events "$QUAVIZ_CHALLENGE_DIR/data/full/events_validation.csv" \
  --assets "$QUAVIZ_CHALLENGE_DIR/data/full/assets.csv" \
  --benign "$QUAVIZ_CHALLENGE_DIR/data/full/benign_windows_train_validation.csv" \
  --split validation
```

## Retrain

See [TRAINING.md](TRAINING.md). Quick path: generate synthetic roles with `synthetic.py`, train views with `run.py` / `train_sensor.py`, assemble with `make_ensemble.py`, then `tune_policy.py`.

A small neural alternative was trained and rejected (79.8 validation under its policy search).

## Tests

```sh
python -m pytest -q --ignore=tests/test_review.py
```
