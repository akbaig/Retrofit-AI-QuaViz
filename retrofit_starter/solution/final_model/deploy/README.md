# Deploy adapter: raw telemetry → best

Frozen **best** still consumes 1-min features. This package is the missing front-end:

```
python -m retrofit_sim replay   (1 s JSONL / MQTT)
        │
        ▼
deploy.MinuteBuffer + official features_1min()
        │
        ▼
OnlineBest (model.joblib + policy.json + assets/context/snapshots)
        │
        ▼
alerts JSONL  (+ optional explainability / what-if)
```

## Run

From `solution/final_model`:

```sh
# Internal replay → aggregate → infer once at end (fast demo)
python -m deploy --asset PUMP-08 --start 2025-09-10T08:00:00Z --end 2025-09-10T11:00:00Z \
  --infer-every 0 --alerts-out /tmp/live_alerts.csv

# Near-live: re-score every 5 completed minutes (default)
python -m deploy --asset PUMP-08 --limit 12000

# Pipe the challenge replay tool
PYTHONPATH=../../src python -m retrofit_sim replay --data ../../data/full --asset PUMP-08 --limit 12000 \
  | python -m deploy --asset PUMP-08 --stdin --infer-every 0

# With operator narration on each new alert
python -m deploy --asset PUMP-08 --start 2025-09-10T08:00:00Z --end 2025-09-10T11:00:00Z \
  --infer-every 0 --explain --provider template
```

## Notes

- Aggregation uses the **same** `retrofit_sim.dataset.features_1min` as the published dataset.
- Context and vibration snapshots are still loaded from `data/full` (nameplate / sparse waveforms are not invented from the 1 s stream).
- Inference is causal: a minute `[t, t+1)` is scored only after it completes (`until = window_start + 1 min`).
- `--infer-every 0` aggregates the whole stream then scores once; use `>0` for periodic online scoring.
- This is near-real-time / edge-shaped, not a write path to drives or valves.
