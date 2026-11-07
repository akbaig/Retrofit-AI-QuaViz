# Fleet detector (second ML approach)

Fleet-wide normal-behavior models, causal window features, a multiclass fault classifier and our own alert policy. Compared with the final ensemble in [`../comparison/`](../comparison/) and told in [`../../SOLUTION.md`](../../SOLUTION.md).

## Layout

- `quaviz_detector/` — package (stage 1–3, baselines harness, simfleet generator, fidelity check)
- `cache/` — fitted stage-1 models, stage-2 tables, stage-3 classifiers and policies
- `results/` — scores on train / validation / held-out sims, stage-1 separability
- `run_detector.py` — harness entry point
- `tests/` — causality checks

## Run

```sh
cd RETROFIT_AI_PARTICIPANTS_2026
source .venv/bin/activate
python -m quaviz_detector.stage1 fit          # if cache/stage1 is empty
python -m quaviz_detector.stage2 build
python solution/fleet_detector/run_detector.py --no-lopo
```

Simulated fleets live at `../../build/sim` (override with `QUAVIZ_SIM`).
