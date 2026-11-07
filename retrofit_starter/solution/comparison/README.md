# Head-to-head comparison

Official-scorer comparison of:

- **astra-best** — the selected final model (`../final_model`)
- **quaviz** — the fleet detector (`../fleet_detector`)
- **B0–B3** — challenge baselines

on four sets: official validation, our held-out fleets s0112/s0113, and the final model's frozen holdout.

`results/comparison.csv` is the source of the headline table in [`../../SOLUTION.md`](../../SOLUTION.md).

```sh
python compare_external.py --reuse          # reprint saved table
python compare_external.py                  # recompute (needs build/sim_snap and build/astra_synth)
```
