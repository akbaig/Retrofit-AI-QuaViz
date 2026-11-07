# Execution and review record

2026-11-07. User authorized dataset exploration, model choice/comparison, execution, embeddings and cluster visualization. Work is isolated in this new experiment directory. No edits to source data, starter application or previous experiments.

1. Read START_HERE.html, data dictionary, benchmark split rules, technical generation model and the existing observation/physics functions.
2. Frozen vendor copies from `experiments/quaviz-learned/`; recorded their hashes. Independent environment created with Python 3.12 and pinned scientific packages.
3. Initial tests for labels, preprocessing and sequence boundaries failed for missing implementations, then passed. Reused and ran causal-feature tests.
4. Ran five unsupervised candidates. Initial raw-PCA macro AP was 0.313. Strong equipment confounding motivated an explicitly label-informed contrastive comparison.
5. Independent review identified display-label precedence hiding unknown overlapping labels, and cache reuse without source fingerprints. Added a failing regression case, independent uncertainty flags, conservative eligibility, active event IDs and fingerprint checks. Archived all provisional output; reran preparation and the comparison. The final raw-PCA result is 0.301. The current report uses corrected numbers only.
6. Ran contrastive seeds 42, 7 and 123. Prespecified seed42 remains the default despite seed123 scoring higher. Test embeddings were produced only after model selection in each experiment and never entered a selection metric. Test data were not plotted or labeled.
7. Independent re-review found both important issues resolved. Reviewer checked all ten tests, artifact fingerprints, exact encoder reload and a controlled mutation experiment showing that non-training data/labels do not affect contrastive weights.
8. Verified all eight exported matrices and aligned Parquet tables, source hashes, test-label treatment and saved-model replay. The inference CLI reproduced a 120-minute PUMP-08 prefix to tolerance 1e-5 without reading labels.
9. Generated six train-fitted UMAP/PCA maps, an offline explorer and three standalone figures. Inspected the figure renders; adjusted an overlapping footer/legend. Browser checks confirmed all four selectors work, point inspection displays real metadata, and no JavaScript errors were reported.

Verification commands (from this directory):

```bash
.venv/bin/python -m pytest -q
.venv/bin/python verify.py
.venv/bin/python encode.py --model results/models/supervised_contrastive.joblib --data-dir ../../BusinessTurku_SinceAI2026_FINAL_BUILDER_PUBLISH/RETROFIT_AI_PARTICIPANTS_2026/data/full --asset-id PUMP-08 --until 2025-09-08T02:00:00Z --output results/inference_smoke.parquet
```

Scope: research representations and offline diagnostics, no deployed classifier, alerts, official challenge scoring, hardware control, publication or changes to original data. The synthetic benchmark's small number of assets/events and partially labeled intervals limit conclusions. Seed variation and per-fault failures are explicitly reported.
