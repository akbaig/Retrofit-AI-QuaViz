# Verification for encoding-astra

Branch prepared on 2026-11-07 from `origin/main` at `596bfdbda8d1cdacfc4534223b20cb69d83f806f`. The separate `astra-best` branch was not modified. Checks used Python 3.12 and the recorded experiment environment (`requirements-lock.txt`).

## Checks executed on the packaged branch

| Check | Result | Evidence |
|---|---|---|
| `python -m pytest -q` | **21 passed** | `provenance/branch_tests.log` |
| `python verify_fault_detection.py` | **192 classifiers verified**, all saved predictions/metrics reproduced; **134 frozen encoder/vector hashes unchanged** | `provenance/branch_detector_audit.log` |
| Repository dataset and public annotation files | All **six input hashes match** the original run | `provenance/branch_data_checks.json` |
| README vector-loading/classifier example | Passed; **10,080 × 32** array, aligned rows, finite classifier scores; splits **6,048 / 1,896 / 1,896** | `provenance/branch_data_checks.json` |
| README label-free `encode.py` command | Encoded **6,120 minutes**, matched saved vectors within `rtol=atol=2e-5` | `provenance/branch_encode.log`, `branch_encode_replay.json` |
| `ENCODING_DATA_DIR` override | Resolved the repository dataset correctly | Executed with the relative dataset path |
| Active documentation links and artifact inventory | **18 links valid**, 67 encoders, 67 full arrays, 192 classifiers; largest experiment file **17.03 MiB** | `provenance/branch_package_checks.json` |
| Detector report generator and chart | Rebuilt successfully; chart visually inspected | `provenance/branch_report.log` |
| Package contents | All delivered file hashes checked; no source-data edits or environment/output files staged | `PACKAGE_MANIFEST.json` |
| Modified Python sources | Compilation passed | `python -m py_compile data.py summarize.py visualize.py fault_detection_report.py` |

The detector audit confirms **zero mixed-class test cases** among fitted targets. It therefore does not certify a test-confirmed winner. It verifies reproducibility and the stated evaluation limits.

## Packaging scope

No encoders or classifiers were retrained. Saved model weights and full vector arrays were preserved. Dataset lookup now defaults to the repository's challenge directory and permits `ENCODING_DATA_DIR`; plot generators use the platform temporary directory. Generated-report wording, documentation and archive notices were clarified. Trailing whitespace in the bundled explorer script was removed. Original `data.py` is retained in `provenance/data_original.py.txt`.

The branch includes current models, full NumPy vectors, metadata, classifier predictions, summaries and visualizations. It omits the large feature cache, redundant vector/split tables and historical binary copies. The exact omitted current-run files are listed in `provenance/omitted_generated_files.json`.

## Historical checks versus this delivery

The full original reconstruction audit of **67 encoders** is recorded in `results/verification.json` and `results/verification.log`. It was **not rerun from this reduced package**, which omits inputs needed by that historical audit. The classifier audit was rerun directly from the shipped models/vectors/metadata/predictions.

The earlier implementation received an independent review with no blocking findings. A regression test was added that changes test vectors and labels and confirms unchanged fitted classifiers, validation thresholds and selected encoder. This is included in the passing suite; packaging did not introduce new model algorithms.

Historical manifests describe the original workspace and its original source hashes. `PACKAGE_MANIFEST.json` records files delivered on this branch. No official challenge score, real-world evaluation, new training benchmark, or fresh untouched test-set claim is made.
