# Submission Pack

This directory contains the documentation and deliverables for the Retrofit AI / QuaViz challenge. It covers the machine learning approach, alert-policy calibration, explainability with sandbox what-if validation, the edge deployment architecture, and retrofit Capex/Opex feasibility.

## Directory Structure

```
submissions/
  README.md                      # Overview, definitions, and reproduction commands
  predictions_test.csv           # Final test-split alert predictions
  system/                        # End-to-end system architecture
  01_training/                   # Model exploration, selection, and benchmark figures
  02_calibration/                # Alert-policy operating point and threshold tuning
  03_whatif/                     # Validation explainability and what-if simulation results
  04_edge/                       # Edge software architecture (MQTT, CPU infer, LLM, LAN UI)
  05_retrofit_feasibility/       # Hardware BOM, Capex/Opex, payback computation
```

## Documentation Index

| Chapter | Document | Focus |
|---|---|---|
| **System** | [`system/system.md`](system/system.md) | End-to-end data flow and architectural boundaries |
| **1. Training** | [`01_training/training.md`](01_training/training.md) | Deep learning exploration vs. feature-engineered tree ensemble |
| **2. Calibration** | [`02_calibration/calibration.md`](02_calibration/calibration.md) | Threshold tuning, persistence windows, and alert suppression |
| **3. What-If** | [`03_whatif/whatif_validation.md`](03_whatif/whatif_validation.md) | LLM narrative validation and simulated maintenance impact |
| **4. Edge design** | [`04_edge/edge_design.md`](04_edge/edge_design.md) + [`edge_design.png`](04_edge/edge_design.png) | On-box CPU inference, MQTT, Gemini/Ollama/template, LAN console |
| **5. Retrofit feasibility** | [`05_retrofit_feasibility/feasibility.md`](05_retrofit_feasibility/feasibility.md) | Sensor pathways, BOM, Capex/Opex, worked payback (B1 / best / holdout) |

Canonical kit source for chapter 5 numbers: [`../retrofit_starter/RETROFIT_FEASIBILITY.md`](../retrofit_starter/RETROFIT_FEASIBILITY.md) and [`../retrofit_starter/examples/business_case.py`](../retrofit_starter/examples/business_case.py).

## Key Definitions

- **Calibration:** Refers to operating-point tuning for the alert policy, configuring per-fault detection thresholds, a 15-minute persistence requirement, and a 240-minute re-arm recovery window into `policy.json`. It is not statistical probability calibration (such as Platt scaling or isotonic regression); reported confidence values remain the raw winning ensemble scores.
- **Fixed (What-If Validation):** A maintenance action is classified as fixed when the sandbox simulation confirms the action is feasible (`accepted = true`) and results in a lower risk proxy (`proposed.risk_proxy < baseline.risk_proxy`). Physical repairs (`clean_strainer`, `replace_bearing`, `realign`) alter physical latent states; inspection actions (`inspect`) verify operational feasibility without modifying physical state.
- **System Boundary:** The system operates strictly as an operator advisor. It provides diagnostics and maintenance recommendations but has no direct control or write access to PLCs, variable-frequency drives, or safety systems.

## Primary Artifacts

| Component | Path | Description |
|---|---|---|
| Frozen Model & Policy | [`../retrofit_starter/solution/final_model/`](../retrofit_starter/solution/final_model/) | Trained model bundle (`model.joblib`) and policy configuration (`policy.json`) |
| Test Predictions | [`predictions_test.csv`](predictions_test.csv) | Final challenge submission file (identical to [`../retrofit_starter/submission/predictions_test.csv`](../retrofit_starter/submission/predictions_test.csv)) |
| Detailed Technical Narrative | [`../retrofit_starter/SOLUTION.md`](../retrofit_starter/SOLUTION.md) | Comprehensive record of experiments, baselines, and design rationale |
| Product Backend | [`../app/backend/`](../app/backend/) | Real-time FastAPI service with MQTT ingestion and streaming inference |
| Operator Frontend | [`../app/frontend/`](../app/frontend/) ([`app/frontend/README.md`](../app/frontend/README.md)) | Local plant console displaying fleet status, alert details, and advice cards |
| Retrofit feasibility | [`05_retrofit_feasibility/feasibility.md`](05_retrofit_feasibility/feasibility.md) | Worked Capex/Opex/payback; kit source [`../retrofit_starter/RETROFIT_FEASIBILITY.md`](../retrofit_starter/RETROFIT_FEASIBILITY.md) |

## Reproducing Results

Activate your virtual environment from the repository root:

```sh
export QUAVIZ_CHALLENGE_DIR="$(pwd)/retrofit_starter"

# Generate test predictions using the frozen model and policy
cd retrofit_starter/solution/final_model
python infer.py --model model.joblib --policy policy.json \
  --data "$QUAVIZ_CHALLENGE_DIR/data/full" --split test \
  --output "$QUAVIZ_CHALLENGE_DIR/../submissions/predictions_test.csv"

# Synchronize with the starter submission directory
cp "$QUAVIZ_CHALLENGE_DIR/../submissions/predictions_test.csv" \
   "$QUAVIZ_CHALLENGE_DIR/submission/predictions_test.csv"

# Run validation explainability and what-if simulation (requires GEMINI_API_KEY in .env)
python -m explainability.batch_validation \
  --provider gemini \
  --output-dir "$QUAVIZ_CHALLENGE_DIR/../submissions/03_whatif/results"
```
