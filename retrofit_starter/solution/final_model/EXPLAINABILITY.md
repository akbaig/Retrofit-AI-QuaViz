# Operator explainability (design)

Layer on top of the frozen submission model (**best** / `astra-best`).  
Does not retrain the detector. Does not control the plant.

## Design principles

1. **Model decides.** Fault type and confidence come only from frozen `model.joblib` + `policy.json`.
2. **Features explain.** Numbers at alert time (residuals, physics departures, snapshot cues, operating mode) are packaged as structured evidence. The LLM does not invent sensors or diagnoses.
3. **LLM narrates.** Turns the explain packet into operator language: what looks wrong, what to check, what not to do, and stated uncertainty.
4. **Sandbox measures.** The recommended physical action is evaluated with `retrofit_sim.what_if` (simulated baseline vs proposed). Advice only — no write path to devices.
5. **Advice only.** Recommendations require a human. No PLC/drive/valve commands. Sensor faults map to inspect/recalibrate, not mechanical repairs.

## Pipeline (per alert)

```
best infer
  → alert (asset, time, fault_type, confidence)
  → explain packet (JSON: measured vs expected, top residuals/phys/snapshot, mode, sandbox_action)
  → LLM (system prompt + packet only)
       → summary, likely cause, checks, do-not-do, uncertainty
  → what_if(sandbox_action) when applicable
       → accepted/rejected, Δflow/power/vib/risk, illustrative cost
  → operator card (UI / notebook / demo)
```

Submission CSV format stays unchanged. Explainability artifacts are additive.

## Explain packet (contents)

At the minute the alert fires, from the same transformed features used by `make_alerts`:

| Field | Source |
|---|---|
| `asset_id`, `alert_time_utc`, `fault_type`, `event_class`, `confidence` | alert |
| `top_residuals` | largest `\|residual__*\|` (healthy-regressor gaps) |
| `physics` | key `phys__*` curve / suction / power departures when present |
| `snapshots` | BPFO/BPFI-style snapshot features when present |
| `context` | speed / running / duty when available |
| `recommended_action_text` | existing static `ACTIONS` string (human-facing default) |
| `sandbox_action` | mapped intervention for `what_if` (see below) |

Healthy / expected behaviour is whatever the healthy-response path already encodes (residuals and physics targets), not a second model invented for the LLM.

## Sandbox action map

`what_if` only accepts: `none`, `speed_setpoint`, `valve_position`, `inspect`, `clean_strainer`, `realign`, `replace_bearing`.

| `fault_type` | `sandbox_action` | Notes |
|---|---|---|
| `SUCTION_STRAINER_CLOGGING` | `clean_strainer` | Clears simulated clog |
| `BEARING_WEAR` | `replace_bearing` | Resets wear; small residual misalignment is normal in the sandbox |
| `MISALIGNMENT` | `realign` | Clears simulated misalignment |
| `CAVITATION` | `clean_strainer` or `inspect` | Prefer strainer clean when suction/clog evidence dominates; else inspect |
| `DISCHARGE_RESTRICTION` | `inspect` | No dedicated “clear restriction” intervention |
| `IMBALANCE` | `inspect` | No matching sandbox repair |
| `IMPELLER_WEAR` | `inspect` | No matching sandbox repair |
| `AIR_ENTRAINMENT` | `inspect` | No matching sandbox repair |
| `SENSOR_*` / `COMMUNICATION_LOSS` | `inspect` | Instrument / link check only |

`recommended_action` (text) and `sandbox_action` (API name) are separate fields.

## LLM contract

- **System prompt:** pump/cooling vocabulary, residual meaning, sensor vs machine, advice-only, never invent channels (`explainability/prompts.py`).
- **User input:** explain packet JSON only.
- **Structured output:** `summary`, `likely_cause`, `what_to_check[]`, `do_not_do[]`, `uncertainty_note`.
- **Fallback:** `LLM_PROVIDER=template` (or gemini init failure) fills the same schema without a network call.

The LLM must not override `fault_type` or invent a higher confidence than the model.

### Provider interface (swap without code changes)

```
explainability/llm/
  base.py       Narrative + LLM protocol
  factory.py    get_llm() from LLM_PROVIDER
  gemini.py     Vertex express mode (api_key + vertexai=True) — matches TS GoogleGenAI
  ollama.py     local Ollama /api/chat
  template.py   offline deterministic fallback
```

Env (repo-root `.env`, see `.env.example`):

| Variable | Role |
|---|---|
| `LLM_PROVIDER` | `gemini` (default) \| `ollama` \| `template` |
| `GEMINI_API_KEY` | Vertex express / Gemini API key |
| `GEMINI_MODEL` | default `gemini-3-flash-preview` |
| `GEMINI_VERTEXAI` | default `true` |
| `OLLAMA_BASE_URL` | default `http://127.0.0.1:11434` |
| `OLLAMA_MODEL` | default `llama3.2` |

```sh
cd solution/final_model
pip install -r explainability/requirements.txt
python -m explainability --asset PUMP-08 --provider gemini
python -m explainability --asset PUMP-08 --provider ollama
python -m explainability --asset PUMP-08 --provider template --no-what-if
```

## Out of scope (for this layer)

- Retraining or retuning `best`
- Calibrated probabilities (separate optional work)
- SHAP / tree importances as a hard dependency (residuals/physics are the primary evidence)
- Live MQTT control; replay may feed the same card later as a read-only demo
- Claiming certified safety or guaranteed savings

## Build order

1. Explain packet builder from frozen infer (one validation alert → JSON) — done
2. Wire `sandbox_action` → `what_if` and show before/after — done
3. LLM client + system prompt + fallback templates — done
4. Thin operator card (notebook or small UI) — CLI card via `python -m explainability`
5. Live ingest: `deploy/` buffers `retrofit_sim replay` → official `features_1min` → frozen best — done

```sh
cd solution/final_model
python -m deploy --asset PUMP-08 --start 2025-09-10T08:00:00Z --end 2025-09-10T11:00:00Z --infer-every 0
```

## Relation to challenge options

Supports: operator evidence + recommendation, sandbox what-if for the action, and (via packet fields) surface physics / snapshot checks already inside `best`.  
Does not by itself deliver: calibrated confidence, full replay live loop, or edge BOM (see `RETROFIT_FEASIBILITY.md` and `SOLUTION.md` §6 for those).
