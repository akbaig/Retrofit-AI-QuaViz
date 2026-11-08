# Backend

Product service: subscribe → aggregate 1 s → 1 min → frozen **best** → explain → HTTP/WS for `../frontend`.

```
MQTT / retrofit_sim replay (publisher only)
        │  topic: retrofit/<site>/<asset>/telemetry
        ▼
app/backend (quaviz_backend)
  subscribe → MinuteBuffer → features_1min
  → OnlineBest (model.joblib + policy + assets/context/snapshots)
  → explainability (Gemini / Ollama / template)
  → REST + /ws/events
```

## Depends on

| Path | Role |
|---|---|
| `../../retrofit_starter/solution/final_model/` | `model.joblib`, `policy.json`, `infer`, `deploy`, `explainability` |
| `../../retrofit_starter/data/full/` | assets, operating_context, vibration_snapshots |
| `../../retrofit_starter/src/retrofit_sim` | `features_1min`, optional `what_if` / replay publisher |

`QUAVIZ_CHALLENGE_DIR` defaults to `../../retrofit_starter` when unset.

## Setup

Use the **repo-root** venv (see root [`README.md`](../../README.md)):

```sh
cd ../..   # Retrofit-AI-QuaViz/
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e ./retrofit_starter --no-deps
cp .env.example .env   # GEMINI_API_KEY / LLM_PROVIDER if needed
```


## Run

**MQTT (live):** start a broker, then:

```sh
cd app/backend
python -m quaviz_backend serve --mqtt-host 127.0.0.1 --port 8000 --provider template
```

In another shell, publish with the kit replay tool:

```sh
# from repo root, with .venv activated
cd retrofit_starter
python -m retrofit_sim replay \
  --data data/full --asset PUMP-08 \
  --start 2025-09-10T08:00:00Z --end 2025-09-10T11:00:00Z \
  --mqtt-host 127.0.0.1 --speed 0
```

**Stdin pipe (no broker):**

```sh
cd retrofit_starter
python -m retrofit_sim replay \
  --data data/full --asset PUMP-08 \
  --start 2025-09-10T08:00:00Z --end 2025-09-10T11:00:00Z --limit 12000 \
| (cd ../app/backend && python -m quaviz_backend stdin --asset PUMP-08 --infer-every 0 --provider template)
```


## API

| Method | Path | |
|---|---|---|
| GET | `/health` | model/data paths |
| GET | `/context` | Fleet operating context (sea temp, duty, schedule, …) |
| GET | `/status` | per-asset progress |
| GET | `/assets` | message/minute/alert counts |
| GET | `/alerts?asset_id=` | recent alerts |
| GET | `/explanations?asset_id=` | narratives + packets |
| POST | `/score/{asset_id}` | force re-infer |
| POST | `/flush` | close open minutes + score |
| WS | `/ws/events` | live alert/explain JSON |

## Env

See repo-root `.env.example`. Backend-specific:

| Var | Default | |
|---|---|---|
| `MQTT_HOST` | unset | broker; required for live subscribe |
| `MQTT_PORT` | `1883` | |
| `MQTT_TOPIC` | `retrofit/+/+/telemetry` | |
| `HTTP_HOST` / `HTTP_PORT` | `0.0.0.0` / `8000` | |
| `INFER_EVERY` | `5` | minutes between online scores (`0` = flush-only) |
| `EXPLAIN` | `true` | narrate each new alert |
| `EXPLAIN_WHAT_IF` | `false` | sandbox measure (heavier) |
| `LLM_PROVIDER` | `gemini` | `gemini` \| `ollama` \| `template` |
| `QUAVIZ_ASSETS` | all known | comma-separated allow-list |
