# Retrofit AI / QuaViz — RETROFIT AI 2026

Team solution for the Since AI 2026 / Business Turku pump-fault challenge.

```
Retrofit-AI-QuaViz/
  .venv/                local Python env (create at repo root with uv)
  requirements.txt      kit + frozen best + backend
  retrofit_starter/     challenge kit + frozen best + research
  app/
    backend/            MQTT → aggregate → best → explain → API
    frontend/           operator UI (empty for now)
```

**Start here:** [`retrofit_starter/SOLUTION.md`](retrofit_starter/SOLUTION.md)

| What you need | Where |
|---|---|
| Final model (frozen) | [`retrofit_starter/solution/final_model/`](retrofit_starter/solution/final_model/) |
| Test-pump submission | [`retrofit_starter/submission/predictions_test.csv`](retrofit_starter/submission/predictions_test.csv) |
| Challenge kit | [`retrofit_starter/`](retrofit_starter/) — see `README_FIRST.md` |
| Backend | [`app/backend/`](app/backend/) |
| Frontend | [`app/frontend/`](app/frontend/) — build from [`PROMPT.md`](app/frontend/PROMPT.md) |

## Setup (once)

Use a **repo-root** `.venv` via [uv](https://docs.astral.sh/uv/). Do not create the venv inside `retrofit_starter/` — absolute paths break if that folder is renamed.

Install uv if needed: `curl -LsSf https://astral.sh/uv/install.sh | sh`

```sh
cd Retrofit-AI-QuaViz

# Prefer 3.12 (matches frozen model pins); 3.13 also works
uv venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

uv pip install -r requirements.txt
uv pip install -e ./retrofit_starter --no-deps   # enables `python -m retrofit_sim`

cp .env.example .env               # optional: GEMINI_API_KEY / LLM_PROVIDER
```

Check you are on the repo venv:

```sh
which python    # .../Retrofit-AI-QuaViz/.venv/bin/python
python -c "import sys; print(sys.prefix)"
```

## Backend (MQTT)

One-shot demo (starts mosquitto if needed, then subscriber + publishers for **all** assets):

```sh
./scripts/start_mqtt_demo.sh
```

Defaults: every pump in `assets.csv`, full time windows, `SPEED=60` (steady; ~2.8h for a full week), `LLM_PROVIDER=gemini`. Progress prints from `/status` every few seconds.

Optional env overrides: `ASSET` / `ASSETS`, `START`, `END`, `SPEED` (`1` = real time, `0` = asap), `HTTP_PORT`, `LLM_PROVIDER`, `INFER_EVERY`, `PROGRESS_EVERY`.

Shorter smoke window example:

```sh
START=2025-09-10T08:00:00Z END=2025-09-10T11:00:00Z ASSET=PUMP-08 ./scripts/start_mqtt_demo.sh
```


Manual two-terminal flow:

```sh
source .venv/bin/activate
export QUAVIZ_CHALLENGE_DIR="$(pwd)/retrofit_starter"

# terminal A
cd app/backend && python -m quaviz_backend serve --mqtt-host 127.0.0.1 --provider template

# terminal B
cd retrofit_starter && python -m retrofit_sim replay \
  --data data/full --asset PUMP-08 --mqtt-host 127.0.0.1 --speed 0
```

No broker? Pipe stdin instead (see [`app/backend/README.md`](app/backend/README.md)).

Synthetic data only. Advice only. No write path to any device.
