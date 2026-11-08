# QuaViz Fleet Console (frontend)

Live operator console for the Retrofit AI / QuaViz pump-fault backend. Read-only: it never commands drives, valves or setpoints.

## Run

```sh
npm install && npm run dev
```

Backend URL defaults to `http://127.0.0.1:8000`. Override with `VITE_API_BASE=http://host:8000` in `.env`, or click the URL in the top strip to change it at runtime.
Start the backend from the repo root: `source .venv/bin/activate && ./scripts/start_mqtt_demo.sh`.

If the backend is unreachable, the console offers an in-browser synthetic demo replay (clearly labelled DEMO).

Status colours: red = alert with confidence ≥ 0.8 · amber = low-confidence alert, data arriving but not yet scored, or backend unreachable · green = scored, no alerts · grey = never seen.
