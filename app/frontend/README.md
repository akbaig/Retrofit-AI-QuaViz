# Frontend

Operator console for industrial engineers — fleet health, alerts, confidence, recommended checks.

**Empty app shell.** Hand the AI build brief to an agent:

→ **[`PROMPT.md`](PROMPT.md)** — full product + API + acceptance spec

## Backend dependency

```sh
# repo root
./scripts/start_mqtt_demo.sh
# API: http://127.0.0.1:8000  docs: /docs  live: /ws/events
```

## After the UI is generated

```sh
cd app/frontend
npm install
npm run dev
```

Point the app at `VITE_API_BASE=http://127.0.0.1:8000`.
