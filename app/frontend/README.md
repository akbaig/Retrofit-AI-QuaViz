# QuaViz Fleet Console

Live operator UI for the Retrofit AI / QuaViz pump-fault backend. Read-only — never commands drives, valves, or setpoints.

Product brief (design reference): [`PROMPT.md`](PROMPT.md).

## Run

**One-click (recommended)** from the repo root — broker, API, publishers, and this UI:

```sh
./start.sh
# UI http://127.0.0.1:5173  ·  API http://127.0.0.1:8000
```

Frontend-only (API already running):

```sh
cd app/frontend
cp -n .env.example .env   # optional; default is already 127.0.0.1:8000
npm install
npm run dev
```

Override the backend with `VITE_API_BASE` or click the URL in the top strip.

If the backend is unreachable, use **Run in-browser demo replay** (labelled DEMO).

## Layout

- **Top:** fleet context — schedule, duty, sea / supply / ambient temps, sea state  
- **Left:** system stats (backend, stream, msgs, alerts, uptime, API URL, demo)  
- **Center:** pump fleet wall  
- **Right:** live event log  
- **Click a pump:** detail side pane (Esc / backdrop to close)

## What it wires

| Backend | Frontend |
|---|---|
| `GET /health` `/status` `/assets` `/alerts` `/explanations` `/context` | Bootstrap on connect + re-bootstrap when recovering from offline |
| `WS /ws/events` | Live alert / explain merge into the store + event log |
| Alert / explain events | In-UI toasts + browser notifications (if permitted) |
| `POST /score/{asset}` | Optional re-infer from asset detail |

Status colours: red = confidence ≥ 0.8 · amber = low-confidence / data without score / backend down · green = scored, no alerts · grey = never seen.
