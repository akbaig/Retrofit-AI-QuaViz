# AI prompt — build the QuaViz operator frontend

Copy everything below the line into a coding agent. Build an **engineering-ready** UI for industrial / marine cooling-pump operators. Advice only — no plant write path.

---

## Mission

Build `app/frontend` as a live operator console for the Retrofit AI / QuaViz pump-fault assistant.

Audience: industrial engineers and plant/vessel operators who need to see **fleet health at a glance**, then drill into **what fault**, **how confident**, **why**, and **what to check next**.

The backend already runs. Do not reimplement detection. Consume its HTTP + WebSocket API.

## Product feel

Industrial operations console — dense, readable, calm under alarm — not a marketing landing page and not a generic SaaS dashboard collage.

- One clear composition for the first viewport: **fleet wall** of all pumps.
- Every pump is a status tile: **green** = healthy / no open alert, **amber** = degraded / low-confidence or recovering, **red** = active fault alert.
- Selecting a pump opens detail: latest alerts, confidence, recommended action, LLM narrative + evidence when present.
- Live updates as MQTT replay / production stream feeds the backend.
- Synthetic demo data; never imply the UI can command drives, valves, or setpoints.

## Visual direction

- Dark industrial control-room palette (steel / graphite / muted cyan accent). Avoid purple-on-white, cream+terracotta, and neon glow stacks.
- Expressive but legible typography (not Inter/Roboto/Arial defaults).
- Status color is the primary signal (green / amber / red). Do not bury it in icons alone.
- Motion: subtle — tile color transitions, soft pulse on new red alerts, list insert for new events. No decorative noise.
- Responsive: fleet wall usable on laptop; detail readable on tablet. Mobile is secondary.

## Backend contract (already implemented)

Base URL default: `http://127.0.0.1:8000` (configurable via env, e.g. `VITE_API_BASE`).

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + model/data paths |
| GET | `/status` | `uptime_s`, per-asset counters, `alert_count`, `explanation_count` |
| GET | `/assets` | List of asset status objects |
| GET | `/alerts?asset_id=&limit=` | Recent alerts (newest first) |
| GET | `/explanations?asset_id=&limit=` | Narratives + explain packets |
| POST | `/score/{asset_id}` | Force re-infer (optional button) |
| POST | `/flush` | Flush open minutes + score (optional / advanced) |
| WS | `/ws/events` | Live JSON events: `{type:"alert"| "explain"| "error", ...}` |

### Asset status object (`/assets`, `/status.assets`)

```json
{
  "asset_id": "PUMP-08",
  "messages": 1173,
  "minutes_aggregated": 19,
  "feature_rows": 19,
  "alerts": 1,
  "last_timestamp_utc": "2025-09-10T09:52:00+00:00",
  "last_infer_utc": "..."
}
```

### Alert object

```json
{
  "id": "uuid",
  "asset_id": "PUMP-08",
  "alert_time_utc": "2025-09-10T09:52:00+00:00",
  "event_class": "MACHINE",
  "fault_type": "BEARING_WEAR",
  "confidence": 0.987,
  "evidence": "...",
  "recommended_action": "Inspect bearing lubrication and condition; confirm with vibration measurements.",
  "received_at": 1731.0
}
```

### Explanation object (when LLM / template runs)

```json
{
  "id": "uuid",
  "asset_id": "PUMP-08",
  "alert_id": "...",
  "alert_time_utc": "...",
  "fault_type": "BEARING_WEAR",
  "narrative": { "summary": "...", "fault_type": "BEARING_WEAR", "...": "..." },
  "packet": {
    "confidence": 0.987,
    "recommended_action_text": "...",
    "top_residuals": [{"name": "...", "value": 0.1}],
    "physics": [],
    "snapshots": [],
    "context": {},
    "sandbox_action": "inspect"
  },
  "sandbox_action": "inspect",
  "provider": "GeminiLLM",
  "what_if": null
}
```

WebSocket event shapes: same payloads with `"type": "alert" | "explain" | "error"`.

## Required screens / regions

### 1. Fleet wall (home)

- Grid of **all known pumps** (typically PUMP-01 … PUMP-12). Prefer union of `/assets` + any ids seen in alerts.
- Each tile shows at minimum:
  - `asset_id`
  - status color (see rules below)
  - latest `fault_type` if red/amber (else “OK” / “Listening”)
  - confidence % when an active alert exists
  - relative time of last alert or last telemetry
- Click / keyboard focus selects the pump → detail panel or route.

**Status color rules (implement exactly):**

| Color | Condition |
|---|---|
| Green | No alerts for this asset in the current session store, or last alert recovered / cleared (if you only have counts: `alerts == 0` and no items in `/alerts?asset_id=`) |
| Red | At least one alert with `confidence >= 0.8` (or any alert if you cannot filter) for that asset |
| Amber | Asset is receiving data (`messages` growing) but no alert yet after infer, **or** alert with `confidence < 0.8`, **or** backend unreachable for that tile |
| Grey / muted | Never seen (`messages == 0`) |

When a new red alert arrives over WS, the matching tile must turn red within ~1s and optionally pulse once.

### 2. Alert stream (fleet or filtered)

- Chronological feed of alerts (from REST bootstrap + WS append).
- Columns/fields: time, asset, fault_type, event_class, confidence, short recommended_action.
- Clicking a row selects that asset and scrolls detail to that alert.

### 3. Asset detail

For the selected pump:

- Header: id, status color, message/minute counters from `/status`.
- **Active alert card**: fault_type, confidence meter/bar, event_class, recommended_action (primary CTA text — human check only).
- **Why**: evidence string; if explanation exists, show narrative summary + top residuals / physics from `packet` (compact tables, not raw JSON dumps).
- **What to do**: emphasize `recommended_action` / `recommended_action_text`; show `sandbox_action` as advisory label only (“sandbox measure: inspect”) — never as an execute button that implies plant control.
- Empty state: “No alerts yet — listening to telemetry.”

### 4. Connection / system strip

- Backend health (`/health`): connected / degraded.
- WS connected indicator.
- Optional: total msgs, alert_count, explanation_count from `/status`.
- Link or note: run backend via `./scripts/start_mqtt_demo.sh` (repo root README).

## Technical requirements

- **Stack:** Vite + React + TypeScript (strict). Prefer modern React; no unnecessary memoization.
- **Data:** 
  - On load: `GET /assets`, `GET /alerts`, `GET /explanations`, `GET /status`.
  - Subscribe `WS /ws/events`; merge into client store (do not require full REST refresh for each event).
  - Poll `/status` every 5–10s as a fallback if WS drops.
- **Config:** `VITE_API_BASE` (http) and derive WS URL (`http`→`ws`, `https`→`wss`) + `/ws/events`.
- **CORS:** backend already allows `*`.
- **Quality:** TypeScript types for API payloads; loading / empty / error states; keyboard-accessible tiles; no crash on malformed WS messages.
- **Do not** call Gemini or load `model.joblib` from the frontend. Do not add MQTT in the browser.
- **Do not** invent write APIs to equipment.

## Suggested file layout

```
app/frontend/
  PROMPT.md          ← this file
  README.md          ← how to run
  package.json
  vite.config.ts
  index.html
  src/
    main.tsx
    App.tsx
    api/client.ts      # REST + WS helpers
    api/types.ts
    state/store.ts     # alerts, explanations, assets
    components/
      FleetWall.tsx
      PumpTile.tsx
      AlertStream.tsx
      AssetDetail.tsx
      SystemStrip.tsx
    styles/
      tokens.css
      app.css
```

## Acceptance checklist

- [ ] All 12 pumps visible on one fleet wall without horizontal chaos on a 1440px laptop
- [ ] Healthy pumps green; injecting/receiving an alert turns that pump red
- [ ] Confidence and recommended action visible without opening a modal maze
- [ ] New alerts appear live via WebSocket while demo script runs
- [ ] Explanation narrative shows when backend has produced one
- [ ] No control/command affordances that suggest writing to the plant
- [ ] `README.md` documents `npm install && npm run dev` and required backend URL

## How to run the backend while developing

From repo root (separate terminal):

```sh
source .venv/bin/activate
./scripts/start_mqtt_demo.sh
# or short window:
START=2025-09-10T08:00:00Z END=2025-09-10T11:00:00Z ASSET=PUMP-08 SPEED=0 ./scripts/start_mqtt_demo.sh
```

API docs: `http://127.0.0.1:8000/docs`

## Out of scope

- Retraining models, challenge scoring UI, editing `policy.json`
- Auth / multi-user accounts (single operator workstation is fine)
- 3D pump models, map GIS, heavy BI chart walls

Build the smallest complete console that an engineer would trust on a shift.
