#!/usr/bin/env bash
# One-click stack: MQTT broker + Quaviz API + publishers + frontend console.
# Default: all assets, full windows, steady pace (SPEED=60), Gemini explain, UI on :5173.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MQTT_HOST="${MQTT_HOST:-127.0.0.1}"
MQTT_PORT="${MQTT_PORT:-1883}"
# Optional: ASSET=PUMP-08 or ASSETS=PUMP-08,PUMP-09 — default = every asset in assets.csv
ASSET="${ASSET:-}"
ASSETS="${ASSETS:-}"
PROVIDER="${LLM_PROVIDER:-gemini}"
HTTP_PORT="${HTTP_PORT:-8000}"
UI_PORT="${UI_PORT:-5173}"
FRONTEND="${FRONTEND:-1}"          # 0 = API + MQTT only
OPEN_BROWSER="${OPEN_BROWSER:-1}"  # 0 = do not open the UI
# 0 = asap; 1 = real time; 60 = 1 sim-min / wall-sec (~2.8h for a full week)
SPEED="${SPEED:-30}"
START="${START:-}"
END="${END:-}"
INFER_EVERY="${INFER_EVERY:-5}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"

VENV_PY="$ROOT/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Missing $VENV_PY - create the root venv first (see README.md):" >&2
  echo "  uv venv --python 3.12 && source .venv/bin/activate" >&2
  echo "  uv pip install -r requirements.txt && uv pip install -e ./retrofit_starter --no-deps" >&2
  exit 1
fi

export QUAVIZ_CHALLENGE_DIR="${QUAVIZ_CHALLENGE_DIR:-$ROOT/retrofit_starter}"
export PYTHONPATH="${PYTHONPATH:-}"
export LLM_PROVIDER="$PROVIDER"
export VITE_API_BASE="http://127.0.0.1:${HTTP_PORT}"

BROKER_PID=""
BACKEND_PID=""
FRONTEND_PID=""
REPLAY_PIDS=()
STARTED_BROKER=0

cleanup() {
  trap - EXIT INT TERM
  echo
  echo "Stopping..."
  local pid
  for pid in "${REPLAY_PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  [[ -n "$FRONTEND_PID" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  if [[ "$STARTED_BROKER" -eq 1 && -n "$BROKER_PID" ]]; then
    kill "$BROKER_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

port_open() {
  local host="$1" port="$2"
  if command -v nc >/dev/null 2>&1; then
    nc -z "$host" "$port" >/dev/null 2>&1
  else
    "$VENV_PY" - "$host" "$port" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
s.settimeout(0.5)
try:
    s.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    s.close()
PY
  fi
}

mqtt_open() { port_open "$MQTT_HOST" "$MQTT_PORT"; }

ensure_broker() {
  if mqtt_open; then
    echo "MQTT broker already on $MQTT_HOST:$MQTT_PORT"
    return
  fi
  if ! command -v mosquitto >/dev/null 2>&1; then
    echo "No broker on $MQTT_HOST:$MQTT_PORT and mosquitto not found." >&2
    echo "Install: brew install mosquitto   (or start your own broker)" >&2
    exit 1
  fi
  echo "Starting mosquitto on :$MQTT_PORT..."
  mosquitto -c "$ROOT/scripts/mosquitto.conf" >/tmp/quaviz-mosquitto.log 2>&1 &
  BROKER_PID=$!
  STARTED_BROKER=1
  for _ in $(seq 1 20); do
    if mqtt_open; then
      echo "Broker ready (pid $BROKER_PID)"
      return
    fi
    sleep 0.25
  done
  echo "Broker failed to listen - see /tmp/quaviz-mosquitto.log" >&2
  exit 1
}

wait_for_health() {
  local url="http://127.0.0.1:${HTTP_PORT}/health"
  echo "Waiting for API at $url ..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      echo "Subscriber exited before becoming healthy." >&2
      exit 1
    fi
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "API ready."
      sleep 0.5
      return
    fi
    sleep 0.25
  done
  echo "Timed out waiting for $url" >&2
  exit 1
}

wait_for_ui() {
  local url="http://127.0.0.1:${UI_PORT}/"
  echo "Waiting for frontend at $url ..."
  for _ in $(seq 1 180); do
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
      echo "Frontend exited before becoming ready - see /tmp/quaviz-frontend.log" >&2
      exit 1
    fi
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "Frontend ready."
      return
    fi
    sleep 0.5
  done
  echo "Timed out waiting for $url - see /tmp/quaviz-frontend.log" >&2
  exit 1
}

ensure_frontend() {
  if [[ "$FRONTEND" != "1" ]]; then
    echo "Frontend skipped (FRONTEND=$FRONTEND)."
    return
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "npm not found — install Node.js or set FRONTEND=0." >&2
    exit 1
  fi
  local fe="$ROOT/app/frontend"
  if [[ ! -d "$fe/node_modules" ]]; then
    echo "Installing frontend dependencies (first run)..."
    (cd "$fe" && npm install) >/tmp/quaviz-frontend-install.log 2>&1 \
      || { echo "npm install failed — see /tmp/quaviz-frontend-install.log" >&2; exit 1; }
  fi
  if port_open 127.0.0.1 "$UI_PORT"; then
    echo "Something already on :$UI_PORT — reusing it (UI_PORT=$UI_PORT)."
    return
  fi
  echo "Starting frontend (http://127.0.0.1:$UI_PORT → API $VITE_API_BASE)..."
  (
    cd "$fe"
    exec npm run dev -- --host 127.0.0.1 --port "$UI_PORT" --strictPort
  ) >/tmp/quaviz-frontend.log 2>&1 &
  FRONTEND_PID=$!
  wait_for_ui
}

open_ui() {
  [[ "$FRONTEND" == "1" && "$OPEN_BROWSER" == "1" ]] || return 0
  local url="http://127.0.0.1:${UI_PORT}/"
  echo "Opening $url ..."
  if command -v open >/dev/null 2>&1; then
    open "$url" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "$url" >/dev/null 2>&1 || true
  fi
}

list_assets() {
  if [[ -n "$ASSET" ]]; then
    echo "$ASSET"
    return
  fi
  if [[ -n "$ASSETS" ]]; then
    echo "$ASSETS" | tr ',' '\n' | sed '/^$/d'
    return
  fi
  "$VENV_PY" - <<PY
import pandas as pd
path = "$ROOT/retrofit_starter/data/full/assets.csv"
for a in pd.read_csv(path)["asset_id"].astype(str):
    print(a)
PY
}

echo "=== QuaViz one-click demo ==="
echo "  API       http://127.0.0.1:$HTTP_PORT"
[[ "$FRONTEND" == "1" ]] && echo "  Frontend  http://127.0.0.1:$UI_PORT"
echo "  Provider  $PROVIDER   SPEED=$SPEED"
echo

ensure_broker

ASSET_LIST=()
while IFS= read -r aid; do
  [[ -n "$aid" ]] && ASSET_LIST+=("$aid")
done < <(list_assets)

if [[ ${#ASSET_LIST[@]} -eq 0 ]]; then
  echo "No assets to replay." >&2
  exit 1
fi

echo "Starting subscriber (API http://127.0.0.1:$HTTP_PORT, provider=$PROVIDER)..."
echo "  accepting assets: ${ASSET_LIST[*]}"
(
  cd "$ROOT/app/backend"
  exec "$VENV_PY" -m quaviz_backend serve \
    --mqtt-host "$MQTT_HOST" \
    --mqtt-port "$MQTT_PORT" \
    --host 127.0.0.1 \
    --port "$HTTP_PORT" \
    --infer-every "$INFER_EVERY" \
    --provider "$PROVIDER"
) &
BACKEND_PID=$!

wait_for_health
ensure_frontend
open_ui

# Rough ETA: full week is 604800 sim-seconds; wall ~= sim/SPEED when SPEED>0.
if [[ "$SPEED" != "0" ]]; then
  eta_min=$(awk -v s="$SPEED" 'BEGIN{printf "%.0f", 604800/s/60}')
  echo "Note: full 7-day windows at SPEED=$SPEED take ~${eta_min} wall-minutes per asset (publishers stay quiet until done)."
fi

echo "Starting publishers (n=${#ASSET_LIST[@]}, speed=$SPEED${START:+, start=$START}${END:+, end=$END})..."
mkdir -p /tmp/quaviz-replay
for aid in "${ASSET_LIST[@]}"; do
  (
    cd "$ROOT/retrofit_starter"
    args=(
      -m retrofit_sim replay
      --data data/full
      --asset "$aid"
      --mqtt-host "$MQTT_HOST"
      --mqtt-port "$MQTT_PORT"
      --speed "$SPEED"
    )
    [[ -n "$START" ]] && args+=(--start "$START")
    [[ -n "$END" ]] && args+=(--end "$END")
    exec "$VENV_PY" "${args[@]}"
  ) >/tmp/quaviz-replay/"$aid".log 2>&1 &
  REPLAY_PIDS+=($!)
  echo "  publisher $aid pid $!  (log /tmp/quaviz-replay/$aid.log)"
done

echo
echo "Running. Ctrl+C stops everything."
[[ "$FRONTEND" == "1" ]] && echo "  UI     http://127.0.0.1:$UI_PORT"
echo "  API    http://127.0.0.1:$HTTP_PORT/status"
echo "  Docs   http://127.0.0.1:$HTTP_PORT/docs"
echo "  Progress every ${PROGRESS_EVERY}s"
echo

publishers_alive() {
  local pid
  for pid in "${REPLAY_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

print_progress() {
  "$VENV_PY" - "$HTTP_PORT" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as r:
        d = json.load(r)
except Exception as e:
    print(f"  status unavailable: {e}")
    raise SystemExit(0)
assets = d.get("assets") or {}
msgs = sum(int(a.get("messages") or 0) for a in assets.values())
mins = sum(int(a.get("minutes_aggregated") or 0) for a in assets.values())
alerts = int(d.get("alert_count") or 0)
expl = int(d.get("explanation_count") or 0)
print(
    f"  uptime={d.get('uptime_s')}s assets={len(assets)} "
    f"msgs={msgs} minutes={mins} alerts={alerts} explanations={expl}"
)
PY
}

while publishers_alive; do
  print_progress || true
  sleep "$PROGRESS_EVERY"
done

for pid in "${REPLAY_PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done
echo "All replays finished — API/UI still up (Ctrl+C to stop)."
print_progress || true
wait "$BACKEND_PID"
