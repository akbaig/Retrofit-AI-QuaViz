#!/usr/bin/env bash
# Start MQTT broker (if needed) + Quaviz subscriber + retrofit_sim publisher(s).
# Default: all assets, full windows, steady pace (SPEED=60), Gemini explain.
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
# 0 = asap; 1 = real time; 60 = 1 sim-min / wall-sec (~2.8h for a full week)
SPEED="${SPEED:-60}"
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

BROKER_PID=""
BACKEND_PID=""
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
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
  if [[ "$STARTED_BROKER" -eq 1 && -n "$BROKER_PID" ]]; then
    kill "$BROKER_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

port_open() {
  if command -v nc >/dev/null 2>&1; then
    nc -z "$MQTT_HOST" "$MQTT_PORT" >/dev/null 2>&1
  else
    "$VENV_PY" - "$MQTT_HOST" "$MQTT_PORT" <<'PY'
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

ensure_broker() {
  if port_open; then
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
    if port_open; then
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
  echo "Waiting for subscriber at $url ..."
  for _ in $(seq 1 120); do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      echo "Subscriber exited before becoming healthy." >&2
      exit 1
    fi
    if curl -sf "$url" >/dev/null 2>&1; then
      echo "Subscriber ready."
      sleep 0.5
      return
    fi
    sleep 0.25
  done
  echo "Timed out waiting for $url" >&2
  exit 1
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
  # No --asset: hub accepts any known asset from assets.csv
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
echo "Running. Ctrl+C to stop. Progress every ${PROGRESS_EVERY}s from /status."
echo "  GET http://127.0.0.1:$HTTP_PORT/status"
echo "  GET http://127.0.0.1:$HTTP_PORT/alerts"
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

# Progress ticker while any publisher is still running.
while publishers_alive; do
  print_progress || true
  sleep "$PROGRESS_EVERY"
done

# Reap children.
for pid in "${REPLAY_PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done
echo "All replays finished - API still up (Ctrl+C to stop)."
print_progress || true
wait "$BACKEND_PID"
