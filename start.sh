#!/usr/bin/env bash
# One-click QuaViz demo (broker + API + MQTT replay + frontend).
exec "$(cd "$(dirname "$0")" && pwd)/scripts/start_mqtt_demo.sh" "$@"
