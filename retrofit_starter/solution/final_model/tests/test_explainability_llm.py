"""Unit tests for explainability LLM plumbing (no network)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

FINAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINAL))

from explainability.llm.parse import parse_narrative  # noqa: E402
from explainability.llm.template import TemplateLLM  # noqa: E402
from explainability.packet import build_explain_packet  # noqa: E402
from explainability.sandbox_map import sandbox_action_for  # noqa: E402


def test_sandbox_map_basic():
    assert sandbox_action_for("BEARING_WEAR") == "replace_bearing"
    assert sandbox_action_for("SENSOR_DRIFT") == "inspect"
    pkt = {"top_residuals": [{"name": "residual__suction_pressure_kpa", "value": -2.0}]}
    assert sandbox_action_for("CAVITATION", pkt) == "clean_strainer"


def test_build_packet_and_template_narrative():
    alert = {
        "asset_id": "PUMP-08",
        "alert_time_utc": "2025-01-01T00:00:00Z",
        "fault_type": "BEARING_WEAR",
        "event_class": "MACHINE",
        "confidence": 0.91,
        "evidence": "learned BEARING_WEAR score=0.910",
        "recommended_action": "Inspect bearing",
    }
    row = pd.Series({
        "residual__vibration_rms_mm_s": 2.5,
        "residual__bearing_temp_c": 1.2,
        "phys__head_curve_residual": -0.1,
        "snapshot__envelope_bpfo_fraction": 0.4,
    })
    packet = build_explain_packet(alert, row)
    assert packet["sandbox_action"] == "replace_bearing"
    assert packet["top_residuals"][0]["name"] == "residual__vibration_rms_mm_s"
    narrative = TemplateLLM().narrate("sys", packet)
    assert narrative.fault_type == "BEARING_WEAR"
    assert "BEARING_WEAR" in narrative.summary


def test_parse_narrative_fenced_json():
    packet = {"fault_type": "IMBALANCE"}
    text = '```json\n{"summary":"s","likely_cause":"c","what_to_check":["a"],"do_not_do":["b"],"uncertainty_note":"u"}\n```'
    n = parse_narrative(text, packet, "gemini", "m")
    assert n.summary == "s" and n.fault_type == "IMBALANCE"
    assert json.loads(json.dumps(n.to_dict()))["what_to_check"] == ["a"]
