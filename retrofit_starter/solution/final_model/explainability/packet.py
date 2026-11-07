"""Build structured explain packets from frozen best inference features."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import json
import pandas as pd

from alerts import ACTIONS, AlertPolicy, make_alerts
from features import build_features
from infer import observed_horizon
from learning import SENSORS

from .sandbox_map import sandbox_action_for


def _finite_series(row: pd.Series, prefix: str) -> pd.Series:
    s = row.filter(like=prefix).dropna()
    return s[pd.to_numeric(s, errors="coerce").notna()].astype(float)


def _top_abs(series: pd.Series, k: int = 5) -> list[dict[str, Any]]:
    if series.empty:
        return []
    idx = series.abs().nlargest(k).index
    return [{"name": str(name), "value": float(series[name])} for name in idx]


def _context_bits(row: pd.Series, meta_row: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ("steady", "running", "available_utc", "window_start_utc"):
        if key in meta_row.index and pd.notna(meta_row[key]):
            val = meta_row[key]
            out[key] = val.isoformat() if hasattr(val, "isoformat") else (bool(val) if key in ("steady", "running") else val)
    for col in row.index:
        name = str(col)
        if name.startswith(("ctx__", "obs__", "rating__")) or name in (
            "motor_speed_rpm", "flow_m3_h", "discharge_pressure_kpa", "suction_pressure_kpa",
            "motor_power_kw", "motor_current_a", "vibration_rms_mm_s",
        ):
            if pd.notna(row[col]):
                try:
                    out[name] = float(row[col])
                except (TypeError, ValueError):
                    out[name] = str(row[col])
    return out


def build_explain_packet(
    alert: pd.Series | dict[str, Any],
    feature_row: pd.Series,
    meta_row: pd.Series | None = None,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Assemble the JSON packet for LLM / UI from one alert + its feature row."""
    a = alert if isinstance(alert, dict) else alert.to_dict()
    fault = str(a["fault_type"])
    residuals = _top_abs(_finite_series(feature_row, "residual__"), top_k)
    physics = _top_abs(_finite_series(feature_row, "phys__"), top_k)
    snaps = _top_abs(_finite_series(feature_row, "snapshot__"), min(top_k, 4))
    meta_row = meta_row if meta_row is not None else pd.Series(dtype=object)
    packet: dict[str, Any] = {
        "asset_id": a["asset_id"],
        "alert_time_utc": a["alert_time_utc"],
        "fault_type": fault,
        "event_class": a.get("event_class") or ("SENSOR" if fault in SENSORS else "MACHINE"),
        "confidence": float(a["confidence"]),
        "evidence": a.get("evidence"),
        "recommended_action_text": a.get("recommended_action") or ACTIONS.get(fault, ""),
        "top_residuals": residuals,
        "physics": physics,
        "snapshots": snaps,
        "context": _context_bits(feature_row, meta_row),
        "normal_behaviour_note": (
            "residual__* = measured − healthy expected at this operating point; "
            "phys__* = pump-curve / thermal / power departures."
        ),
    }
    packet["sandbox_action"] = sandbox_action_for(fault, packet)
    return packet


def explain_alert(
    *,
    model,
    policy: AlertPolicy,
    features: pd.DataFrame,
    asset: pd.Series,
    context: pd.DataFrame | None = None,
    snapshots: pd.DataFrame | None = None,
    until=None,
    alert_index: int = 0,
) -> dict[str, Any]:
    """Run frozen infer for one asset and return packet for the selected alert."""
    horizon = pd.Timestamp(until) if until is not None else observed_horizon(features, asset, context, snapshots)
    meta, x = build_features(features, asset, context, snapshots, until=horizon)
    transformed = model.transform(x, meta)
    probabilities = (
        model.probabilities_from_transformed(transformed, meta)
        if getattr(model, "is_composite", False)
        else model.probabilities_from_transformed(transformed)
    )
    alerts = make_alerts(meta, probabilities, policy, x=transformed)
    if len(alerts):
        t = pd.to_datetime(alerts.alert_time_utc, utc=True)
        alerts = alerts[t.lt(pd.Timestamp(asset.window_end_utc)) & t.le(horizon)].reset_index(drop=True)
    if alerts.empty:
        raise ValueError(f"No alerts for {asset.asset_id} in the observed window")
    if alert_index < 0 or alert_index >= len(alerts):
        raise IndexError(f"alert_index {alert_index} out of range (n={len(alerts)})")
    alert = alerts.iloc[alert_index]
    # Match alert time to the meta row used for evidence (available_utc).
    alert_ts = pd.Timestamp(alert.alert_time_utc)
    times = pd.to_datetime(meta.available_utc, utc=True)
    idx = int((times - alert_ts).abs().to_numpy().argmin())
    packet = build_explain_packet(alert, transformed.iloc[idx], meta.iloc[idx])
    packet["n_alerts_on_asset"] = int(len(alerts))
    packet["alert_index"] = int(alert_index)
    return {"alert": alert.to_dict(), "packet": packet, "alerts": alerts}


def load_frozen(model_path: Path, policy_path: Path):
    model = joblib.load(model_path)
    policy = AlertPolicy(**json.loads(Path(policy_path).read_text()))
    return model, policy
