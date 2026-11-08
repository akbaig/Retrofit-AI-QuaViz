"""Fleet operating-context snapshot for the operator dashboard."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class ContextCache:
    """Latest operating_context rows keyed by asset (as-of last telemetry time)."""

    KEYS = (
        "schedule_state",
        "duty_demand_pct",
        "supply_water_temp_c",
        "sea_temp_c",
        "ambient_temp_c",
        "sea_state_hs_m",
        "suction_static_head_m",
        "controlled_setpoint",
    )

    def __init__(self, data_dir: Path):
        path = Path(data_dir) / "operating_context.csv"
        self.path = path
        self._by_asset: dict[str, pd.DataFrame] = {}
        if path.exists():
            df = pd.read_csv(path)
            if "timestamp_utc" in df.columns:
                df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
                df = df.dropna(subset=["timestamp_utc"]).sort_values("timestamp_utc")
            for aid, g in df.groupby("asset_id", sort=False):
                self._by_asset[str(aid)] = g.reset_index(drop=True)

    def _row_asof(self, asset_id: str, as_of: str | None) -> dict[str, Any] | None:
        g = self._by_asset.get(asset_id)
        if g is None or g.empty:
            return None
        if as_of:
            t = pd.Timestamp(as_of)
            if t.tzinfo is None:
                t = t.tz_localize("UTC")
            sub = g[g["timestamp_utc"] <= t]
            row = sub.iloc[-1] if not sub.empty else g.iloc[0]
        else:
            row = g.iloc[-1]
        out: dict[str, Any] = {"asset_id": asset_id}
        ts = row.get("timestamp_utc")
        out["timestamp_utc"] = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        for k in self.KEYS:
            if k not in row.index:
                continue
            v = row[k]
            if pd.isna(v):
                out[k] = None
            elif hasattr(v, "item"):
                try:
                    out[k] = v.item()
                except Exception:
                    out[k] = v
            else:
                out[k] = v
        return out

    def snapshot(self, assets: dict[str, Any]) -> dict[str, Any]:
        """assets: asset_id → status dict with optional last_timestamp_utc."""
        per: list[dict[str, Any]] = []
        for aid, st in assets.items():
            as_of = None
            if isinstance(st, dict):
                as_of = st.get("last_timestamp_utc")
            else:
                as_of = getattr(st, "last_timestamp_utc", None)
            row = self._row_asof(str(aid), str(as_of) if as_of else None)
            if row:
                per.append(row)

        def _mean(key: str) -> float | None:
            vals = [float(r[key]) for r in per if r.get(key) is not None]
            if not vals:
                return None
            return round(sum(vals) / len(vals), 2)

        schedules = [r["schedule_state"] for r in per if r.get("schedule_state")]
        schedule = max(set(schedules), key=schedules.count) if schedules else None

        return {
            "ok": bool(self._by_asset),
            "source": str(self.path) if self.path.exists() else None,
            "fleet": {
                "n_assets": len(per),
                "schedule_state": schedule,
                "duty_demand_pct": _mean("duty_demand_pct"),
                "sea_temp_c": _mean("sea_temp_c"),
                "supply_water_temp_c": _mean("supply_water_temp_c"),
                "ambient_temp_c": _mean("ambient_temp_c"),
                "sea_state_hs_m": _mean("sea_state_hs_m"),
            },
            "assets": {r["asset_id"]: r for r in per},
        }
