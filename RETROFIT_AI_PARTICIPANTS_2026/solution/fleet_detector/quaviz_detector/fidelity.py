"""Do simulated fleets resemble the official pumps? Compares healthy operating levels, noise, event mix and
benign-window rates per profile/control group, and writes a markdown report.

    python -m quaviz_detector.fidelity --sim build/sim/s01* --out results/fidelity.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .data import ts
from .features import minute_frame
from .paths import RESULTS

CHANNELS = ["n", "i", "pw", "pd", "ps", "q", "vp", "tf", "bt", "wt", "v", "a", "c"]


def healthy_minutes(ds: data.Dataset) -> pd.DataFrame:
    """Steady running minutes outside every labeled event window (all events, observable or not)."""
    out = []
    ev = ds.events.copy()
    for aid in ds.ids():
        fr = minute_frame(ds, aid)
        bad = np.zeros(len(fr), dtype=bool)
        a = ds.asset(aid)
        we = ts(pd.Series([a["window_end_utc"]]))[0]
        for _, e in ev[ev["asset_id"] == aid].iterrows():
            s = ts(pd.Series([e["onset_utc"]]))[0]
            t = ts(pd.Series([e["end_utc"]]))[0] if pd.notna(e["end_utc"]) else we
            bad |= (fr.index >= s) & (fr.index <= t)
        fr = fr[fr["steady"] & ~bad].copy()
        fr["group"] = f"{a['profile']} / {a['control']}"
        out.append(fr)
    return pd.concat(out)


def level_table(off: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    """Each official pump's median level and noise against the spread of per-pump values in the simulated group."""
    rows = []
    per_pump = lambda df, col: df.groupby("asset_id")[col].median().dropna()
    for g in sorted(set(off["group"]) & set(sim["group"])):
        o, s = off[off["group"] == g], sim[sim["group"] == g]
        for k in CHANNELS:
            if k not in o or o[k].notna().sum() < 100 or k not in s:
                continue
            sl, sn = per_pump(s, k), per_pump(s, k + "_mstd")
            if len(sl) < 5:
                continue
            lo, hi = np.percentile(sl, [5, 95])
            nlo, nhi = np.percentile(sn, [5, 95]) if len(sn) >= 5 else (np.nan, np.nan)
            o_noise = per_pump(o, k + "_mstd")
            for aid, v in per_pump(o, k).items():
                nv = o_noise.get(aid, np.nan)      # NaN for sparse wireless channels (one sample per minute or less)
                rows.append({"group": g, "channel": k, "official_pump": aid, "official_median": v,
                             "sim_pumps": len(sl), "sim_p5": lo, "sim_p95": hi, "level_inside": bool(lo <= v <= hi),
                             "official_noise": nv, "sim_noise_p5": nlo, "sim_noise_p95": nhi,
                             "noise_inside": bool(nlo <= nv <= nhi) if np.isfinite(nv) and np.isfinite(nlo) else True})
    return pd.DataFrame(rows)


def event_table(off: data.Dataset, sims: list[data.Dataset]) -> pd.DataFrame:
    def counts(ds):
        e = ds.events
        days = len(ds.ids()) * 7.0
        g = e.groupby("fault_type")
        return pd.DataFrame({"n": g.size(), "observable": g["observable"].sum(), "per_100_asset_days": g.size() / days * 100})
    o = counts(off.subset(off.ids("train") + off.ids("validation")))
    s = counts(data.concat(sims, "sim"))
    return o.join(s, lsuffix="_official", rsuffix="_sim", how="outer").fillna(0).reset_index()


def timing_table(off: data.Dataset, sims: list[data.Dataset], near_h: float = 6.0) -> pd.DataFrame:
    """How fault onsets are spread over each pump's week. Repeated cavitation episodes of one incident are counted
    separately, because they are the simulator's own physics rather than a planning choice."""
    def stats(ds):
        rows = []
        e = ds.events.assign(s=ts(ds.events["onset_utc"]), t=ts(ds.events["end_utc"]))
        for aid, g in e.groupby("asset_id"):
            g = g.assign(t=g["t"].fillna(ts(pd.Series([ds.asset(aid)["window_end_utc"]]))[0])).reset_index(drop=True)
            for i, x in g[g["observable"]].iterrows():
                other = g.index != i
                near = other & ((g["s"] - x["s"]).abs() < pd.Timedelta(hours=near_h))
                near_cav = near & (g["fault_type"] == "CAVITATION") & (x["fault_type"] == "CAVITATION")
                rows.append({"overlap": bool((other & (g["s"] < x["t"]) & (g["t"] > x["s"])).any()),
                             "near": bool(near.any()), "near_excl_cav": bool((near & ~near_cav).any()),
                             "pre": bool(x["pre_existing"])})
        r = pd.DataFrame(rows).mean()
        return {"events_per_pump_week": len(e) / len(ds.ids()), "overlapping_another": r["overlap"],
                f"other_onset_within_{near_h:g}h": r["near"], f"other_onset_within_{near_h:g}h_excl_cavitation_repeats": r["near_excl_cav"],
                "pre_existing": r["pre"]}
    labeled = off.subset(off.ids("train") + off.ids("validation"))
    return pd.DataFrame({"official": stats(labeled), "simulated": stats(data.concat(sims, "sim"))}).reset_index(names="measure")


def benign_table(off: data.Dataset, sims: list[data.Dataset]) -> pd.DataFrame:
    def rate(ds):
        return ds.benign.groupby("benign_type").size() / (len(ds.ids()) * 7.0)
    labeled = off.subset(off.ids("train") + off.ids("validation"))
    return pd.DataFrame({"official_per_asset_day": rate(labeled), "sim_per_asset_day": rate(data.concat(sims, "sim"))}).fillna(0).reset_index()


def _md(df: pd.DataFrame) -> str:
    fmt = lambda v: f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(lines + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m quaviz.fidelity")
    p.add_argument("--sim", nargs="+", required=True)
    p.add_argument("--out", default=str(RESULTS / "fidelity.md"))
    a = p.parse_args(argv)
    off = data.load()
    labeled = off.subset(off.ids("train") + off.ids("validation"))
    sims = [data.load(x) for x in a.sim]
    hm_off = healthy_minutes(labeled)
    hm_sim = pd.concat([healthy_minutes(s) for s in sims])
    lv = level_table(hm_off, hm_sim)
    ev, bn, tm = event_table(off, sims), benign_table(off, sims), timing_table(off, sims)
    text = ["# Simulated fleets vs official pumps", "",
            f"Simulated fleets: {len(sims)} ({sum(len(s.ids()) for s in sims)} pumps). Official labeled pumps: {len(labeled.ids())}.", "",
            f"Healthy steady operation, per official pump and channel: the official median level lies inside the 5-95 % "
            f"range of simulated pumps of the same profile/control for {lv['level_inside'].mean():.0%} of {len(lv)} cases; "
            f"the official noise level (median 1-min std) for {lv['noise_inside'].mean():.0%}.", "",
            "## Healthy steady operation (nameplate-scaled values)", "", _md(lv), "",
            "## Events per fault type", "", _md(ev), "",
            "## Event timing (shares of scored events)", "", _md(tm), "",
            "## Benign windows per asset-day", "", _md(bn), ""]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text[:6]))
    mis = lv[~lv["level_inside"] | ~lv["noise_inside"]]
    print("\nOfficial pump/channel outside the simulated range:\n" + (mis.round(3).to_string(index=False) if len(mis) else "none"))
    print("\n" + ev.round(2).to_string(index=False))
    print("\n" + tm.round(3).to_string(index=False))
    print("\n" + bn.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
