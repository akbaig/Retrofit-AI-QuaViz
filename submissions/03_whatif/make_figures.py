"""Paper figures for validation what-if chapter."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
RESULTS = HERE / "results" / "whatif_validation.csv"


def acceptance_bars(df: pd.DataFrame) -> None:
    n = len(df)
    n_acc = int(df["accepted"].astype(bool).sum())
    n_fix = int(df["fixed"].astype(bool).sum())
    labels = ["Accepted", "Fixed\n(accepted + Δrisk<0)", "Not fixed"]
    vals = [n_acc, n_fix, n - n_fix]
    colors = ["#2a6f97", "#2a9d8f", "#adb5bd"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    ax.set_ylabel("Alerts")
    ax.set_title(f"Validation what-if outcomes (n={n})")
    ax.set_ylim(0, max(vals) * 1.25 + 0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, str(v), ha="center", va="bottom")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "01_acceptance_fixed.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def delta_risk(df: pd.DataFrame) -> None:
    sub = df.dropna(subset=["delta_risk_proxy"]).copy()
    sub["label"] = sub["asset_id"] + " #" + sub["alert_index"].astype(str) + "\n" + sub["sandbox_action"]
    order = sub["delta_risk_proxy"].argsort()
    sub = sub.iloc[order]
    colors = ["#2a9d8f" if v < 0 else "#c1121f" if v > 0 else "#8d99ae" for v in sub["delta_risk_proxy"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.barh(sub["label"], sub["delta_risk_proxy"], color=colors)
    ax.axvline(0, color="#212529", linewidth=0.8)
    ax.set_xlabel("Δ risk_proxy (proposed − baseline)")
    ax.set_title("What-if risk change by validation alert")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "02_delta_risk.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def action_breakdown(df: pd.DataFrame) -> None:
    g = (
        df.groupby("sandbox_action", dropna=False)
        .agg(n=("sandbox_action", "size"), fixed=("fixed", "sum"), accepted=("accepted", "sum"))
        .reset_index()
        .sort_values("n", ascending=False)
    )
    x = range(len(g))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    w = 0.35
    ax.bar([i - w / 2 for i in x], g["n"], width=w, label="Alerts", color="#8d99ae")
    ax.bar([i + w / 2 for i in x], g["fixed"].astype(int), width=w, label="Fixed", color="#2a9d8f")
    ax.set_xticks(list(x))
    ax.set_xticklabels(g["sandbox_action"])
    ax.set_ylabel("Count")
    ax.set_title("Recommended sandbox action vs fixed")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "03_action_breakdown.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    if not RESULTS.exists():
        raise SystemExit(f"Missing {RESULTS}; run batch_validation first")
    df = pd.read_csv(RESULTS)
    # CSV may store bools as strings
    for col in ("accepted", "fixed"):
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin(("true", "1", "yes"))
    acceptance_bars(df)
    delta_risk(df)
    action_breakdown(df)


if __name__ == "__main__":
    main()
