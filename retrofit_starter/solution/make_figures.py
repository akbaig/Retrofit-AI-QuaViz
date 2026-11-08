"""Build the figures referenced by SOLUTION.md into solution/figures/."""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(parents=True, exist_ok=True)

NAMES = {"astra-best": "best", "b1": "B1", "b3": "B3", "b2": "B2", "b0": "B0"}
COLORS = {"astra-best": "#1b9e77", "b1": "#7570b3", "b3": "#e7298a", "b2": "#66a61e", "b0": "#a6761d"}


def headline():
    s = pd.read_csv(HERE / "comparison" / "results" / "comparison.csv")
    s = s[s["method"].isin(NAMES) & (s["set"] == "validation")]
    methods = list(NAMES)
    vals = [float(s[s.method == m]["composite_score"].iloc[0]) for m in methods]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(methods))
    ax.bar(x, vals, width=0.62, color=[COLORS[m] for m in methods])
    ax.set_xticks(x)
    ax.set_xticklabels([NAMES[m] for m in methods])
    ax.set_ylabel("Official composite / 100")
    ax.set_ylim(0, 100)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_title("Official validation (PUMP-08/09): best vs baselines")
    fig.tight_layout()
    fig.savefig(OUT / "01_headline_composites.png", dpi=140)
    plt.close(fig)


def validation_prf():
    s = pd.read_csv(HERE / "comparison" / "results" / "comparison.csv")
    s = s[s["method"].isin(NAMES) & (s["set"] == "validation")].set_index("method")
    methods = list(NAMES)
    metrics = [
        ("event_recall", "Event recall"),
        ("alert_precision", "Alert precision"),
        ("event_f1", "Event F1"),
    ]
    x = np.arange(len(metrics))
    w = 0.15
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, m in enumerate(methods):
        vals = [float(s.loc[m, key]) for key, _ in metrics]
        ax.bar(x + (i - (len(methods) - 1) / 2) * w, vals, w, label=NAMES[m], color=COLORS[m])
        for xi, v in zip(x + (i - (len(methods) - 1) / 2) * w, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylabel("Score (0–1)")
    ax.set_ylim(0, 1.18)
    ax.legend(ncol=5, fontsize=8, loc="upper right")
    ax.set_title("Official validation (PUMP-08/09): precision, recall, and F1")
    ax.axhline(1.0, color="#bbbbbb", lw=0.8, ls=":")
    fig.tight_layout()
    fig.savefig(OUT / "08_validation_prf.png", dpi=140)
    plt.close(fig)


def per_type(set_name: str, outfile: str, title: str, figsize=(8, 5.5)) -> None:
    pe = pd.read_csv(HERE / "comparison" / "results" / "per_event.csv")
    methods = list(NAMES)
    pe = pe[pe["method"].isin(methods) & (pe["set"] == set_name)].copy()
    pe["named"] = pe["type_correct"].astype(int)
    g = pe.groupby(["fault_type", "method"]).agg(n=("detected", "size"), named=("named", "sum"))
    rate = (g["named"] / g["n"]).unstack("method").reindex(columns=methods)
    n = g["n"].unstack("method")["astra-best"]
    rate = rate.sort_values("astra-best", ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(rate.to_numpy(float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([NAMES[m] for m in methods])
    ax.set_yticks(range(len(rate)))
    ax.set_yticklabels([f"{t.replace('SENSOR_', 'S_').replace('_', ' ').title()} ({int(n[t])})" for t in rate.index],
                       fontsize=8)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.to_numpy(float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v < 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, label="First-alert type correct")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / outfile, dpi=140)
    plt.close(fig)


def validation_timeline():
    assets = pd.read_csv(CHALLENGE / "data" / "full" / "assets.csv")
    ev = pd.read_csv(CHALLENGE / "data" / "full" / "events_validation.csv")
    al = pd.read_csv(HERE / "final_model" / "results" / "original_validation_predictions.csv")
    ev["onset"] = pd.to_datetime(ev["onset_utc"], utc=True)
    ev["end"] = pd.to_datetime(ev["end_utc"], utc=True)
    al["t"] = pd.to_datetime(al["alert_time_utc"], utc=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5))
    for ax, aid in zip(axes, ["PUMP-08", "PUMP-09"]):
        a = assets.set_index("asset_id").loc[aid]
        t0 = pd.to_datetime(a["window_start_utc"], utc=True)
        t1 = pd.to_datetime(a["window_end_utc"], utc=True)
        for _, r in ev[ev.asset_id == aid].iterrows():
            end = r["end"] if pd.notna(r["end"]) else t1
            ax.axvspan(r["onset"], end, color="#fdae6b", alpha=0.35)
            ax.text(r["onset"], 0.85, r["fault_type"].replace("SENSOR_", "S_").replace("_", "\n"), fontsize=6, va="top")
        aa = al[al.asset_id == aid]
        ax.scatter(aa["t"], np.full(len(aa), 0.4), marker="v", s=60, color="#1b9e77", zorder=3, label="best: alert")
        for _, r in aa.iterrows():
            ax.text(r["t"], 0.15, r["fault_type"].replace("SENSOR_", "S_")[:14], fontsize=6, rotation=45, ha="right")
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(aid)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Validation: labeled events (orange) and alerts of the best model (green)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "03_validation_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def stage1_separability():
    p = HERE / "fleet_detector" / "results" / "stage1" / "separability_per_event.csv"
    res = pd.read_csv(p)
    res = res[res["set"] == "official"]
    order = ["B1 z (first 24 h)", "ours A (context)", "ours B (consistency)", "ours A+B relative to own past"]
    short = ["B1 z", "Context", "Consistency", "Relative\nto own past"]
    vals = [res[res["score"] == s]["auc"].dropna() for s in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(vals, tick_labels=short, showfliers=False)
    ax.scatter(np.repeat(np.arange(1, 5), [len(v) for v in vals]), np.concatenate(vals), alpha=0.35, s=12, color="#444")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_ylabel("Event AUROC")
    ax.set_ylim(0, 1.05)
    ax.set_title("Normal model: fault window vs healthy minutes\n(official pumps, 33 events)")
    fig.tight_layout()
    fig.savefig(OUT / "04_stage1_separability.png", dpi=140)
    plt.close(fig)


def copy_static():
    enc = HERE / "encoding"
    for src, dst in [(enc / "latent-space" / "results" / "figures" / "per_pump_label_informed.png", "05_encoding_label_informed.png"),
                     (enc / "latent-space" / "results" / "figures" / "per_pump_cluster.png", "06_encoding_clusters.png"),
                     (enc / "encoding-expanded" / "results" / "comparison.png", "07_encoding_expanded.png")]:
        if src.exists():
            shutil.copy2(src, OUT / dst)


def main():
    for old in [
        "02_per_type_holdouts.png",
        "08_sim_coverage.png",
        "09_sim_event_mix.png",
    ]:
        (OUT / old).unlink(missing_ok=True)
    headline()
    per_type(
        "validation",
        "02_per_type_validation.png",
        "Official validation: fraction of events named correctly first",
        figsize=(8, 5.5),
    )
    per_type(
        "astra holdout",
        "02_per_type_holdout.png",
        "Frozen holdout (78 events): fraction of events named correctly first",
        figsize=(8.5, 6),
    )
    validation_prf()
    validation_timeline()
    stage1_separability()
    copy_static()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
