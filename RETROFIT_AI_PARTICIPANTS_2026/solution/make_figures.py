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


def headline():
    s = pd.read_csv(HERE / "comparison" / "results" / "comparison.csv")
    keep = ["astra-best", "quaviz", "b1", "b3", "b2", "b0"]
    s = s[s["method"].isin(keep)]
    order = ["validation", "our holdout s0112", "our holdout s0113", "astra holdout"]
    labels = {"validation": "Validation\n(PUMP-08/09)", "our holdout s0112": "Our holdout\ns0112",
              "our holdout s0113": "Our holdout\ns0113", "astra holdout": "Astra holdout\n(frozen)"}
    methods = [m for m in keep if m in set(s["method"])]
    colors = {"astra-best": "#1b9e77", "quaviz": "#d95f02", "b1": "#7570b3", "b3": "#e7298a", "b2": "#66a61e",
              "b0": "#a6761d"}
    x = np.arange(len(order))
    w = 0.13
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, m in enumerate(methods):
        vals = [float(s[(s.method == m) & (s.set == st)]["composite_score"].iloc[0]) for st in order]
        ax.bar(x + (i - len(methods) / 2) * w + w / 2, vals, w, label=m, color=colors[m])
    ax.set_xticks(x)
    ax.set_xticklabels([labels[st] for st in order])
    ax.set_ylabel("Official composite / 100")
    ax.set_ylim(0, 100)
    ax.axhline(78.0, color="#7570b3", ls=":", lw=1, alpha=0.7)
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.set_title("Same scorer, four evaluation sets")
    fig.tight_layout()
    fig.savefig(OUT / "01_headline_composites.png", dpi=140)
    plt.close(fig)


def per_type():
    pe = pd.read_csv(HERE / "comparison" / "results" / "per_event.csv")
    pe = pe[pe["method"].isin(["astra-best", "quaviz", "b1"])]
    pe = pe[pe["set"] != "validation"]  # holdouts only (enough events)
    pe["named"] = pe["type_correct"].astype(int)
    g = pe.groupby(["fault_type", "method"]).agg(n=("detected", "size"), named=("named", "sum"))
    rate = (g["named"] / g["n"]).unstack("method").reindex(columns=["astra-best", "quaviz", "b1"])
    rate = rate.sort_values("astra-best", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(rate.to_numpy(float), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["astra-best", "fleet detector", "B1"])
    ax.set_yticks(range(len(rate)))
    ax.set_yticklabels([t.replace("SENSOR_", "S_").replace("_", " ").title() for t in rate.index], fontsize=8)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            v = rate.to_numpy(float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v < 0.55 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, label="First-alert type correct")
    ax.set_title("Holdouts pooled: fraction of events named correctly first")
    fig.tight_layout()
    fig.savefig(OUT / "02_per_type_holdouts.png", dpi=140)
    plt.close(fig)


def validation_timeline():
    assets = pd.read_csv(CHALLENGE / "data" / "full" / "assets.csv")
    ev = pd.read_csv(CHALLENGE / "data" / "full" / "events_validation.csv")
    al = pd.read_csv(HERE / "final_model" / "results" / "original_validation_predictions.csv")
    ev["onset"] = pd.to_datetime(ev["onset_utc"], utc=True)
    ev["end"] = pd.to_datetime(ev["end_utc"], utc=True)
    al["t"] = pd.to_datetime(al["alert_time_utc"], utc=True)
    pumps = ["PUMP-08", "PUMP-09"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=False)
    for ax, aid in zip(axes, pumps):
        a = assets.set_index("asset_id").loc[aid]
        t0 = pd.to_datetime(a["window_start_utc"], utc=True)
        t1 = pd.to_datetime(a["window_end_utc"], utc=True)
        e = ev[ev.asset_id == aid].copy()
        for _, r in e.iterrows():
            end = r["end"] if pd.notna(r["end"]) else t1
            ax.axvspan(r["onset"], end, color="#fdae6b", alpha=0.35)
            ax.text(r["onset"], 0.85, r["fault_type"].replace("SENSOR_", "S_").replace("_", "\n"),
                    fontsize=6, rotation=0, va="top")
        aa = al[al.asset_id == aid]
        ax.scatter(aa["t"], np.full(len(aa), 0.4), marker="v", s=60, color="#1b9e77", zorder=3, label="astra-best alert")
        for _, r in aa.iterrows():
            ax.text(r["t"], 0.15, r["fault_type"].replace("SENSOR_", "S_")[:14], fontsize=6, rotation=45, ha="right")
        ax.set_xlim(t0, t1)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_title(aid)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Validation: labeled events (orange) and final-model alerts (green)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "03_validation_timeline.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def stage1_separability():
    p = HERE / "fleet_detector" / "results" / "stage1" / "separability_per_event.csv"
    if not p.exists():
        return
    res = pd.read_csv(p)
    res = res[res["set"].str.startswith("official") | res["set"].str.startswith("sim")]
    order = ["B1 z (first 24 h)", "ours A (context)", "ours B (consistency)", "ours A+B relative to own past"]
    short = {"B1 z (first 24 h)": "B1 z", "ours A (context)": "Ours A", "ours B (consistency)": "Ours B",
             "ours A+B relative to own past": "Ours relative"}
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for ax, (label, mask) in zip(axes, [("Official pumps", res["set"] == "official"),
                                        ("Held-out sims", res["set"].str.startswith("sim"))]):
        sub = res[mask]
        vals = [sub[sub["score"] == s]["auc"].dropna() for s in order]
        ax.boxplot(vals, tick_labels=[short[s] for s in order], showfliers=False)
        ax.scatter(np.repeat(np.arange(1, 5), [len(v) for v in vals]), np.concatenate(vals),
                   alpha=0.35, s=12, color="#444")
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_title(label)
        ax.set_ylabel("Event AUROC" if ax is axes[0] else "")
        ax.set_ylim(0, 1.05)
    fig.suptitle("Stage 1: how well deviations separate the first 6 h of a fault from healthy minutes")
    fig.tight_layout()
    fig.savefig(OUT / "04_stage1_separability.png", dpi=140)
    plt.close(fig)


def copy_static():
    copies = [
        (HERE / "encoding" / "latent-space" / "results" / "figures" / "per_pump_label_informed.png",
         OUT / "05_encoding_label_informed.png"),
        (HERE / "encoding" / "latent-space" / "results" / "figures" / "per_pump_cluster.png",
         OUT / "06_encoding_clusters.png"),
        (HERE / "encoding" / "encoding-expanded" / "results" / "comparison.png",
         OUT / "07_encoding_expanded.png"),
    ]
    # fidelity report figure from logs if present
    repo = CHALLENGE.parent
    for cand in [repo / "logs" / "step1" / "figures", repo / "results"]:
        pass
    fidelity_figs = list((repo / "logs").glob("**/0*.png")) if (repo / "logs").exists() else []
    for src in fidelity_figs:
        if "fidelity" in src.name.lower() or src.name.startswith("09"):
            copies.append((src, OUT / f"08_{src.name}"))
    # also try common step1 figures
    for name in ["01_fleet_coverage.png", "09_fidelity_before_after.png", "08_b1_by_type.png"]:
        src = repo / "logs" / "step1" / name
        if src.exists():
            copies.append((src, OUT / f"08_{name}"))
    for src, dst in copies:
        if src.exists():
            shutil.copy2(src, dst)


def main():
    headline()
    per_type()
    validation_timeline()
    stage1_separability()
    copy_static()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
