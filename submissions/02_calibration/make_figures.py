"""Paper figures for alert-policy calibration chapter."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
ROOT = HERE.parents[1]
POLICY = ROOT / "retrofit_starter" / "solution" / "final_model" / "policy.json"
SELECTION = (
    ROOT
    / "retrofit_starter"
    / "solution"
    / "final_model"
    / "results"
    / "development_search"
    / "selection.json"
)


def per_fault_thresholds() -> None:
    policy = json.loads(POLICY.read_text())
    global_thr = float(policy["threshold"])
    overrides = dict(policy.get("thresholds") or {})
    # Show overrides + a "GLOBAL (default)" bar for context
    labels = list(overrides.keys()) + ["GLOBAL_DEFAULT"]
    values = [float(overrides[k]) for k in overrides] + [global_thr]
    order = np.argsort(values)
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = ["#2a6f97" if lab != "GLOBAL_DEFAULT" else "#8d99ae" for lab in labels]
    ax.barh(labels, values, color=colors, height=0.7)
    ax.axvline(global_thr, color="#c1121f", linestyle="--", linewidth=1.2, label=f"global={global_thr}")
    ax.set_xlabel("Score threshold")
    ax.set_title("AlertPolicy per-fault thresholds (frozen)")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "01_per_fault_thresholds.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def selection_composites() -> None:
    sel = json.loads(SELECTION.read_text())
    reports = sel["reports"]
    names = ["original_validation", "development", "selection_mean"]
    vals = [
        float(reports["original_validation"]["composite_score"]),
        float(reports["development"]["composite_score"]),
        float(sel["selection_mean"]),
    ]
    labels = ["Official validation", "Synthetic development", "Equal-weight mean"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars = ax.bar(labels, vals, color=["#2a6f97", "#61a5c2", "#014f86"], width=0.55)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Official composite")
    ax.set_title("Policy selection objective (trial 243)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG / "02_selection_composites.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    per_fault_thresholds()
    selection_composites()


if __name__ == "__main__":
    main()
