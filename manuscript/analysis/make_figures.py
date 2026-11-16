"""Figures for the manuscript, from manuscript/generated/ and the inference JSON files.

Run after build_numbers.py, from the repository root:
    uv run python manuscript/analysis/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "manuscript" / "generated"
FIG = ROOT / "manuscript" / "figures"
PUB = ROOT / "results" / "publication"

# Categorical slots 1, 2, 3, 5 of the validated reference palette (light mode); marker
# shape repeats the dataset identity because two adjacent slots sit in the CVD floor band.
STYLE = {"ECL": ("#2a78d6", "o"), "ETTh2": ("#eb6834", "s"),
         "weather": ("#1baf7a", "^"), "exchange_rate": ("#e87ba4", "D")}
NAME = {"ECL": "ECL", "ETTh2": "ETTh2", "weather": "Weather", "exchange_rate": "Exchange"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
MIN_BLOCKS = 10

plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})


def forest() -> None:
    ecl = json.loads((PUB / "paired_interval_inference_ECL.json").read_text())["cells"]
    exposed = json.loads((PUB / "paired_interval_inference_exposed.json").read_text())["cells"]
    order = ["ECL", "ETTh2", "exchange_rate", "weather"]
    rows = sorted(ecl + exposed, key=lambda c: (order.index(c["data"]), c["horizon"]))
    # One full-width, short panel: cells along x, grouped by dataset with a gap between groups.
    pos, groups, x = [], {}, 0.0
    for i, c in enumerate(rows):
        if i and c["data"] != rows[i - 1]["data"]:
            x += 0.8
        pos.append(x)
        groups.setdefault(c["data"], []).append(x)
        x += 1
    fig, ax = plt.subplots(figsize=(5.4, 2.1))  # elsarticle text width, 390 pt
    ticks = []
    for xi, c in zip(pos, rows):
        color, marker = STYLE[c["data"]]
        thin = c["n_origins"] / c["block_len"] < MIN_BLOCKS
        lo, mid, hi = (100 * c[k] for k in ("relative_ci_low", "relative_mean", "relative_ci_high"))
        ax.plot([xi, xi], [lo, hi], color=color, lw=2, alpha=0.45 if thin else 1.0, solid_capstyle="round")
        ax.plot(xi, mid, marker=marker, ms=6, color=color, mfc="white" if thin else color, mew=1.5, ls="none")
        if c["data"] == "ECL" and c["horizon"] == 336:
            ax.annotate("primary", (xi, hi), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=INK)
        ticks.append(str(c["horizon"]))  # thin cells are marked by hollow markers only
    ax.axhline(0, color=INK, lw=0.8)
    ax.axhline(-1, color=MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.text(pos[-1] + 0.75, -1, "$-1\\%$\nmargin", color=MUTED, ha="left", va="center", fontsize=7)
    ax.axvspan(groups["ECL"][0] - 0.5, groups["ECL"][-1] + 0.5, color="#eef4fc", zorder=0)
    ax.set_xticks(pos, ticks)
    ax.tick_params(axis="x", length=0)
    for data, xs in groups.items():
        ax.text(sum(xs) / len(xs), -0.13, NAME[data], transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=INK)
    ax.set_xlim(pos[0] - 0.7, pos[-1] + 0.7)
    ax.set_ylabel("Winkler change\nvs flat CP (%)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(FIG / "forest_winkler.pdf")
    fig.savefig(FIG / "forest_winkler.png", dpi=200)
    plt.close(fig)


def trade() -> None:
    cells = pd.read_csv(GEN / "cells.csv")
    cells = cells[cells.data != "ETTh1"]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for data, grp in cells.groupby("data"):
        color, marker = STYLE[data]
        ax.plot(100 * grp.width_rel, 100 * grp.cov_delta, marker=marker, ms=6, color=color, ls="none",
                mec="white", mew=0.8, label=NAME[data])
    ax.axhline(0, color=INK, lw=0.8)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("Mean width, cluster SCCP vs flat CP (% change)")
    ax.set_ylabel("Coverage, cluster minus flat\n(percentage points)")
    ax.grid(color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=7, loc="lower center", ncol=2, handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "coverage_width_trade.pdf")
    fig.savefig(FIG / "coverage_width_trade.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    forest()
    trade()
    print(f"wrote figures to {FIG}")
