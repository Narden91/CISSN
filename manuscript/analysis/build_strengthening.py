"""Tables and figures from the separate frozen-checkpoint comparison suite."""
from __future__ import annotations

import argparse
import json
from itertools import groupby
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

LABEL = {"flat_paired": "Flat paired", "flat_full": "Flat full", "phase": "Phase", "state": "State",
         "phase_state": "Phase+State", "aci": "ACI (capped)", "phase_coarse": "Coarse phase",
         "phase_coarse_state": "Coarse phase+State", "flat_scaled": "Flat scaled"}
METHODS = list(LABEL)
COLORS = ["#555555", "#111111", "#0072b2", "#d55e00", "#009e73", "#cc79a7", "#56b4e9", "#e69f00", "#999933"]


def tex(value):
    return str(value).replace("_", r"\_")


def table(path, headers, rows, columns, long=False):
    # Separate dataset/horizon blocks allow page breaks between tables.
    groups = [list(group) for _, group in groupby(rows, key=lambda row: (row[0], row[1]))] if long else [rows]
    lines = []
    for group in groups:
        if long:
            lines.append(r"\begin{center}")
        lines += [rf"\begin{{tabular}}{{{columns}}}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
        lines += [" & ".join(map(str, row)) + r" \\" for row in group]
        lines += [r"\bottomrule", r"\end{tabular}"]
        if long:
            lines.append(r"\end{center}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/paper_strengthening"))
    parser.add_argument("--prefix", choices=("strengthening", "cluster"), default="strengthening")
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--table-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    prefix = args.prefix
    root = Path(__file__).resolve().parents[1]
    gen, figdir = args.generated_dir or root / "generated", args.figure_dir or root / "figures"
    tables = args.table_dir or args.generated_dir or root / "tables"
    gen.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.results / "summary.csv")
    methods = [method for method in METHODS if method in set(frame.method)]
    frame["miss"] = frame.winkler - frame.width
    uncertainty = pd.read_csv(args.results / "uncertainty.csv")
    phases = pd.read_csv(args.results / "phase_effects.csv")
    inventory = json.loads((args.results / "inventory.json").read_text())
    main_rows, alpha_rows, strong_rows = [], [], []
    for (dataset, horizon, model), group in frame.groupby(["dataset", "horizon", "model"]):
        at_alpha = group[np.isclose(group.alpha, 0.1)].set_index("method")
        for method in methods:
            main_rows.append([tex(dataset), horizon, tex(model), LABEL[method],
                              *[f"{at_alpha.loc[method, metric]:.4f}" for metric in ("coverage", "width", "winkler", "miss")]])
        for row in group.itertuples():
            alpha_rows.append([tex(dataset), horizon, tex(model), LABEL[row.method], f"{row.alpha:.2f}", f"{row.coverage:.4f}",
                               f"{row.absolute_coverage_error:.4f}", f"{row.width:.4f}", f"{row.winkler:.4f}"])
        if model != "cissn":
            for row in at_alpha.itertuples():
                strong_rows.append([tex(dataset), horizon, tex(model), LABEL[row.Index], f"{row.mae:.4f}", f"{row.mse:.4f}",
                                    f"{row.coverage:.4f}", f"{row.width:.4f}", f"{row.winkler:.4f}"])
    table(tables / f"tab_{prefix}_main.tex", ["Dataset", "$H$", "Model", "Method", "Coverage", "Width", "Winkler", "Miss cost"],
          main_rows, "lrllrrrr")
    horizon_rows = []
    for key, group in frame.groupby(["dataset", "horizon", "model", "alpha", "method"]):
        dataset, horizon, model, alpha, method = key
        runs = [r for r in inventory if r["status"] == "completed" and r["dataset"] == dataset
                and r["horizon"] == horizon and r["model"] == model]
        paths = [args.results / "scores" / r["run"] / f"{method}_alpha{alpha}_horizon.npy" for r in runs]
        if paths:
            coverage_steps = np.mean([np.load(path)[:, 0] for path in paths], axis=0)
            horizon_rows.append([tex(dataset), horizon, tex(model), LABEL[method], f"{alpha:.2f}",
                                 *[f"{value:.4f}" for value in np.quantile(coverage_steps, [0, 0.05, 0.5])]])
    table(tables / f"tab_{prefix}_horizon_coverage.tex", ["Dataset", "$H$", "Model", "Method", r"$\alpha$",
          "Minimum", "5th percentile", "Median"], horizon_rows, "lrllrrrr", long=True)
    table(tables / f"tab_{prefix}_alpha.tex", ["Dataset", "$H$", "Model", "Method", r"$\alpha$", "Coverage", "Error", "Width", "Winkler"], alpha_rows, "lrllrrrrr", long=True)
    if strong_rows:
        table(tables / f"tab_{prefix}_strong.tex", ["Dataset", "$H$", "Model", "Method", "MAE", "MSE", "Coverage", "Width", "Winkler"], strong_rows, "lrllrrrrr")
    else:
        (tables / f"tab_{prefix}_strong.tex").write_text(
            "No matched strong-forecaster comparison was run: the DLinear and PatchTST publication weights are absent.\n", encoding="utf-8")
    # Keep the complete numerical evidence beside the manuscript.
    for name in ("summary.csv", "uncertainty.csv", "phase_effects.csv"):
        (gen / f"{prefix}_{name}").write_bytes((args.results / name).read_bytes())
    block = uncertainty[(uncertainty.block_multiplier == 1) & uncertainty.kind.isna() & np.isclose(uncertainty.alpha, 0.1)]
    ci_rows = []
    for row in block.itertuples():
        ci_rows.append([tex(row.dataset), row.horizon, tex(row.model), LABEL[row.baseline], row.metric,
                        f"{row.mean:.4f}", f"[{row.ci_low:.4f}, {row.ci_high:.4f}]", f"{row.n_origins / row.block_len:.1f}"])
    table(tables / f"tab_{prefix}_ci.tex", ["Dataset", "$H$", "Model", "Baseline", "Metric", r"$\Delta$", r"95\% CI", "Blocks"], ci_rows, "lrlllrlr", long=True)
    extra_rows = []
    extra = uncertainty[(uncertainty.block_multiplier == 1) & uncertainty.kind.isin(["nested", "ratio_of_means"])
                        & np.isclose(uncertainty.alpha, 0.1)]
    for row in extra.itertuples():
        comparison = ("Coarse Phase+State minus Coarse Phase" if row.kind == "nested"
                      else "State/Flat paired minus 1")
        extra_rows.append([tex(row.dataset), row.horizon, tex(row.model), comparison, row.metric,
                           f"{row.mean:.4f}", f"[{row.ci_low:.4f}, {row.ci_high:.4f}]"])
    table(tables / f"tab_{prefix}_additional_ci.tex", ["Dataset", "$H$", "Model", "Comparison", "Metric",
          r"$\Delta$", r"95\% CI"], extra_rows, "lrlllrl", long=True)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    cells = list(frame.groupby(["dataset", "horizon", "model"]))
    fig, axes = plt.subplots((len(cells) + 1) // 2, 2, figsize=(8, 2.7 * ((len(cells) + 1) // 2)), squeeze=False)
    for ax, ((dataset, horizon, model), group) in zip(axes.flat, cells):
        for method in methods:
            color = COLORS[METHODS.index(method)]
            rows = group[group.method == method].sort_values("alpha")
            ax.plot(rows.coverage, rows.width, "-", label=LABEL[method], color=color)
            for row in rows.itertuples():
                ax.plot(row.coverage, row.width, marker={0.05: "^", 0.1: "o", 0.2: "s"}[row.alpha], color=color, ms=4)
        ax.set(title=f"{dataset}, H={horizon}, {model}", xlabel="Empirical cellwise coverage", ylabel="Width (training SD units)")
    for ax in list(axes.flat)[len(cells):]:
        ax.set_visible(False)
    axes[0, 0].legend(ncol=2, fontsize=6)
    fig.legend(handles=[Line2D([], [], color="#333333", marker=m, linestyle="none", label=f"alpha={a}")
                        for a, m in ((0.05, "^"), (0.10, "o"), (0.20, "s"))], loc="upper center", ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, figdir, f"{prefix}_coverage_width")
    # Plot available phase results; do not label another dataset as ECL.
    select = phases[(phases.baseline == "flat_paired") & (phases.metric == "winkler") & np.isclose(phases.alpha, 0.1)]
    cells_phase = list(select.groupby(["dataset", "horizon", "model"]))
    fig, axes = plt.subplots((len(cells_phase) + 1) // 2, 2, figsize=(8, 2.5 * ((len(cells_phase) + 1) // 2)), squeeze=False)
    for ax, ((dataset, horizon, model), group) in zip(axes.flat, cells_phase):
        group = group.sort_values("phase")
        ax.plot(group.phase, group.state_minus_baseline, "o-", ms=3)
        ax.axhline(0, color="#777777", lw=0.7)
        ax.set(title=f"{dataset}, H={horizon}, {model}", xlabel="Phase of first target", ylabel="State minus Flat Winkler")
    fig.tight_layout()
    for ax in list(axes.flat)[len(cells_phase):]:
        ax.set_visible(False)
    save(fig, figdir, f"{prefix}_phase_effects")
    diagnostics = []
    for path in sorted(args.results.rglob("diagnostics.json")):
        if path.parents[1].name != "scores":
            continue
        config = json.loads((path.parents[2] / "cache" / path.parent.name / "metadata.json").read_text())["provenance"]["config"]
        diagnostics.append((path, config, json.loads(path.read_text())))
    # One seed per cell, selected by seed number before seeing diagnostic values.
    chosen = {}
    for path, config, data in diagnostics:
        key = (config["data"], config["pred_len"])
        if key not in chosen or config["seed"] < chosen[key][1]["seed"]:
            chosen[key] = (path, config, data)
    fig, axes = plt.subplots(max(1, (len(chosen) + 1) // 2), 2, figsize=(8, 2.5 * max(1, (len(chosen) + 1) // 2)), squeeze=False)
    for ax, (key, (_, config, data)) in zip(axes.flat, sorted(chosen.items())):
        counts = np.asarray(data["cluster_phase_counts"], dtype=float)
        proportions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(proportions, aspect="auto", origin="lower", cmap="Blues")
        ax.set(title=f"{key[0]}, H={key[1]}, seed {config['seed']}: NMI={data['cluster_phase_nmi']:.3f}",
               xlabel="Phase of first target", ylabel="State cluster")
        fig.colorbar(im, ax=ax, label="Phase fraction within cluster")
    fig.tight_layout()
    for ax in list(axes.flat)[len(chosen):]:
        ax.set_visible(False)
    save(fig, figdir, f"{prefix}_cluster_phase")
    plot = block[(block.metric == "winkler") & block.baseline.isin(["flat_paired", "flat_full", "phase", "phase_coarse", "flat_scaled", "aci"])]
    fig, ax = plt.subplots(figsize=(6.5, max(3, 0.22 * len(plot))))
    for y, row in enumerate(plot.itertuples()):
        ax.plot([row.ci_low, row.ci_high], [y, y], color="#0072b2")
        ax.plot(row.mean, y, "o", color="#d55e00", ms=4, mfc="white" if row.n_origins / row.block_len < 10 else "#d55e00")
    ax.set_yticks(range(len(plot)), [f"{r.dataset} H={r.horizon}, {r.model}, {LABEL[r.baseline]}" for r in plot.itertuples()])
    ax.axvline(0, color="#777777", lw=0.7)
    ax.set(xlabel="State minus baseline Winkler (95% moving-block CI)")
    fig.tight_layout()
    save(fig, figdir, f"{prefix}_paired_ci")
    completed = sum(r["status"] == "completed" for r in inventory)
    blocked = sum(r["status"] == "blocked" for r in inventory)
    (gen / f"{prefix}_status.tex").write_text(
        f"The follow-up completed {completed} frozen-checkpoint runs and {len(frame)} method--alpha cell summaries. "
        f"It recorded {blocked} selected runs as blocked by missing weights. All completed dataset splits were previously exposed.\n", encoding="utf-8")
    current = frame[np.isclose(frame.alpha, 0.1)]
    phase_scores = current.pivot(index=["dataset", "horizon", "model"], columns="method", values="winkler")
    coverage = current.pivot(index=["dataset", "horizon", "model"], columns="method", values="coverage")
    loss = 100 * (coverage["phase"] - coverage["state"])
    exchange = phase_scores.loc[("exchange_rate", 96, "cissn")] if ("exchange_rate", 96, "cissn") in phase_scores.index else None
    macro_lines = [rf"\newcommand{{\FollowupRuns}}{{{completed}}}", rf"\newcommand{{\FollowupCells}}{{{len(phase_scores)}}}",
                   rf"\newcommand{{\FollowupPhaseWins}}{{{int((phase_scores.state < phase_scores.phase).sum())}}}",
                   rf"\newcommand{{\FollowupCoverageLossMin}}{{{loss.min():.2f}}}", rf"\newcommand{{\FollowupCoverageLossMax}}{{{loss.max():.2f}}}"]
    if exchange is not None:
        macro_lines.append(rf"\newcommand{{\FollowupExchangeFullLoss}}{{{100 * (exchange.state / exchange.flat_full - 1):.2f}}}")
    (gen / f"{prefix}_numbers.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")
    count_groups = {}
    for path in args.results.rglob("calibration_alpha0.1.json"):
        config = json.loads((path.parents[2] / "cache" / path.parent.name / "metadata.json").read_text())["provenance"]["config"]
        model = "cissn" if config.get("architecture") == "legacy" else config["model"]
        count_groups.setdefault((config["data"], config["pred_len"], model), []).append(json.loads(path.read_text()))
    count_rows = []
    for (dataset, horizon, model), values in sorted(count_groups.items()):
        phase_counts = [n for value in values for n in value["phase"]["counts"].values()]
        cross_fallbacks = [sum(v is not None for k, v in value["phase_state"]["fallbacks"].items() if ":" in k) for value in values]
        count_rows.append([tex(dataset), horizon, tex(model), len(values[0]["paired_indices"]), values[0]["n_full"],
                           len(values[0]["phase"]["counts"]), f"{min(phase_counts)}--{max(phase_counts)}",
                           f"{min(cross_fallbacks)}--{max(cross_fallbacks)}"])
    table(tables / f"tab_{prefix}_counts.tex", ["Dataset", "$H$", "Model", "Paired", "Full", "Phase bins", "Counts/bin", "Cross fallbacks"], count_rows, "lrllrrrr")
    print(f"Wrote tables and four figure pairs from {len(frame)} measured summaries.")


if __name__ == "__main__":
    main()
