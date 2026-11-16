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
# Each suite gets its own macro prefix: main.tex inputs both macro files, and \newcommand rejects duplicates.
TAG = {"strengthening": "Followup", "cluster": "Cluster"}
NAME = {"ECL": "ECL", "solar": "Solar", "ETTh2": "ETTh2", "weather": "Weather", "exchange_rate": "Exchange",
        "cissn": "CISSN", "dlinear": "DLinear", "patchtst": "PatchTST-lite"}
# TeX macro names hold letters only.
MACRO = {"ECL": "Ecl", "solar": "Solar", "ETTh2": "EttTwo", "weather": "Weather", "exchange_rate": "Exchange",
         96: "Short", 336: "Long", "cissn": "", "dlinear": "Dlinear", "patchtst": "Patchtst"}
CALENDAR = ("phase", "phase_coarse")


def tex(value):
    return str(value).replace("_", r"\_")


def signed(value):
    return f"{value:+.2f}" if abs(value) < 10 else f"{value:+.1f}"


def sci(value):
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def camel(method):
    return "".join(part.capitalize() for part in method.split("_"))


def reading_outcome(intervals):
    """Follow-up reading rule of docs/prespecification_solar.md.

    ``intervals`` maps a control to the (low, high) limits of State minus control.
    """
    if any(intervals[control][1] >= 0 for control in CALENDAR):
        return "A"
    return "C" if intervals["flat_scaled"][1] >= 0 else "B"


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


def score_path(results, run, filename):
    """Support standalone suites and merged suites with dataset subdirectories."""
    candidates = [results / "scores" / run["run"] / filename,
                  results / run["dataset"] / "scores" / run["run"] / filename]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("Missing score artifact; checked: " + ", ".join(map(str, candidates)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/paper_strengthening_v2"))
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
    runs_by_name = {run["run"]: run for run in inventory}
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
          main_rows, "lrllrrrr", long=True)
    horizon_rows = []
    for key, group in frame.groupby(["dataset", "horizon", "model", "alpha", "method"]):
        dataset, horizon, model, alpha, method = key
        runs = [r for r in inventory if r["status"] == "completed" and r["dataset"] == dataset
                and r["horizon"] == horizon and r["model"] == model]
        paths = [score_path(args.results, r, f"{method}_alpha{alpha}_horizon.npy") for r in runs]
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
        run = runs_by_name[path.parent.name]
        diagnostics.append((path, run, json.loads(path.read_text())))
    # One seed per cell, selected by seed number before seeing diagnostic values.
    chosen = {}
    for path, run, data in diagnostics:
        key = (run["dataset"], run["horizon"])
        if key not in chosen or run["seed"] < chosen[key][1]["seed"]:
            chosen[key] = (path, run, data)
    fig, axes = plt.subplots(max(1, (len(chosen) + 1) // 2), 2, figsize=(8, 2.5 * max(1, (len(chosen) + 1) // 2)), squeeze=False)
    for ax, (key, (_, run, data)) in zip(axes.flat, sorted(chosen.items())):
        counts = np.asarray(data["cluster_phase_counts"], dtype=float)
        proportions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(proportions, aspect="auto", origin="lower", cmap="Blues")
        ax.set(title=f"{key[0]}, H={key[1]}, seed {run['seed']}: NMI={data['cluster_phase_nmi']:.3f}",
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
        f"It recorded {blocked} selected runs as blocked by missing weights. Every test split in it was scored before.\n", encoding="utf-8")
    tag = TAG[prefix]
    current = frame[np.isclose(frame.alpha, 0.1)]
    phase_scores = current.pivot(index=["dataset", "horizon", "model"], columns="method", values="winkler")
    coverage = current.pivot(index=["dataset", "horizon", "model"], columns="method", values="coverage")
    loss = 100 * (coverage["phase"] - coverage["state"])
    exchange = phase_scores.loc[("exchange_rate", 96, "cissn")] if ("exchange_rate", 96, "cissn") in phase_scores.index else None
    macros = {"Runs": completed, "Cells": len(phase_scores),
              "PhaseWins": int((phase_scores.state < phase_scores.phase).sum()),
              "CoverageLossMin": f"{loss.min():.2f}", "CoverageLossMax": f"{loss.max():.2f}"}
    if exchange is not None:
        macros["ExchangeFullLoss"] = f"{100 * (exchange.state / exchange.flat_full - 1):.2f}"
    count_groups = {}
    for path in args.results.rglob("calibration_alpha0.1.json"):
        if path.parents[1].name != "scores":
            continue
        run = runs_by_name[path.parent.name]
        count_groups.setdefault((run["dataset"], run["horizon"], run["model"]), []).append(json.loads(path.read_text()))
    count_rows = []
    for (dataset, horizon, model), values in sorted(count_groups.items()):
        phase_counts = [n for value in values for n in value["phase"]["counts"].values()]
        cross_fallbacks = [sum(v is not None for k, v in value["phase_state"]["fallbacks"].items() if ":" in k) for value in values]
        count_rows.append([tex(dataset), horizon, tex(model), len(values[0]["paired_indices"]), values[0]["n_full"],
                           len(values[0]["phase"]["counts"]), f"{min(phase_counts)}--{max(phase_counts)}",
                           f"{min(cross_fallbacks)}--{max(cross_fallbacks)}"])
    table(tables / f"tab_{prefix}_counts.tex", ["Dataset", "$H$", "Model", "Paired", "Full", "Phase bins", "Counts/bin", "Cross fallbacks"], count_rows, "lrllrrrr")

    # Paired differences at alpha 0.1, relative to the comparator's mean score (fixed denominator).
    scores = current.set_index(["dataset", "horizon", "model", "method"])
    pairs = uncertainty[(uncertainty.block_multiplier == 1) & np.isclose(uncertainty.alpha, 0.1)
                        & (uncertainty.kind.isna() | (uncertainty.kind == "nested"))].copy()
    pairs["pair"] = np.where(pairs.kind == "nested", "nested", pairs.baseline)
    pairs = pairs.set_index(["dataset", "horizon", "model", "pair", "metric"])

    def relative(cell, pair):
        row = pairs.loc[(*cell, pair, "winkler")]
        base = scores.loc[(*cell, "phase_coarse" if pair == "nested" else pair), "winkler"]
        return [100 * row[key] / base for key in ("mean", "ci_low", "ci_high")]

    def interval(cell, pair):
        mean, low, high = relative(cell, pair)
        return f"${signed(mean)}$ [${signed(low)}$, ${signed(high)}$]"

    def label(cell):
        return [NAME.get(cell[0], tex(cell[0])), cell[1], NAME.get(cell[2], tex(cell[2]))]

    controls = [m for m in ("flat_paired", "flat_scaled", "phase", "phase_coarse") if m in methods]
    others = [m for m in ("flat_full", "aci") if m in methods] + (["nested"] if "phase_coarse_state" in methods else [])
    cell_keys = list(phase_scores.index)
    blocks = {cell: pairs.loc[(*cell, "flat_paired", "winkler"), "effective_blocks"] for cell in cell_keys}
    table(tables / f"tab_{prefix}_relative.tex",
          ["Dataset", "$H$", "Forecaster", "Blocks", *[LABEL[m] for m in controls]],
          [[*label(cell), f"{blocks[cell]:.1f}" + (r"$^\ast$" if blocks[cell] < 10 else ""),
            *[interval(cell, m) for m in controls]] for cell in cell_keys], "lrlr" + "l" * len(controls))
    table(tables / f"tab_{prefix}_relative_other.tex",
          ["Dataset", "$H$", "Forecaster", *[("Coarse phase+State minus Coarse phase" if m == "nested" else LABEL[m]) for m in others]],
          [[*label(cell), *[interval(cell, m) for m in others]] for cell in cell_keys], "lrl" + "l" * len(others))
    longest = int(frame.horizon.max())
    score_rows = []
    for cell in cell_keys:
        if cell[1] == longest and cell[2] == "cissn":
            flat = scores.loc[(*cell, "flat_paired")]
            for method in methods:
                row = scores.loc[(*cell, method)]
                score_rows.append([NAME.get(cell[0], tex(cell[0])), LABEL[method], f"{row.coverage:.3f}", f"{row.width:.3f}",
                                   f"{row.winkler:.3f}", f"{row.miss:.3f}", f"${signed(100 * (row.winkler / flat.winkler - 1))}$"])
    table(tables / f"tab_{prefix}_scores.tex", ["Dataset", "Calibrator", "Coverage", "Width", "Winkler", "Miss cost",
          r"Winkler vs.\ paired Flat (\%)"], score_rows, "llrrrrr")

    # Calibration-only calendar diagnostic and state-phase association, mean over seeds.
    calendar, nmi = {}, {}
    for path in args.results.rglob("calendar_diagnostic.json"):
        if path.parents[1].name == "scores":
            run = runs_by_name[path.parent.name]
            calendar.setdefault((run["dataset"], run["horizon"], run["model"]), []).append(json.loads(path.read_text()))
    for _, run, data in diagnostics:
        nmi.setdefault((run["dataset"], run["horizon"], run["model"]), []).append(data["cluster_phase_nmi"])
    scale = {cell: [value["flat_scale"] for value in values if "flat_scale" in value] for cell, values in count_groups.items()}
    calendar_rows = []
    for cell in cell_keys:
        values = [value for value in calendar.get(cell, []) if "profile_correlation" in value]
        if values and scale.get(cell):
            calendar_rows.append([*label(cell), values[0]["period"],
                                  f"{np.mean([v['profile_correlation'] for v in values]):.2f}",
                                  f"{np.mean([v['equal_arc_share'] for v in values]):.2f}",
                                  f"{values[0]['random_group_reference']:.2f}", f"{np.mean(nmi[cell]):.2f}",
                                  f"{min(scale[cell]):.2f}--{max(scale[cell]):.2f}"])
    table(tables / f"tab_{prefix}_calendar.tex", ["Dataset", "$H$", "Forecaster", "Period", "Profile corr.",
          "Equal-arc share", "Chance", "Cluster--phase NMI", "Flat scale $c$"], calendar_rows, "lrlrrrrrl")

    # One macro per reported number, named <suite><dataset><horizon><forecaster><quantity>.
    runs = pd.read_csv(args.results / "runs.csv")
    for cell in cell_keys:
        if cell[0] not in MACRO or cell[1] not in MACRO or cell[2] not in MACRO:
            continue
        name = MACRO[cell[0]] + MACRO[cell[1]] + MACRO[cell[2]]
        # The intervals cover origin sampling only, so the per-seed sign is reported beside them.
        seeds = runs[(runs.dataset == cell[0]) & (runs.horizon == cell[1]) & (runs.model == cell[2])
                     & np.isclose(runs.alpha, 0.1)].pivot(index="seed", columns="method", values="winkler")
        flat = scores.loc[(*cell, "flat_paired")]
        macros[name + "Origins"] = int(flat.n_origins)
        macros[name + "Blocks"] = f"{blocks[cell]:.1f}"
        macros[name + "Mse"] = f"{flat.mse:.3f}"
        for method in methods:
            row = scores.loc[(*cell, method)]
            macros.update({name + camel(method) + "Cov": f"{row.coverage:.3f}", name + camel(method) + "Width": f"{row.width:.3f}",
                           name + camel(method) + "Winkler": f"{row.winkler:.3f}", name + camel(method) + "Miss": f"{row.miss:.3f}",
                           name + camel(method) + "RelFlat": signed(100 * (row.winkler / flat.winkler - 1))})
        for pair in [m for m in methods if m != "state"] + (["nested"] if "phase_coarse_state" in methods else []):
            stem = name + ("Nested" if pair == "nested" else "Vs" + camel(pair))
            base = "phase_coarse" if pair == "nested" else pair
            macros.update(zip((stem + "Rel", stem + "Lo", stem + "Hi"), map(signed, relative(cell, pair))))
            macros[stem + "CovPp"] = signed(100 * pairs.loc[(*cell, pair, "coverage"), "mean"])
            macros[stem + "WidthRel"] = signed(100 * pairs.loc[(*cell, pair, "width"), "mean"] / scores.loc[(*cell, base), "width"])
            if pair != "nested":
                macros[stem + "SeedsHigher"] = f"{int((seeds.state > seeds[pair]).sum())}/{len(seeds)}"
        # The written rule names ECL and Solar at the long horizon only.
        if cell[0] in ("ECL", "solar") and cell[2] == "cissn" and cell[1] == longest and "flat_scaled" in methods:
            macros[name + "Outcome"] = reading_outcome({control: (pairs.loc[(*cell, control, "winkler"), "ci_low"],
                                                                  pairs.loc[(*cell, control, "winkler"), "ci_high"])
                                                        for control in (*CALENDAR, "flat_scaled")})
        values = [value for value in calendar.get(cell, []) if "profile_correlation" in value]
        if values and scale.get(cell):
            macros.update({name + "ProfileCorr": f"{np.mean([v['profile_correlation'] for v in values]):.2f}",
                           name + "ArcShare": f"{np.mean([v['equal_arc_share'] for v in values]):.2f}",
                           name + "ClusterPhaseNmi": f"{np.mean(nmi[cell]):.2f}",
                           name + "ScaleRange": " to ".join(sorted({f"{min(scale[cell]):.2f}", f"{max(scale[cell]):.2f}"}))})
    all_scales = [c for values in scale.values() for c in values]
    macros["ScaleRuns"] = len(all_scales)
    macros["ScaleAtLimit"] = int(sum(np.isclose(c, 0.8) or np.isclose(c, 1.2) for c in all_scales))
    macros["CalendarFallbacks"] = sum(v is not None for values in count_groups.values() for value in values
                                      for control in CALENDAR if control in value and "fallbacks" in value[control]
                                      for v in value[control]["fallbacks"].values())
    # Frozen reloads against the stored State and paired Flat per-origin scores.
    reports = [(runs_by_name[path.parent.name], json.loads(path.read_text()))
               for path in args.results.rglob("reproduction_*.json") if path.parents[1].name == "scores"]
    if reports and all("metrics" in report for _, report in reports):
        macros["ReproScoreDrift"] = sci(max(report["metrics"][m]["max_relative_drift"] for _, report in reports for m in ("width", "winkler")))
        macros["ReproCoverageDrift"] = sci(max(report["metrics"]["coverage"]["max_relative_drift"] for _, report in reports))
        for run, report in reports:
            if run["dataset"] in MACRO and run["horizon"] in MACRO:
                macros[MACRO[run["dataset"]] + MACRO[run["horizon"]] + "ReproOrigins"] = report["n_common_origins"]
    (gen / f"{prefix}_numbers.tex").write_text(
        "".join(rf"\newcommand{{\{tag}{key}}}{{{value}}}" + "\n" for key, value in macros.items()), encoding="utf-8")
    print(f"Wrote tables and four figure pairs from {len(frame)} measured summaries.")


if __name__ == "__main__":
    main()
