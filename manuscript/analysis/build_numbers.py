"""Build every number and table the manuscript cites, from results/ only.

Run from the repository root:
    uv run python manuscript/analysis/build_numbers.py

Reads results/publication/*/{config,metrics,environment}.json, the per-origin arrays,
the inference JSON files written by scripts/paired_interval_inference.py, the hybrid
check, the ablations, the selection records, and manuscript/hpc/ecl_forecast_rank.json.
Writes macros and data to manuscript/generated/, and LaTeX tables to manuscript/tables/.
No value in the manuscript should be typed by hand; cite a macro or a generated table.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from cissn.evaluation.inference import moving_block_bootstrap_ci

ROOT = Path(__file__).resolve().parents[2]
PUB = ROOT / "results" / "publication"
OUT = ROOT / "manuscript" / "generated"
TABLES = ROOT / "manuscript" / "tables"
SELECTION = ROOT / "results" / "selection"

# Prespecified (docs/prespecification.md, sections 4, 5, 8).
N_BOOT, BOOT_SEED, MARGIN = 5000, 0, 0.01
# Not prespecified: below this many blocks a percentile CI is too coarse to read.
MIN_BLOCKS = 10
COVERAGE_BAND = (0.87, 0.93)
DATASET_ROLE = {"ECL": "confirmatory", "solar": "confirmatory", "ETTh2": "exposed", "weather": "exposed",
                "exchange_rate": "exposed", "ETTh1": "development"}
DATASET_TEX = {"ECL": "ECL", "solar": "Solar", "ETTh1": "ETTh1", "ETTh2": "ETTh2", "weather": "Weather",
               "exchange_rate": "Exchange"}
DATASET_MACRO = {"ECL": "Ecl", "solar": "Solar", "ETTh1": "EttOne", "ETTh2": "EttTwo", "weather": "Weather",
                 "exchange_rate": "Exchange"}
SOLAR_PERIOD = 144  # steps per day on the 10-minute solar data
SOLAR_PHASE_BINS = 6
MODEL_TEX = {"cissn": "CISSN", "dlinear": "DLinear", "patchtst": "PatchTST-lite",
             "deepstate": "DeepState-lite", "mc_dropout": "MC-Dropout$^\\dagger$",
             "deep_ensemble": "Deep Ens.$^\\dagger$", "hybrid": "Hybrid"}
ABLATION_TEX = {"full": "Full model", "no_structured_A": "Dense transition",
                "no_disentanglement_loss": "No disentanglement penalty", "flat_cp": "Flat CP",
                "no_correction_mlp": "No nonlinear correction", "state_dim_4": "Four-dimensional state"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_runs() -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(PUB.glob("*/metrics.json")):
        run = metrics_path.parent
        m, c, e = _json(metrics_path), _json(run / "config.json"), _json(run / "environment.json")
        is_cissn = run.name.startswith("CISSN_")
        model = ("hybrid" if c.get("architecture") == "hybrid" else "cissn") if is_cissn else c["model"]
        row = {
            "run": run.name, "model": model, "data": c["data"], "horizon": int(c["pred_len"]),
            "seed": int(c["seed"]), "revin": bool(c.get("revin")), "commit": e["git_commit"][:7],
            "device": e["cuda_device_name"], "structural_passed": m["structural_passed"],
            "n_quality_flags": len(m.get("quality_flags", [])),
            "mse": m["point"]["mse"], "mae": m["point"]["mae"],
            "variance_ratio": m["point"].get("variance_ratio"),
            "stride": int(c.get("test_origin_stride", 1)),
        }
        blocks = {"primary": "interval"}
        if is_cissn:
            blocks.update(flat="interval_flat_cp", cluster="interval_cluster_cp", scaled="interval_state_scaled")
        for tag, key in blocks.items():
            block = m[key]
            row[f"{tag}_coverage"] = block["coverage_primary"] if "coverage_primary" in block else block["coverage"]
            row[f"{tag}_width"] = block["mean_width"]
            row[f"{tag}_winkler"] = block["winkler"]
            cc = block.get("conditional_coverage") or {}
            row[f"{tag}_worst_bin"] = cc.get("worst_prespecified_bin_coverage")
            row[f"{tag}_max_dev"] = cc.get("max_coverage_deviation")
        row["interval_origin"] = m["interval"].get("interval_origin")
        stats = run / "cluster_stats.json"
        if is_cissn and stats.exists():
            clusters = _json(stats)["clusters"].values()
            row["n_fallback_clusters"] = sum(cl.get("fallback") is not None for cl in clusters)
        rows.append(row)
    return pd.DataFrame(rows)


def check_per_origin(runs: pd.DataFrame, tol: float = 1e-4) -> int:
    """The per-origin Winkler mean must reproduce metrics.json for every CISSN run."""
    checked = 0
    for r in runs[runs.model == "cissn"].itertuples():
        for tag, name in (("cluster", "per_origin_interval_metrics.npy"),
                          ("flat", "per_origin_interval_flat_cp_metrics.npy")):
            got = float(np.load(PUB / r.run / name)[:, 2].mean())
            want = getattr(r, f"{tag}_winkler")
            if abs(got - want) > tol * abs(want):
                raise SystemExit(f"{r.run} {tag}: per-origin mean {got} != metrics.json {want}")
            checked += 1
    return checked


def coverage_ci(runs: pd.DataFrame, column: int = 0) -> dict:
    """Block-bootstrap CI of marginal coverage per cell, cluster SCCP and flat CP (secondary 3)."""
    out = {}
    for (data, h), cell in runs[runs.model == "cissn"].groupby(["data", "horizon"]):
        dirs = [PUB / r for r in cell.run]
        stride = int(cell.stride.iloc[0])
        block = int(math.ceil(h / stride))
        res = {}
        for tag, name in (("cluster", "per_origin_interval_metrics.npy"),
                          ("flat", "per_origin_interval_flat_cp_metrics.npy")):
            per_origin = np.mean([np.load(d / name)[:, column] for d in dirs], axis=0)
            res[tag] = moving_block_bootstrap_ci(per_origin, block, N_BOOT, BOOT_SEED)
        out[(data, h)] = res
    return out


def reading(lo: float, hi: float) -> str:
    """Prespecified reading rule, docs/prespecification.md section 4."""
    if hi < -MARGIN:
        return "better, beyond the margin"
    if hi < 0:
        return "better, within the margin"
    if lo > 0:
        return "worse"
    if -MARGIN <= lo and hi <= MARGIN:
        return "no material difference"
    return "inconclusive"


def pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:+.{digits}f}"


def num(x: float, digits: int = 3) -> str:
    return f"{x:.{digits}f}"


class Macros:
    def __init__(self) -> None:
        self.lines = ["% Generated by manuscript/analysis/build_numbers.py. Do not edit."]

    def add(self, name: str, value) -> None:
        assert name.isalpha(), name
        self.lines.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def write(self, path: Path) -> None:
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def tabular(header: list[str], rows: list[list[str]], align: str) -> str:
    body = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    body += [" & ".join(r) + r" \\" for r in rows]
    body += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(body) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs.structural_passed.all():
        raise SystemExit("A run in results/publication failed the structural validity check.")
    runs.to_csv(OUT / "runs.csv", index=False)
    mac = Macros()
    mac.add("NPerOriginChecks", check_per_origin(runs))

    # Run inventory.
    cissn = runs[runs.model == "cissn"]
    mac.add("NRunsTotal", len(runs))
    mac.add("NRunsCissn", len(cissn))
    mac.add("NRunsHybrid", int((runs.model == "hybrid").sum()))
    mac.add("NRunsBaseline", int((~runs.model.isin(["cissn", "hybrid"])).sum()))
    for model, name in (("dlinear", "Dlinear"), ("patchtst", "Patchtst"), ("deepstate", "Deepstate"),
                        ("mc_dropout", "McDropout"), ("deep_ensemble", "DeepEnsemble")):
        mac.add("NRuns" + name, int((runs.model == model).sum()))

    # Primary endpoint and secondary 1 (ECL).
    ecl = {c["horizon"]: c for c in _json(PUB / "paired_interval_inference_ECL.json")["cells"]}
    for h, word in ((336, "Primary"), (96, "SecOne")):
        c = ecl[h]
        mac.add(f"{word}Rel", pct(c["relative_mean"]))
        mac.add(f"{word}Lo", pct(c["relative_ci_low"]))
        mac.add(f"{word}Hi", pct(c["relative_ci_high"]))
        mac.add(f"{word}Abs", num(c["mean"], 4))
        mac.add(f"{word}AbsLo", num(c["ci_low"], 4))
        mac.add(f"{word}AbsHi", num(c["ci_high"], 4))
        mac.add(f"{word}Flat", num(c["flat_winkler_mean"]))
        mac.add(f"{word}Origins", c["n_origins"])
        mac.add(f"{word}Block", c["block_len"])
        mac.add(f"{word}Blocks", num(c["n_origins"] / c["block_len"], 1))
        if word == "Primary":  # the outcome words are defined for the primary endpoint only
            mac.add(f"{word}Reading", reading(c["relative_ci_low"], c["relative_ci_high"]))
    rows = []
    for h in (336, 96):
        c = ecl[h]
        cell = cissn[(cissn.data == "ECL") & (cissn.horizon == h)]
        wins = int((cell.cluster_winkler < cell.flat_winkler).sum())
        rows.append([f"{h}{' (primary)' if h == 336 else ''}", str(c["n_origins"]), str(c["block_len"]),
                     num(c["flat_winkler_mean"]), num(cell.cluster_winkler.mean()),
                     num(c["mean"], 4), f"{pct(c['relative_mean'])}\\%",
                     f"[{pct(c['relative_ci_low'])}, {pct(c['relative_ci_high'])}]",
                     f"{wins}/{len(cell)}",
                     reading(c["relative_ci_low"], c["relative_ci_high"]) if h == 336 else "descriptive"])
    (TABLES / "tab_primary.tex").write_text(tabular(
        ["Horizon", "Origins", "Block", "Flat CP", "Cluster SCCP", "$\\Delta$", "Rel.", "95\\% CI (\\%)",
         "Seeds", "Reading"], rows, "lrrrrrrlcl"), encoding="utf-8")

    # Secondary 2 (exposed grid).
    exposed = _json(PUB / "paired_interval_inference_exposed.json")
    s = exposed["summary"]
    mac.add("ExpCells", s["n_cells"])
    mac.add("ExpCellsBetter", s["cells_with_primary_better"])
    mac.add("ExpMeanDelta", num(s["mean_of_cell_means"], 4))
    mac.add("ExpWilcoxonP", num(s["wilcoxon_two_sided_p"], 2))
    for d, v in s["dataset_mean_delta"].items():
        mac.add("ExpDelta" + DATASET_MACRO[d], num(v, 4))
    rows, cells = [], []
    for c in exposed["cells"]:
        cell = cissn[(cissn.data == c["data"]) & (cissn.horizon == c["horizon"])]
        wins = int((cell.cluster_winkler < cell.flat_winkler).sum())
        blocks = c["n_origins"] / c["block_len"]
        degenerate = blocks < MIN_BLOCKS
        rows.append([DATASET_TEX[c["data"]], str(c["horizon"]), str(c["n_origins"]), str(c["block_len"]),
                     num(blocks, 1) + ("$^\\ast$" if degenerate else ""),
                     num(c["flat_winkler_mean"]), f"{pct(c['relative_mean'])}",
                     f"[{pct(c['relative_ci_low'])}, {pct(c['relative_ci_high'])}]", f"{wins}/{len(cell)}"])
        cells.append({**c, "seed_wins": wins, "block_over_half": degenerate})
    pd.DataFrame(cells).to_csv(OUT / "cells_exposed.csv", index=False)
    mac.add("ExpDegenerate", sum(c["block_over_half"] for c in cells))
    # Sign of each CI, split by whether the cell has enough blocks for the CI to be read.
    for few, word in ((False, "Read"), (True, "Few")):
        group = [c for c in cells if c["block_over_half"] == few]
        mac.add(f"Exp{word}Cells", len(group))
        mac.add(f"Exp{word}Better", sum(c["relative_ci_high"] < 0 for c in group))
        mac.add(f"Exp{word}Worse", sum(c["relative_ci_low"] > 0 for c in group))
        mac.add(f"Exp{word}Names" + "Worse", ", ".join(f"{DATASET_TEX[c['data']]} $H={c['horizon']}$"
                                                       for c in group if c["relative_ci_low"] > 0) or "none")
        mac.add(f"Exp{word}Names" + "Better", ", ".join(f"{DATASET_TEX[c['data']]} $H={c['horizon']}$"
                                                        for c in group if c["relative_ci_high"] < 0) or "none")
    (TABLES / "tab_exposed.tex").write_text(tabular(
        ["Dataset", "$H$", "Origins", "Block", "Blocks", "Flat CP", "Rel. (\\%)", "95\\% CI (\\%)", "Seeds"],
        rows, "lrrrrrrlc"), encoding="utf-8")

    # Mechanism comparison, all CISSN cells (descriptive), with coverage and conditional coverage.
    cov = coverage_ci(runs)
    order = ["ECL", "ETTh2", "weather", "exchange_rate", "ETTh1"]
    rows, cell_rows = [], []
    for data in order:
        for h in sorted(cissn[cissn.data == data].horizon.unique()):
            cell = cissn[(cissn.data == data) & (cissn.horizon == h)]
            cc = cov[(data, h)]
            lo, hi = cc["cluster"]["ci_low"], cc["cluster"]["ci_high"]
            short = cc["cluster"]["n_origins"] / cc["cluster"]["block_len"] < MIN_BLOCKS
            band = COVERAGE_BAND[0] <= lo and hi <= COVERAGE_BAND[1]
            rec = {"data": data, "horizon": h, "role": DATASET_ROLE[data], "n_seeds": len(cell),
                   **{f"{t}_{k}": cell[f"{t}_{k}"].mean() for t in ("flat", "cluster", "scaled")
                      for k in ("winkler", "coverage", "width", "worst_bin", "max_dev")},
                   "cluster_cov_ci_low": lo, "cluster_cov_ci_high": hi, "block_over_half": short,
                   "flat_cov_ci_low": cc["flat"]["ci_low"], "flat_cov_ci_high": cc["flat"]["ci_high"],
                   "coverage_claim_allowed": band,
                   "cov_delta": cell.cluster_coverage.mean() - cell.flat_coverage.mean(),
                   "width_rel": cell.cluster_width.mean() / cell.flat_width.mean() - 1,
                   "winkler_rel": cell.cluster_winkler.mean() / cell.flat_winkler.mean() - 1,
                   # Worst-bin coverage minus the method's own marginal coverage, so a lower
                   # marginal level does not by itself make the conditional picture look worse.
                   "cluster_bin_gap": (cell.cluster_worst_bin - cell.cluster_coverage).mean(),
                   "flat_bin_gap": (cell.flat_worst_bin - cell.flat_coverage).mean(),
                   "fallback_clusters": int(cell.n_fallback_clusters.sum()),
                   "variance_ratio": cell.variance_ratio.mean()}
            cell_rows.append(rec)
            rows.append([DATASET_TEX[data] + ("$^\\S$" if data == "ETTh1" else ""), str(h),
                         num(rec["flat_winkler"]), num(rec["cluster_winkler"]), num(rec["scaled_winkler"]),
                         num(rec["cluster_coverage"]), f"[{num(lo)}, {num(hi)}]" + ("$^\\ast$" if short else ""), num(rec["flat_coverage"]),
                         num(rec["cluster_worst_bin"]), num(rec["flat_worst_bin"]), num(rec["variance_ratio"], 2)])
    cells_df = pd.DataFrame(cell_rows)
    cells_df.to_csv(OUT / "cells.csv", index=False)
    graded = cells_df[cells_df.data != "ETTh1"]
    # Counts over the graded cells only: ETTh1 is the development dataset and supports no claim.
    mac.add("AllCells", len(graded))
    mac.add("CovLowerCells", int((graded.cov_delta < 0).sum()))
    mac.add("NarrowerCells", int((graded.width_rel < 0).sum()))
    mac.add("MaxDevLowerCells", int((graded.cluster_max_dev < graded.flat_max_dev).sum()))
    mac.add("BinGapSmallerCells", int((graded.cluster_bin_gap > graded.flat_bin_gap).sum()))
    over = graded[graded.cluster_cov_ci_high > COVERAGE_BAND[1]]
    over = over[over.cluster_coverage > COVERAGE_BAND[1]]
    mac.add("CovOverCells", ", ".join(f"{DATASET_TEX[r.data]} $H={r.horizon}$" for r in over.itertuples()))
    under = graded[graded.cluster_coverage < COVERAGE_BAND[0]]
    mac.add("CovUnderCells", ", ".join(f"{DATASET_TEX[r.data]} $H={r.horizon}$" for r in under.itertuples()))
    # ECL H=96: the one large worst-bin shortfall; report the bin's size with it.
    ecl96 = [_json(PUB / r / "metrics.json")["interval_cluster_cp"]["conditional_coverage"]
             for r in cissn[(cissn.data == "ECL") & (cissn.horizon == 96)].run]
    worst = [min(cc["bins"].values(), key=lambda b: b["coverage"]) for cc in ecl96]
    mac.add("EclWorstBinCov", num(np.mean([b["coverage"] for b in worst])))
    mac.add("EclWorstBinN", f"{min(b['n_samples'] for b in worst)}--{max(b['n_samples'] for b in worst)}")
    mac.add("EclWorstBinFlat", num(ecl_worst_flat := float(np.mean(
        [_json(PUB / r / "metrics.json")["interval_flat_cp"]["conditional_coverage"]["worst_prespecified_bin_coverage"]
         for r in cissn[(cissn.data == "ECL") & (cissn.horizon == 96)].run]))))
    mac.add("CovDeltaMin", num(graded.cov_delta.min()))
    mac.add("CovDeltaMax", num(graded.cov_delta.max()))
    ecl_cells = cells_df[cells_df.data == "ECL"].set_index("horizon")
    for h, word in ((336, "Primary"), (96, "SecOne")):
        mac.add(f"{word}CovDelta", num(ecl_cells.loc[h, "cov_delta"]))
        mac.add(f"{word}WidthRel", pct(ecl_cells.loc[h, "width_rel"]))
        mac.add(f"{word}CovCluster", num(ecl_cells.loc[h, "cluster_coverage"]))
        mac.add(f"{word}CovFlat", num(ecl_cells.loc[h, "flat_coverage"]))
        mac.add(f"{word}CovLo", num(ecl_cells.loc[h, "cluster_cov_ci_low"]))
        mac.add(f"{word}CovHi", num(ecl_cells.loc[h, "cluster_cov_ci_high"]))
    trade_rows = [[DATASET_TEX[r.data] + ("$^\\S$" if r.data == "ETTh1" else ""), str(r.horizon),
                   pct(r.winkler_rel), pct(r.width_rel), f"{r.cov_delta:+.3f}", f"{r.cluster_bin_gap:+.3f}",
                   f"{r.flat_bin_gap:+.3f}", num(r.cluster_max_dev), num(r.flat_max_dev)]
                  for r in cells_df.itertuples()]
    (TABLES / "tab_trade.tex").write_text(tabular(
        ["Dataset", "$H$", "Winkler (\\%)", "Width (\\%)", "$\\Delta$ cov.", "Bin gap", "Bin gap flat",
         "Max dev.", "Max dev. flat"], trade_rows, "lrrrrrrrr"), encoding="utf-8")
    mac.add("CovCellsGraded", len(graded))
    mac.add("CovCellsInBand", int(graded.coverage_claim_allowed.sum()))
    mac.add("FallbackClustersTotal", int(cells_df.fallback_clusters.sum()))
    mac.add("CovMinCluster", num(graded.cluster_coverage.min()))
    mac.add("CovMaxCluster", num(graded.cluster_coverage.max()))
    mac.add("VarRatioMin", num(graded.variance_ratio.min(), 2))
    mac.add("VarRatioMax", num(graded.variance_ratio.max(), 2))
    scaled_vs_flat = (graded.scaled_winkler - graded.flat_winkler) / graded.flat_winkler
    mac.add("ScaledRelMean", pct(scaled_vs_flat.mean()))
    mac.add("ScaledCellsBetter", int((scaled_vs_flat < 0).sum()))
    (TABLES / "tab_mechanisms.tex").write_text(tabular(
        ["Dataset", "$H$", "Flat", "Cluster", "Scalar $\\sigma$", "Cov.", "Cov. 95\\% CI", "Cov. flat",
         "Worst bin", "Worst bin flat", "VR"], rows, "lrrrrrlrrrr"), encoding="utf-8")

    # Point accuracy within the RevIN regime. DeepState has no RevIN arm, so it gets its own
    # table (prespecification section 6: compare point accuracy only within one regime).
    models = ["cissn", "dlinear", "patchtst", "mc_dropout", "deep_ensemble"]
    pt = runs[runs.model.isin(models + ["deepstate"])]
    grid = pt.groupby(["data", "horizon", "model"]).agg(mse=("mse", "mean"), mse_sd=("mse", "std"),
                                                        mae=("mae", "mean"), n=("mse", "size")).reset_index()
    grid.to_csv(OUT / "point_grid.csv", index=False)
    rows, graded_rank = [], []
    for data in order:
        for h in sorted(grid[grid.data == data].horizon.unique()):
            g = grid[(grid.data == data) & (grid.horizon == h) & (grid.model != "deepstate")].set_index("model")
            best = g.mse.idxmin()
            # A rank means something only where CISSN has at least one comparator.
            if data != "ETTh1" and "cissn" in g.index and len(g) > 1:
                graded_rank.append(int(g.mse.rank().loc["cissn"]))
            cells_tex = []
            for mdl in models:
                if mdl not in g.index:
                    # ECL has no baseline runs; the other gaps are the 12 runs awaiting the encoder-fix rerun.
                    cells_tex.append("--" if data == "ECL" else "$\\ddagger$")
                    continue
                v = num(g.loc[mdl, "mse"]) + ("" if g.loc[mdl, "n"] == 3 else f"$^{{({int(g.loc[mdl, 'n'])})}}$")
                v = f"\\textbf{{{v}}}" if mdl == best else v
                cells_tex.append(v + f" {{\\scriptsize({num(g.loc[mdl, 'mse_sd'])})}}")
            rows.append([DATASET_TEX[data] + ("$^\\S$" if data == "ETTh1" else ""), str(h), *cells_tex])
    (TABLES / "tab_point.tex").write_text(tabular(
        ["Dataset", "$H$", *[MODEL_TEX[m] for m in models]], rows, "lr" + "r" * len(models)), encoding="utf-8")
    mac.add("CissnBestCells", sum(r == 1 for r in graded_rank))
    ds_rows = [[DATASET_TEX[r.data] + ("$^\\S$" if r.data == "ETTh1" else ""), str(r.horizon), num(r.mse), num(r.mae)]
               for r in grid[grid.model == "deepstate"].itertuples()]
    (TABLES / "tab_deepstate.tex").write_text(tabular(["Dataset", "$H$", "MSE", "MAE"], ds_rows, "lrrr"),
                                           encoding="utf-8")
    hyb_base = _json(PUB / "hybrid_vs_base_ECL.json")["runs"]
    ecl_cissn = cissn[(cissn.data == "ECL") & (cissn.horizon == 336)].mse.mean()
    base = float(np.mean([r["base_mse"] for r in hyb_base]))
    mac.add("EclCissnMse", num(ecl_cissn))
    mac.add("EclBaseMse", num(base))
    mac.add("EclCissnOverBase", pct(ecl_cissn / base - 1, 0))
    mac.add("PointCells", len(graded_rank))
    dl = grid[grid.model == "dlinear"].set_index(["data", "horizon"]).mse
    cs = grid[grid.model == "cissn"].set_index(["data", "horizon"]).mse
    ratio = (cs / dl).dropna()
    ratio = ratio[[d != "ETTh1" for d, _ in ratio.index]]
    mac.add("CissnOverDlinearMedian", num(ratio.median(), 2))
    mac.add("CissnOverDlinearMin", num(ratio.min(), 2))
    mac.add("CissnOverDlinearMax", num(ratio.max(), 2))
    mac.add("CissnBeatsDlinearCells", int((ratio < 1).sum()))
    mac.add("CissnDlinearCells", len(ratio))

    # ECL baselines (descriptive): scored on the same daily origins as the ECL CISSN runs.
    ecl_mse = grid[grid.data == "ECL"].set_index(["horizon", "model"]).mse
    for h, word in ((336, "Long"), (96, "Short")):
        for mdl, name in (("cissn", "Cissn"), ("dlinear", "Dlinear"), ("patchtst", "Patchtst"),
                          ("mc_dropout", "McDropout"), ("deep_ensemble", "DeepEnsemble")):
            mac.add(f"EclMse{name}{word}", num(ecl_mse[(h, mdl)]))
        mac.add(f"EclCissnOverDlinear{word}", num(ecl_mse[(h, "cissn")] / ecl_mse[(h, "dlinear")], 2))
        mac.add(f"EclCissnOverPatchtst{word}", num(ecl_mse[(h, "cissn")] / ecl_mse[(h, "patchtst")], 2))

    # Hybrid on ECL (prespecified point check).
    hyb = _json(PUB / "hybrid_vs_base_ECL.json")
    rows = []
    for r in hyb["runs"]:
        change = np.asarray(r["per_feature_change"])
        rows.append([str(r["seed"]), str(r["n_origins"]), num(r["base_mse"], 4), num(r["hybrid_mse"], 4),
                     f"{pct(r['mse_ratio'] - 1)}\\%", f"{int((change < 0).sum())}/{change.size}",
                     f"{pct(change.max(), 1)}\\%", f"{r['reproduction_max_abs_drift']:.1e}"])
    (TABLES / "tab_hybrid.tex").write_text(tabular(
        ["Seed", "Origins", "Base MSE", "Hybrid MSE", "Change", "Features improved", "Worst feature", "Drift"],
        rows, "rrrrrrrr"), encoding="utf-8")
    mac.add("HybridSeedsBetter", hyb["summary"]["seeds_hybrid_better"])
    mac.add("HybridMeanRatio", f"{pct(hyb['summary']['mean_mse_ratio'] - 1)}")
    mac.add("HybridWorstFeature", f"{pct(hyb['summary']['worst_feature_change_any_seed'], 0)}")

    # ECL forecast rank (computed on the cluster from pred.npy / true.npy).
    rank = pd.DataFrame(_json(ROOT / "manuscript" / "hpc" / "ecl_forecast_rank.json"))
    rank["arch"] = np.where(rank.run_dir.str.contains("hybrid"), "hybrid", "cissn")
    leg = rank[(rank.arch == "cissn") & (rank.horizon == 336)]
    hyb_r = rank[rank.arch == "hybrid"]
    mac.add("RankCissnCentered", f"{leg.pred_centered_rank99.min()}--{leg.pred_centered_rank99.max()}")
    mac.add("RankHybridCentered", f"{hyb_r.pred_centered_rank99.min()}--{hyb_r.pred_centered_rank99.max()}")
    mac.add("RankTrueCentered", str(leg.true_centered_rank99.iloc[0]))
    mac.add("RankBound", 5 + 2 * 321)

    # Ablations, ETTh1-h336 (diagnostic; check the RevIN flag of the runs).
    abl = [_json(PUB / f"ablations_ETTh1_h336_s{s}.json") for s in (42, 123, 456)]
    rows = []
    for name in abl[0]:
        res = [a[name] for a in abl]
        mse = [r["point"]["mse"] for r in res]
        # The flat_cp variant has no separate flat comparator.
        flat = np.mean([r["interval_flat_cp"].get("winkler", np.nan) for r in res])
        rows.append([ABLATION_TEX.get(name, name), f"{np.mean(mse):.3f} $\\pm$ {np.std(mse, ddof=1):.3f}",
                     num(np.mean([r["point"].get("variance_ratio", np.nan) for r in res]), 2),
                     num(np.mean([r["interval"]["coverage_primary"] for r in res])),
                     num(np.mean([r["interval"]["winkler"] for r in res])),
                     num(flat) if np.isfinite(flat) else "--"])
    (TABLES / "tab_ablation.tex").write_text(tabular(
        ["Variant", "MSE", "VR", "Coverage", "Winkler", "Flat Winkler"], rows, "lrrrrr"), encoding="utf-8")
    # config holds the ablation name; the RevIN arm is tagged "fullrevin" in the setting name.
    mac.add("AblationRevin", "yes" if any("fullrevin" in a["full"]["setting"] for a in abl) else "no")
    mac.add("AblationVarRatio", num(np.mean([a["full"]["point"]["variance_ratio"] for a in abl]), 2))

    # Step 3b.2: promote state-scaled CP only if it beats flat and cluster on >= 2 of 3 seeds (validation).
    sel = [_json(p) for p in sorted(SELECTION.glob("*scalecond*/selection.json"))]
    wk = [{k: s[k]["winkler"] for k in ("interval_flat_cp", "interval_cluster_cp", "interval_state_scaled")}
          for s in sel]
    mac.add("SelScaleWins", f"{sum(w['interval_state_scaled'] < min(w['interval_flat_cp'], w['interval_cluster_cp']) for w in wk)}/{len(wk)}")
    spread = max((max(w.values()) - min(w.values())) / w["interval_flat_cp"] for w in wk)
    mac.add("SelScaleSpread", num(100 * spread, 1))

    # Step 5: hybrid variant selection on validation, each run against its own frozen DLinear base.
    hfm = _json(SELECTION / "hybrid_feature_mse.json")["runs"]
    for variant, word in (("hybrid_anchored_lc", "Anchored"), ("hybrid_legacy", "Legacy"),
                          ("hybrid_anchored_revin", "AnchoredRevin")):
        rel = [np.mean(r["hybrid_mse"]) / np.mean(r["base_mse"]) - 1 for r in hfm if variant in r["setting"]]
        if rel:
            mac.add(f"SelHybrid{word}", pct(float(np.mean(rel))))
            mac.add(f"SelHybrid{word}Seeds", f"{sum(x < 0 for x in rel)}/{len(rel)}")

    # Validation-only selection decisions.
    r1 = _json(SELECTION / "r1_decision.json")["datasets"]
    for d, v in r1.items():
        tag = DATASET_MACRO[d]
        mac.add(f"SelValMse{tag}Revin", num(v["val_mse_revin"]))
        mac.add(f"SelValMse{tag}Plain", num(v["val_mse_no_revin"]))
        mac.add(f"SelCov{tag}Revin", num(v["coverage_revin"]))
        mac.add(f"SelCov{tag}Plain", num(v["coverage_no_revin"]))

    mac.add("CissnTrailsDlinearCells", int((ratio > 1).sum()))
    dev = cells_df[cells_df.data == "ETTh1"].winkler_rel
    mac.add("DevRelBest", pct(dev.min(), 1))
    mac.add("DevRelWorst", pct(dev.max(), 1))
    band = cells_df[(cells_df.data != "ETTh1") & cells_df.coverage_claim_allowed]
    mac.add("CovBandCells", ", ".join(f"{DATASET_TEX[r.data]} $H={r.horizon}$" for r in band.itertuples()))

    # ETTh1-h336 headroom diagnostic: reuses ETTh1 test arrays, so it is development evidence only.
    head = [_json(p) for p in sorted((ROOT / "results" / "validation").glob("conditioning_headroom_revin_seed*.json"))]
    assert all(h["evidence_role"] == "exploratory_test_reuse" for h in head)
    flat_mean = np.mean([c["flat_cp"]["winkler"] for h in head for c in h["per_cut"]])
    for key, word in (("cluster_cp", "Cluster"), ("state_scaled_cp", "Scalar"),
                      ("state_scaled_per_cell_cp", "PerCell"), ("label_informed_per_sample_reference", "Label")):
        delta = np.mean([h["summary_vs_flat"][key]["mean_winkler_delta"] for h in head])
        won = sum(h["summary_vs_flat"][key]["cuts_won"] for h in head)
        total = sum(h["summary_vs_flat"][key]["cuts_total"] for h in head)
        mac.add(f"Diag{word}Rel", pct(delta / flat_mean, 1))
        mac.add(f"Diag{word}Won", f"{won}/{total}")
    vd = [h["variance_decomposition"] for h in head]
    mac.add("DiagPerSampleShare", num(100 * np.mean([v["per_sample_fraction"] for v in vd]), 1))
    mac.add("DiagPerCellShare", num(100 * np.mean([v["per_cell_fraction"] for v in vd]), 1))

    # Second confirmatory test (solar): docs/prespecification_solar.md.
    def solar_cell(method: str, conf: str, h: int) -> dict:
        d = _json(PUB / f"paired_interval_inference_solar_{method}_{conf}.json")
        return next(c for c in d["cells"] if c["horizon"] == h)

    endpoints = {"One": "primary", "Two": "state_scaled"}  # E1 cluster SCCP, E2 per-cell scaled CP
    wide = {w: solar_cell(m, "0.975", 336) for w, m in endpoints.items()}
    narrow = {w: solar_cell(m, "0.95", 336) for w, m in endpoints.items()}
    # Holm steps (section 3): an endpoint whose 97.5% upper bound is below zero is read at 97.5%,
    # and the other one at 95%; if neither is below zero, both are read at 97.5%.
    any_below = any(c["relative_ci_high"] < 0 for c in wide.values())
    for w in endpoints:
        used = wide[w] if (wide[w]["relative_ci_high"] < 0 or not any_below) else narrow[w]
        short = solar_cell(endpoints[w], "0.95", 96)
        mac.add(f"Solar{w}Rel", pct(narrow[w]["relative_mean"], 1))
        mac.add(f"Solar{w}Lo", pct(narrow[w]["relative_ci_low"], 1))
        mac.add(f"Solar{w}Hi", pct(narrow[w]["relative_ci_high"], 1))
        mac.add(f"Solar{w}LoWide", pct(wide[w]["relative_ci_low"], 1))
        mac.add(f"Solar{w}HiWide", pct(wide[w]["relative_ci_high"], 1))
        mac.add(f"Solar{w}Reading", reading(used["relative_ci_low"], used["relative_ci_high"]))
        mac.add(f"Solar{w}ShortRel", pct(short["relative_mean"], 1))
        mac.add(f"Solar{w}ShortLo", pct(short["relative_ci_low"], 1))
        mac.add(f"Solar{w}ShortHi", pct(short["relative_ci_high"], 1))
    c1, c1s = narrow["One"], solar_cell("primary", "0.95", 96)
    mac.add("SolarFlat", num(c1["flat_winkler_mean"]))
    mac.add("SolarOrigins", c1["n_origins"])
    mac.add("SolarBlock", c1["block_len"])
    mac.add("SolarBlocks", num(c1["n_origins"] / c1["block_len"], 1))
    mac.add("SolarShortBlocks", num(c1s["n_origins"] / c1s["block_len"], 0))
    sol = cissn[(cissn.data == "solar") & (cissn.horizon == 336)]
    for tag, word in (("flat", "Flat"), ("cluster", "Cluster"), ("scaled", "Scaled")):
        mac.add(f"SolarCov{word}", num(sol[f"{tag}_coverage"].mean()))
        mac.add(f"SolarWidth{word}", num(sol[f"{tag}_width"].mean()))
    mac.add("SolarWidthRel", pct(sol.cluster_width.mean() / sol.flat_width.mean() - 1, 1))
    mac.add("SolarSeedsCluster", f"{int((sol.cluster_winkler < sol.flat_winkler).sum())}/{len(sol)}")
    mac.add("SolarSeedsScaled", f"{int((sol.scaled_winkler < sol.flat_winkler).sum())}/{len(sol)}")
    # Effect by time of day (secondary 3): the origin index modulo the daily period, in equal bins.
    sol_dirs = [PUB / r for r in sol.run]
    sol_origins = [np.load(d / "test_origin_indices.npy") for d in sol_dirs]
    assert all(np.array_equal(sol_origins[0], o) for o in sol_origins[1:])
    cl = np.mean([np.load(d / "per_origin_interval_metrics.npy")[:, 2] for d in sol_dirs], axis=0)
    fl = np.mean([np.load(d / "per_origin_interval_flat_cp_metrics.npy")[:, 2] for d in sol_dirs], axis=0)
    phase = sol_origins[0] % SOLAR_PERIOD
    step = SOLAR_PERIOD // SOLAR_PHASE_BINS
    by_phase = np.array([(cl[m] - fl[m]).mean() / fl[m].mean()
                         for m in ((phase >= b * step) & (phase < (b + 1) * step) for b in range(SOLAR_PHASE_BINS))])
    mac.add("SolarPhaseMin", pct(by_phase.min(), 1))
    mac.add("SolarPhaseMax", pct(by_phase.max(), 1))
    mac.add("SolarPhaseBetter", f"{int((by_phase < 0).sum())}/{len(by_phase)}")
    rows = []
    for h in (336, 96):
        a, b = solar_cell("primary", "0.95", h), solar_cell("state_scaled", "0.95", h)
        rows.append([str(h), str(a["n_origins"]), num(a["n_origins"] / a["block_len"], 1), num(a["flat_winkler_mean"]),
                     f"{pct(a['relative_mean'], 1)}\\%", f"[{pct(a['relative_ci_low'], 1)}, {pct(a['relative_ci_high'], 1)}]",
                     f"{pct(b['relative_mean'], 1)}\\%", f"[{pct(b['relative_ci_low'], 1)}, {pct(b['relative_ci_high'], 1)}]"])
    (TABLES / "tab_solar.tex").write_text(tabular(
        ["$H$", "Origins", "Blocks", "Flat CP", "E1 rel.", "E1 95\\% CI (\\%)", "E2 rel.", "E2 95\\% CI (\\%)"],
        rows, "rrrrrlrl"), encoding="utf-8")

    # Adaptive conformal inference against flat CP (descriptive; ACI uses test errors as they arrive).
    aci = []
    for name in ("ECL", "solar", "exposed"):
        for c in _json(PUB / f"aci_vs_flat_{name}.json")["cells"]:
            pairs = [r["n_capped"] / (len(np.load(PUB / r["run_dir"] / "test_origin_indices.npy"))
                                      * c["horizon"] * _json(PUB / r["run_dir"] / "config.json")["enc_in"])
                     for r in c["runs"]]
            aci.append({**c, "capped_share": float(np.mean(pairs)),
                        "few_blocks": c["n_origins"] / c["block_len"] < MIN_BLOCKS})
    aci_df = pd.DataFrame(aci)
    aci_df["rank"] = aci_df.data.map({d: i for i, d in enumerate(["ECL", "solar", "ETTh2", "weather", "exchange_rate"])})
    aci_df = aci_df.sort_values(["rank", "horizon"])
    rows = [[DATASET_TEX[r.data], str(r.horizon), f"{pct(r.relative_mean, 1)}\\%",
             f"[{pct(r.relative_ci_low, 1)}, {pct(r.relative_ci_high, 1)}]" + ("$^\\ast$" if r.few_blocks else ""),
             num(r.flat_coverage), num(r.aci_coverage), num(r.flat_width, 2), num(r.aci_width, 2),
             f"{100 * r.capped_share:.0f}"] for r in aci_df.itertuples()]
    (TABLES / "tab_aci.tex").write_text(tabular(
        ["Dataset", "$H$", "Rel. Winkler", "95\\% CI (\\%)", "Cov. flat", "Cov. ACI", "Width flat", "Width ACI",
         "Capped (\\%)"], rows, "lrrlrrrrr"), encoding="utf-8")
    mac.add("AciCells", len(aci_df))
    mac.add("AciCellsBetter", int((aci_df.relative_mean < 0).sum()))

    def aci_cell(data: str, h: int):
        return aci_df[(aci_df.data == data) & (aci_df.horizon == h)].iloc[0]

    for data, word in (("ECL", "Ecl"), ("solar", "Solar")):
        for h, suffix in ((336, ""), (96, "Short")):
            r = aci_cell(data, h)
            mac.add(f"Aci{word}{suffix}Rel", pct(r.relative_mean, 1))
            mac.add(f"Aci{word}{suffix}Lo", pct(r.relative_ci_low, 1))
            mac.add(f"Aci{word}{suffix}Hi", pct(r.relative_ci_high, 1))
    r = aci_cell("ECL", 336)
    mac.add("AciEclCovFlat", num(r.flat_coverage))
    mac.add("AciEclCovAci", num(r.aci_coverage))
    for data, word in (("ETTh2", "EttTwo"), ("weather", "Weather"), ("exchange_rate", "Exchange")):
        sub = aci_df[aci_df.data == data]
        mac.add(f"Aci{word}Min", pct(sub.relative_mean.min(), 1))
        mac.add(f"Aci{word}Max", pct(sub.relative_mean.max(), 1))
    wsub = aci_df[aci_df.data == "weather"]
    mac.add("AciWeatherWidthMax", pct((wsub.aci_width / wsub.flat_width - 1).max(), 0))
    mac.add("AciWeatherCappedMax", f"{100 * wsub.capped_share.max():.0f}")
    mac.add("AciWeatherCovAci", num(wsub.aci_coverage.mean()))
    mac.add("AciWeatherCovFlat", num(wsub.flat_coverage.mean()))

    drift = max(r["reproduction_max_relative_drift"] for name in ("ECL", "solar", "exposed")
                for c in _json(PUB / f"aci_vs_flat_{name}.json")["cells"] for r in c["runs"])
    mantissa, exponent = f"{drift:.1e}".split("e")
    mac.add("AciMaxDrift", f"${mantissa}\\times 10^{{{int(exponent)}}}$")

    # Phase dependence on ETTh2 (post hoc, exposed test split): effect by origin index modulo 24.
    for h, word in ((336, "Long"), (96, "Short")):
        ett = cissn[(cissn.data == "ETTh2") & (cissn.horizon == h)]
        ett_dirs = [PUB / r for r in ett.run]
        ett_idx = np.load(ett_dirs[0] / "test_origin_indices.npy") % 24
        e_cl = np.mean([np.load(d / "per_origin_interval_metrics.npy")[:, 2] for d in ett_dirs], axis=0)
        e_fl = np.mean([np.load(d / "per_origin_interval_flat_cp_metrics.npy")[:, 2] for d in ett_dirs], axis=0)
        e_rel = np.array([(e_cl[ett_idx == p] - e_fl[ett_idx == p]).mean() / e_fl[ett_idx == p].mean()
                          for p in range(24)])
        mac.add(f"PhaseEttTwoMin{word}", pct(e_rel.min(), 1))
        mac.add(f"PhaseEttTwoMax{word}", pct(e_rel.max(), 1))
        mac.add(f"PhaseEttTwoAll{word}", pct((e_cl - e_fl).mean() / e_fl.mean(), 1))
        mac.add(f"PhaseEttTwoZero{word}", pct(e_rel[0], 1))

    # Which commit produced which runs.
    step7 = sol.commit.iloc[0]
    assert sol.commit.nunique() == 1
    four = runs[runs.model.isin(["dlinear", "patchtst", "mc_dropout", "deep_ensemble"]) & (runs.data != "ECL")]
    mac.add("CommitStepSeven", step7)
    mac.add("CommitFirstBaselines", four[four.commit != step7].commit.mode().iloc[0])
    mac.add("NRerun", int((four.commit == step7).sum()))
    mac.add("CommitEcl", cissn[cissn.data == "ECL"].commit.mode().iloc[0])
    mac.add("CommitExposed", cissn[cissn.data.isin(["ETTh2", "weather", "exchange_rate", "ETTh1"])].commit.mode().iloc[0])

    # Datasets: chronological split sizes (windows) at the primary horizon, from a real run config.
    from cissn.data.data_loader import get_data_loader
    rows = []
    for data in ["ECL", "solar", "ETTh2", "weather", "exchange_rate", "ETTh1"]:
        cfg = json.loads((PUB / cissn[(cissn.data == data) & (cissn.horizon == 336)].run.iloc[0]
                          / "config.json").read_text(encoding="utf-8"))
        cfg = {**cfg, "root_path": str(ROOT / cfg["root_path"]), "num_workers": 0}
        sizes = [len(get_data_loader(cfg, flag)[0]) for flag in ("train", "cal", "val", "test")]
        freq = {"h": "1 h", "t": "10 min", "d": "1 day"}[cfg["freq"]]
        rows.append([DATASET_TEX[data], str(cfg["enc_in"]), freq, DATASET_ROLE[data], *map(str, sizes)])
    (TABLES / "tab_datasets.tex").write_text(tabular(
        ["Dataset", "Channels", "Freq.", "Role", "Train", "Cal.", "Val.", "Test"], rows, "lrllrrrr"),
        encoding="utf-8")

    # Provenance: which machine produced each group of runs. The manuscript names machines by
    # letter and omits commits; runs.csv keeps the recorded device name and commit.
    prov = runs.groupby(["model", "device"]).size().reset_index(name="runs")
    machine = {d: chr(ord("A") + i) for i, d in enumerate(sorted(runs.device.unique()))}
    rows = [[MODEL_TEX.get(r.model, r.model), machine[r.device], str(r.runs)] for r in prov.itertuples()]
    (TABLES / "tab_provenance.tex").write_text(tabular(["Model", "Machine", "Runs"], rows, "lll"),
                                            encoding="utf-8")

    mac.write(OUT / "numbers.tex")
    print(f"wrote {len(list(OUT.iterdir()))} files to {OUT}")


if __name__ == "__main__":
    main()
