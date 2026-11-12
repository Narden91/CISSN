"""Paired post-hoc conformal comparisons from frozen publication checkpoints.

Run: python -m experiments.run_paper_suite --config experiments/paper_suite.yaml
Missing resources are recorded, never replaced by retraining or fabricated scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, Subset

from cissn.baselines.flat_conformal import FlatConformal
from cissn.conformal.aci import adaptive_conformal_quantiles
from cissn.conformal.phase import PhaseConformal, phase_from_timestamps
from cissn.conformal.state_conditional import StateConditionalConformal
from cissn.data.registry import get_dataset_spec
from cissn.evaluation.inference import moving_block_bootstrap_ci, moving_block_bootstrap_ratio_ci
from cissn.evaluation.metrics import per_origin_interval_scores
from experiments.run_benchmark import Experiment, environment_snapshot, set_random_seed

METRICS = ("coverage", "width", "winkler")
METHODS = ("flat_paired", "flat_full", "phase", "state", "phase_state", "aci",
           "phase_coarse", "phase_coarse_state", "flat_scaled")


def choose_flat_scale(cond_residuals, q, alpha, grid):
    """Minimise conditioning-half Winkler loss, breaking ties toward unit scale."""
    residuals = np.asarray(cond_residuals)
    q = np.asarray(q)
    grid = np.asarray(grid, dtype=float)
    if (residuals.ndim < 1 or residuals.size == 0 or not np.all(np.isfinite(residuals)) or np.any(residuals < 0)
            or not np.all(np.isfinite(q)) or np.any(q < 0) or not 0 < alpha < 1
            or grid.ndim != 1 or not grid.size or not np.all(np.isfinite(grid)) or np.any(grid <= 0)):
        raise ValueError("Provide finite non-negative residuals/quantiles and a positive scale grid.")
    q = np.broadcast_to(q, residuals.shape[1:])
    best, best_loss = None, np.inf
    # Bound temporary memory independently of the calibration length and grid size.
    for scale in grid:
        widths = scale * q
        total = 0.0
        for start in range(0, len(residuals), 64):
            excess = np.maximum(residuals[start:start + 64] - widths, 0)
            total += float(excess.sum(dtype=np.float64))
        loss = 2 * float(widths.mean()) + 2 / alpha * total / residuals.size
        if loss < best_loss or (loss == best_loss and (abs(scale - 1), scale) < (abs(best - 1), best)):
            best, best_loss = float(scale), loss
    return best


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def calibration_selection(n, stride):
    if stride < 1:
        raise ValueError("Calibration stride must be positive.")
    selected = np.arange(0, n, stride, dtype=np.int64)
    conditioning, paired = Experiment._split_calibration_indices(selected)
    return conditioning, paired, np.arange(n, dtype=np.int64)


def fit_methods(residuals, states, phases, alpha, n_clusters, seed, stride, period):
    """Fit without any test data; full flat includes unthinned calibration.

    Keep the state partition fixed across alpha so the alpha comparison changes
    quantile levels only. Its occupancy rule uses the original alpha=0.1.
    """
    conditioning, paired, full = calibration_selection(len(residuals), stride)
    cluster = StateConditionalConformal(alpha=0.1, n_clusters=n_clusters, random_state=seed)
    cluster.fit_partition(states[conditioning])
    cluster.alpha = alpha
    cluster.calibrate(states[paired], residuals[paired])
    labels = cluster.assign_clusters(states[paired])
    flat = FlatConformal(alpha)
    flat.fit(residuals[paired])
    full_flat = FlatConformal(alpha)
    full_flat.fit(residuals[full])
    phase = PhaseConformal(alpha, period).fit(residuals[paired], phases[paired])
    cross = PhaseConformal(alpha, period).fit(residuals[paired], phases[paired], labels, cluster.kmeans.n_clusters)
    groups = cluster.kmeans.n_clusters
    phase_map = np.arange(period) * groups // period
    coarse = PhaseConformal(alpha, groups).fit(residuals[paired], phase_map[phases[paired]])
    coarse_cross = PhaseConformal(alpha, groups).fit(
        residuals[paired], phase_map[phases[paired]], labels, groups)
    scale = choose_flat_scale(residuals[conditioning], flat.quantile_, alpha, np.arange(80, 121, 2) / 100)
    return {"cluster": cluster, "flat_paired": flat, "flat_full": full_flat, "phase": phase, "phase_state": cross,
            "phase_coarse": coarse, "phase_coarse_state": coarse_cross, "phase_map": phase_map,
            "flat_scale": scale,
            "conditioning_indices": conditioning, "paired_indices": paired, "full_indices": full}


def method_widths(method, fitted, phases, labels, residuals, errors, origins, horizon, alpha, gamma):
    """Dispatch explicitly so a newly registered method cannot silently become ACI."""
    capped = 0
    if method in ("flat_paired", "flat_full"):
        widths = fitted[method].quantile_
    elif method == "flat_scaled":
        widths = fitted["flat_scale"] * fitted["flat_paired"].quantile_
    elif method == "state":
        widths = np.stack([fitted["cluster"].quantiles[int(k)] for k in labels])
    elif method == "phase":
        widths = fitted[method].half_widths(phases)
    elif method == "phase_state":
        widths = fitted[method].half_widths(phases, labels)
    elif method == "phase_coarse":
        widths = fitted[method].half_widths(fitted["phase_map"][phases])
    elif method == "phase_coarse_state":
        widths = fitted[method].half_widths(fitted["phase_map"][phases], labels)
    elif method == "aci":
        widths, capped = adaptive_conformal_quantiles(
            residuals[fitted["paired_indices"]], errors, origins, horizon, alpha, gamma)
    else:
        raise ValueError(f"Unknown interval method: {method}")
    return np.broadcast_to(widths, errors.shape), capped


def calendar_diagnostic(residuals, phases, conditioning, paired, period, groups, seed=0):
    """Compare phase partitions on the later calibration half, without test inputs.

    Profiles average channels but retain horizon steps. Shares are between-phase
    variance explained on the later half; K-means labels use the earlier half.
    Missing phases and zero variance make these quantities unidentified.
    """
    profiles, counts = [], []
    for indices in (conditioning, paired):
        profile = []
        count = np.bincount(phases[indices], minlength=period)
        counts.append(count.tolist())
        if np.any(count == 0):
            return {"status": "missing_calibration_phases", "phase_counts": counts, "period": period}
        for phase in range(period):
            selected = indices[phases[indices] == phase]
            total = np.zeros(residuals.shape[1], dtype=np.float64)
            for start in range(0, len(selected), 64):
                total += residuals[selected[start:start + 64]].sum(axis=(0, 2), dtype=np.float64)
            profile.append(total / (len(selected) * residuals.shape[2]))
        profiles.append(np.asarray(profile))
    early, late = profiles
    total_variance = float(np.square(late - late.mean(axis=0)).sum())
    if total_variance == 0 or period < 2:
        return {"status": "zero_phase_variance", "phase_counts": counts, "period": period}

    def share(labels):
        within = sum(float(np.square(late[labels == k] - late[labels == k].mean(axis=0)).sum())
                     for k in np.unique(labels))
        return 1 - within / total_variance

    arcs = np.arange(period) * groups // period
    learned = KMeans(n_clusters=min(groups, period), n_init=10, random_state=seed).fit_predict(early)
    ranked = np.empty(period, dtype=int)
    ranked[np.argsort(early.mean(axis=1), kind="stable")] = arcs
    # Remove the shared horizon trend; correlation must measure phase structure.
    early_phase, late_phase = early - early.mean(axis=0), late - late.mean(axis=0)
    correlation = (float(np.corrcoef(early_phase.ravel(), late_phase.ravel())[0, 1])
                   if early_phase.std() > 0 and late_phase.std() > 0 else None)
    variations = []
    for profile in profiles:
        means = profile.mean(axis=0)
        variations.append(float(np.mean(np.divide(profile.std(axis=0), means,
                                                 out=np.zeros_like(means), where=means != 0))))
    return {"status": "ok", "period": period, "groups": groups, "phase_counts": counts,
            "phase_variation_per_step": variations, "profile_correlation": correlation,
            "equal_arc_share": share(arcs), "ranked_level_share": share(ranked),
            "kmeans_share": share(learned), "random_group_reference": (min(groups, period) - 1) / (period - 1),
            "early_profile": early.tolist(), "late_profile": late.tolist(),
            "correlation_definition": "flattened_profiles_centred_over_phases_at_each_step",
            "kmeans_phase_labels": learned.tolist(), "role": "calibration_only_descriptive"}


def score_bounds(pred, true, widths, alpha):
    lower, upper = pred - widths, pred + widths
    per_origin = per_origin_interval_scores(lower, upper, true, alpha)
    # Preserve each horizon's cellwise means without storing all intervals.
    per_horizon = np.stack([
        ((true >= lower) & (true <= upper)).mean(axis=(0, 2)),
        (upper - lower).mean(axis=(0, 2)),
        ((upper - lower) + 2 / alpha * (np.maximum(lower - true, 0) + np.maximum(true - upper, 0))).mean(axis=(0, 2)),
    ], axis=1)
    return per_origin, per_horizon


def assert_aligned(left_origins, right_origins, left_targets, right_targets):
    if not np.array_equal(left_origins, right_origins) or not np.array_equal(left_targets, right_targets):
        raise ValueError("Paired forecasts must have identical origins and targets.")


def first_target_phases(dataset, config, flag, origins, period):
    path = Path(config["root_path"]) / config["data_path"]
    if config["data"] == "solar":
        with path.open("rb") as stream:
            n = sum(1 for _ in stream)
        dates = pd.date_range("2006-01-01", periods=n, freq="10min")
    else:
        dates = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(path, usecols=["date"])["date"]))
    borders, _ = dataset._get_borders(pd.DataFrame(index=np.arange(len(dates))))
    split_index = dataset._resolve_split_index(len(borders))
    positions = borders[split_index] + dataset.seq_len + origins
    return phase_from_timestamps(dates[positions], period)


def inventory(settings):
    runs = []
    for file in sorted(Path(settings["results_root"]).glob("*/config.json")):
        config = json.loads(file.read_text(encoding="utf-8"))
        if (config.get("data") in settings["datasets"] and config.get("pred_len") in settings["horizons"]
                and config.get("seed") in settings["seeds"] and config.get("revin")):
            model = "cissn" if config.get("architecture") == "legacy" else config.get("model")
            if model in settings["models"]:
                runs.append((file.parent, config, model))
    return runs


def prepare_cache(run, config, model, encoder_run, settings):
    """Stream predictions/residuals to disk; reuse only matching complete caches."""
    from experiments.run_baseline import build_single_model, load_single_checkpoint, load_revin_checkpoint
    from cissn.models.revin import RevIN

    ckpt_root = Path(settings["checkpoints_root"])
    encoder_dir = ckpt_root / encoder_run.name
    point_dir = ckpt_root / run.name
    weights = [encoder_dir / "checkpoint.pth", encoder_dir / "checkpoint_head.pth", encoder_dir / "checkpoint_revin.pth"]
    if model != "cissn":
        weights += [point_dir / "checkpoint.pth", point_dir / "checkpoint_revin.pth"]
    missing = [str(p) for p in weights if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen weights: " + ", ".join(missing))
    encoder_config = json.loads((encoder_run / "config.json").read_text(encoding="utf-8"))
    for field in ("data", "pred_len", "seq_len", "features", "cal_fraction", "root_path", "data_path"):
        if config[field] != encoder_config[field]:
            raise ValueError(f"Point forecaster and state encoder disagree on {field}.")
    stride = settings["test_strides"][config["data"]]
    inference_batch_size = min(settings["batch_size"], config["batch_size"], encoder_config["batch_size"])
    provenance = {"config": config, "encoder_config": encoder_config, "test_stride": stride,
                  "inference_batch_size": inference_batch_size,
                  "weights_sha256": {str(p): sha256(p) for p in weights},
                  "dataset_sha256": sha256(Path(config["root_path"]) / config["data_path"]),
                  "source_sha256": {str(p): sha256(p) for p in (
                      Path(__file__), Path("cissn/data/dataset.py"), Path("experiments/run_benchmark.py"),
                      Path("experiments/run_baseline.py"), *sorted(Path("cissn/models").glob("*.py")))}}
    expected = encoder_config.get("protocol", {}).get("dataset", {}).get("actual_sha256")
    if expected and provenance["dataset_sha256"] != expected:
        raise ValueError("Dataset bytes differ from the frozen encoder run.")
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = Path(settings["output_root"]) / "cache" / run.name
    meta_file = cache / "metadata.json"
    if meta_file.exists() and json.loads(meta_file.read_text())["fingerprint"] == fingerprint:
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    args = Namespace(**{**encoder_config, "batch_size": inference_batch_size, "num_workers": 0, "no_progress": True,
                        "evidence_role": "descriptive", "require_gpu": False})
    set_random_seed(int(config["seed"]), strict=True)
    exp = Experiment(args)
    exp._load_checkpoint(encoder_dir)
    exp._set_train_mode(False)
    point_model, point_revin = None, None
    if model != "cissn":
        point_args = Namespace(**config)
        point_model = build_single_model(point_args).to(exp.device)
        point_revin = RevIN(config["enc_in"]).to(exp.device) if config.get("revin") else None
        load_single_checkpoint(point_dir, point_model, exp.device)
        load_revin_checkpoint(point_revin, point_dir, exp.device)
        point_model.eval()
        if point_revin is not None:
            point_revin.eval()
    period = get_dataset_spec(config["data"])["seasonal_period"]
    for flag in ("cal", "test"):
        dataset, _ = exp._get_data(flag)
        origins = np.arange(0, len(dataset), 1 if flag == "cal" else stride, dtype=np.int64)
        loader = DataLoader(Subset(dataset, origins.tolist()), batch_size=args.batch_size, shuffle=False, num_workers=0)
        np.save(cache / f"{flag}_origins.npy", origins)
        np.save(cache / f"{flag}_phases.npy", first_target_phases(dataset, config, flag, origins, period))
        arrays, offset = {}, 0
        with torch.no_grad():
            for x, y, _, _ in loader:
                state, pred, target = exp._forward_and_slice(x, y)
                if point_model is not None:
                    point_input = x.float().to(exp.device)
                    if point_revin is not None:
                        point_input = point_revin(point_input, "norm")
                    pred = point_model(point_input)
                    if point_revin is not None:
                        pred = point_revin(pred, "denorm")
                    pred = pred[:, -config["pred_len"]:, -1 if config["features"] == "MS" else 0:]
                values = {"states": state.cpu().numpy(), "residuals": (pred - target).abs().cpu().numpy()}
                if flag == "test":
                    values.update(pred=pred.cpu().numpy(), true=target.cpu().numpy())
                for name, value in values.items():
                    if name not in arrays:
                        arrays[name] = np.lib.format.open_memmap(cache / f"{flag}_{name}.npy", mode="w+", dtype=value.dtype,
                                                              shape=(len(origins), *value.shape[1:]))
                    arrays[name][offset:offset + len(value)] = value
                offset += len(x)
        for value in arrays.values():
            value.flush()
        if offset != len(origins):
            raise RuntimeError("Prediction cache is incomplete.")
    write_json(meta_file, {"fingerprint": fingerprint, "provenance": provenance, "period": period,
                          "environment": environment_snapshot(exp.device)})
    return cache


def diagnostics(fitted, cal_states, cal_residuals, phases, test_states, test_residuals, test_phases):
    cluster = fitted["cluster"]
    paired = fitted["paired_indices"]
    cal_labels = cluster.assign_clusters(cal_states[paired])
    labels = cluster.assign_clusters(test_states)
    scale = cal_residuals[paired].mean(axis=(1, 2))
    edges = np.quantile(scale, [1 / 3, 2 / 3])
    test_scale = test_residuals.mean(axis=(1, 2))
    table = np.zeros((cluster.kmeans.n_clusters, fitted["phase"].period), dtype=int)
    np.add.at(table, (labels, test_phases), 1)
    groups = []
    for k in range(cluster.kmeans.n_clusters):
        r = test_residuals[labels == k]
        groups.append({"cluster": k, "n_origins": len(r),
                       "residual_quantiles": np.quantile(r, [0.1, 0.5, 0.9]).tolist() if len(r) else [],
                       "horizon_mean_residual": r.mean(axis=(0, 2)).tolist() if len(r) else [],
                       "channel_mean_residual": r.mean(axis=(0, 1)).tolist() if len(r) else []})
    return {"calibration_cluster_counts": np.bincount(cal_labels, minlength=cluster.kmeans.n_clusters).tolist(),
            "test_cluster_counts": np.bincount(labels, minlength=cluster.kmeans.n_clusters).tolist(),
            "cluster_phase_counts": table.tolist(), "cluster_phase_nmi": float(normalized_mutual_info_score(labels, test_phases)),
            "cluster_residual_regime_nmi": float(normalized_mutual_info_score(labels, np.digitize(test_scale, edges))),
            "residual_regime_edges_from_calibration": edges.tolist(), "groups": groups,
            "role": "descriptive_test_diagnostic_not_selection"}


def score_cache(cache, config, settings, model):
    arrays = {name: np.load(cache / f"{name}.npy", mmap_mode="r") for name in (
        "cal_states", "cal_residuals", "cal_phases", "test_states", "test_pred", "test_true", "test_phases", "test_origins")}
    out = Path(settings["output_root"]) / "scores" / cache.name
    out.mkdir(parents=True, exist_ok=True)
    residuals = arrays["cal_residuals"]
    pred, true = arrays["test_pred"], arrays["test_true"]
    rows = []
    for alpha in settings["alphas"]:
        fitted = fit_methods(residuals, arrays["cal_states"], arrays["cal_phases"], alpha,
                             int(config.get("n_clusters", 5)), int(config["seed"]), int(config.get("calibration_stride", 1)),
                             get_dataset_spec(config["data"])["seasonal_period"])
        cluster = fitted["cluster"]
        labels = cluster.assign_clusters(arrays["test_states"])
        phase, cross = fitted["phase"], fitted["phase_state"]
        stats = {"phase": phase.stats(), "phase_state": cross.stats(),
                 "phase_coarse": fitted["phase_coarse"].stats(),
                 "phase_coarse_state": fitted["phase_coarse_state"].stats(),
                 "coarse_groups": cluster.kmeans.n_clusters, "phase_to_group": fitted["phase_map"].tolist(),
                 "flat_scale": fitted["flat_scale"],
                 "state_counts": cluster.cluster_sizes_, "state_fallbacks": cluster.cluster_fallbacks_,
                 "conditioning_indices": fitted["conditioning_indices"].tolist(),
                 "paired_indices": fitted["paired_indices"].tolist(), "n_full": len(fitted["full_indices"])}
        if alpha == 0.1:
            write_json(out / "calendar_diagnostic.json", calendar_diagnostic(
                residuals, arrays["cal_phases"], fitted["conditioning_indices"], fitted["paired_indices"],
                phase.period, cluster.kmeans.n_clusters))
            write_json(out / "diagnostics.json", diagnostics(fitted, arrays["cal_states"], residuals, arrays["cal_phases"],
                       arrays["test_states"], np.abs(pred - true), arrays["test_phases"]))
        write_json(out / f"calibration_alpha{alpha}.json", stats)
        for method in METHODS:
            widths, capped = method_widths(method, fitted, arrays["test_phases"], labels, residuals,
                                          np.abs(pred - true), arrays["test_origins"], config["pred_len"],
                                          alpha, settings["aci_gamma"])
            per_origin, per_horizon = score_bounds(pred, true, widths, alpha)
            np.save(out / f"{method}_alpha{alpha}.npy", per_origin)
            np.save(out / f"{method}_alpha{alpha}_horizon.npy", per_horizon)
            row = {"run": cache.name, "dataset": config["data"], "horizon": config["pred_len"], "model": model,
                   "seed": config["seed"], "alpha": alpha, "method": method, "stride": settings["test_strides"][config["data"]],
                   "n_origins": len(pred), "nominal_coverage": 1 - alpha, "aci_capped_cells": capped,
                   "n_calibration": len(fitted["full_indices"] if method == "flat_full" else fitted["paired_indices"]),
                   "mae": float(np.abs(pred - true).mean()), "mse": float(np.square(pred - true).mean())}
            row.update(zip(METRICS, map(float, per_origin.mean(axis=0))))
            row["absolute_coverage_error"] = abs(row["coverage"] - (1 - alpha))
            rows.append(row)
        # Reproduce the old paired result where the origins match the publication.
        original = Path(settings["results_root"]) / cache.name
        if alpha == 0.1 and model == "cissn":
            original_origins = np.load(original / "test_origin_indices.npy")
            positions = np.searchsorted(arrays["test_origins"], original_origins)
            valid = positions < len(pred)
            valid[valid] &= arrays["test_origins"][positions[valid]] == original_origins[valid]
            for method, filename in (("state", "per_origin_interval_metrics.npy"), ("flat_paired", "per_origin_interval_flat_cp_metrics.npy")):
                if np.any(valid):
                    saved = np.load(original / filename)[valid]
                    new = np.load(out / f"{method}_alpha{alpha}.npy")[positions[valid]]
                    drift = float(np.max(np.abs(new - saved) / np.maximum(np.abs(saved), 1e-8)))
                    write_json(out / f"reproduction_{method}.json", {"n_common_origins": int(valid.sum()), "max_relative_drift": drift})
                    if drift > 1e-4:
                        raise ValueError(f"Frozen reload failed to reproduce {cache.name}: {method} drift={drift}.")
    return rows


def summarize(rows, settings):
    root = Path(settings["output_root"])
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    frame.to_csv(root / "runs.csv", index=False)
    summaries, uncertainty, phase_rows = [], [], []
    for key, group in frame.groupby(["dataset", "horizon", "model", "alpha"]):
        dataset, horizon, model, alpha = key
        run_names = sorted(group.run.unique())
        origins = [np.load(root / "cache" / name / "test_origins.npy") for name in run_names]
        phases = [np.load(root / "cache" / name / "test_phases.npy") for name in run_names]
        if any(not np.array_equal(origins[0], o) for o in origins[1:]) or any(not np.array_equal(phases[0], p) for p in phases[1:]):
            raise ValueError("Seeds have different forecast origins or phases.")
        reference_targets = np.load(root / "cache" / run_names[0] / "test_true.npy", mmap_mode="r")
        for name, origin in zip(run_names[1:], origins[1:]):
            targets = np.load(root / "cache" / name / "test_true.npy", mmap_mode="r")
            assert_aligned(origins[0], origin, reference_targets, targets)
        values = {method: np.mean([np.load(root / "scores" / name / f"{method}_alpha{alpha}.npy") for name in run_names], axis=0) for method in METHODS}
        base = {"dataset": dataset, "horizon": int(horizon), "model": model, "alpha": float(alpha),
                "n_seeds": len(run_names), "n_origins": len(origins[0])}
        for method, scores in values.items():
            means = scores.mean(axis=0)
            summaries.append({**base, "method": method, **dict(zip(METRICS, means.tolist())),
                              "nominal_coverage": 1 - alpha, "absolute_coverage_error": float(abs(means[0] - (1 - alpha))),
                              "mae": float(group[group.method == method].mae.mean()), "mse": float(group[group.method == method].mse.mean())})
        pairs = [("state", baseline) for baseline in METHODS if baseline != "state"]
        pairs.append(("phase_coarse_state", "phase_coarse"))
        for method, baseline in pairs:
            delta = values[method] - values[baseline]
            delta = np.column_stack((delta, delta[:, 2] - delta[:, 1]))
            for multiplier in settings["bootstrap"]["multipliers"]:
                block = math.ceil(multiplier * horizon / settings["test_strides"][dataset])
                for column, metric in enumerate((*METRICS, "miss")):
                    record = {**base, "baseline": baseline, "metric": metric, "block_multiplier": multiplier,
                              "block_len": block, "effective_blocks": len(delta) / block, "mean": float(delta[:, column].mean())}
                    if method != "state":
                        record.update(kind="nested", method=method)
                    if block > len(delta):
                        record.update(ci_low=None, ci_high=None, status="block_longer_than_series")
                    else:
                        ci = moving_block_bootstrap_ci(delta[:, column], block, settings["bootstrap"]["n_boot"],
                            settings["bootstrap"]["seed"], settings["bootstrap"]["confidence"])
                        record.update(ci, status="descriptive_fewer_than_10_blocks" if len(delta) / block < 10 else "descriptive")
                    uncertainty.append(record)
                if method == "state" and baseline == "flat_paired" and block <= len(delta):
                    ratio = moving_block_bootstrap_ratio_ci(values[method][:, 2], values[baseline][:, 2], block,
                        settings["bootstrap"]["n_boot"], settings["bootstrap"]["seed"], settings["bootstrap"]["confidence"])
                    uncertainty.append({**base, "baseline": baseline, "metric": "winkler", "kind": "ratio_of_means",
                                        "block_multiplier": multiplier, **ratio})
            for phase in np.unique(phases[0]):
                mask = phases[0] == phase
                for column, metric in enumerate(METRICS):
                    phase_rows.append({**base, "baseline": baseline, "method": method, "metric": metric, "phase": int(phase),
                                       "n_origins_phase": int(mask.sum()), "state_minus_baseline": float(delta[mask, column].mean())})
        # Separate CIs for individual metrics use the same chronological blocks.
        for method, scores in values.items():
            block = math.ceil(horizon / settings["test_strides"][dataset])
            if block <= len(scores):
                for column, metric in enumerate(METRICS):
                    ci = moving_block_bootstrap_ci(scores[:, column], block, settings["bootstrap"]["n_boot"],
                        settings["bootstrap"]["seed"], settings["bootstrap"]["confidence"])
                    uncertainty.append({**base, "baseline": method, "metric": metric, "kind": "method_mean", "block_multiplier": 1, **ci})
    pd.DataFrame(summaries).to_csv(root / "summary.csv", index=False)
    pd.DataFrame(uncertainty).to_csv(root / "uncertainty.csv", index=False)
    pd.DataFrame(phase_rows).to_csv(root / "phase_effects.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args(argv)
    settings = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = Path(settings["output_root"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    source_paths = [*sorted(Path("cissn").rglob("*.py")), Path(__file__), args.config]
    write_json(root / "analysis_manifest.json", {"command": sys.argv, "settings": settings,
        "source_sha256": {str(p): sha256(p) for p in source_paths},
        "environment": environment_snapshot(torch.device("cuda" if torch.cuda.is_available() else "cpu"))})
    runs = inventory(settings)
    encoders = {(c["data"], c["pred_len"], c["seed"]): run for run, c, m in runs if m == "cissn"}
    records, rows = [], []
    for run, config, model in runs:
        key = (config["data"], config["pred_len"], config["seed"])
        record = {"run": run.name, "dataset": config["data"], "horizon": config["pred_len"], "model": model, "seed": config["seed"]}
        try:
            if key not in encoders:
                raise FileNotFoundError("No matched CISSN state-encoder run.")
            required = [Path(settings["checkpoints_root"]) / encoders[key].name / name for name in ("checkpoint.pth", "checkpoint_head.pth", "checkpoint_revin.pth")]
            if model != "cissn":
                required += [Path(settings["checkpoints_root"]) / run.name / name for name in ("checkpoint.pth", "checkpoint_revin.pth")]
            missing = [str(p) for p in required if not p.is_file()]
            if missing:
                raise FileNotFoundError("Missing frozen weights: " + ", ".join(missing))
            record["status"] = "ready"
            if not args.inventory_only:
                print(f"Scoring {run.name}", flush=True)
                cache = prepare_cache(run, config, model, encoders[key], settings)
                rows.extend(score_cache(cache, config, settings, model))
                record["status"] = "completed"
        except FileNotFoundError as error:
            record.update(status="blocked", reason=str(error))
        except Exception as error:
            record.update(status="failed", reason=f"{type(error).__name__}: {error}")
            records.append(record)
            write_json(root / "inventory.json", records)
            raise
        records.append(record)
        write_json(root / "inventory.json", records)
    summarize(rows, settings)
    print(f"Runs: {len(records)}, completed: {sum(r['status'] == 'completed' for r in records)}, blocked: {sum(r['status'] == 'blocked' for r in records)}")


if __name__ == "__main__":
    main(sys.argv[1:])
