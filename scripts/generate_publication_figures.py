#!/usr/bin/env python
"""Generate publication figures from run artifacts when available."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from cissn.evaluation import calibration_curve, interval_width_plot, reliability_diagram


def _safe_read_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_array(path: Path) -> np.ndarray | None:
    try:
        return np.load(path)
    except Exception:
        return None


def _collect_run_dirs(results_root: Path) -> list[Path]:
    return sorted({p.parent for p in results_root.glob("**/metrics.json")})


def _make_calibration_plots(run_dirs: list[Path], out_dir: Path) -> bool:
    empirical, nominal, pincp = [], [], []
    for run in run_dirs:
        payload = _safe_read_json(run / "metrics.json")
        if not payload:
            continue
        interval = payload.get("interval", {})
        cov = interval.get("coverage_primary")
        alpha = interval.get("alpha")
        if cov is None or alpha is None:
            continue
        empirical.append(float(cov))
        nominal.append(float(alpha))
        pincp.append(float(cov))
    if not empirical:
        return False

    fig, ax = plt.subplots(figsize=(6, 5))
    calibration_curve(empirical_coverages=empirical, nominal_alphas=nominal, ax=ax, color="tab:blue")
    fig.tight_layout()
    fig.savefig(out_dir / "calibration_curve.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    reliability_diagram(pincp=pincp, bins=min(10, max(3, len(pincp) // 2)), ax=ax, color="tab:orange")
    fig.tight_layout()
    fig.savefig(out_dir / "reliability_diagram.png", dpi=200)
    plt.close(fig)
    return True


def _pick_state_run(run_dirs: list[Path]) -> Path | None:
    for run in run_dirs:
        if (run / "states.npy").exists() and (run / "lower.npy").exists() and (run / "upper.npy").exists():
            return run
    return None


def _make_state_plots(run_dirs: list[Path], out_dir: Path) -> dict[str, bool]:
    status = {"interval_vs_state": False, "state_pca": False, "state_tsne": False, "contribution_heatmap": False}
    run = _pick_state_run(run_dirs)
    if run is None:
        return status

    states = _load_array(run / "states.npy")
    lower = _load_array(run / "lower.npy")
    upper = _load_array(run / "upper.npy")
    labels = _load_array(run / "cluster_labels.npy")
    if states is None:
        return status

    states_2d = states.reshape(states.shape[0], -1)
    state_norm = np.linalg.norm(states_2d, axis=1)

    if lower is not None and upper is not None:
        width = np.mean((upper - lower).reshape(upper.shape[0], -1), axis=1)
        fig, ax = plt.subplots(figsize=(8, 5))
        interval_width_plot(
            state_norm=state_norm,
            interval_width=width,
            cluster_labels=labels if labels is not None else None,
            ax=ax,
        )
        fig.tight_layout()
        fig.savefig(out_dir / "interval_width_vs_state.png", dpi=200)
        plt.close(fig)
        status["interval_vs_state"] = True

    if states_2d.shape[0] >= 3 and states_2d.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=42)
        xy = pca.fit_transform(states_2d)
        fig, ax = plt.subplots(figsize=(7, 6))
        if labels is not None and len(labels) == len(xy):
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels, s=9, alpha=0.7, cmap="tab10")
            fig.colorbar(sc, ax=ax, label="Cluster")
        else:
            ax.scatter(xy[:, 0], xy[:, 1], s=9, alpha=0.7)
        ax.set_title("State PCA (2D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "state_pca.png", dpi=200)
        plt.close(fig)
        status["state_pca"] = True

    n = min(len(states_2d), 2000)
    if n >= 50 and states_2d.shape[1] >= 2:
        subset = states_2d[:n]
        tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, min(30, n // 50)))
        xy = tsne.fit_transform(subset)
        fig, ax = plt.subplots(figsize=(7, 6))
        if labels is not None and len(labels) >= n:
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels[:n], s=9, alpha=0.7, cmap="tab10")
            fig.colorbar(sc, ax=ax, label="Cluster")
        else:
            ax.scatter(xy[:, 0], xy[:, 1], s=9, alpha=0.7)
        ax.set_title("State t-SNE (2D)")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "state_tsne.png", dpi=200)
        plt.close(fig)
        status["state_tsne"] = True

    # Contribution proxy heatmap: state-dimension correlation matrix.
    if states_2d.shape[1] >= 2:
        corr = np.corrcoef(states_2d, rowvar=False)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        fig.colorbar(im, ax=ax, label="Correlation")
        ax.set_title("State Contribution Proxy Heatmap")
        ax.set_xlabel("State Dimension")
        ax.set_ylabel("State Dimension")
        fig.tight_layout()
        fig.savefig(out_dir / "contribution_heatmap.png", dpi=200)
        plt.close(fig)
        status["contribution_heatmap"] = True

    return status


def _make_runtime_plot(run_dirs: list[Path], out_dir: Path) -> bool:
    points = []
    for run in run_dirs:
        runtime = _safe_read_json(run / "runtime.json")
        config = _safe_read_json(run / "config.json")
        if not runtime or not config:
            continue
        pred = config.get("pred_len")
        sec = runtime.get("train_seconds")
        if pred is None or sec is None:
            continue
        points.append((int(pred), float(sec)))
    if not points:
        return False

    arr = np.array(points)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(arr[:, 0], arr[:, 1], alpha=0.7, s=18)
    unique_pred = sorted(set(arr[:, 0].astype(int)))
    med = [np.median(arr[arr[:, 0] == p, 1]) for p in unique_pred]
    ax.plot(unique_pred, med, color="tab:red", marker="o", linewidth=1.5, label="Median train time")
    ax.set_title("Runtime Scaling vs Forecast Horizon")
    ax.set_xlabel("pred_len")
    ax.set_ylabel("train_seconds")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "runtime_scaling.png", dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate publication figures from result artifacts.")
    parser.add_argument("--results_root", type=Path, default=Path("./results"))
    parser.add_argument("--output_dir", type=Path, default=Path("./results/publication_figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dirs = _collect_run_dirs(args.results_root)

    status = {}
    status["calibration_figures"] = _make_calibration_plots(run_dirs, args.output_dir)
    status.update(_make_state_plots(run_dirs, args.output_dir))
    status["runtime_scaling"] = _make_runtime_plot(run_dirs, args.output_dir)
    status["runs_detected"] = len(run_dirs)

    with (args.output_dir / "figure_generation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()

