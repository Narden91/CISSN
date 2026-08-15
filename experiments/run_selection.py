#!/usr/bin/env python
"""Chronological validation-only selection controller.

`run_benchmark.py`'s `enforce_evidence_contract` hard-raises on
`--evidence_role selection` and names the reason: the test-evaluation runner
always constructs a test loader, so nothing that runs through it can be a
clean pre-test selection procedure. This module is the controller it names.

It trains and calibrates exactly like `run_benchmark.py` (same `Experiment`/
`HybridExperiment.train()`, same conditioning-fit and quantile-calibration
split), then scores every interval mechanism on the VALIDATION split instead
of test. The test loader is never constructed: `_get_data('test')` is
overridden to raise, and `test()` is overridden to raise, so a future edit
that reaches for test data inside inherited code fails loudly rather than
silently leaking test information into a selection decision.

Chronology is sound because calibration is a tail of train and strictly
precedes validation (dataset.py): conditioning-fit -> quantile-calibration ->
validation scoring has no lookahead.

Usage mirrors run_benchmark.py, with --evidence_role forced to "selection":
    uv run python experiments/run_selection.py --data ETTh1 --pred_len 336 \
        --seed 42 --revin --conformal_conditioning scale --scale_geometry per_cell
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    from .run_benchmark import (
        Experiment,
        HybridExperiment,
        StateConditionalConformal,
        build_protocol_manifest,
        build_run_setting,
        build_setting_name,
        conditional_coverage_by_bin,
        environment_snapshot,
        fit_coverage_bin_edges,
        mean_absolute_error,
        mean_squared_error,
        parse_args as parse_benchmark_args,
        per_origin_interval_scores,
        print_run_header,
        require_clean_source,
        save_json,
        set_random_seed,
    )
except ImportError:
    from run_benchmark import (
        Experiment,
        HybridExperiment,
        StateConditionalConformal,
        build_protocol_manifest,
        build_run_setting,
        build_setting_name,
        conditional_coverage_by_bin,
        environment_snapshot,
        fit_coverage_bin_edges,
        mean_absolute_error,
        mean_squared_error,
        parse_args as parse_benchmark_args,
        per_origin_interval_scores,
        print_run_header,
        require_clean_source,
        save_json,
        set_random_seed,
    )


def enforce_selection_contract(args) -> None:
    """Symmetric to run_benchmark.py's enforce_evidence_contract: this
    runner exists ONLY for selection. Any other evidence role is refused so
    the two runners stay mutually exclusive -- a selection decision must
    come from this controller, and a confirmation/development run must come
    from run_benchmark.py, never a driver that could accidentally read test.
    """
    role = getattr(args, "evidence_role", "development")
    if role != "selection":
        raise ValueError(
            f"run_selection.py only supports --evidence_role selection; got {role!r}. "
            "Use run_benchmark.py for development or confirmation runs."
        )


class _NoTestDataMixin:
    """Fail-closed guard: raise the instant anything asks for the test split.

    This is the structural guarantee, not a convention: every data access in
    Experiment/HybridExperiment goes through _get_data, so intercepting it
    here means a future edit to inherited train()/test() logic that reaches
    for test data fails immediately and loudly, rather than silently
    leaking test information into a selection decision.
    """

    def _get_data(self, flag):
        if flag == "test":
            raise RuntimeError(
                "Selection runs must never construct the test loader. "
                "This experiment scores validation only."
            )
        return super()._get_data(flag)

    def test(self, setting):
        raise RuntimeError(
            "Selection runs do not evaluate on test; call evaluate_on_validation() instead."
        )


class SelectionExperiment(_NoTestDataMixin, Experiment):
    """Legacy-architecture experiment restricted to train/val/cal splits."""


class HybridSelectionExperiment(_NoTestDataMixin, HybridExperiment):
    """Hybrid-architecture experiment restricted to train/val/cal splits."""


def evaluate_on_validation(exp, setting: str) -> dict:
    """Score every calibrated interval mechanism on the validation split.

    A focused analogue of Experiment.test(): reuses the same forward pass
    (_forward_and_slice), the same interval-scoring helpers
    (_predict_intervals, _compare_against_flat_conformal,
    _compare_against_secondary_conformal) that test() uses against test
    data, applied here to validation data instead. Deliberately does not
    write pred.npy/true.npy/states.npy or run check_forecast_sanity --
    those are publication artifacts for a sealed confirmation run, not a
    selection probe. selection.json uses the same field names as
    metrics.json so a downstream reader does not need a second schema.
    """
    vali_data, vali_loader = exp._get_data(flag="val")

    path = Path(exp.args.checkpoints) / setting
    exp._load_checkpoint(path)
    eval_start = time.time()

    preds, trues, vali_states = [], [], []
    exp._set_train_mode(False)
    with torch.no_grad():
        for batch_x, batch_y, _batch_x_mark, _batch_y_mark in vali_loader:
            final_state, outputs, batch_y = exp._forward_and_slice(batch_x, batch_y)
            preds.append(outputs.detach().cpu().numpy())
            trues.append(batch_y.detach().cpu().numpy())
            vali_states.append(final_state.detach().cpu().numpy())
    preds = exp._concatenate_batches(preds, "prediction")
    trues = exp._concatenate_batches(trues, "target")
    vali_states = exp._concatenate_batches(vali_states, "state")

    mae = mean_absolute_error(trues.flatten(), preds.flatten())
    mse = mean_squared_error(trues.flatten(), preds.flatten())
    rmse = float(np.sqrt(mse))

    interval_metrics: dict = {}
    flat_comparison: dict = {}
    secondary_comparison: dict = {}
    conditioning_mode = getattr(exp.args, "conformal_conditioning", "cluster")
    if hasattr(exp, "conformal") and exp.conformal.calibrated:
        lower_np, upper_np, cluster_labels = exp._predict_intervals(vali_states, preds)
        interval_metrics = exp._score_interval_comparator(
            lower_np, upper_np, trues, exp.conformal.coverage_scope, test_states=vali_states
        )
        flat_comparison = exp._compare_against_flat_conformal(preds, trues, test_states=vali_states)
        secondary_comparison = exp._compare_against_secondary_conformal(vali_states, preds, trues)

    cluster_result = interval_metrics if conditioning_mode == "cluster" else secondary_comparison
    scaled_result = secondary_comparison if conditioning_mode == "cluster" else interval_metrics

    return {
        "setting": setting,
        "evidence_role": "selection",
        "split_scored": "validation",
        "test_loader_constructed": False,
        "validation_mse": mse,
        "validation_mae": mae,
        "validation_rmse": rmse,
        "interval": interval_metrics,
        "interval_flat_cp": flat_comparison,
        "interval_cluster_cp": cluster_result,
        "interval_state_scaled": scaled_result,
        "conditioning_mode": conditioning_mode,
        "eval_seconds": time.time() - eval_start,
        "config": {k: v for k, v in vars(exp.args).items() if k != "protocol"},
        "protocol_hash": exp.args.protocol.get("protocol_hash") if hasattr(exp.args, "protocol") else None,
    }


def parse_args(argv: Optional[list[str]] = None):
    """Reuses run_benchmark.py's full CLI unchanged, so a selection command
    and its eventual confirmation command differ only in runner name and
    --evidence_role -- never in what flags exist."""
    args = parse_benchmark_args(argv)
    args.evidence_role = "selection"
    return args


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    enforce_selection_contract(args)
    require_clean_source(args)
    set_random_seed(args.seed, strict=args.strict_determinism)
    args.protocol = build_protocol_manifest(args)

    setting = build_run_setting(args, build_setting_name(args))
    print_run_header("CISSN selection", args, setting)

    exp = HybridSelectionExperiment(args) if args.architecture == "hybrid" else SelectionExperiment(args)
    print("\n[1/2] Training and calibration (train/val/cal only)")
    exp.train(setting)
    print("\n[2/2] Validation scoring (test loader never constructed)")
    result = evaluate_on_validation(exp, setting)

    output_dir = Path(args.results_dir) / setting
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "selection.json", result)
    save_json(output_dir / "environment.json", environment_snapshot(exp.device))
    save_json(output_dir / "protocol.json", args.protocol)

    winkler = result["interval"].get("winkler") if result["interval"] else None
    flat_winkler = result["interval_flat_cp"].get("winkler") if result["interval_flat_cp"] else None
    if winkler is not None and flat_winkler is not None:
        print(f"Validation | primary winkler={winkler:.4f} | flat CP winkler={flat_winkler:.4f}")
    print(f"Saved selection artifacts to {output_dir}")


if __name__ == "__main__":
    main()
