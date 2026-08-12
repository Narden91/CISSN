"""
Convergence and degenerate-prediction sanity gate.

Catches the failure mode where a run completes without error but the model
learned nothing useful -- e.g. training on corrupted/white-noise data, or an
LR schedule that decays to ~0 before the model has moved from initialization.
None of these raise exceptions on their own; they only show up as a model
that predicts close to the training mean, which check_forecast_sanity()
flags directly from the saved pred/true arrays and training history.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class SanityCheckFailure(RuntimeError):
    """Raised by check_forecast_sanity(strict=True) when a hard check fails."""


def check_forecast_sanity(
    preds: np.ndarray,
    trues: np.ndarray,
    history: Optional[list[dict]] = None,
    strict: bool = False,
) -> dict:
    """Run a small set of convergence/degeneracy checks on a completed run.

    Args:
        preds, trues: Test-set predictions and targets, any matching shape.
        history: Optional per-epoch training history as saved to history.json
            (list of dicts with at least 'vali_loss' and 'lr' keys).
        strict: If True, raise SanityCheckFailure on any hard failure instead
            of only recording it.

    Returns:
        {'passed': bool, 'failures': [...], 'warnings': [...]}
    """
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    failures: list[str] = []
    warnings: list[str] = []

    pred_std = float(preds.std())
    true_std = float(trues.std())
    if pred_std < 0.1 * true_std:
        failures.append(
            f"pred.std()={pred_std:.6f} is below 10% of true.std()={true_std:.6f} "
            "-- predictions look like a near-constant output, not a fitted forecaster."
        )

    mse = float(np.mean((preds - trues) ** 2))
    true_var = float(trues.var())
    if mse >= 0.9 * true_var:
        failures.append(
            f"test MSE={mse:.6f} is not below 90% of true.var()={true_var:.6f} "
            "-- the model is no better than predicting the mean."
        )

    pred_mean = float(preds.mean())
    true_mean = float(trues.mean())
    if abs(pred_mean - true_mean) >= 0.5 * true_std:
        warnings.append(
            f"pred.mean()={pred_mean:.6f} deviates from true.mean()={true_mean:.6f} "
            f"by >= 0.5 * true.std()={true_std:.6f} -- possible systematic bias."
        )

    if history:
        if history[-1]["vali_loss"] >= history[0]["vali_loss"]:
            warnings.append(
                f"validation loss did not improve from epoch 1 ({history[0]['vali_loss']:.6f}) "
                f"to the final epoch ({history[-1]['vali_loss']:.6f})."
            )
        if len(history) <= 1:
            warnings.append(f"training ran for only {len(history)} epoch(s) before stopping.")
        final_lr = history[-1].get("lr")
        if final_lr is not None and final_lr <= 1e-5:
            warnings.append(
                f"final learning rate {final_lr:.2e} <= 1e-5 -- the LR schedule may have "
                "collapsed before the model finished learning (see run_benchmark.py's "
                "adjust_learning_rate 'type1' policy paired with a large train_epochs)."
            )

    report = {"passed": not failures, "failures": failures, "warnings": warnings}
    if failures and strict:
        raise SanityCheckFailure("; ".join(failures))
    return report
