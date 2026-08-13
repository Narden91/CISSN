"""
Run-artifact validation, split into structural validity and forecast quality.

These are two different questions and only one of them may exclude a run from
publication.

*Structural* validity asks whether the saved arrays describe a well-formed run:
finite, non-empty, shape-consistent, and internally reproducible. A structural
failure means the artifact cannot be interpreted at all, so it is a hard
failure and must exit nonzero.

*Quality* asks how good the forecast is. A poor-but-finite forecast is a valid
experimental result -- often the entire point, when the result is negative --
so quality never excludes a run. It is reported as advisory flags.

The quality references are computed from the **training** split only
(train-fitted mean, last-value persistence, seasonal-naive). An earlier version
scored against test-set variance, which made a run's publication eligibility
depend on test data; that is a test-set leak in the reporting path, so no test
statistic may be used as a reference here.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# A forecast is flagged as weak when it fails to beat the best training-split
# reference by this margin. Advisory only -- it never sets structural_passed.
_QUALITY_IMPROVEMENT_MARGIN = 0.1


def _reference_mses(
    trues: np.ndarray,
    y_train: Optional[np.ndarray],
    seasonal_period: Optional[int],
    horizon: Optional[int] = None,
) -> dict[str, float]:
    """Train-fitted reference MSEs. No test statistic may enter this function.

    Every reference is evaluated at the **same forecast horizon** as the model
    being scored. A one-step-ahead persistence error is not a valid reference
    for an h-step forecast: it is far smaller by construction, so comparing
    against it would flag good long-horizon models as weak.

    Returns an empty dict when no training data is supplied, in which case the
    quality section reports references as unavailable rather than falling back
    to a test-derived baseline.
    """
    if y_train is None:
        return {}

    y_train = np.asarray(y_train, dtype=float)
    if y_train.size == 0:
        return {}

    # Horizon at which the model was scored; trues is (samples, horizon, ...).
    if horizon is None:
        horizon = trues.shape[1] if trues.ndim >= 2 else 1
    horizon = max(1, int(horizon))

    references: dict[str, float] = {}

    # Train-fitted mean, broadcast per feature when the trailing axes line up.
    train_mean = y_train.reshape(-1, y_train.shape[-1]).mean(axis=0)
    if train_mean.shape[-1] == trues.shape[-1]:
        references["train_mean"] = float(np.mean((trues - train_mean) ** 2))
    else:
        references["train_mean"] = float(np.mean((trues - y_train.mean()) ** 2))

    def _lagged_mse(lag: int) -> Optional[float]:
        """MSE of predicting y[t] from y[t - lag] on the training split."""
        if not 0 < lag < y_train.shape[0]:
            return None
        return float(np.mean((y_train[lag:] - y_train[:-lag]) ** 2))

    # Seasonal-naive at the forecast horizon: repeat the value one full seasonal
    # cycle back, which for an h-step forecast means a lag of ceil(h/m)*m.
    if seasonal_period and seasonal_period > 0:
        cycles = int(np.ceil(horizon / seasonal_period))
        seasonal_mse = _lagged_mse(cycles * seasonal_period)
        if seasonal_mse is not None:
            references["seasonal_naive"] = seasonal_mse

    # Persistence (random walk) held over the full horizon, not one step.
    persistence_mse = _lagged_mse(horizon)
    if persistence_mse is not None:
        references["persistence"] = persistence_mse

    return {k: v for k, v in references.items() if np.isfinite(v)}


def check_structural_validity(
    preds: np.ndarray,
    trues: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> dict:
    """Validate that a run's arrays are well-formed and interpretable.

    Every failure here means the artifact cannot be read as a result at all.
    This is the only check permitted to exclude a run from publication.
    """
    preds = np.asarray(preds)
    trues = np.asarray(trues)
    failures: list[str] = []

    if preds.size == 0 or trues.size == 0:
        failures.append(
            f"empty forecast arrays: preds.size={preds.size}, trues.size={trues.size}."
        )
        return {"passed": False, "failures": failures}

    if preds.shape != trues.shape:
        failures.append(f"shape mismatch: preds{preds.shape} vs trues{trues.shape}.")
        return {"passed": False, "failures": failures}

    for name, array in (("preds", preds), ("trues", trues)):
        n_bad = int((~np.isfinite(array)).sum())
        if n_bad:
            failures.append(f"{name} contains {n_bad} non-finite value(s) (NaN or Inf).")

    if lower is not None and upper is not None:
        lower = np.asarray(lower)
        upper = np.asarray(upper)
        if lower.shape != preds.shape or upper.shape != preds.shape:
            failures.append(
                f"interval shape mismatch: lower{lower.shape}, upper{upper.shape}, preds{preds.shape}."
            )
        # All-NaN bounds mean intervals were never produced, which is valid for
        # a point-only run. Partially-NaN bounds mean a corrupted interval.
        elif not (np.isnan(lower).all() and np.isnan(upper).all()):
            n_bad = int((~np.isfinite(lower)).sum() + (~np.isfinite(upper)).sum())
            if n_bad:
                failures.append(f"interval bounds contain {n_bad} non-finite value(s).")
            n_inverted = int((upper < lower).sum())
            if n_inverted:
                failures.append(
                    f"{n_inverted} interval(s) have upper < lower, which cannot be a valid interval."
                )

    return {"passed": not failures, "failures": failures}


def check_forecast_quality(
    preds: np.ndarray,
    trues: np.ndarray,
    history: Optional[Sequence[dict]] = None,
    y_train: Optional[np.ndarray] = None,
    seasonal_period: Optional[int] = None,
    horizon: Optional[int] = None,
) -> dict:
    """Score a well-formed forecast against training-split references.

    Advisory only: the returned report has no pass/fail verdict, because a poor
    forecast is a valid result and must remain visible in every table.
    """
    preds = np.asarray(preds, dtype=float)
    trues = np.asarray(trues, dtype=float)
    flags: list[str] = []

    mse = float(np.mean((preds - trues) ** 2))
    pred_std = float(preds.std())
    true_std = float(trues.std())

    if pred_std < 0.1 * true_std:
        flags.append(
            f"pred.std()={pred_std:.6f} is below 10% of true.std()={true_std:.6f} "
            "-- predictions look like a near-constant output, not a fitted forecaster."
        )

    references = _reference_mses(trues, y_train, seasonal_period, horizon=horizon)
    if references:
        best_name = min(references, key=references.get)
        best_mse = references[best_name]
        if mse >= (1.0 - _QUALITY_IMPROVEMENT_MARGIN) * best_mse:
            flags.append(
                f"test MSE={mse:.6f} does not improve by "
                f"{_QUALITY_IMPROVEMENT_MARGIN:.0%} over the best training-split "
                f"reference ({best_name} MSE={best_mse:.6f})."
            )

    pred_mean = float(preds.mean())
    true_mean = float(trues.mean())
    if abs(pred_mean - true_mean) >= 0.5 * true_std:
        flags.append(
            f"pred.mean()={pred_mean:.6f} deviates from true.mean()={true_mean:.6f} "
            f"by >= 0.5 * true.std()={true_std:.6f} -- possible systematic bias."
        )

    if history:
        if history[-1]["vali_loss"] >= history[0]["vali_loss"]:
            flags.append(
                f"validation loss did not improve from epoch 1 ({history[0]['vali_loss']:.6f}) "
                f"to the final epoch ({history[-1]['vali_loss']:.6f})."
            )
        if len(history) <= 1:
            flags.append(f"training ran for only {len(history)} epoch(s) before stopping.")
        final_lr = history[-1].get("lr")
        if final_lr is not None and final_lr <= 1e-5:
            flags.append(
                f"final learning rate {final_lr:.2e} <= 1e-5 -- the LR schedule may have "
                "collapsed before the model finished learning."
            )

    return {
        "mse": mse,
        "reference_mse": references,
        "flags": flags,
    }


def check_forecast_sanity(
    preds: np.ndarray,
    trues: np.ndarray,
    history: Optional[Sequence[dict]] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    seasonal_period: Optional[int] = None,
    horizon: Optional[int] = None,
) -> dict:
    """Produce the combined ``sanity.json`` payload for a completed run.

    ``passed`` reflects **structural validity only**. Poor forecasts stay
    publication-visible with advisory quality flags attached.
    """
    structural = check_structural_validity(preds, trues, lower=lower, upper=upper)

    if not structural["passed"]:
        return {
            "passed": False,
            "structural_passed": False,
            "failures": structural["failures"],
            "quality": None,
            "warnings": [],
        }

    quality = check_forecast_quality(
        preds,
        trues,
        history=history,
        y_train=y_train,
        seasonal_period=seasonal_period,
        horizon=horizon,
    )
    return {
        "passed": True,
        "structural_passed": True,
        "failures": [],
        "quality": quality,
        "warnings": quality["flags"],
    }
