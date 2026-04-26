"""
Evaluation metrics for time-series forecasting and uncertainty quantification.

Covers: point metrics (MSE, MAE, RMSE, MAPE), interval metrics (PICP, MPIW),
scoring rules (Winkler, CRPS), and calibration diagnostics.
"""
import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def compute_picp(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> float:
    """
    Prediction Interval Coverage Probability (PICP):
    fraction of true values falling within the prediction interval.
    """
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def compute_mpiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """
    Mean Prediction Interval Width (MPIW):
    average width of the prediction interval.
    """
    return float(np.mean(upper - lower))


def winkler_score(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Winkler (interval) score: penalizes wide intervals and rewards coverage.

    Score = width + (2/alpha) * max(lower - y_true, 0) + (2/alpha) * max(y_true - upper, 0)

    Lower is better.
    """
    width = upper - lower
    penalty_low = (2.0 / alpha) * np.maximum(lower - y_true, 0)
    penalty_high = (2.0 / alpha) * np.maximum(y_true - upper, 0)
    return float(np.mean(width + penalty_low + penalty_high))


def crps_gaussian(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
) -> float:
    """
    Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions.

    CRPS = σ * [ν * (2·Φ(ν) − 1) + 2·φ(ν) − 1/√π]

    where ν = (y_true − μ) / σ, and φ, Φ are the standard normal PDF and CDF.

    Args:
        y_true: Ground truth values.
        y_pred: Mean predictions (μ).
        y_std: Standard deviation estimates (σ).

    Returns:
        Mean CRPS across all samples.
    """
    from scipy.stats import norm

    y_std = np.maximum(y_std, 1e-8)
    nu = (y_true - y_pred) / y_std
    phi = norm.pdf(nu)
    Phi = norm.cdf(nu)
    crps = y_std * (nu * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(np.pi))
    return float(np.mean(crps))


def calibration_error(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Calibration error: abs(empirical coverage − nominal coverage).

    Values near 0 indicate well-calibrated intervals.
    """
    empirical_coverage = compute_picp(lower, upper, y_true)
    nominal_coverage = 1.0 - alpha
    return float(abs(empirical_coverage - nominal_coverage))
