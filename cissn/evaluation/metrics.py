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


def compute_joint_picp(lower: np.ndarray, upper: np.ndarray, y_true: np.ndarray) -> float:
    """
    Joint Prediction Interval Coverage Probability:
    fraction of samples where ALL horizon and feature elements are simultaneously covered.
    """
    covered = (y_true >= lower) & (y_true <= upper)
    if covered.ndim > 1:
        axes = tuple(range(1, covered.ndim))
        covered_joint = np.all(covered, axis=axes)
    else:
        covered_joint = covered
    return float(np.mean(covered_joint))


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


_SEASONAL_PERIOD_BY_FREQ = {
    "h": 24,     # hourly data -> daily seasonality
    "t": 96,     # 15-minute data -> daily seasonality (4 * 24)
    "d": 7,      # daily data -> weekly seasonality
    "w": 52,     # weekly data -> yearly seasonality
}


def seasonal_period_for_freq(freq: str) -> int:
    """Map a dataset's sampling frequency to its naive-seasonal lag for MSIS scaling."""
    key = freq.lower()
    if key not in _SEASONAL_PERIOD_BY_FREQ:
        raise ValueError(f"No seasonal period configured for freq={freq!r}. Known: {sorted(_SEASONAL_PERIOD_BY_FREQ)}.")
    return _SEASONAL_PERIOD_BY_FREQ[key]


def mean_scaled_interval_score(
    lower: np.ndarray,
    upper: np.ndarray,
    y_true: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int,
    alpha: float = 0.1,
) -> float:
    """
    Mean Scaled Interval Score (MSIS): the Winkler interval score scaled by the
    in-sample seasonal-naive MAE, making it comparable across series of
    different scale (Gneiting & Raftery 2007; M4 competition convention).

    MSIS = winkler_score(lower, upper, y_true, alpha) / seasonal_naive_mae(y_train)

    Args:
        y_train: In-sample (training-split) target values used to compute the
            scaling denominator; must be 1-D or flattened before scaling.
        seasonal_period: Naive-forecast lag (e.g. 24 for hourly data with daily
            seasonality). Must be < len(y_train).
    """
    y_train = np.asarray(y_train).reshape(-1)
    if seasonal_period <= 0 or seasonal_period >= y_train.shape[0]:
        raise ValueError(
            f"seasonal_period must satisfy 0 < seasonal_period < len(y_train); "
            f"got seasonal_period={seasonal_period}, len(y_train)={y_train.shape[0]}."
        )
    denom = float(np.mean(np.abs(y_train[seasonal_period:] - y_train[:-seasonal_period])))
    if denom <= 1e-8:
        raise ValueError("Seasonal-naive scaling denominator is ~0; MSIS is undefined for a constant series.")
    return winkler_score(lower, upper, y_true, alpha=alpha) / denom
