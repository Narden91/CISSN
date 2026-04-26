"""
Publication-quality plotting utilities for CISSN experiments.

Covers: calibration curves, reliability diagrams, forecast decomposition plots,
refinement ratio visualization, and interval width analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def calibration_curve(
    empirical_coverages: list[float],
    nominal_alphas: list[float],
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Plot empirical coverage vs. nominal coverage.

    The diagonal line represents perfect calibration. Points above the diagonal
    indicate conservative intervals; points below indicate overconfident intervals.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    nominal = [1.0 - a for a in nominal_alphas]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Perfect calibration')
    ax.plot(nominal, empirical_coverages, 'o-', markersize=6, **kwargs)
    ax.set_xlabel('Nominal Coverage')
    ax.set_ylabel('Empirical Coverage')
    ax.set_title('Calibration (Reliability) Curve')
    ax.set_xlim(0.7, 1.0)
    ax.set_ylim(0.7, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def reliability_diagram(
    pincp: list[float],
    bins: int = 10,
    ax: plt.Axes | None = None,
    label: str = 'CISSN',
    **kwargs,
) -> plt.Axes:
    """
    Reliability diagram: expected coverage vs observed coverage across bins.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    obs = np.zeros(bins)
    for i in range(bins):
        mask = (np.array(pincp) >= bin_edges[i]) & (np.array(pincp) < bin_edges[i + 1])
        if mask.sum() > 0:
            obs[i] = np.array(pincp)[mask].mean()
        else:
            obs[i] = np.nan
    valid = ~np.isnan(obs)
    ax.plot(bin_centers[valid], obs[valid], 's-', label=label, **kwargs)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_xlabel('Expected Coverage (binned)')
    ax.set_ylabel('Observed Coverage')
    ax.set_title('Reliability Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def decomposition_plot(
    time_axis: np.ndarray,
    ground_truth: np.ndarray,
    forecast: np.ndarray,
    level: np.ndarray,
    trend: np.ndarray,
    seasonal: np.ndarray,
    residual: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    title: str = 'Forecast Decomposition',
    **kwargs,
) -> plt.Axes:
    """
    Stacked decomposition plot showing the contribution of each state
    component to the final forecast.

    Args:
        time_axis: x-axis values (timesteps).
        ground_truth: Actual observed values.
        forecast: Point forecast values.
        level, trend, seasonal, residual: Per-component contributions.
        lower, upper: Optional prediction interval bounds.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 5))

    T = len(time_axis)
    ax.plot(time_axis, ground_truth, 'k-', linewidth=1.5, label='Ground Truth', alpha=0.7)
    ax.plot(time_axis, forecast, 'b-', linewidth=1.5, label='Forecast', alpha=0.8)

    if lower is not None and upper is not None:
        ax.fill_between(time_axis, lower, upper, alpha=0.15, color='blue',
                        label='Prediction Interval')

    # Stacked area for components
    components = np.column_stack([
        level[:T] if len(level) >= T else np.pad(level, (0, T - len(level)), constant_values=np.nan),
        trend[:T] if len(trend) >= T else np.pad(trend, (0, T - len(trend)), constant_values=np.nan),
        seasonal[:T] if len(seasonal) >= T else np.pad(seasonal, (0, T - len(seasonal)), constant_values=np.nan),
        residual[:T] if len(residual) >= T else np.pad(residual, (0, T - len(residual)), constant_values=np.nan),
    ])
    ax.stackplot(time_axis, *components.T,
                 labels=['Level', 'Trend', 'Seasonal', 'Residual'],
                 alpha=0.4)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend(loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    return ax


def refinement_ratio_plot(
    epochs: list[int],
    refinement_ratios: list[float],
    ax: plt.Axes | None = None,
    prefix: str = 'CISSN',
    **kwargs,
) -> plt.Axes:
    """Plot refinement ratio over training epochs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, refinement_ratios, 'o-', markersize=4, **kwargs)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Equal split')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Refinement Ratio')
    ax.set_title(f'{prefix} — Refinement Ratio Over Training\n'
                 '(0 = pure linear, 1 = pure non-linear)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    return ax


def interval_width_plot(
    state_norm: np.ndarray,
    interval_width: np.ndarray,
    cluster_labels: np.ndarray | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
) -> plt.Axes:
    """
    Scatter plot: interval width vs. state magnitude, colored by cluster.

    Demonstrates adaptivity: different state regimes get different interval widths.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    if cluster_labels is not None:
        scatter = ax.scatter(state_norm.flatten(), interval_width.flatten(),
                             c=cluster_labels.flatten(), cmap='tab10',
                             alpha=0.6, s=20)
        plt.colorbar(scatter, ax=ax, label='Cluster')
    else:
        ax.scatter(state_norm.flatten(), interval_width.flatten(),
                   alpha=0.5, s=20, **kwargs)
    ax.set_xlabel('State Magnitude ||s||')
    ax.set_ylabel('Interval Half-Width')
    ax.set_title('Prediction Interval Width vs. State Magnitude')
    ax.grid(True, alpha=0.3)
    return ax
