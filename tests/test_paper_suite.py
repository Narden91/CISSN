"""Checks for the calendar control and paired frozen-checkpoint analysis."""
import numpy as np
import pytest
import torch
from types import SimpleNamespace

from cissn.baselines.flat_conformal import FlatConformal
from cissn.conformal.aci import adaptive_conformal_quantiles
from cissn.conformal.phase import PhaseConformal, phase_from_timestamps
from cissn.conformal.state_conditional import split_conformal_q_level
from experiments.run_paper_suite import assert_aligned, calibration_selection, fit_methods, score_bounds
from experiments.run_benchmark import Experiment


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
@pytest.mark.parametrize("n", [20, 99, 100, 101])
def test_order_statistic_index(alpha, n):
    residuals = np.arange(1, n + 1, dtype=float)[:, None, None]
    flat = FlatConformal(alpha)
    flat.fit(residuals)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    assert flat.quantile_.item() == rank
    phase = PhaseConformal(alpha, 1).fit(residuals, np.zeros(n, dtype=int))
    np.testing.assert_array_equal(phase.half_widths([0])[0], flat.quantile_)


def test_too_small_quantile_is_capped_not_exact_finite_sample():
    assert split_conformal_q_level(3, 0.05) == 1
    flat = FlatConformal(0.05)
    flat.fit(np.arange(3))
    assert flat.quantile_ == 2  # The proper unattainable order statistic would be infinity.


@pytest.mark.parametrize("period,dates,expected", [
    (24, ["2016-07-01 02:00", "2016-07-02 00:00"], [2, 0]),
    (144, ["2006-01-01 00:10", "2006-01-01 23:50"], [1, 143]),
    (96, ["2016-07-01 00:15", "2016-07-01 23:45"], [1, 95]),
    (7, ["2026-10-12", "2026-10-18"], [0, 6]),
    (52, ["2020-12-31", "2021-01-04"], [0, 0]),
])
def test_calendar_phase(period, dates, expected):
    np.testing.assert_array_equal(phase_from_timestamps(dates, period), expected)


def test_sparse_phase_and_cross_fallback_and_counts():
    residuals = np.arange(60, dtype=float)[:, None, None]
    phases = np.r_[np.zeros(30, int), np.ones(25, int), np.full(5, 2)]
    labels = np.r_[np.zeros(29, int), np.ones(31, int)]
    fitted = PhaseConformal(0.1, 4).fit(residuals, phases, labels, 2)
    assert fitted.group_counts_["0:1"] == 1
    assert fitted.group_counts_["3"] == 0
    assert fitted.fallbacks_["0:1"] == "phase"
    assert fitted.fallbacks_["2:1"] == "paired_global"
    np.testing.assert_array_equal(fitted.half_widths([0], [1])[0], fitted.phase_quantiles_[0])
    np.testing.assert_array_equal(fitted.half_widths([3], [1])[0], fitted.global_quantile_)
    assert sum(fitted.group_counts_[str(p)] for p in range(4)) == len(residuals)


def test_full_flat_uses_all_origins_and_paired_uses_exact_subset():
    states = np.arange(120, dtype=np.float32).reshape(60, 2)
    residuals = np.r_[np.full(30, 100.), np.ones(30)][:, None, None]
    phases = np.arange(60) % 2
    fitted = fit_methods(residuals, states, phases, 0.1, 2, 42, 3, 2)
    conditioning, paired, full = calibration_selection(60, 3)
    np.testing.assert_array_equal(conditioning, np.arange(0, 30, 3))
    np.testing.assert_array_equal(paired, np.arange(30, 60, 3))
    np.testing.assert_array_equal(full, np.arange(60))
    assert fitted["flat_paired"].quantile_.item() == 1
    assert fitted["flat_full"].quantile_.item() == 100
    assert fitted["phase"].calibration_samples_ == len(paired)


def test_fitting_and_assignment_do_not_use_test_labels():
    rng = np.random.default_rng(4)
    states, residuals = rng.normal(size=(100, 3)).astype("float32"), rng.uniform(size=(100, 2, 1))
    fitted = fit_methods(residuals, states, np.arange(100) % 2, 0.1, 2, 42, 1, 2)
    test_states = rng.normal(size=(20, 3)).astype("float32")
    centers = fitted["cluster"].kmeans.cluster_centers_.copy()
    before = fitted["cluster"].assign_clusters(test_states)
    first = fitted["phase"].half_widths(np.arange(20) % 2)
    score_bounds(np.zeros((20, 2, 1)), np.zeros((20, 2, 1)), first, 0.1)
    score_bounds(np.zeros((20, 2, 1)), np.full((20, 2, 1), 1e8), first, 0.1)
    np.testing.assert_array_equal(centers, fitted["cluster"].kmeans.cluster_centers_)
    np.testing.assert_array_equal(before, fitted["cluster"].assign_clusters(test_states))
    np.testing.assert_array_equal(first, fitted["phase"].half_widths(np.arange(20) % 2))


def test_metrics_exact_and_alignment():
    pred = np.zeros((2, 1, 1))
    true = np.array([0., 2.])[:, None, None]
    per_origin, per_horizon = score_bounds(pred, true, 1., 0.1)
    np.testing.assert_allclose(per_origin, [[1, 2, 2], [0, 2, 22]])
    np.testing.assert_allclose(per_horizon, [[0.5, 2, 12]])
    assert_aligned(np.arange(2), np.arange(2), true, true.copy())
    with pytest.raises(ValueError):
        assert_aligned(np.arange(2), np.arange(2)[::-1], true, true)
    with pytest.raises(ValueError):
        assert_aligned(np.arange(2), np.arange(2), true, true + 1)


def test_seed_and_alpha_do_not_change_partition():
    rng = np.random.default_rng(10)
    states = rng.normal(size=(200, 3)).astype("float32")
    residuals, phases = rng.uniform(size=(200, 2, 1)), np.arange(200) % 2
    first = fit_methods(residuals, states, phases, 0.1, 3, 42, 1, 2)
    repeat = fit_methods(residuals, states, phases, 0.1, 3, 42, 1, 2)
    other_alpha = fit_methods(residuals, states, phases, 0.05, 3, 42, 1, 2)
    for result in (repeat, other_alpha):
        np.testing.assert_array_equal(first["cluster"].kmeans.cluster_centers_, result["cluster"].kmeans.cluster_centers_)


def test_feedback_cannot_read_future_labels():
    rng = np.random.default_rng(0)
    cal = np.abs(rng.normal(size=(100, 4, 1)))
    test = np.abs(rng.normal(size=(20, 4, 1)))
    q, _ = adaptive_conformal_quantiles(cal, test, np.arange(20), 4, 0.1, 0.005)
    changed = test.copy()
    changed[6:] = 1e6
    q_changed, _ = adaptive_conformal_quantiles(cal, changed, np.arange(20), 4, 0.1, 0.005)
    np.testing.assert_array_equal(q[:10], q_changed[:10])


def test_state_encoder_and_point_prediction_do_not_read_targets():
    torch.manual_seed(1)
    args = SimpleNamespace(enc_in=2, c_out=2, state_dim=5, d_model=8, dropout=0.0,
                           pred_len=3, data="ETTh1", features="M", revin=True, require_gpu=False)
    experiment = Experiment(args)
    experiment._set_train_mode(False)
    x, y = torch.randn(4, 12, 2), torch.randn(4, 5, 2)
    with torch.no_grad():
        state, pred, _ = experiment._forward_and_slice(x, y)
        changed_state, changed_pred, _ = experiment._forward_and_slice(x, y + 1e6)
    torch.testing.assert_close(state, changed_state, rtol=0, atol=0)
    torch.testing.assert_close(pred, changed_pred, rtol=0, atol=0)


def test_heteroskedastic_exchangeable_regimes():
    rng = np.random.default_rng(17)
    phase = rng.integers(0, 2, size=4000)
    residuals = np.abs(rng.normal(size=4000) * np.where(phase == 0, 1, 4))[:, None, None]
    states = phase[:, None].astype("float32")
    fitted = fit_methods(residuals, states, phase, 0.1, 2, 42, 1, 2)
    test_phase = rng.integers(0, 2, size=20000)
    true = (rng.normal(size=20000) * np.where(test_phase == 0, 1, 4))[:, None, None]
    pred = np.zeros_like(true)
    q = fitted["phase"].half_widths(test_phase)
    labels = fitted["cluster"].assign_clusters(test_phase[:, None].astype("float32"))
    state_q = np.stack([fitted["cluster"].quantiles[int(k)] for k in labels])
    np.testing.assert_allclose(state_q, q)
    assert q[test_phase == 1].mean() > 3 * q[test_phase == 0].mean()
    flat_q = fitted["flat_paired"].quantile_
    assert flat_q.shape == (1, 1)
    for p in (0, 1):
        cov = score_bounds(pred[test_phase == p], true[test_phase == p], q[test_phase == p], 0.1)[0][:, 0].mean()
        assert 0.87 < cov < 0.93


@pytest.mark.parametrize("scores,phases", [(np.array([1., np.nan]), [0, 1]), (np.array([-1., 2.]), [0, 1]), (np.ones(2), [0, 3])])
def test_invalid_phase_data_rejected(scores, phases):
    with pytest.raises(ValueError):
        PhaseConformal(0.1, 2).fit(scores, np.array(phases))
