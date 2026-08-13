"""
Contract tests for HybridCISSN.

The hybrid is only publishable because of two guarantees, so both are asserted
here rather than left to inspection:

1. At correction-stage epoch 0 the hybrid is *exactly* the frozen DLinear base.
2. The frozen base cannot change during correction training.

If either regresses, the claim "the hybrid cannot underperform its own base
except through a training choice validation can reject" is no longer true.
"""
import math
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from cissn.baselines.dlinear import DLinear
from cissn.models.hybrid import HybridCISSN, LinearCorrectionHead


def _make_hybrid(**kwargs) -> HybridCISSN:
    torch.manual_seed(0)
    params = dict(
        input_dim=7, seq_len=96, pred_len=24, output_dim=7, hidden_dim=16, dropout=0.0
    )
    params.update(kwargs)
    return HybridCISSN(**params)


class TestHybridInitialisation(unittest.TestCase):
    def test_correction_head_is_zero_initialised(self):
        head = LinearCorrectionHead(state_dim=5, horizon=24, output_dim=7)

        self.assertTrue(torch.all(head.weight == 0))
        self.assertTrue(torch.all(head.bias == 0))

    def test_epoch_zero_output_exactly_equals_frozen_dlinear(self):
        """The defining guarantee: zero-init makes the hybrid start as DLinear."""
        model = _make_hybrid().eval()
        x = torch.randn(8, 96, 7)

        with torch.no_grad():
            hybrid_out = model(x)
            base_out = model.base(x)

        # Exact equality, not allclose: the correction is identically zero, so
        # any drift here means the additive structure has been broken.
        self.assertTrue(torch.equal(hybrid_out, base_out))

    def test_epoch_zero_exactness_holds_under_state_revin(self):
        model = _make_hybrid(state_revin=True).eval()
        x = torch.randn(8, 96, 7)

        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), model.base(x)))

    def test_hybrid_matches_standalone_dlinear_given_same_weights(self):
        model = _make_hybrid().eval()
        reference = DLinear(input_dim=7, seq_len=96, pred_len=24, output_dim=7).eval()
        reference.load_state_dict(model.base.state_dict())
        x = torch.randn(4, 96, 7)

        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), reference(x)))


class TestComponentDecomposition(unittest.TestCase):
    def test_components_sum_exactly_to_total(self):
        model = _make_hybrid().eval()
        # Break out of zero-init so the correction path is actually exercised.
        with torch.no_grad():
            model.correction.weight.normal_(0, 0.1)
            model.correction.bias.normal_(0, 0.1)
        x = torch.randn(8, 96, 7)

        with torch.no_grad():
            parts = model.forward_components(x)
            forward_out = model(x)

        summed = (
            parts["state_level"]
            + parts["state_trend"]
            + parts["state_seasonal"]
            + parts["state_residual"]
            + parts["bias"]
        )
        torch.testing.assert_close(summed, parts["total_correction"])
        torch.testing.assert_close(parts["base"] + parts["total_correction"], parts["total"])
        torch.testing.assert_close(parts["total"], forward_out)

    def test_components_expose_all_required_terms(self):
        model = _make_hybrid().eval()
        with torch.no_grad():
            parts = model.forward_components(torch.randn(2, 96, 7))

        for key in (
            "base", "state_level", "state_trend", "state_seasonal",
            "state_residual", "bias", "state", "total_correction", "total",
        ):
            self.assertIn(key, parts)

    def test_forward_output_shape(self):
        model = _make_hybrid().eval()
        with torch.no_grad():
            out = model(torch.randn(5, 96, 7))
        self.assertEqual(out.shape, (5, 24, 7))


class TestFrozenBase(unittest.TestCase):
    def test_freeze_base_disables_base_gradients_only(self):
        model = _make_hybrid().freeze_base()

        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.correction.parameters()))
        self.assertTrue(any(p.requires_grad for p in model.encoder.parameters()))

    def test_base_parameters_never_change_during_correction_training(self):
        model = _make_hybrid().freeze_base()
        before = {k: v.clone() for k, v in model.base.state_dict().items()}
        optimizer = torch.optim.Adam(model.correction_parameters(), lr=1e-2)

        x = torch.randn(16, 96, 7)
        y = torch.randn(16, 24, 7)
        for _ in range(3):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(x), y).backward()
            optimizer.step()

        for key, original in before.items():
            self.assertTrue(
                torch.equal(original, model.base.state_dict()[key]),
                f"frozen base parameter {key} changed during correction training",
            )

    def test_correction_parameters_receive_gradients(self):
        model = _make_hybrid().freeze_base()
        x = torch.randn(16, 96, 7)
        y = torch.randn(16, 24, 7)

        torch.nn.functional.mse_loss(model(x), y).backward()

        self.assertIsNotNone(model.correction.weight.grad)
        self.assertTrue(torch.any(model.correction.weight.grad != 0))

    def test_correction_parameters_excludes_base(self):
        model = _make_hybrid().freeze_base()
        base_ids = {id(p) for p in model.base.parameters()}

        self.assertFalse(base_ids & {id(p) for p in model.correction_parameters()})

    def test_frozen_base_stays_in_eval_mode_after_train_call(self):
        """model.train() must not silently reactivate base train-mode behaviour."""
        model = _make_hybrid(dropout=0.1).freeze_base()

        model.train()

        self.assertFalse(model.base.training)
        self.assertTrue(model.encoder.training)

    def test_training_can_reduce_loss_below_frozen_base(self):
        """Sanity: the correction path is expressive enough to matter at all."""
        model = _make_hybrid().freeze_base()
        x = torch.randn(32, 96, 7)
        y = torch.randn(32, 24, 7)
        loss_fn = torch.nn.functional.mse_loss

        with torch.no_grad():
            baseline_loss = loss_fn(model.base(x), y).item()

        optimizer = torch.optim.Adam(model.correction_parameters(), lr=1e-2)
        for _ in range(30):
            optimizer.zero_grad()
            loss_fn(model(x), y).backward()
            optimizer.step()

        with torch.no_grad():
            final_loss = loss_fn(model(x), y).item()

        self.assertLess(final_loss, baseline_loss)


class TestAnchoredDynamics(unittest.TestCase):
    def test_omega_is_fixed_to_the_seasonal_period_and_never_trains(self):
        from cissn.models.anchored import AnchoredStateEncoder

        encoder = AnchoredStateEncoder(input_dim=7, seasonal_period=24, hidden_dim=16)

        self.assertFalse(encoder.omega.requires_grad)
        self.assertAlmostEqual(
            float(encoder.omega.detach()), 2.0 * math.pi / 24, places=6
        )

    def test_level_integrates_trend(self):
        """The coupling term is what makes this a local *linear trend*."""
        from cissn.models.anchored import AnchoredStateEncoder

        encoder = AnchoredStateEncoder(input_dim=7, seasonal_period=24, hidden_dim=16)
        matrix = encoder._dynamics_matrix(encoder._structured_dynamics())

        self.assertNotEqual(float(matrix[0, 1]), 0.0)

    def test_legacy_dynamics_keep_level_and_trend_independent(self):
        from cissn.models.encoder import DisentangledStateEncoder

        encoder = DisentangledStateEncoder(input_dim=7, hidden_dim=16)
        matrix = encoder._dynamics_matrix(encoder._structured_dynamics())

        self.assertEqual(float(matrix[0, 1]), 0.0)

    def test_anchored_hybrid_requires_a_seasonal_period(self):
        with self.assertRaises(ValueError):
            HybridCISSN(
                input_dim=7, seq_len=96, pred_len=24, output_dim=7,
                state_dynamics="anchored", seasonal_period=None,
            )

    def test_anchored_hybrid_preserves_epoch_zero_exactness(self):
        model = _make_hybrid(state_dynamics="anchored", seasonal_period=24).eval()
        x = torch.randn(4, 96, 7)

        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), model.base(x)))


class TestValidation(unittest.TestCase):
    def test_rejects_unknown_state_dynamics(self):
        with self.assertRaises(ValueError):
            HybridCISSN(
                input_dim=7, seq_len=96, pred_len=24, output_dim=7,
                state_dynamics="nonexistent",
            )

    def test_rejects_non_structured_state_dim(self):
        with self.assertRaises(ValueError):
            LinearCorrectionHead(state_dim=4, horizon=24, output_dim=7)

    def test_rejects_malformed_state_shape(self):
        head = LinearCorrectionHead(state_dim=5, horizon=24, output_dim=7)

        with self.assertRaises(ValueError):
            head(torch.randn(8, 96, 5))  # 3-D, missing reduction to final state
        with self.assertRaises(ValueError):
            head(torch.randn(8, 4))      # wrong trailing dim


if __name__ == "__main__":
    unittest.main()
