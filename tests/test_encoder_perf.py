"""Encoder sequence-loop contracts: numerical parity and per-forward spectral norm."""
import unittest

import torch

from cissn.models.encoder import DisentangledStateEncoder


class TestEncoderSequence(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.enc = DisentangledStateEncoder(input_dim=7, hidden_dim=64)
        self.x = torch.randn(8, 96, 7)

    def test_spectral_norm_updates_once_per_forward(self):
        """Power iteration must not run per timestep.

        The Lipschitz bound 1 + L_G*beta assumes one fixed spectrally-normalised
        weight for the whole sequence; a per-step update changes the transition
        matrix mid-sequence and invalidates the bound. Counting hook invocations
        pins the behaviour: it was once per timestep (96), it must now be once.
        """
        self.enc.train()
        module = self.enc.correction_mlp[0]
        hook_id = next(iter(module._forward_pre_hooks))
        original = module._forward_pre_hooks[hook_id]
        calls = []

        def counting_hook(mod, inputs):
            calls.append(1)
            return original(mod, inputs)

        module._forward_pre_hooks[hook_id] = counting_hook
        try:
            self.enc(self.x)
        finally:
            module._forward_pre_hooks[hook_id] = original

        self.assertEqual(
            len(calls), 1,
            f"spectral norm ran {len(calls)} times per forward; expected exactly 1",
        )

    def test_all_states_matches_final_state(self):
        """return_all_states=True last slice must equal return_all_states=False."""
        self.enc.eval()
        with torch.no_grad():
            final = self.enc(self.x, return_all_states=False)
            allst = self.enc(self.x, return_all_states=True)
        self.assertTrue(torch.allclose(final, allst[:, -1, :], atol=1e-6))

    def test_gradients_flow_to_all_params(self):
        self.enc.train()
        self.enc(self.x).sum().backward()
        for name, p in self.enc.named_parameters():
            self.assertIsNotNone(p.grad, f"{name} has no gradient")
            self.assertTrue(torch.isfinite(p.grad).all(), f"{name} grad not finite")


if __name__ == "__main__":
    unittest.main()
