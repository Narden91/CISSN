import unittest
import inspect
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler


from cissn.data.data_loader import get_data_loader
from cissn.data.dataset import Dataset_ETT_hour
from experiments.run_benchmark import Experiment


class IdentityEncoder(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


class IdentityHead(nn.Module):
    def forward(self, state):
        return state.unsqueeze(1)


class MeanFirstFeatureHead:
    def get_refinement_ratio(self, state):
        return float(state[:, 0].mean().item())


class TestTrainingPipeline(unittest.TestCase):
    @patch('cissn.data.dataset.pd.read_csv')
    def test_get_data_loader_uses_deterministic_eval_policy(self, mock_read_csv):
        dates = pd.date_range(start='2020-01-01', periods=64, freq='h')
        df = pd.DataFrame({'date': dates, 'OT': np.random.randn(64)})
        mock_read_csv.return_value = df

        args = SimpleNamespace(
            data='ETTh1',
            root_path='.',
            data_path='ignored.csv',
            seq_len=4,
            label_len=2,
            pred_len=2,
            features='S',
            target='OT',
            batch_size=4,
            freq='h',
            num_workers=0,
        )

        borders = ([0, 20, 32], [20, 32, 48])
        with patch.object(Dataset_ETT_hour, '_get_borders', return_value=borders):
            _, train_loader = get_data_loader(args, 'train')
            _, val_loader = get_data_loader(args, 'val')
            _, cal_loader = get_data_loader(args, 'cal')
            _, test_loader = get_data_loader(args, 'test')
            _, pred_loader = get_data_loader(args, 'pred')

        # Training keeps the final partial batch: the models use LayerNorm only,
        # so a short batch is harmless, and dropping it would discard data.
        self.assertFalse(train_loader.drop_last)
        self.assertIsInstance(train_loader.sampler, RandomSampler)

        self.assertFalse(val_loader.drop_last)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)

        self.assertFalse(cal_loader.drop_last)
        self.assertIsInstance(cal_loader.sampler, SequentialSampler)

        self.assertFalse(test_loader.drop_last)
        self.assertIsInstance(test_loader.sampler, SequentialSampler)

        self.assertFalse(pred_loader.drop_last)
        self.assertIsInstance(pred_loader.sampler, SequentialSampler)
        self.assertEqual(pred_loader.batch_size, 1)

    @patch('cissn.data.dataset.pd.read_csv')
    def test_training_epoch_sees_every_sample(self, mock_read_csv):
        """No training sample may be silently dropped by batching.

        drop_last=True discards len(split) % batch_size samples per epoch, which
        grows with batch size (45% of exchange_rate at batch 2048). Results would
        then not be comparable to baselines trained on the full split.
        """
        dates = pd.date_range(start='2020-01-01', periods=200, freq='h')
        df = pd.DataFrame({'date': dates, 'OT': np.random.randn(200)})
        mock_read_csv.return_value = df

        borders = ([0, 120, 160], [120, 160, 200])
        for batch_size in (7, 16, 32):
            args = SimpleNamespace(
                data='ETTh1', root_path='.', data_path='ignored.csv',
                seq_len=4, label_len=2, pred_len=2, features='S', target='OT',
                batch_size=batch_size, freq='h', num_workers=0,
            )
            with patch.object(Dataset_ETT_hour, '_get_borders', return_value=borders):
                dataset, loader = get_data_loader(args, 'train')

            seen = sum(batch[0].shape[0] for batch in loader)
            self.assertEqual(
                seen, len(dataset),
                f"batch_size={batch_size} dropped {len(dataset) - seen} of {len(dataset)} samples",
            )

    @patch('cissn.data.dataset.pd.read_csv')
    def test_dataset_raises_for_too_short_split(self, mock_read_csv):
        dates = pd.date_range(start='2020-01-01', periods=24, freq='h')
        df = pd.DataFrame({'date': dates, 'OT': np.random.randn(24)})
        mock_read_csv.return_value = df

        with patch.object(Dataset_ETT_hour, '_get_borders', return_value=([0, 12, 18], [12, 18, 24])):
            with self.assertRaisesRegex(ValueError, 'too short'):
                Dataset_ETT_hour(root_path='.', flag='train', size=[10, 4, 4])

    def test_vali_weights_partial_batches_by_element_count(self):
        experiment = Experiment.__new__(Experiment)
        experiment.model = IdentityEncoder()
        experiment.head = IdentityHead()
        experiment.device = torch.device('cpu')
        experiment.args = SimpleNamespace(features='S', pred_len=1)

        criterion = nn.MSELoss()
        loader = [
            (
                torch.zeros(2, 1, 1),
                torch.ones(2, 1, 1),
                torch.zeros(2, 1, 1),
                torch.zeros(2, 1, 1),
            ),
            (
                torch.ones(1, 1, 1),
                torch.ones(1, 1, 1),
                torch.zeros(1, 1, 1),
                torch.zeros(1, 1, 1),
            ),
        ]

        loss = Experiment.vali(experiment, loader, criterion)

        self.assertAlmostEqual(loss, 2.0 / 3.0, places=6)

    def test_baseline_train_epoch_weights_partial_batches(self):
        """Epoch training loss must be element-weighted, like validation.

        Keeping the final partial batch (instead of dropping it) makes unequal
        batch sizes routine, so an unweighted mean over per-batch means would
        over-weight the samples in that short batch.
        """
        from cissn.baselines.training import train_baseline_epoch

        class ZeroForecast(nn.Module):
            """Always predicts zero, so the loss equals the squared target."""

            def __init__(self):
                super().__init__()
                self.scale = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                return x * self.scale

        model = ZeroForecast()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)  # lr=0 keeps predictions at zero
        # Batch A: 2 samples, squared error 1.0 each. Batch B: 1 sample, error 4.0.
        loader = [
            (torch.zeros(2, 1, 1), torch.ones(2, 1, 1), torch.zeros(2, 1, 1), torch.zeros(2, 1, 1)),
            (torch.zeros(1, 1, 1), torch.full((1, 1, 1), 2.0), torch.zeros(1, 1, 1), torch.zeros(1, 1, 1)),
        ]

        loss = train_baseline_epoch(
            model=model, loader=loader, optimizer=optimizer, criterion=nn.MSELoss(),
            device=torch.device('cpu'), pred_len=1, features='S', grad_clip=0.0,
        )

        # Element-weighted: (1.0*2 + 4.0*1) / 3 = 2.0. Unweighted would give 2.5.
        self.assertAlmostEqual(loss, 2.0, places=6)

    def test_concatenate_batches_handles_variable_batch_sizes(self):
        combined = Experiment._concatenate_batches(
            [np.zeros((2, 1, 1), dtype=np.float32), np.ones((1, 1, 1), dtype=np.float32)],
            'prediction',
        )

        self.assertEqual(combined.shape, (3, 1, 1))
        self.assertEqual(combined[0, 0, 0], 0.0)
        self.assertEqual(combined[-1, 0, 0], 1.0)

    def test_epoch_diagnostics_aggregate_all_training_batches(self):
        criterion = nn.MSELoss()
        disentangle = type(
            'DisentangleProbe',
            (),
            {
                'get_metrics': staticmethod(lambda states: {'mean_abs_off_diag_corr': float(states[:, :, 0].mean().item())}),
            },
        )()
        state_batches = [
            torch.ones(2, 1, 5),
            torch.full((1, 1, 5), 4.0),
        ]
        final_state_batches = [
            torch.ones(2, 5),
            torch.full((1, 5), 4.0),
        ]

        metrics, refinement_ratio = Experiment._summarize_epoch_diagnostics(
            disentangle,
            MeanFirstFeatureHead(),
            state_batches,
            final_state_batches,
        )

        self.assertAlmostEqual(metrics['mean_abs_off_diag_corr'], 2.0, places=6)
        self.assertAlmostEqual(refinement_ratio, 2.0, places=6)
        self.assertNotAlmostEqual(metrics['mean_abs_off_diag_corr'], 4.0, places=6)
        self.assertNotAlmostEqual(refinement_ratio, 4.0, places=6)

    def test_train_protocol_does_not_touch_test_split(self):
        source = inspect.getsource(Experiment.train)
        self.assertNotIn("_get_data(flag='test')", source)
        self.assertNotIn("test_loss", source)
        self.assertIn("_get_data(flag='cal')", source)


if __name__ == '__main__':
    unittest.main()
 
