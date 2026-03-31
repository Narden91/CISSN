import unittest
import sys
import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import RandomSampler, SequentialSampler

# Ensure the project root is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cissn.data.data_loader import get_data_loader
from cissn.data.dataset import Dataset_ETT_hour
from experiments.run_benchmark import Experiment


class IdentityEncoder(nn.Module):
    def forward(self, x):
        return x[:, 0, :]


class IdentityHead(nn.Module):
    def forward(self, state):
        return state.unsqueeze(1)


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
            _, test_loader = get_data_loader(args, 'test')
            _, pred_loader = get_data_loader(args, 'pred')

        self.assertTrue(train_loader.drop_last)
        self.assertIsInstance(train_loader.sampler, RandomSampler)

        self.assertFalse(val_loader.drop_last)
        self.assertIsInstance(val_loader.sampler, SequentialSampler)

        self.assertFalse(test_loader.drop_last)
        self.assertIsInstance(test_loader.sampler, SequentialSampler)

        self.assertFalse(pred_loader.drop_last)
        self.assertIsInstance(pred_loader.sampler, SequentialSampler)
        self.assertEqual(pred_loader.batch_size, 1)

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

    def test_concatenate_batches_handles_variable_batch_sizes(self):
        combined = Experiment._concatenate_batches(
            [np.zeros((2, 1, 1), dtype=np.float32), np.ones((1, 1, 1), dtype=np.float32)],
            'prediction',
        )

        self.assertEqual(combined.shape, (3, 1, 1))
        self.assertEqual(combined[0, 0, 0], 0.0)
        self.assertEqual(combined[-1, 0, 0], 1.0)


if __name__ == '__main__':
    unittest.main()
 
