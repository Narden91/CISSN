import unittest
import torch
import sys
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cissn.data.dataset import BaseETTDataset, Dataset_ETT_hour, Dataset_ETT_minute
from cissn.models import ForecastHead
from cissn.explanations import ForecastExplainer

class TestComponents(unittest.TestCase):
    
    def test_forecast_explainer_structure(self):
        """Test that ForecastExplainer returns correct structure"""
        torch.manual_seed(0)
        state_dim = 5
        horizon = 3
        batch_size = 3
        
        head = ForecastHead(state_dim=state_dim, output_dim=2, horizon=horizon)
        explainer = ForecastExplainer(head)
        
        # Mock state
        state = torch.randn(batch_size, state_dim)
        horizon_idx = 2
        output_idx = 1
        
        results = explainer.explain(state, horizon_idx=horizon_idx, output_idx=output_idx)
        forecasts = head(state)[:, horizon_idx, output_idx]
        
        self.assertEqual(len(results), batch_size)
        self.assertTrue(hasattr(results[0], 'level_contribution'))
        self.assertTrue(hasattr(results[0], 'trend_contribution'))
        self.assertTrue(hasattr(results[0], 'seasonal_contribution'))
        self.assertTrue(hasattr(results[0], 'linear_prediction'))
        self.assertTrue(hasattr(results[0], 'refinement_contribution'))

        for result, forecast in zip(results, forecasts):
            self.assertAlmostEqual(
                result.linear_prediction + result.refinement_contribution,
                result.total_prediction,
                places=6,
            )
            self.assertAlmostEqual(result.total_prediction, float(forecast.item()), places=6)
        
    @patch('cissn.data.dataset.pd.read_csv')
    def test_dataset_refactoring(self, mock_read_csv):
        """Test that subclasses inherit correctly from BaseETTDataset"""
        # Mock DataFrame
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='h')
        df = pd.DataFrame({'date': dates, 'OT': np.random.randn(1000)})
        mock_read_csv.return_value = df
        
        # Test Hourly
        # Mocking get_borders to fit our small mock data
        with patch.object(Dataset_ETT_hour, '_get_borders', return_value=([0, 10, 20], [10, 20, 30])):
             ds_hour = Dataset_ETT_hour(root_path='.', flag='train', size=[5, 2, 2])
             self.assertIsInstance(ds_hour, BaseETTDataset)
             self.assertEqual(ds_hour.data_stamp.shape[1], 4) # 4 time features
             
        # Test Minute
        with patch.object(Dataset_ETT_minute, '_get_borders', return_value=([0, 10, 20], [10, 20, 30])):
             ds_min = Dataset_ETT_minute(root_path='.', flag='train', size=[5, 2, 2])
             self.assertIsInstance(ds_min, BaseETTDataset)
             self.assertEqual(ds_min.data_stamp.shape[1], 5) # 5 time features (w/ minute)

if __name__ == '__main__':
    unittest.main()
