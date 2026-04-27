import torch
import unittest


from cissn.models import DisentangledStateEncoder, ForecastHead

class TestCISSNModel(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.state_dim = 5
        self.hidden_dim = 32
        self.horizon = 5
        self.batch_size = 8
        self.seq_len = 15
        
        self.encoder = DisentangledStateEncoder(
            input_dim=self.input_dim,
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim
        )
        self.head = ForecastHead(
            state_dim=self.state_dim,
            output_dim=1,
            horizon=self.horizon
        )

    def test_encoder_output_shape(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        state = self.encoder(x)
        self.assertEqual(state.shape, (self.batch_size, self.state_dim))

    def test_encoder_return_all_states(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        states = self.encoder(x, return_all_states=True)
        self.assertEqual(states.shape, (self.batch_size, self.seq_len, self.state_dim))

    def test_forecast_head_output_shape(self):
        state = torch.randn(self.batch_size, self.state_dim)
        forecast = self.head(state)
        self.assertEqual(forecast.shape, (self.batch_size, self.horizon, 1))

    def test_integration(self):
        x = torch.randn(self.batch_size, self.seq_len, self.input_dim)
        state = self.encoder(x)
        forecast = self.head(state)
        self.assertEqual(forecast.shape, (self.batch_size, self.horizon, 1))

if __name__ == '__main__':
    unittest.main()
