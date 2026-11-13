import torch

from src.models.fnn import FeedForwardNetwork
from src.models.lstm import LSTMNetwork


def test_fnn_output_shape():
    model = FeedForwardNetwork(window_size=100, feature_count=4, hidden_sizes=[32, 16], dropout=0.1)
    assert model(torch.zeros(8, 100, 4)).shape == (8,)


def test_lstm_output_shape():
    model = LSTMNetwork(feature_count=4, hidden_size=16, num_layers=1, dropout=0.0)
    assert model(torch.zeros(8, 100, 4)).shape == (8,)
