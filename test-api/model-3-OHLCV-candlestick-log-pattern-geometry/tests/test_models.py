import torch

from src.models.fnn import FeedForwardNetwork
from src.models.lstm import LSTMNetwork


def test_flattened_fnn_output_shape():
    model = FeedForwardNetwork(100, 96, [32, 16], 0.1, "flatten_sequence")
    assert model(torch.zeros(8, 100, 96)).shape == (8,)


def test_summary_fnn_uses_96_inputs():
    model = FeedForwardNetwork(100, 96, [64, 32], 0.3, "latest_summary")
    assert model.network[0].in_features == 96
    assert model(torch.zeros(8, 100, 96)).shape == (8,)


def test_lstm_output_shape_and_projection():
    model = LSTMNetwork(96, 32, 1, 0.3, projection_size=24)
    assert model.input_projection[0].in_features == 96
    assert model.input_projection[0].out_features == 24
    assert model(torch.zeros(8, 100, 96)).shape == (8,)
