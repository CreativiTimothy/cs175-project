import torch
from torch import nn


class FeedForwardNetwork(nn.Module):
    """FNN that flattens the OHLC window before applying fully connected layers."""

    def __init__(self, window_size, feature_count, hidden_sizes, dropout):
        super().__init__()
        layers = []
        input_size = window_size * feature_count
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout),
            ])
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.flatten(start_dim=1)).squeeze(-1)
