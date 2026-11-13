from torch import nn


class LSTMNetwork(nn.Module):
    """Optional sequence model that keeps the time dimension of the OHLC window."""

    def __init__(self, feature_count, hidden_size, num_layers, dropout):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=feature_count,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        return self.output(sequence_output[:, -1, :]).squeeze(-1)
