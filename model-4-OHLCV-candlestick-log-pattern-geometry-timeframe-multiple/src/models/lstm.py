from torch import nn


class LSTMNetwork(nn.Module):
    """
    Sequence model for 100-bar geometric features.

    A projection layer first compresses the 96 engineered channels at each
    bar. This reduces LSTM parameter count and lets the recurrent layer focus
    on how the compressed representation changes through time.
    """

    def __init__(
        self,
        feature_count,
        hidden_size,
        num_layers,
        dropout,
        projection_size=24,
    ):
        super().__init__()
        self.input_projection = nn.Sequential(
            nn.Linear(feature_count, projection_size),
            nn.LayerNorm(projection_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        projected = self.input_projection(x)
        sequence_output, _ = self.lstm(projected)
        return self.output(sequence_output[:, -1, :]).squeeze(-1)
