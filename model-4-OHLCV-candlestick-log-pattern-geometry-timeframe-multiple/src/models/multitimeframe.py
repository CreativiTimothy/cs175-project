import torch
from torch import nn


class MultiTimeframeHybrid(nn.Module):
    """Target-timeframe sequence plus one summary vector per configured timeframe."""

    def __init__(self, feature_count, timeframe_count, direct_indices, projection=16, hidden=32, context_hidden=128, dropout=0.30):
        super().__init__()
        self.register_buffer("direct_indices", torch.tensor(direct_indices, dtype=torch.long))
        self.sequence_projection = nn.Sequential(
            nn.Linear(len(direct_indices), projection), nn.LayerNorm(projection), nn.ReLU(), nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(projection, hidden, batch_first=True)
        self.context = nn.Sequential(
            nn.Linear(timeframe_count * feature_count, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(context_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden + 64, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, target_sequence, timeframe_summaries):
        direct = torch.index_select(target_sequence, 2, self.direct_indices)
        sequence, _ = self.lstm(self.sequence_projection(direct))
        sequence_vector = sequence[:, -1, :]
        context_vector = self.context(timeframe_summaries.flatten(start_dim=1))
        return self.output(torch.cat([sequence_vector, context_vector], dim=1)).squeeze(-1)
