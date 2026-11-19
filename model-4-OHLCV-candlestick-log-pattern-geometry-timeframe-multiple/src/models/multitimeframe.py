import torch
from torch import nn


class MultiTimeframeModel(nn.Module):
    """Comparable FNN, RNN, GRU, and LSTM models for synchronized inputs."""

    def __init__(self, model_type, feature_count, timeframe_count, direct_indices,
                 projection_size=16, hidden_size=32, context_hidden=128,
                 fusion_hidden=32, dropout=0.30):
        super().__init__()
        if model_type not in {"fnn", "rnn", "gru", "lstm"}:
            raise ValueError("model.type must be fnn, rnn, gru, or lstm")
        self.model_type = model_type
        self.register_buffer("direct_indices", torch.tensor(direct_indices, dtype=torch.long))
        self.context = nn.Sequential(
            nn.Linear(timeframe_count * feature_count, context_hidden),
            nn.LayerNorm(context_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(context_hidden, 64), nn.ReLU(), nn.Dropout(dropout),
        )
        if model_type == "fnn":
            fusion_input = 64
        else:
            self.projection = nn.Sequential(
                nn.Linear(len(direct_indices), projection_size),
                nn.LayerNorm(projection_size), nn.ReLU(), nn.Dropout(dropout),
            )
            recurrent = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[model_type]
            kwargs = dict(input_size=projection_size, hidden_size=hidden_size, batch_first=True)
            if model_type == "rnn":
                kwargs["nonlinearity"] = "tanh"
            self.recurrent = recurrent(**kwargs)
            fusion_input = hidden_size + 64
        self.output = nn.Sequential(
            nn.Linear(fusion_input, fusion_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(fusion_hidden, 1)
        )

    def forward(self, target_sequence, summaries):
        context = self.context(summaries.flatten(start_dim=1))
        if self.model_type == "fnn":
            combined = context
        else:
            direct = torch.index_select(target_sequence, 2, self.direct_indices)
            sequence, _ = self.recurrent(self.projection(direct))
            combined = torch.cat([sequence[:, -1, :], context], dim=1)
        return self.output(combined).squeeze(-1)
