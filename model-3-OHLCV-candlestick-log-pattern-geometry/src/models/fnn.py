from torch import nn


class FeedForwardNetwork(nn.Module):
    """
    Feedforward classifier/regressor for either a flattened sequence or the
    final multi-scale summary vector.

    latest_summary uses x[:, -1, :]. The final row still contains trailing
    5-100 bar returns, slopes, ranges, volatility, and volume context.
    """

    def __init__(
        self,
        window_size,
        feature_count,
        hidden_sizes,
        dropout,
        input_mode="flatten_sequence",
    ):
        super().__init__()
        valid_modes = {"flatten_sequence", "latest_summary"}
        if input_mode not in valid_modes:
            raise ValueError(f"input_mode must be one of {sorted(valid_modes)}")

        self.input_mode = input_mode
        input_size = (
            feature_count
            if input_mode == "latest_summary"
            else window_size * feature_count
        )

        layers = []
        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(dropout),
                ]
            )
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        if self.input_mode == "latest_summary":
            model_input = x[:, -1, :]
        else:
            model_input = x.flatten(start_dim=1)
        return self.network(model_input).squeeze(-1)
