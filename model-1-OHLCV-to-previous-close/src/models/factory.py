from src.models.fnn import FeedForwardNetwork
from src.models.lstm import LSTMNetwork


def create_model(config, window_size, feature_count):
    model_config = config["model"]
    if model_config["type"] == "fnn":
        return FeedForwardNetwork(
            window_size=window_size,
            feature_count=feature_count,
            hidden_sizes=model_config["hidden_sizes"],
            dropout=float(model_config["dropout"]),
        )
    if model_config["type"] == "lstm":
        return LSTMNetwork(
            feature_count=feature_count,
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
        )
    raise ValueError(f"Unknown model type: {model_config['type']}")
