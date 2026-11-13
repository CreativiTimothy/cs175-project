from src.models.fnn import FeedForwardNetwork
from src.models.lstm import LSTMNetwork


def create_model(config, window_size, feature_count):
    model_config = config["model"]
    model_type = model_config["type"]

    if model_type == "fnn":
        return FeedForwardNetwork(
            window_size=window_size,
            feature_count=feature_count,
            hidden_sizes=model_config["hidden_sizes"],
            dropout=float(model_config["dropout"]),
            input_mode=model_config.get("input_mode", "flatten_sequence"),
        )

    if model_type == "lstm":
        return LSTMNetwork(
            feature_count=feature_count,
            hidden_size=int(model_config["hidden_size"]),
            num_layers=int(model_config["num_layers"]),
            dropout=float(model_config["dropout"]),
            projection_size=int(model_config.get("projection_size", 24)),
        )

    raise ValueError(f"Unknown model type: {model_type}")
