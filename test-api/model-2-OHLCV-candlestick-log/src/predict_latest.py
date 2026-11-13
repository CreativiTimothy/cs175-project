import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data.preprocess import add_normalized_features, load_bars
from src.models.factory import create_model


def latest_windows(config):
    data_config = config["data"]
    bars = load_bars(data_config["path"], data_config.get("symbols") or None)
    raw_features = list(data_config["features"])
    if data_config.get("include_volume") and "volume" not in raw_features:
        raw_features.append("volume")
    feature_map = {
        "open": "open_norm",
        "high": "high_norm",
        "low": "low_norm",
        "close": "close_norm",
        "volume": "volume_norm",
    }
    feature_names = [feature_map[name] for name in raw_features]
    window_size = int(data_config["window_size"])
    sequences, rows = [], []

    for symbol, symbol_rows in bars.groupby("symbol", sort=False):
        normalized = add_normalized_features(symbol_rows, feature_names).dropna(subset=feature_names)
        if len(normalized) < window_size:
            continue
        window = normalized.iloc[-window_size:][feature_names]
        sequences.append(window.to_numpy(dtype=np.float32))
        rows.append({
            "symbol": symbol,
            "prediction_as_of": normalized.iloc[-1]["timestamp"],
            "last_close": float(normalized.iloc[-1]["close"]),
        })

    if not sequences:
        raise ValueError("No symbol has enough valid bars for a latest prediction window.")
    return np.stack(sequences), pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_config(args.config)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu")
    scaler = joblib.load(run_dir / "scaler.joblib")
    X, result = latest_windows(config)
    X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)

    model = create_model(config, checkpoint["window_size"], checkpoint["feature_count"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        raw = model(torch.from_numpy(X))
        if checkpoint["task"] == "classification":
            result["probability_up"] = torch.sigmoid(raw).numpy()
        else:
            result["predicted_return"] = raw.numpy()
            result["probability_up"] = 1.0 / (1.0 + np.exp(-result["predicted_return"] * 100.0))

    result = result.sort_values("probability_up", ascending=False)
    output = Path(args.output) if args.output else run_dir / "latest_predictions.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"Saved latest rankings to {output}")


if __name__ == "__main__":
    main()
