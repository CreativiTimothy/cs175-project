import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data.preprocess import add_pattern_features, get_feature_names, load_bars
from src.models.factory import create_model


def latest_windows(config):
    data = config["data"]
    include_volume = bool(data.get("include_volume", True))
    names = get_feature_names(include_volume)
    bars = load_bars(data["path"], data.get("symbols") or None)
    window_size = int(data["window_size"])
    sequences, rows = [], []

    for symbol, raw in bars.groupby("symbol", sort=False):
        featured = add_pattern_features(raw, include_volume).dropna(subset=names)
        if len(featured) < window_size:
            continue
        latest = featured.iloc[-window_size:]
        sequences.append(latest[names].to_numpy(np.float32))
        rows.append({
            "symbol": symbol,
            "prediction_as_of": latest.iloc[-1]["timestamp"],
            "last_close": float(latest.iloc[-1]["close"]),
        })

    if not sequences:
        raise ValueError("No symbol has enough valid bars for a prediction window.")
    return np.stack(sequences), pd.DataFrame(rows), names


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
    X, result, names = latest_windows(config)
    if names != checkpoint["feature_names"]:
        raise ValueError("Current feature order differs from the trained model.")

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
