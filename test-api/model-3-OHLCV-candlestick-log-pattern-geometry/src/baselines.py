import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.data.preprocess import prepare_data
from src.metrics import classification_metrics


def select_baseline_features(X, input_mode):
    """Match the logistic baseline input to the FNN representation."""
    if input_mode == "latest_summary":
        return X[:, -1, :]
    if input_mode == "flatten_sequence":
        return X.reshape(len(X), -1)
    raise ValueError(f"Unsupported baseline input_mode: {input_mode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["model"]["task"] != "classification":
        raise ValueError("Baselines are defined for classification.")

    output_dir = Path(config["project"]["output_dir"]) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_data(config, output_dir)

    y_test = prepared.y_test.astype(int)
    training_up_rate = float(np.mean(prepared.y_train))
    always_up = np.ones_like(y_test, dtype=float)
    constant = np.full_like(y_test, training_up_rate, dtype=float)

    input_mode = config["model"].get("input_mode", "flatten_sequence")
    if config["model"]["type"] == "lstm":
        input_mode = "latest_summary"

    X_train = select_baseline_features(prepared.X_train, input_mode)
    X_test = select_baseline_features(prepared.X_test, input_mode)
    scaler = StandardScaler().fit(X_train)

    # L2 regularization is retained; class_weight is omitted so predicted
    # probabilities reflect the observed class frequency more naturally.
    logistic = LogisticRegression(C=0.1, max_iter=3000, solver="lbfgs")
    logistic.fit(scaler.transform(X_train), prepared.y_train.astype(int))
    logistic_probabilities = logistic.predict_proba(scaler.transform(X_test))[:, 1]
    joblib.dump(
        {"scaler": scaler, "model": logistic, "input_mode": input_mode},
        output_dir / "logistic_regression.joblib",
    )

    comparison = {
        "always_up": classification_metrics(y_test, always_up),
        "constant_training_up_rate": classification_metrics(y_test, constant),
        "logistic_regression": classification_metrics(y_test, logistic_probabilities),
    }
    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
