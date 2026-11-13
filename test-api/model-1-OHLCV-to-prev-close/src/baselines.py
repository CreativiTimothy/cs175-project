import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.data.preprocess import prepare_data
from src.metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config["model"]["task"] != "classification":
        raise ValueError("Baselines are defined for the direction-classification task.")

    output_dir = Path(config["project"]["output_dir"]) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_data(config, output_dir)

    y_test = prepared.y_test.astype(int)
    training_up_rate = float(np.mean(prepared.y_train))
    always_up_probabilities = np.ones_like(y_test, dtype=float)
    constant_probabilities = np.full_like(y_test, training_up_rate, dtype=float)

    X_train = prepared.X_train.reshape(len(prepared.X_train), -1)
    X_test = prepared.X_test.reshape(len(prepared.X_test), -1)
    scaler = StandardScaler().fit(X_train)
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced")
    logistic.fit(scaler.transform(X_train), prepared.y_train.astype(int))
    logistic_probabilities = logistic.predict_proba(scaler.transform(X_test))[:, 1]
    joblib.dump({"scaler": scaler, "model": logistic}, output_dir / "logistic_regression.joblib")

    comparison = {
        "always_up": classification_metrics(y_test, always_up_probabilities),
        "constant_training_up_rate": classification_metrics(y_test, constant_probabilities),
        "logistic_regression": classification_metrics(y_test, logistic_probabilities),
    }
    with open(output_dir / "baseline_metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(comparison, metrics_file, indent=2)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
