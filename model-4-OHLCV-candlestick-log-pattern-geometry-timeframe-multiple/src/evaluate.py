import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from src.metrics import classification_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    if "probability_up" not in predictions:
        raise ValueError("Evaluation plots require classification probabilities.")

    y_true = predictions["direction"].astype(int).to_numpy()
    probabilities = predictions["probability_up"].to_numpy()
    metrics = classification_metrics(y_true, probabilities)
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=["Down", "Up"]).plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(run_dir / "confusion_matrix.png", dpi=200)
    plt.close()

    RocCurveDisplay.from_predictions(y_true, probabilities)
    plt.tight_layout()
    plt.savefig(run_dir / "roc_curve.png", dpi=200)
    plt.close()

    observed, predicted = calibration_curve(y_true, probabilities, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    plt.plot(predicted, observed, marker="o", label="Model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction of up days")
    plt.title("Probability Calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "calibration.png", dpi=200)
    plt.close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
