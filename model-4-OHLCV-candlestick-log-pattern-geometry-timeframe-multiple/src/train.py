import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.config import load_config, save_config, set_seed
from src.data.preprocess import prepare_data
from src.datasets import WindowDataset
from src.metrics import classification_metrics, regression_metrics
from src.models.factory import create_model


def predict_batches(model, loader, device, task):
    model.eval()
    outputs = []
    with torch.no_grad():
        for X_batch, _ in loader:
            raw = model(X_batch.to(device))
            values = torch.sigmoid(raw) if task == "classification" else raw
            outputs.append(values.cpu().numpy())
    return np.concatenate(outputs)


def train_model(config):
    seed = int(config["project"]["seed"])
    set_seed(seed)
    output_dir = Path(config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config_used.yaml")

    prepared = prepare_data(config, output_dir)
    batch_size = int(config["training"]["batch_size"])
    loaders = {
        "train": DataLoader(WindowDataset(prepared.X_train, prepared.y_train), batch_size=batch_size, shuffle=True),
        "validation": DataLoader(WindowDataset(prepared.X_validation, prepared.y_validation), batch_size=batch_size, shuffle=False),
        "test": DataLoader(WindowDataset(prepared.X_test, prepared.y_test), batch_size=batch_size, shuffle=False),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(config, prepared.X_train.shape[1], prepared.X_train.shape[2]).to(device)
    task = config["model"]["task"]
    criterion = nn.BCEWithLogitsLoss() if task == "classification" else nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    best_state = None
    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        model.train()
        train_losses = []
        for X_batch, y_batch in loaders["train"]:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for X_batch, y_batch in loaders["validation"]:
                validation_losses.append(criterion(model(X_batch.to(device)), y_batch.to(device)).item())

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"Epoch {epoch:03d} train={train_loss:.6f} validation={validation_loss:.6f}")

        if validation_loss < best_validation_loss - 1e-6:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(config["training"]["patience"]):
                print("Early stopping: validation loss stopped improving.")
                break

    model.load_state_dict(best_state)
    torch.save({
        "state_dict": model.state_dict(),
        "window_size": prepared.X_train.shape[1],
        "feature_count": prepared.X_train.shape[2],
        "feature_names": prepared.feature_names,
        "task": task,
    }, output_dir / "best_model.pt")

    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "history.csv", index=False)
    plt.figure(figsize=(8, 5))
    plt.plot(history_frame["epoch"], history_frame["train_loss"], label="Training")
    plt.plot(history_frame["epoch"], history_frame["validation_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=200)
    plt.close()

    test_values = predict_batches(model, loaders["test"], device, task)
    predictions = prepared.meta_test.copy()
    if task == "classification":
        predictions["probability_up"] = test_values
        predictions["predicted_direction"] = (test_values >= float(config["training"]["decision_threshold"])).astype(int)
        metrics = classification_metrics(prepared.y_test, test_values, float(config["training"]["decision_threshold"]))
    else:
        predictions["predicted_return"] = test_values
        predictions["probability_up"] = 1.0 / (1.0 + np.exp(-test_values * 100.0))
        metrics = regression_metrics(prepared.y_test, test_values)

    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)
    print(json.dumps(metrics, indent=2))
    return output_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train_model(load_config(args.config))


if __name__ == "__main__":
    main()
