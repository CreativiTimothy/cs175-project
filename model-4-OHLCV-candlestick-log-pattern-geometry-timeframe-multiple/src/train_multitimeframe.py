import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.data.multitimeframe import prepare_multitimeframe_data
from src.metrics import classification_metrics
from src.models.multitimeframe import MultiTimeframeHybrid

DIRECT_NAMES = [
    "open_prev_close_log", "high_prev_close_log", "low_prev_close_log", "close_return_1",
    "body_log", "open_to_high_log", "open_to_low_log", "range_log",
    "upper_wick_fraction", "lower_wick_fraction", "body_fraction", "close_location",
    "open_location", "bullish_bar", "volume_change_1",
]


class MultiDataset(Dataset):
    def __init__(self, target, context, labels):
        self.target = torch.tensor(target, dtype=torch.float32)
        self.context = torch.tensor(context, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.target[index], self.context[index], self.labels[index]


def raw_predictions(model, loader, device):
    values = []
    model.eval()
    with torch.no_grad():
        for target, context, _ in loader:
            values.append(model(target.to(device), context.to(device)).cpu().numpy())
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    output = Path(config["project"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_multitimeframe_data(config, output)

    torch.manual_seed(int(config["project"].get("seed", 42)))
    direct_indices = [data.feature_names.index(name) for name in DIRECT_NAMES]
    model = MultiTimeframeHybrid(
        len(data.feature_names),
        len(data.timeframe_names),
        direct_indices,
        projection=int(config["model"].get("projection_size", 16)),
        hidden=int(config["model"].get("lstm_hidden_size", 32)),
        context_hidden=int(config["model"].get("context_hidden_size", 128)),
        dropout=float(config["model"].get("dropout", 0.30)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}; device={device}")

    batch_size = int(config["training"].get("batch_size", 128))
    train_loader = DataLoader(MultiDataset(data.target_train, data.context_train, data.y_train), batch_size, shuffle=True)
    validation_loader = DataLoader(MultiDataset(data.target_validation, data.context_validation, data.y_validation), batch_size)
    test_loader = DataLoader(MultiDataset(data.target_test, data.context_test, data.y_test), batch_size)
    holdout_loader = DataLoader(MultiDataset(data.target_holdout, data.context_holdout, data.y_holdout), batch_size)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 0.0001)),
        weight_decay=float(config["training"].get("weight_decay", 0.001)),
    )
    best_state, best_loss, stale = None, float("inf"), 0
    for epoch in range(1, int(config["training"].get("epochs", 100)) + 1):
        model.train()
        train_losses = []
        for target, context, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(target.to(device), context.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        model.eval()
        validation_losses = []
        with torch.no_grad():
            for target, context, labels in validation_loader:
                validation_losses.append(criterion(model(target.to(device), context.to(device)), labels.to(device)).item())
        train_loss, validation_loss = np.mean(train_losses), np.mean(validation_losses)
        print(f"Epoch {epoch:03d} train={train_loss:.6f} validation={validation_loss:.6f}")
        if validation_loss < best_loss - 1e-6:
            best_loss, stale, best_state = validation_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= int(config["training"].get("patience", 15)):
                print("Early stopping")
                break
    model.load_state_dict(best_state)

    validation_raw = raw_predictions(model, validation_loader, device)
    calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(validation_raw.reshape(-1, 1), data.y_validation.astype(int))
    test_raw = raw_predictions(model, test_loader, device)
    test_probability = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    prediction_frame = data.meta_test.copy()
    prediction_frame["raw_probability_up"] = torch.sigmoid(torch.tensor(test_raw)).numpy()
    prediction_frame["probability_up"] = test_probability
    prediction_frame.to_csv(output / "test_predictions.csv", index=False)
    metrics = classification_metrics(data.y_test, test_probability)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_names": data.feature_names,
            "timeframe_names": data.timeframe_names,
            "target_timeframe": config["data"]["target_timeframe"],
        },
        output / "best_model.pt",
    )

    if config.get("evaluation", {}).get("unlock_final_holdout", False):
        holdout_raw = raw_predictions(model, holdout_loader, device)
        holdout_probability = calibrator.predict_proba(holdout_raw.reshape(-1, 1))[:, 1]
        holdout_frame = data.meta_holdout.copy()
        holdout_frame["probability_up"] = holdout_probability
        holdout_frame.to_csv(output / "final_holdout_predictions.csv", index=False)
        holdout_metrics = classification_metrics(data.y_holdout, holdout_probability)
        (output / "final_holdout_metrics.json").write_text(json.dumps(holdout_metrics, indent=2))
        print("FINAL HOLDOUT")
        print(json.dumps(holdout_metrics, indent=2))


if __name__ == "__main__":
    main()
