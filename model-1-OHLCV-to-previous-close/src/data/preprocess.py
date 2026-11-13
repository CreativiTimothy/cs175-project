from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    X_train: np.ndarray
    y_train: np.ndarray
    meta_train: pd.DataFrame
    X_validation: np.ndarray
    y_validation: np.ndarray
    meta_validation: pd.DataFrame
    X_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame
    scaler: StandardScaler
    feature_names: list


def load_bars(path, symbols=None):
    bars = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if symbols:
        bars = bars[bars["symbol"].isin(symbols)]
    bars = bars.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])
    if bars.empty:
        raise ValueError("No rows remain after loading and symbol filtering.")
    return bars


def add_normalized_features(group, feature_names):
    """Convert OHLC prices into scale-independent changes known at each timestamp."""
    group = group.sort_values("timestamp").copy()
    previous_close = group["close"].shift(1)
    group["open_norm"] = group["open"] / previous_close - 1.0
    group["high_norm"] = group["high"] / previous_close - 1.0
    group["low_norm"] = group["low"] / previous_close - 1.0
    group["close_norm"] = group["close"] / previous_close - 1.0
    group["volume_norm"] = np.log1p(group["volume"]).diff()
    return group


def build_windows(bars, window_size, raw_features, task):
    feature_map = {
        "open": "open_norm",
        "high": "high_norm",
        "low": "low_norm",
        "close": "close_norm",
        "volume": "volume_norm",
    }
    feature_names = [feature_map[name] for name in raw_features]
    sequences, targets, metadata = [], [], []

    for symbol, symbol_rows in bars.groupby("symbol", sort=False):
        rows = add_normalized_features(symbol_rows, feature_names).reset_index(drop=True)
        rows["next_close"] = rows["close"].shift(-1)
        rows["next_open"] = rows["open"].shift(-1)
        rows["target_return"] = rows["next_close"] / rows["close"] - 1.0
        rows["executable_return"] = rows["next_close"] / rows["next_open"] - 1.0
        rows["direction"] = (rows["target_return"] > 0).astype(float)

        for end_index in range(window_size - 1, len(rows) - 1):
            window = rows.loc[end_index - window_size + 1:end_index, feature_names]
            if window.isna().any().any():
                continue
            target = rows.loc[end_index, "direction" if task == "classification" else "target_return"]
            sequences.append(window.to_numpy(dtype=np.float32))
            targets.append(np.float32(target))
            metadata.append({
                "symbol": symbol,
                "feature_end": rows.loc[end_index, "timestamp"],
                "target_date": rows.loc[end_index + 1, "timestamp"],
                "current_close": rows.loc[end_index, "close"],
                "next_open": rows.loc[end_index, "next_open"],
                "next_close": rows.loc[end_index, "next_close"],
                "target_return": rows.loc[end_index, "target_return"],
                "executable_return": rows.loc[end_index, "executable_return"],
                "direction": rows.loc[end_index, "direction"],
            })

    if not sequences:
        raise ValueError("No windows were created. Increase the date range or reduce window_size.")
    return np.stack(sequences), np.asarray(targets), pd.DataFrame(metadata), feature_names


def assign_chronological_splits(metadata, train_fraction, validation_fraction):
    unique_dates = np.array(sorted(pd.to_datetime(metadata["target_date"]).unique()))
    train_end = max(1, int(len(unique_dates) * train_fraction))
    validation_end = max(train_end + 1, int(len(unique_dates) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(unique_dates) - 1)
    train_dates = set(unique_dates[:train_end])
    validation_dates = set(unique_dates[train_end:validation_end])
    test_dates = set(unique_dates[validation_end:])

    dates = pd.to_datetime(metadata["target_date"])
    split = np.where(dates.isin(train_dates), "train", np.where(dates.isin(validation_dates), "validation", "test"))
    metadata = metadata.copy()
    metadata["split"] = split
    return metadata


def scale_from_training_only(X, metadata):
    scaler = StandardScaler()
    train_mask = metadata["split"].to_numpy() == "train"
    train_flat = X[train_mask].reshape(-1, X.shape[-1])
    scaler.fit(train_flat)
    transformed = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape).astype(np.float32)
    return transformed, scaler


def prepare_data(config, output_dir):
    data_config = config["data"]
    bars = load_bars(data_config["path"], data_config.get("symbols") or None)
    features = list(data_config["features"])
    if data_config.get("include_volume") and "volume" not in features:
        features.append("volume")

    X, y, metadata, feature_names = build_windows(
        bars,
        window_size=int(data_config["window_size"]),
        raw_features=features,
        task=config["model"]["task"],
    )
    metadata = assign_chronological_splits(
        metadata,
        float(data_config["train_fraction"]),
        float(data_config["validation_fraction"]),
    )
    X, scaler = scale_from_training_only(X, metadata)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_dir / "scaler.joblib")
    processed_path = Path(data_config["processed_rows_path"])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(processed_path, index=False)

    parts = {}
    for split in ["train", "validation", "test"]:
        mask = metadata["split"].to_numpy() == split
        parts[split] = (X[mask], y[mask], metadata.loc[mask].reset_index(drop=True))
        if len(parts[split][0]) == 0:
            raise ValueError(f"Chronological split {split} is empty. Use a longer date range.")

    return PreparedData(
        X_train=parts["train"][0], y_train=parts["train"][1], meta_train=parts["train"][2],
        X_validation=parts["validation"][0], y_validation=parts["validation"][1], meta_validation=parts["validation"][2],
        X_test=parts["test"][0], y_test=parts["test"][1], meta_test=parts["test"][2],
        scaler=scaler, feature_names=feature_names,
    )
