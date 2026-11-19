from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.preprocess import add_pattern_features, get_feature_names, load_bars


@dataclass
class MultiTimeframeData:
    target_train: np.ndarray
    context_train: np.ndarray
    y_train: np.ndarray
    meta_train: pd.DataFrame
    target_validation: np.ndarray
    context_validation: np.ndarray
    y_validation: np.ndarray
    meta_validation: pd.DataFrame
    target_test: np.ndarray
    context_test: np.ndarray
    y_test: np.ndarray
    meta_test: pd.DataFrame
    target_holdout: np.ndarray
    context_holdout: np.ndarray
    y_holdout: np.ndarray
    meta_holdout: pd.DataFrame
    feature_names: list
    timeframe_names: list


def parse_duration(value):
    value = value.lower().strip()
    units = {"min": "minutes", "hour": "hours", "day": "days", "week": "weeks"}
    for suffix, keyword in units.items():
        if value.endswith(suffix):
            return pd.Timedelta(**{keyword: int(value[:-len(suffix)])})
    raise ValueError(f"Unsupported duration: {value}")


def feature_frame(path, duration, include_volume):
    bars = load_bars(path)
    names = get_feature_names(include_volume)
    frames = []
    for symbol, group in bars.groupby("symbol", sort=False):
        featured = add_pattern_features(group, include_volume).copy()
        featured["available_at"] = featured["timestamp"] + parse_duration(duration)
        featured["symbol"] = symbol
        frames.append(featured.dropna(subset=names))
    return pd.concat(frames, ignore_index=True), names


def build_multitimeframe_examples(config):
    data = config["data"]
    include_volume = bool(data.get("include_volume", True))
    target_name = data.get("target_timeframe", "1day")
    timeframe_names = list(data["timeframes"])
    if target_name not in timeframe_names:
        raise ValueError("target_timeframe must be present in data.timeframes")

    frames = {}
    feature_names = None
    for name, settings in data["timeframes"].items():
        print(f"Loading {name} from {settings['path']}")
        frame, names = feature_frame(settings["path"], settings["duration"], include_volume)
        frames[name] = frame
        feature_names = names

    target_window = int(data.get("target_window", 100))
    target = frames[target_name]
    target_sequences, context_vectors, labels, metadata = [], [], [], []

    common_symbols = set(target["symbol"])
    for name in timeframe_names:
        common_symbols &= set(frames[name]["symbol"])
    print(f"Symbols present in every timeframe: {len(common_symbols):,}")

    for symbol in sorted(common_symbols):
        target_rows = target[target["symbol"] == symbol].sort_values("available_at").reset_index(drop=True)
        target_rows["next_close"] = target_rows["close"].shift(-1)
        target_rows["next_open"] = target_rows["open"].shift(-1)
        target_rows["target_return"] = target_rows["next_close"] / target_rows["close"] - 1.0
        target_rows["executable_return"] = target_rows["next_close"] / target_rows["next_open"] - 1.0
        target_rows["direction"] = (target_rows["target_return"] > 0).astype(np.float32)

        lookups = {
            name: frames[name][frames[name]["symbol"] == symbol].sort_values("available_at").reset_index(drop=True)
            for name in timeframe_names
        }
        available_arrays = {
            name: lookups[name]["available_at"].astype("int64").to_numpy()
            for name in timeframe_names
        }

        for end in range(target_window - 1, len(target_rows) - 1):
            decision_time = target_rows.loc[end, "available_at"]
            target_sequence = target_rows.loc[end - target_window + 1:end, feature_names]
            if target_sequence.isna().any().any():
                continue

            summaries = []
            valid = True
            for name in timeframe_names:
                positions = available_arrays[name]
                index = np.searchsorted(positions, decision_time.value, side="right") - 1
                if index < 0:
                    valid = False
                    break
                summaries.append(lookups[name].loc[index, feature_names].to_numpy(np.float32))
            if not valid:
                continue

            target_sequences.append(target_sequence.to_numpy(np.float32))
            context_vectors.append(np.stack(summaries))
            labels.append(target_rows.loc[end, "direction"])
            metadata.append(
                {
                    "symbol": symbol,
                    "feature_end": target_rows.loc[end, "timestamp"],
                    "decision_time": decision_time,
                    "target_date": target_rows.loc[end + 1, "timestamp"],
                    "target_return": target_rows.loc[end, "target_return"],
                    "executable_return": target_rows.loc[end, "executable_return"],
                    "direction": target_rows.loc[end, "direction"],
                }
            )

    if not target_sequences:
        raise ValueError("No synchronized multi-timeframe examples could be created")
    return (
        np.stack(target_sequences),
        np.stack(context_vectors),
        np.asarray(labels, np.float32),
        pd.DataFrame(metadata),
        feature_names,
        timeframe_names,
    )


def assign_splits(metadata, data_config):
    dates = pd.to_datetime(metadata["target_date"], utc=True)
    holdout_start = pd.Timestamp(data_config["final_holdout_start"], tz="UTC")
    development_dates = np.array(sorted(dates[dates < holdout_start].unique()))
    holdout_dates = dates >= holdout_start
    if not len(development_dates) or not holdout_dates.any():
        raise ValueError("Need target rows both before and after final_holdout_start")
    train_fraction = float(data_config.get("train_fraction", 0.70))
    validation_fraction = float(data_config.get("validation_fraction", 0.15))
    train_end = int(len(development_dates) * train_fraction)
    validation_end = int(len(development_dates) * (train_fraction + validation_fraction))
    train_dates = set(development_dates[:train_end])
    validation_dates = set(development_dates[train_end:validation_end])
    test_dates = set(development_dates[validation_end:])
    result = metadata.copy()
    result["split"] = np.where(
        holdout_dates,
        "holdout",
        np.where(dates.isin(train_dates), "train", np.where(dates.isin(validation_dates), "validation", "test")),
    )
    return result


def prepare_multitimeframe_data(config, output_dir):
    target, context, y, metadata, names, timeframe_names = build_multitimeframe_examples(config)
    metadata = assign_splits(metadata, config["data"])
    train_mask = metadata["split"].to_numpy() == "train"

    target_scaler = StandardScaler().fit(target[train_mask].reshape(-1, target.shape[-1]))
    target = target_scaler.transform(target.reshape(-1, target.shape[-1])).reshape(target.shape).astype(np.float32)
    context_scalers = []
    for timeframe_index in range(context.shape[1]):
        scaler = StandardScaler().fit(context[train_mask, timeframe_index, :])
        context[:, timeframe_index, :] = scaler.transform(context[:, timeframe_index, :])
        context_scalers.append(scaler)
    context = context.astype(np.float32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"target_scaler": target_scaler, "context_scalers": context_scalers, "timeframe_names": timeframe_names},
        output_dir / "multitimeframe_scalers.joblib",
    )
    metadata.to_csv(output_dir / "multitimeframe_rows.csv", index=False)

    values = {}
    for split in ["train", "validation", "test", "holdout"]:
        mask = metadata["split"].to_numpy() == split
        values[split] = (target[mask], context[mask], y[mask], metadata.loc[mask].reset_index(drop=True))
        if not len(values[split][0]):
            raise ValueError(f"Split {split} is empty")
    print(
        f"Prepared {len(y):,} synchronized examples across {len(timeframe_names)} timeframes: "
        + ", ".join(f"{key}={len(value[0]):,}" for key, value in values.items())
    )
    return MultiTimeframeData(
        *values["train"],
        *values["validation"],
        *values["test"],
        *values["holdout"],
        names,
        timeframe_names,
    )
