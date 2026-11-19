from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RETURN_WINDOWS = (2, 3, 5, 10, 20, 25, 50)
STRUCTURE_WINDOWS = (5, 10, 20, 25, 50, 100)
SLOPE_WINDOWS = (5, 10, 20, 25)
VOLATILITY_WINDOWS = (5, 10, 20, 25, 50)


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
    """Load, validate, UTC-normalize, sort, and de-duplicate OHLCV bars."""
    bars = pd.read_csv(path)
    required = {"symbol", "timestamp", "open", "high", "low", "close", "volume"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    bars["timestamp"] = pd.to_datetime(bars["timestamp"], utc=True)
    numbers = ["open", "high", "low", "close", "volume"]
    bars[numbers] = bars[numbers].apply(pd.to_numeric, errors="coerce")
    bars = bars.dropna(subset=numbers)
    bars = bars[(bars[["open", "high", "low", "close"]] > 0).all(axis=1)]
    bars = bars[bars["volume"] >= 0]
    if symbols:
        bars = bars[bars["symbol"].isin(symbols)]
    bars = bars.sort_values(["symbol", "timestamp"]).drop_duplicates(["symbol", "timestamp"])
    if bars.empty:
        raise ValueError("No valid rows remain after loading and filtering.")
    return bars.reset_index(drop=True)


def _safe_divide(numerator, denominator):
    denominator = denominator.replace(0, np.nan)
    return (numerator / denominator).replace([np.inf, -np.inf], np.nan)


def _rolling_log_slope(series, window):
    """Trailing least-squares slope of log price on a normalized time axis."""
    x = np.linspace(0.0, 1.0, window)
    xc = x - x.mean()
    denominator = np.sum(xc ** 2)

    def slope(values):
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            return np.nan
        y = np.log(values)
        return float(np.sum(xc * (y - y.mean())) / denominator)

    return series.rolling(window, min_periods=window).apply(slope, raw=True)


def add_pattern_features(group, include_volume=True):
    """
    Create general pattern primitives from trailing information only.

    No named chart pattern is hardcoded. The channels describe candle shape,
    returns, range location, trend geometry, curvature, compression, volatility,
    and volume so the model can learn combinations that resemble many patterns.
    """
    g = group.sort_values("timestamp").copy()
    pc = g["close"].shift(1)
    candle_range = g["high"] - g["low"]
    body_top = g[["open", "close"]].max(axis=1)
    body_bottom = g[["open", "close"]].min(axis=1)

    # Previous-close log geometry and candle geometry.
    g["open_prev_close_log"] = np.log(g["open"] / pc)
    g["high_prev_close_log"] = np.log(g["high"] / pc)
    g["low_prev_close_log"] = np.log(g["low"] / pc)
    g["close_return_1"] = np.log(g["close"] / pc)
    g["body_log"] = np.log(g["close"] / g["open"])
    g["open_to_high_log"] = np.log(g["high"] / g["open"])
    g["open_to_low_log"] = np.log(g["low"] / g["open"])
    g["range_log"] = np.log(g["high"] / g["low"])
    g["upper_wick_fraction"] = _safe_divide(g["high"] - body_top, candle_range)
    g["lower_wick_fraction"] = _safe_divide(body_bottom - g["low"], candle_range)
    g["body_fraction"] = _safe_divide((g["close"] - g["open"]).abs(), candle_range)
    g["close_location"] = _safe_divide(g["close"] - g["low"], candle_range)
    g["open_location"] = _safe_divide(g["open"] - g["low"], candle_range)
    g["bullish_bar"] = (g["close"] > g["open"]).astype(float)

    # Returns at pattern-relevant horizons.
    for w in RETURN_WINDOWS:
        g[f"close_return_{w}"] = np.log(g["close"] / g["close"].shift(w))

    # Support/resistance proximity and rolling-range shape.
    for w in STRUCTURE_WINDOWS:
        rh = g["high"].rolling(w, min_periods=w).max()
        rl = g["low"].rolling(w, min_periods=w).min()
        width = np.log(rh / rl)
        g[f"close_to_high_{w}"] = np.log(g["close"] / rh)
        g[f"close_to_low_{w}"] = np.log(g["close"] / rl)
        g[f"range_location_{w}"] = _safe_divide(g["close"] - rl, rh - rl)
        g[f"rolling_width_{w}"] = width
        g[f"width_change_{w}"] = width.diff()

    # Trend-line geometry and convergence.
    for w in SLOPE_WINDOWS:
        cs = _rolling_log_slope(g["close"], w)
        hs = _rolling_log_slope(g["high"], w)
        ls = _rolling_log_slope(g["low"], w)
        g[f"close_slope_{w}"] = cs
        g[f"high_slope_{w}"] = hs
        g[f"low_slope_{w}"] = ls
        g[f"trendline_convergence_{w}"] = ls - hs

    # Curvature and momentum changes.
    g["close_slope_accel_5_20"] = g["close_slope_5"] - g["close_slope_20"]
    g["close_slope_accel_10_25"] = g["close_slope_10"] - g["close_slope_25"]
    g["momentum_change_5"] = g["close_return_5"] - g["close_return_5"].shift(5)
    g["momentum_change_10"] = g["close_return_10"] - g["close_return_10"].shift(10)

    # Volatility, true range, and compression/expansion.
    true_range = pd.concat(
        [g["high"] - g["low"], (g["high"] - pc).abs(), (g["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    normalized_tr = _safe_divide(true_range, pc)
    for w in VOLATILITY_WINDOWS:
        g[f"volatility_{w}"] = g["close_return_1"].rolling(w, min_periods=w).std()
        g[f"mean_range_{w}"] = g["range_log"].rolling(w, min_periods=w).mean()
        g[f"atr_fraction_{w}"] = normalized_tr.rolling(w, min_periods=w).mean()
    g["range_ratio_5_20"] = _safe_divide(g["mean_range_5"], g["mean_range_20"])
    g["range_ratio_10_25"] = _safe_divide(g["mean_range_10"], g["mean_range_25"])
    g["volatility_ratio_5_20"] = _safe_divide(g["volatility_5"], g["volatility_20"])
    g["volatility_ratio_10_50"] = _safe_divide(g["volatility_10"], g["volatility_50"])

    if include_volume:
        log_volume = np.log1p(g["volume"])
        vm5 = g["volume"].rolling(5, min_periods=5).mean()
        vm20 = g["volume"].rolling(20, min_periods=20).mean()
        vs20 = g["volume"].rolling(20, min_periods=20).std()
        g["volume_change_1"] = log_volume.diff()
        g["volume_ratio_5"] = _safe_divide(g["volume"], vm5)
        g["volume_ratio_20"] = _safe_divide(g["volume"], vm20)
        g["volume_trend_5_20"] = _safe_divide(vm5, vm20)
        g["volume_zscore_20"] = _safe_divide(g["volume"] - vm20, vs20)
        g["return_volume_corr_20"] = g["close_return_1"].rolling(20, min_periods=20).corr(log_volume)

    return g.replace([np.inf, -np.inf], np.nan)


def get_feature_names(include_volume=True):
    names = [
        "open_prev_close_log", "high_prev_close_log", "low_prev_close_log",
        "close_return_1", "body_log", "open_to_high_log", "open_to_low_log",
        "range_log", "upper_wick_fraction", "lower_wick_fraction",
        "body_fraction", "close_location", "open_location", "bullish_bar",
    ]
    names += [f"close_return_{w}" for w in RETURN_WINDOWS]
    for w in STRUCTURE_WINDOWS:
        names += [f"close_to_high_{w}", f"close_to_low_{w}", f"range_location_{w}",
                  f"rolling_width_{w}", f"width_change_{w}"]
    for w in SLOPE_WINDOWS:
        names += [f"close_slope_{w}", f"high_slope_{w}", f"low_slope_{w}",
                  f"trendline_convergence_{w}"]
    names += ["close_slope_accel_5_20", "close_slope_accel_10_25",
              "momentum_change_5", "momentum_change_10"]
    for w in VOLATILITY_WINDOWS:
        names += [f"volatility_{w}", f"mean_range_{w}", f"atr_fraction_{w}"]
    names += ["range_ratio_5_20", "range_ratio_10_25",
              "volatility_ratio_5_20", "volatility_ratio_10_50"]
    if include_volume:
        names += ["volume_change_1", "volume_ratio_5", "volume_ratio_20",
                  "volume_trend_5_20", "volume_zscore_20", "return_volume_corr_20"]
    return names


def build_windows(bars, window_size, task, include_volume=True):
    names = get_feature_names(include_volume)
    sequences, targets, metadata = [], [], []
    for symbol, raw in bars.groupby("symbol", sort=False):
        rows = add_pattern_features(raw, include_volume).reset_index(drop=True)
        rows["next_close"] = rows["close"].shift(-1)
        rows["next_open"] = rows["open"].shift(-1)
        rows["target_return"] = rows["next_close"] / rows["close"] - 1.0
        rows["executable_return"] = rows["next_close"] / rows["next_open"] - 1.0
        rows["direction"] = (rows["target_return"] > 0).astype(float)
        for end in range(window_size - 1, len(rows) - 1):
            window = rows.loc[end - window_size + 1:end, names]
            if window.isna().any().any():
                continue
            target = rows.loc[end, "direction" if task == "classification" else "target_return"]
            sequences.append(window.to_numpy(np.float32))
            targets.append(np.float32(target))
            metadata.append({
                "symbol": symbol, "feature_end": rows.loc[end, "timestamp"],
                "target_date": rows.loc[end + 1, "timestamp"],
                "current_close": rows.loc[end, "close"], "next_open": rows.loc[end, "next_open"],
                "next_close": rows.loc[end, "next_close"], "target_return": rows.loc[end, "target_return"],
                "executable_return": rows.loc[end, "executable_return"],
                "direction": rows.loc[end, "direction"],
            })
    if not sequences:
        raise ValueError("No windows created. Use at least about 200 valid bars per symbol.")
    return np.stack(sequences), np.asarray(targets), pd.DataFrame(metadata), names


def assign_chronological_splits(metadata, train_fraction, validation_fraction):
    dates = pd.to_datetime(metadata["target_date"], utc=True)
    unique = np.array(sorted(dates.unique()))
    train_end = max(1, int(len(unique) * train_fraction))
    val_end = min(max(train_end + 1, int(len(unique) * (train_fraction + validation_fraction))), len(unique) - 1)
    train_dates, val_dates = set(unique[:train_end]), set(unique[train_end:val_end])
    result = metadata.copy()
    result["target_date"] = dates
    if "feature_end" in result:
        result["feature_end"] = pd.to_datetime(result["feature_end"], utc=True)
    result["split"] = np.where(dates.isin(train_dates), "train",
                               np.where(dates.isin(val_dates), "validation", "test"))
    return result


def scale_from_training_only(X, metadata):
    mask = metadata["split"].to_numpy() == "train"
    scaler = StandardScaler().fit(X[mask].reshape(-1, X.shape[-1]))
    scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    return scaled.astype(np.float32), scaler


def prepare_data(config, output_dir):
    dc = config["data"]
    bars = load_bars(dc["path"], dc.get("symbols") or None)
    include_volume = bool(dc.get("include_volume", True))
    X, y, metadata, names = build_windows(
        bars, int(dc["window_size"]), config["model"]["task"], include_volume
    )
    metadata = assign_chronological_splits(
        metadata, float(dc["train_fraction"]), float(dc["validation_fraction"])
    )
    X, scaler = scale_from_training_only(X, metadata)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_dir / "scaler.joblib")
    processed = Path(dc["processed_rows_path"])
    processed.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(processed, index=False)
    pd.Series(names, name="feature_name").to_csv(output_dir / "feature_names.csv", index=False)

    parts = {}
    for split in ("train", "validation", "test"):
        mask = metadata["split"].to_numpy() == split
        parts[split] = (X[mask], y[mask], metadata.loc[mask].reset_index(drop=True))
        if not len(parts[split][0]):
            raise ValueError(f"Chronological split '{split}' is empty.")
    print(f"Prepared {len(X):,} examples: {X.shape[1]} bars x {X.shape[2]} feature channels.")
    return PreparedData(
        parts["train"][0], parts["train"][1], parts["train"][2],
        parts["validation"][0], parts["validation"][1], parts["validation"][2],
        parts["test"][0], parts["test"][1], parts["test"][2], scaler, names
    )
