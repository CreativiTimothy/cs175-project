from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    """Stores the prepared training, validation, and testing datasets."""

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
    """
    Load historical OHLCV bars from a CSV file.

    Timestamps are converted to UTC so daylight-saving-time offsets,
    such as -04:00 and -05:00, do not create mixed-timezone errors.
    """

    bars = pd.read_csv(path)

    required = {
        "symbol",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }

    missing = required.difference(bars.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # Market timestamps can use different offsets because of
    # daylight-saving time. Converting everything to UTC keeps
    # timestamps comparable and chronologically sortable.
    bars["timestamp"] = pd.to_datetime(
        bars["timestamp"],
        utc=True,
    )

    if symbols:
        bars = bars[
            bars["symbol"].isin(symbols)
        ]

    bars = (
        bars
        .sort_values(["symbol", "timestamp"])
        .drop_duplicates(["symbol", "timestamp"])
        .reset_index(drop=True)
    )

    if bars.empty:
        raise ValueError(
            "No rows remain after loading and symbol filtering."
        )

    return bars


def add_normalized_features(group, feature_names):
    """
    Represent each OHLC bar as geometric log distances relative to the
    preceding bar's closing price.

    This removes the absolute price scale while preserving the current
    bar's overnight gap, high, low, and closing movement. As a result,
    securities with very different price levels can share one model.
    """

    group = (
        group
        .sort_values("timestamp")
        .copy()
    )

    previous_close = group["close"].shift(1)

    # Describe the complete geometry of the current candlestick relative
    # to the preceding close rather than using absolute dollar prices.
    group["open_norm"] = np.log(
        group["open"] / previous_close
    )

    group["high_norm"] = np.log(
        group["high"] / previous_close
    )

    group["low_norm"] = np.log(
        group["low"] / previous_close
    )

    group["close_norm"] = np.log(
        group["close"] / previous_close
    )

    # Express volume as the log ratio between consecutive bars.
    # Volume is not currently enabled in daily_fnn.yaml, but this feature
    # is ready if include_volume is enabled later.
    previous_volume = group["volume"].shift(1)

    group["volume_norm"] = np.log(
        group["volume"] / previous_volume
    )

    # Zero or invalid price/volume values can produce positive or
    # negative infinity. Mark those values as missing so the window
    # construction code safely excludes the affected examples.
    group.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return group


def build_windows(
    bars,
    window_size,
    raw_features,
    task,
):
    """
    Convert chronological OHLCV data into fixed-length model examples.

    Each input example contains the previous window_size bars. The
    corresponding target is the direction or percentage return of the
    immediately following bar.
    """

    feature_map = {
        "open": "open_norm",
        "high": "high_norm",
        "low": "low_norm",
        "close": "close_norm",
        "volume": "volume_norm",
    }

    invalid_features = set(raw_features).difference(
        feature_map
    )

    if invalid_features:
        raise ValueError(
            "Unsupported features: "
            f"{sorted(invalid_features)}"
        )

    feature_names = [
        feature_map[name]
        for name in raw_features
    ]

    sequences = []
    targets = []
    metadata = []

    for symbol, symbol_rows in bars.groupby(
        "symbol",
        sort=False,
    ):
        rows = add_normalized_features(
            symbol_rows,
            feature_names,
        ).reset_index(drop=True)

        # Values from the next period are used only as labels and
        # evaluation metadata. They are not included in the input window.
        rows["next_close"] = (
            rows["close"].shift(-1)
        )

        rows["next_open"] = (
            rows["open"].shift(-1)
        )

        rows["target_return"] = (
            rows["next_close"]
            / rows["close"]
            - 1.0
        )

        # The historical backtest assumes that a prediction produced
        # after the current close enters at the next period's open.
        rows["executable_return"] = (
            rows["next_close"]
            / rows["next_open"]
            - 1.0
        )

        rows["direction"] = (
            rows["target_return"] > 0
        ).astype(float)

        # The last row has no known next-period result and therefore
        # cannot be used as a labeled training example.
        for end_index in range(
            window_size - 1,
            len(rows) - 1,
        ):
            start_index = (
                end_index
                - window_size
                + 1
            )

            window = rows.loc[
                start_index:end_index,
                feature_names,
            ]

            if window.isna().any().any():
                continue

            if task == "classification":
                target = rows.loc[
                    end_index,
                    "direction",
                ]

            elif task == "regression":
                target = rows.loc[
                    end_index,
                    "target_return",
                ]

            else:
                raise ValueError(
                    "The task must be either "
                    "'classification' or 'regression'."
                )

            sequences.append(
                window.to_numpy(
                    dtype=np.float32
                )
            )

            targets.append(
                np.float32(target)
            )

            metadata.append(
                {
                    "symbol": symbol,

                    "feature_end": rows.loc[
                        end_index,
                        "timestamp",
                    ],

                    "target_date": rows.loc[
                        end_index + 1,
                        "timestamp",
                    ],

                    "current_close": rows.loc[
                        end_index,
                        "close",
                    ],

                    "next_open": rows.loc[
                        end_index,
                        "next_open",
                    ],

                    "next_close": rows.loc[
                        end_index,
                        "next_close",
                    ],

                    "target_return": rows.loc[
                        end_index,
                        "target_return",
                    ],

                    "executable_return": rows.loc[
                        end_index,
                        "executable_return",
                    ],

                    "direction": rows.loc[
                        end_index,
                        "direction",
                    ],
                }
            )

    if not sequences:
        raise ValueError(
            "No model windows were created. "
            "Increase the data date range or reduce window_size."
        )

    return (
        np.stack(sequences),
        np.asarray(targets),
        pd.DataFrame(metadata),
        feature_names,
    )


def assign_chronological_splits(
    metadata,
    train_fraction,
    validation_fraction,
):
    """
    Assign complete target dates to chronological dataset splits.

    All examples sharing the same target date remain in the same split.
    This prevents observations from the same market date from being
    divided between training and evaluation datasets.

    The function requires target_date. The feature_end column is
    normalized to UTC only when that optional column is present.
    """

    if "target_date" not in metadata.columns:
        raise ValueError(
            "Metadata must contain a 'target_date' column."
        )

    if train_fraction <= 0:
        raise ValueError(
            "train_fraction must be greater than zero."
        )

    if validation_fraction <= 0:
        raise ValueError(
            "validation_fraction must be greater than zero."
        )

    if (
        train_fraction
        + validation_fraction
        >= 1.0
    ):
        raise ValueError(
            "The training and validation fractions must leave "
            "some observations for the test split."
        )

    # Convert all target dates to one timezone before sorting. This
    # resolves mixed -04:00 and -05:00 daylight-saving-time offsets.
    target_dates = pd.to_datetime(
        metadata["target_date"],
        utc=True,
    )

    unique_dates = np.array(
        sorted(target_dates.unique())
    )

    if len(unique_dates) < 3:
        raise ValueError(
            "At least three unique target dates are required "
            "to create training, validation, and test splits."
        )

    train_end = max(
        1,
        int(
            len(unique_dates)
            * train_fraction
        ),
    )

    validation_end = max(
        train_end + 1,
        int(
            len(unique_dates)
            * (
                train_fraction
                + validation_fraction
            )
        ),
    )

    # Reserve at least one target date for the test split.
    validation_end = min(
        validation_end,
        len(unique_dates) - 1,
    )

    train_dates = set(
        unique_dates[:train_end]
    )

    validation_dates = set(
        unique_dates[
            train_end:validation_end
        ]
    )

    test_dates = set(
        unique_dates[
            validation_end:
        ]
    )

    split = np.where(
        target_dates.isin(train_dates),
        "train",
        np.where(
            target_dates.isin(
                validation_dates
            ),
            "validation",
            "test",
        ),
    )

    metadata = metadata.copy()

    # Store the required target dates in a consistent UTC format.
    metadata["target_date"] = target_dates

    # Production metadata includes feature_end, but the splitting
    # function and unit test also support metadata with target_date only.
    if "feature_end" in metadata.columns:
        metadata["feature_end"] = pd.to_datetime(
            metadata["feature_end"],
            utc=True,
        )

    metadata["split"] = split

    return metadata


def scale_from_training_only(
    X,
    metadata,
):
    """
    Standardize features using statistics from the training split only.

    Fitting the scaler on validation or test observations would leak
    information from future periods into model training.
    """

    scaler = StandardScaler()

    train_mask = (
        metadata["split"].to_numpy()
        == "train"
    )

    if not train_mask.any():
        raise ValueError(
            "The training split is empty."
        )

    train_flat = X[
        train_mask
    ].reshape(
        -1,
        X.shape[-1],
    )

    scaler.fit(train_flat)

    transformed = scaler.transform(
        X.reshape(
            -1,
            X.shape[-1],
        )
    )

    transformed = transformed.reshape(
        X.shape
    ).astype(
        np.float32
    )

    return transformed, scaler


def prepare_data(
    config,
    output_dir,
):
    """
    Run the complete preprocessing pipeline.

    The pipeline loads bars, creates fixed-length windows, performs a
    chronological split, fits normalization on training observations,
    and returns each dataset with its associated metadata.
    """

    data_config = config["data"]

    bars = load_bars(
        data_config["path"],
        data_config.get("symbols") or None,
    )

    features = list(
        data_config["features"]
    )

    if (
        data_config.get(
            "include_volume"
        )
        and "volume" not in features
    ):
        features.append("volume")

    (
        X,
        y,
        metadata,
        feature_names,
    ) = build_windows(
        bars=bars,
        window_size=int(
            data_config["window_size"]
        ),
        raw_features=features,
        task=config["model"]["task"],
    )

    metadata = assign_chronological_splits(
        metadata=metadata,
        train_fraction=float(
            data_config["train_fraction"]
        ),
        validation_fraction=float(
            data_config[
                "validation_fraction"
            ]
        ),
    )

    X, scaler = scale_from_training_only(
        X,
        metadata,
    )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        scaler,
        output_dir / "scaler.joblib",
    )

    processed_path = Path(
        data_config[
            "processed_rows_path"
        ]
    )

    processed_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata.to_csv(
        processed_path,
        index=False,
    )

    parts = {}

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        mask = (
            metadata["split"].to_numpy()
            == split_name
        )

        parts[split_name] = (
            X[mask],
            y[mask],
            metadata.loc[
                mask
            ].reset_index(
                drop=True
            ),
        )

        if len(
            parts[split_name][0]
        ) == 0:
            raise ValueError(
                "Chronological split "
                f"'{split_name}' is empty. "
                "Use a longer data range."
            )

    return PreparedData(
        X_train=parts["train"][0],
        y_train=parts["train"][1],
        meta_train=parts["train"][2],

        X_validation=parts[
            "validation"
        ][0],
        y_validation=parts[
            "validation"
        ][1],
        meta_validation=parts[
            "validation"
        ][2],

        X_test=parts["test"][0],
        y_test=parts["test"][1],
        meta_test=parts["test"][2],

        scaler=scaler,
        feature_names=feature_names,
    )