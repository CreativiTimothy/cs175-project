import numpy as np
import pandas as pd

from src.data.preprocess import assign_chronological_splits, scale_from_training_only


def test_splits_are_chronological():
    metadata = pd.DataFrame({"target_date": pd.date_range("2020-01-01", periods=20)})
    result = assign_chronological_splits(metadata, 0.70, 0.15)
    train_max = result.loc[result["split"] == "train", "target_date"].max()
    validation_min = result.loc[result["split"] == "validation", "target_date"].min()
    validation_max = result.loc[result["split"] == "validation", "target_date"].max()
    test_min = result.loc[result["split"] == "test", "target_date"].min()
    assert train_max < validation_min
    assert validation_max < test_min


def test_scaler_is_fit_only_on_training_data():
    X = np.zeros((6, 2, 1), dtype=np.float32)
    X[:3] = 1.0
    X[3:] = 1000.0
    metadata = pd.DataFrame({"split": ["train"] * 3 + ["test"] * 3})
    transformed, scaler = scale_from_training_only(X, metadata)
    assert np.isclose(scaler.mean_[0], 1.0)
    assert transformed[3:].mean() > 100.0
