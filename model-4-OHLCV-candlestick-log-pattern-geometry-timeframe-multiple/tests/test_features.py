import numpy as np
import pandas as pd
from src.data.preprocess import add_pattern_features, get_feature_names


def bars(n=240):
    rng = np.random.default_rng(7)
    close = 100 * np.exp(np.cumsum(rng.normal(0, .01, n)))
    open_ = close * np.exp(rng.normal(0, .003, n))
    high = np.maximum(open_, close) * np.exp(abs(rng.normal(.002, .001, n)))
    low = np.minimum(open_, close) / np.exp(abs(rng.normal(.002, .001, n)))
    return pd.DataFrame({"symbol":"TEST", "timestamp":pd.date_range('2020-01-01', periods=n, freq='B', tz='UTC'),
                         "open":open_, "high":high, "low":low, "close":close,
                         "volume":rng.integers(100000, 2000000, n)})


def test_feature_set_is_large_and_finite():
    names = get_feature_names(True)
    valid = add_pattern_features(bars(), True)[names].dropna()
    assert len(names) >= 80
    assert len(valid) > 0
    assert np.isfinite(valid.to_numpy()).all()


def test_fraction_features_are_bounded():
    frame = add_pattern_features(bars(), False).dropna()
    for name in ['upper_wick_fraction','lower_wick_fraction','body_fraction','close_location']:
        assert frame[name].between(-1e-7, 1 + 1e-7).all()
