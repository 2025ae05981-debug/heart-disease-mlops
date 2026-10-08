"""Unit tests for the preprocessing pipeline."""
import numpy as np
import pandas as pd

from src.preprocessing import (CATEGORICAL_FEATURES, NUMERIC_FEATURES,
                               build_preprocessor)


def make_df(n=20):
    rng = np.random.default_rng(0)
    data = {c: rng.normal(50, 10, n) for c in NUMERIC_FEATURES}
    for c in CATEGORICAL_FEATURES:
        data[c] = rng.integers(0, 3, n)
    return pd.DataFrame(data)


def test_preprocessor_output_shape():
    df = make_df()
    out = build_preprocessor().fit_transform(df)
    assert out.shape[0] == len(df)
    assert out.shape[1] >= len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)


def test_preprocessor_handles_missing_values():
    df = make_df()
    df.loc[0, "chol"] = np.nan
    df.loc[1, "ca"] = np.nan
    out = build_preprocessor().fit_transform(df)
    assert not np.isnan(np.asarray(out)).any()


def test_numeric_features_are_scaled():
    df = make_df(200)
    prep = build_preprocessor().fit(df)
    out = np.asarray(prep.transform(df))
    assert abs(out[:, 0].mean()) < 1e-6
    assert abs(out[:, 0].std() - 1) < 1e-2


def test_unknown_category_does_not_crash():
    df = make_df()
    prep = build_preprocessor().fit(df)
    new = make_df(1)
    new["cp"] = 99
    prep.transform(new)
