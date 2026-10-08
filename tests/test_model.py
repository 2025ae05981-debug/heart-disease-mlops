"""Unit tests for the trained model and prediction helper."""
import pandas as pd
import pytest

from src.predict import FEATURES, load_model, predict

SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
          "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
          "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_predict_returns_expected_keys(model):
    result = predict(model, SAMPLE)
    assert set(result) == {"prediction", "confidence", "probability_disease"}


def test_prediction_is_binary(model):
    assert predict(model, SAMPLE)["prediction"] in (0, 1)


def test_confidence_in_valid_range(model):
    conf = predict(model, SAMPLE)["confidence"]
    assert 0.5 <= conf <= 1.0


def test_model_accepts_dataframe(model):
    df = pd.DataFrame([SAMPLE])[FEATURES]
    assert model.predict_proba(df).shape == (1, 2)


def test_missing_feature_raises(model):
    bad = {k: v for k, v in SAMPLE.items() if k != "age"}
    with pytest.raises(KeyError):
        predict(model, bad)
