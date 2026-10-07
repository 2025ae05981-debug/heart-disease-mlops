"""Load the trained pipeline and return predictions with confidence."""
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_model(path: Path = MODEL_PATH):
    return joblib.load(path)


def predict(model, record: dict) -> dict:
    X = pd.DataFrame([record])[FEATURES]
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "prediction": int(proba >= 0.5),
        "confidence": round(max(proba, 1 - proba), 4),
        "probability_disease": round(proba, 4),
    }


if __name__ == "__main__":
    sample = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
              "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
              "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6}
    print(predict(load_model(), sample))
