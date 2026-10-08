"""FastAPI application for Heart Disease model inference."""

from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Heart Disease Prediction API")

# Resolve model path relative to project root
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_model.pkl"  # Adjust filename if your saved model uses a different name

# Load trained model/pipeline
try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None


class PatientInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/")
def read_root():
    return {"status": "healthy", "message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(patient: PatientInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Model file not loaded")

    # Convert incoming JSON data to DataFrame
    data = pd.DataFrame([patient.dict()])

    # Run inference
    prediction = int(model.predict(data)[0])
    
    # Calculate probability if model supports it
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(data)[0]
        confidence = float(probs[prediction])
    else:
        confidence = 1.0

    return {
        "prediction": prediction,
        "risk_status": "Heart Disease Detected" if prediction == 1 else "No Heart Disease",
        "confidence": round(confidence, 4),
    }