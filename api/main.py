"""FastAPI service for heart disease risk prediction."""
import logging
import time

from fastapi import FastAPI, Request, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel, Field

from src.predict import load_model, predict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-api")

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")
model = load_model()

REQUESTS = Counter("api_requests_total", "Total API requests",
                   ["method", "path", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency",
                    ["path"])
PREDICTIONS = Counter("api_predictions_total", "Predictions by class",
                      ["prediction"])


class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=1, le=4)
    trestbps: float = Field(..., ge=50, le=300)
    chol: float = Field(..., ge=50, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=40, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=1, le=3)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=3, le=7)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    path = request.url.path
    REQUESTS.labels(request.method, path, response.status_code).inc()
    LATENCY.labels(path).observe(elapsed)
    logger.info("%s %s -> %s (%.1f ms)", request.method, path,
                response.status_code, elapsed * 1000)
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict_endpoint(patient: PatientData):
    result = predict(model, patient.model_dump())
    PREDICTIONS.labels(str(result["prediction"])).inc()
    logger.info("prediction=%s confidence=%s", result["prediction"],
                result["confidence"])
    return result


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
