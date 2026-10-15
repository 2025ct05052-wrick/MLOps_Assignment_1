"""FastAPI service exposing heart-disease predictions and Prometheus metrics."""

import logging
import os
import time
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.config import DEFAULT_ARTIFACT_DIR, FEATURE_COLUMNS
from src.modeling import probability_for_positive_class

LOGGER = logging.getLogger("heart_disease_api")
REQUEST_COUNT = Counter(
    "heart_disease_api_requests_total",
    "HTTP requests handled by this API.",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "heart_disease_api_request_duration_seconds",
    "HTTP request duration in seconds.",
    ["method", "path"],
)
PREDICTION_COUNT = Counter(
    "heart_disease_predictions_total",
    "Predictions returned by class.",
    ["prediction"],
)
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_ARTIFACT_DIR / "heart_disease_model.joblib")))
_model = None
_model_error: str | None = None

app = FastAPI(
    title="Heart Disease Risk API",
    version="1.0.0",
    description="Predicts binary disease status from 13 clinical features.",
)


class PredictionRequest(BaseModel):
    """Input fields use the UCI Cleveland feature names and value ranges."""

    age: Annotated[float, Field(ge=1, le=120)]
    sex: Annotated[int, Field(ge=0, le=1)]
    cp: Annotated[int, Field(ge=1, le=4)]
    trestbps: Annotated[float, Field(ge=50, le=300)]
    chol: Annotated[float, Field(ge=50, le=700)]
    fbs: Annotated[int, Field(ge=0, le=1)]
    restecg: Annotated[int, Field(ge=0, le=2)]
    thalach: Annotated[float, Field(ge=40, le=250)]
    exang: Annotated[int, Field(ge=0, le=1)]
    oldpeak: Annotated[float, Field(ge=0, le=10)]
    slope: Annotated[int, Field(ge=1, le=3)]
    ca: Annotated[float | None, Field(default=None, ge=0, le=3)]
    thal: Annotated[float | None, Field(default=None, ge=3, le=7)]


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    model_version: str


def _load_model():
    global _model, _model_error
    if _model is not None:
        return _model
    try:
        _model = joblib.load(MODEL_PATH)
        _model_error = None
        LOGGER.info("model_loaded path=%s", MODEL_PATH)
        return _model
    except (FileNotFoundError, OSError, ValueError) as error:
        _model_error = str(error)
        LOGGER.warning("model_unavailable path=%s reason=%s", MODEL_PATH, error)
        raise


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start
        path = request.url.path
        REQUEST_COUNT.labels(request.method, path, str(status)).inc()
        REQUEST_LATENCY.labels(request.method, path).observe(duration)
        LOGGER.info(
            "request_completed method=%s path=%s status=%s duration_ms=%.2f",
            request.method, path, status, duration * 1000,
        )


@app.get("/health")
def health():
    """Report readiness and model availability."""
    try:
        _load_model()
        return {"status": "ok", "model_loaded": True}
    except (FileNotFoundError, OSError, ValueError):
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "model_loaded": False, "detail": _model_error},
        )


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Expose request and prediction counters for Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict binary class and return the estimated probability of disease."""
    try:
        model = _load_model()
    except (FileNotFoundError, OSError, ValueError) as error:
        raise HTTPException(status_code=503, detail="Model artifact is not available.") from error
    record = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
    try:
        prediction = int(model.predict(record)[0])
        confidence = float(probability_for_positive_class(model, record)[0])
    except (AttributeError, TypeError, ValueError) as error:
        LOGGER.exception("prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction could not be computed.") from error
    PREDICTION_COUNT.labels(str(prediction)).inc()
    return PredictionResponse(
        prediction=prediction,
        label="higher_risk" if prediction else "lower_risk",
        confidence=round(confidence, 4),
        model_version="1.0.0",
    )

