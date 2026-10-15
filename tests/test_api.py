import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app import main as api


class DummyModel:
    def predict(self, frame):
        assert isinstance(frame, pd.DataFrame)
        return [1]

    def predict_proba(self, frame):
        return np.array([[0.2, 0.8]])


def test_health_and_prediction_contract(monkeypatch):
    monkeypatch.setattr(api, "_model", DummyModel())
    client = TestClient(api.app)
    assert client.get("/health").json()["model_loaded"] is True
    response = client.post(
        "/predict",
        json={
            "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 3, "ca": None, "thal": None,
        },
    )
    assert response.status_code == 200
    assert response.json() == {
        "prediction": 1,
        "label": "higher_risk",
        "confidence": 0.8,
        "model_version": "1.0.0",
    }
    assert "heart_disease_predictions_total" in client.get("/metrics").text


def test_prediction_rejects_invalid_input(monkeypatch):
    monkeypatch.setattr(api, "_model", DummyModel())
    client = TestClient(api.app)
    response = client.post("/predict", json={"age": 500})
    assert response.status_code == 422

