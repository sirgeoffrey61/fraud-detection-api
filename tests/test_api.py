import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data
    assert "best_threshold" in data


def test_predict_not_fraud(client):
    payload = {
        "Claim_Amount": 500,
        "Patient_Age": 30,
        "Patient_Gender": "M",
        "Provider_Type": "General",
        "Provider_Specialty": "GP",
        "Admission_Type": "Routine",
        "Service_Type": "Outpatient",
        "Provider_Patient_Distance_Miles": 5.0,
        "Claim_Submitted_Late": 0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "Not Fraud"
    assert data["probability"] < 0.2725


def test_predict_returns_required_fields(client):
    payload = {
        "Claim_Amount": 15000,
        "Patient_Age": 65,
        "Patient_Gender": "M",
        "Provider_Type": "Specialist",
        "Provider_Specialty": "Cardiology",
        "Admission_Type": "Emergency",
        "Service_Type": "Inpatient",
        "Provider_Patient_Distance_Miles": 120.5,
        "Claim_Submitted_Late": 1,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    for field in ("label", "probability", "confidence", "threshold_used"):
        assert field in data
