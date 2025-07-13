import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from src.donations.api import app
import datetime

# Mock model and scalers for predict endpoint
def mock_transform(self, x):
    return x

def mock_inverse_transform(self, x):
    return x

class MockScaler:
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

class MockModel:
    def predict(self, inputs):
        # Return a fixed value for testing
        import numpy as np
        return np.array([[42]])

# Patch the model and scalers in the api module
import src.donations.api as api
api.scaler_x = MockScaler()
api.scaler_y = MockScaler()
api.model = MockModel()

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid():
    payload = {
        "lag1": 10,
        "lag2": 11,
        "lag3": 12,
        "lag4": 13,
        "lag5": 14,
        "lag6": 15,
        "lag7": 16,
        "nextday": datetime.datetime.now().strftime('%Y%m%d'),
        "high_donation_holiday": 0,
        "low_donation_holiday": 0,
        "religion_or_culture_holiday": 0,
        "other_holiday": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], float)

def test_predict_invalid_nextday():
    payload = {
        "lag1": 10,
        "lag2": 11,
        "lag3": 12,
        "lag4": 13,
        "lag5": 14,
        "lag6": 15,
        "lag7": 16,
        "nextday": "2025-07-13",  # Invalid format
        "high_donation_holiday": 0,
        "low_donation_holiday": 0,
        "religion_or_culture_holiday": 0,
        "other_holiday": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_predict_invalid_lags():
    payload = {
        "lag1": 10,
        "lag2": 10,
        "lag3": 10,
        "lag4": 10,
        "lag5": 10,
        "lag6": 10,
        "lag7": 10,
        "nextday": datetime.datetime.now().strftime('%Y%m%d'),
        "high_donation_holiday": 0,
        "low_donation_holiday": 0,
        "religion_or_culture_holiday": 0,
        "other_holiday": 0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "detail" in response.json()
