import pytest
import requests
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_e2e_prediction():
    url = "http://localhost:8000/model/api/v2/predict"
    data = {"message": "This is the best day ever!"}
    response = requests.post(url, json=data)
    assert response.status_code == 200
    assert response.json()["sentiment"] in ["Positive", "Negative", "Neutral"]


def test_health_endpoint():
    response = client.get("http://localhost:8000/model/api/v2/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "sentiment-analysis"}
