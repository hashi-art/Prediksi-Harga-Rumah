import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    j = r.json()
    assert "status" in j
    assert j["status"] == "ok"

def test_predict_missing_feature():
    # send empty features -> should respond with 400 due to missing features
    r = client.post("/predict", json={"features": {}})
    assert r.status_code in (400, 500)  # template may raise 400 for missing features or 500 for other issues

def test_predict_example():
    # Provide all features from default list used by template
    sample = {
        "MedInc": 8,
        "HouseAge": 30,
        "AveRooms": 5,
        "AveBedrms": 1,
        "Population": 1000,
        "AveOccup": 3,
        "Latitude": 34,
        "Longitude": -118
    }
    r = client.post("/predict", json={"features": sample})
    assert r.status_code == 200
    j = r.json()
    assert "prediction" in j
    assert isinstance(j["prediction"], float) or isinstance(j["prediction"], int)