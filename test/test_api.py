from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "TinyBERT Sentiment Analysis API is running."}

def test_predict_positive_sentiment():
    response = client.post("/predict/", json={"text": "I love this product!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] in ["positive", "neutral", "negative"]  # Basic check

def test_predict_negative_sentiment():
    response = client.post("/predict/", json={"text": "This is terrible and I hate it."})
    assert response.status_code == 200
    assert response.json()["sentiment"] in ["positive", "neutral", "negative"]

def test_predict_empty_text():
    response = client.post("/predict/", json={"text": ""})
    assert response.status_code == 200
    assert response.json()["sentiment"] in ["positive", "neutral", "negative"]
