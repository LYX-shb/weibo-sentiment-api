import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_predict_happy():
    """测试积极情感"""
    response = client.post("/predict", json={"text": "今天天气真好，心情特别开心！"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "happy"
    assert data["confidence"] > 0.9

def test_predict_angry():
    """测试愤怒情感"""
    response = client.post("/predict", json={"text": "气死我了，简直无法忍受！"})
    assert response.status_code == 200
    data = response.json()
    assert data["label"] == "angry"
    assert data["confidence"] > 0.5

def test_predict_short_text():
    """测试短文本"""
    response = client.post("/predict", json={"text": "哈哈"})
    assert response.status_code == 200
    assert "label" in response.json()

def test_predict_long_text():
    """测试长文本"""
    long_text = "今天" * 100
    response = client.post("/predict", json={"text": long_text})
    assert response.status_code == 200

def test_health_check():
    """测试健康检查接口"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"