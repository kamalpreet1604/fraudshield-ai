"""Automatic checks for the API. Run with:  uv run pytest

They need the saved model, so run  uv run python -m src.train  once first.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api import app

pytestmark = pytest.mark.skipif(
    not Path("models/fraud_model.joblib").exists(),
    reason="model not trained yet: run  uv run python -m src.train",
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    # each test gets its own empty database, so the real history is never touched
    monkeypatch.setenv("FRAUD_DB_PATH", str(tmp_path / "test.db"))
    return TestClient(app)


def test_health_is_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_score_and_decision(client):
    r = client.post("/predict", json={"TransactionAmt": 500, "ProductCD": "C"})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["fraud_score"] <= 1
    assert body["decision"] in ("allow", "review")
    assert body["cutoff"] == 0.45


def test_decision_follows_the_cutoff(client):
    body = client.post("/predict", json={"TransactionAmt": 120}).json()
    expected = "review" if body["fraud_score"] >= body["cutoff"] else "allow"
    assert body["decision"] == expected


def test_missing_amount_is_rejected(client):
    r = client.post("/predict", json={"ProductCD": "C"})
    assert r.status_code == 422   # 422 = the request is not valid


def test_text_amount_is_rejected(client):
    r = client.post("/predict", json={"TransactionAmt": "hello"})
    assert r.status_code == 422


def test_unseen_category_does_not_crash(client):
    r = client.post("/predict", json={"TransactionAmt": 50, "P_emaildomain": "never-seen-before.xyz"})
    assert r.status_code == 200


def test_prediction_is_saved_in_history(client):
    first = client.post("/predict", json={"TransactionAmt": 75, "ProductCD": "W"}).json()
    second = client.post("/predict", json={"TransactionAmt": 300, "ProductCD": "C"}).json()

    history = client.get("/history").json()
    assert len(history) == 2
    assert history[0]["id"] == second["id"]   # newest first
    assert history[1]["id"] == first["id"]
    assert history[0]["amount"] == 300
    assert history[0]["product"] == "C"
    assert history[0]["decision"] == second["decision"]
