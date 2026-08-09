from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_health_contract():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "time" in body


def test_portfolio_contract():
    response = client.get("/portfolio")
    assert response.status_code == 200
    body = response.json()
    assert body["starting_equity"] == 10_000.0
    assert body["equity"] == 10_000.0
    assert body["open_positions"] == 0
    assert body["trade_log"] == []


def test_empty_signals_request_is_rejected():
    response = client.post("/signals", json={"ohlcv": {}})
    assert response.status_code == 400
    assert response.json()["detail"] == "No OHLCV data supplied"
