import time

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1000")
    import research_langgraph.api.security as sec

    monkeypatch.setattr(sec, "RATE_LIMIT_PER_MIN", 1000)
    sec.reset_rate_limits()
    from research_langgraph.api.main import app

    with TestClient(app) as c:
        yield c


H = {"X-API-Key": "test-key"}


def _wait(client, run_id, pred, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/research/{run_id}", headers=H).json()
        if pred(body):
            return body
        time.sleep(0.2)
    raise AssertionError(f"timed out; last={body}")


def test_requires_api_key(client):
    assert client.post("/research", json={"topic": "small language models"}).status_code == 401
    assert client.get("/research/x", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/health").status_code == 200


def test_input_validation(client):
    assert client.post("/research", headers=H, json={"topic": "ab"}).status_code == 422
    assert client.post("/research", headers=H,
                       json={"topic": "valid topic", "budget_mode": "bogus"}).status_code == 422


def test_background_run_lifecycle_and_events(client):
    r = client.post("/research", headers=H, json={"topic": "small language models", "budget_mode": "fast"})
    assert r.status_code == 202 and r.json()["status"] == "queued"
    run_id = r.json()["run_id"]

    body = _wait(client, run_id, lambda b: b["awaiting_approval"])
    assert body["plan"]["dimensions"]

    # approving twice quickly must not start two executions
    a1 = client.post(f"/research/{run_id}/approve-plan", headers=H, json={"approved": True})
    assert a1.status_code == 202
    a2 = client.post(f"/research/{run_id}/approve-plan", headers=H, json={"approved": True})
    assert a2.status_code == 409

    done = _wait(client, run_id, lambda b: b["status"] in ("done", "failed"), timeout=120)
    assert done["status"] == "done" and done["final_report"]

    stream = client.get(f"/research/{run_id}/stream", headers=H)
    assert stream.status_code == 200
    text = stream.text
    assert '"type": "node_started"' in text and '"type": "node_finished"' in text and '"type": "done"' in text
    assert '"phase": "start"' in text and '"phase": "end"' in text and "event:" not in text


def test_unknown_run_is_404_everywhere(client):
    assert client.get("/research/nope", headers=H).status_code == 404
    assert client.get("/research/nope/stream", headers=H).status_code == 404
    assert client.post("/research/nope/approve-plan", headers=H, json={}).status_code == 404


def test_rate_limit(client, monkeypatch):
    import research_langgraph.api.security as sec

    monkeypatch.setattr(sec, "RATE_LIMIT_PER_MIN", 2)
    sec.reset_rate_limits()
    codes = [client.post("/research", headers=H, json={"topic": "rate limit topic"}).status_code
             for _ in range(4)]
    assert codes[:2] == [202, 202] and 429 in codes[2:]
