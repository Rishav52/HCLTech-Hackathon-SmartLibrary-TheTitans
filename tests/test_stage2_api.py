import pytest
from fastapi.testclient import TestClient

from app import db as dbmod
from app.seed import seed


@pytest.fixture()
def client(tmp_path, monkeypatch):
    path = str(tmp_path / "lib.db")
    monkeypatch.setattr(dbmod, "DB_PATH", path)
    c = dbmod.connect(path)
    dbmod.init_db(c)
    seed(c)
    c.close()
    from app.main import app
    return TestClient(app)


def test_availability_public(client):
    r = client.get("/tools/availability", params={"title": "Fluent Python"}).json()
    assert r["status"] == "ok" and r["trace_id"]


def test_loans_needs_identity_header(client):
    assert client.get("/tools/loans").json()["status"] == "denied"
    r = client.get("/tools/loans", headers={"X-User-Email": "aarav@univ.example"}).json()
    assert r["status"] == "ok" and len(r["loans"]) == 3


def test_unknown_email_is_unauthenticated(client):
    assert client.get("/tools/loans", headers={"X-User-Email": "nobody@evil.example"}).json()["status"] == "denied"


def test_search_endpoint(client):
    from app import search
    from app.embedders import HashingEmbedder
    search.set_embedder(HashingEmbedder())
    r = client.get("/tools/search", params={"q": "python programming", "type": ["book"], "top_k": 3}).json()
    assert r["status"] == "ok" and len(r["results"]) == 3 and all(h["type"] == "book" for h in r["results"])
    assert client.get("/tools/search", params={"q": "medieval falconry techniques"}).json()["status"] == "not_found"
    search.set_embedder(None)


def test_recommend_endpoint(client):
    from app import search
    from app.embedders import HashingEmbedder
    search.set_embedder(HashingEmbedder())
    anon = client.get("/tools/recommend").json()
    assert anon["status"] == "invalid"
    r = client.get("/tools/recommend", params={"topic": "data science", "level": "beginner"}, headers={"X-User-Email": "aarav@univ.example"}).json()
    assert r["status"] == "ok" and r["personalization"]["history_used"] is True and r["results"][0]["reasons"]
    search.set_embedder(None)
