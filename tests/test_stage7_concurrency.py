"""Regression: a real server hops worker threads between a request's dependency setup and its endpoint.
TestClient does not reproduce that, so these tests run an actual uvicorn server and hit it in parallel."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
import uvicorn

from app import db as dbmod, search
from app.embedders import HashingEmbedder
from app.seed import seed


@pytest.fixture()
def server(tmp_path, monkeypatch):
    path = str(tmp_path / "lib.db")
    monkeypatch.setattr(dbmod, "DB_PATH", path)
    search.set_embedder(HashingEmbedder())
    c = dbmod.connect(path)
    dbmod.init_db(c)
    seed(c)
    c.close()
    from app.main import app
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started, "server did not start"
    yield f"http://127.0.0.1:{srv.servers[0].sockets[0].getsockname()[1]}"
    srv.should_exit = True
    t.join(timeout=5)
    search.set_embedder(None)


def test_parallel_analytics_requests(server):
    hdr = {"X-User-Email": "lena@univ.example"}
    urls = [f"{server}/analytics/{r}" for r in ("dashboard", "waitlists", "catalog_gaps", "forecast", "circulation", "overdue", "utilization")] * 6
    with httpx.Client(timeout=30) as c, ThreadPoolExecutor(10) as ex:
        res = list(ex.map(lambda u: c.get(u, headers=hdr), urls))
    assert [r.status_code for r in res] == [200] * len(urls)
    assert all(r.json()["status"] == "ok" for r in res)


def test_parallel_chat_with_audit_writes(server):
    hdr = {"X-User-Email": "diya@univ.example"}
    with httpx.Client(timeout=30) as c, ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda i: c.post(f"{server}/chat", json={"message": "Find books on Python programming"}, headers=hdr), range(24)))
    assert all(r.status_code == 200 and r.json()["grounded"] for r in res)
