import pytest

from app import catalog, search, tools
from app.db import connect, init_db
from app.embedders import HashingEmbedder
from app.seed import seed


@pytest.fixture()
def conn():
    search.set_embedder(HashingEmbedder())  # offline, deterministic; also clears the process cache
    c = connect(":memory:")
    init_db(c)
    seed(c)
    yield c
    c.close()
    search.set_embedder(None)


def run(conn, q, **kw):
    return search.search_resources(tools.Context(conn=conn), query=q, **kw)


def titles(r):
    return [h["title"] for h in r["results"]]


# --- relevance & citation ----------------------------------------------------------------------
def test_python_query_ranks_python_books_first(conn):
    r = run(conn, "Find books on Python programming")
    assert r["status"] == "ok"
    assert set(titles(r)[:3]) == {"Python Crash Course", "Automate the Boring Stuff with Python", "Fluent Python"}


def test_results_carry_citation_metadata(conn):
    h = run(conn, "retrieval-augmented generation")["results"][0]
    assert h["title"].startswith("Retrieval-Augmented Generation") and h["source"] == "research_archive"
    assert h["access_url"] and h["matched_terms"] and h["snippet"] and 0 < h["score"] <= 1


def test_papers_filter(conn):
    r = run(conn, "Find papers related to retrieval-augmented generation", types=["paper"])
    assert r["status"] == "ok" and {h["type"] for h in r["results"]} == {"paper"}
    assert titles(r)[0].startswith("Retrieval-Augmented Generation")


def test_level_filter_and_beginner_intent(conn):
    r = run(conn, "data science", level="beginner")
    assert r["status"] == "ok" and {h["level"] for h in r["results"]} == {"beginner"}


def test_generative_ai_beginner_finds_ebook_and_path(conn):
    top2 = set(titles(run(conn, "beginner resources for learning generative AI", top_k=2)))
    assert top2 == {"Generative AI with Large Language Models: A Beginner's Guide", "Generative AI Learning Path (Beginner)"}


# --- abstention --------------------------------------------------------------------------------
@pytest.mark.parametrize("q", ["Quantum Basket Weaving for Beginners", "zzzz qqqq", "medieval falconry techniques"])
def test_unanswerable_returns_not_found(conn, q):
    assert run(conn, q)["status"] == "not_found"


def test_empty_query_invalid(conn):
    assert run(conn, "   ")["status"] == "invalid"


# --- freshness: no code change, no manual reindex ----------------------------------------------
def test_new_resource_becomes_searchable_immediately(conn):
    assert run(conn, "quantum computing")["status"] == "not_found"
    out = catalog.add_resources(conn, [{"type": "ebook", "title": "Quantum Computing for Computer Scientists", "authors": "Yanofsky",
                                        "year": 2008, "abstract": "Introduction to quantum computing and qubits.", "subjects": "quantum computing",
                                        "level": "beginner", "source": "digital_repository", "access_url": "https://digital.library.example/q"}])
    assert len(out["added"]) == 1
    r = run(conn, "quantum computing")
    assert r["status"] == "ok" and titles(r)[0] == "Quantum Computing for Computer Scientists"


def test_updated_metadata_is_reembedded(conn):
    run(conn, "python")  # warm the cache
    rid = conn.execute("SELECT id FROM resources WHERE title='Clean Code'").fetchone()["id"]
    conn.execute("UPDATE resources SET subjects='rust,systems programming', abstract='Learn Rust ownership and borrowing.' WHERE id=?", (rid,))
    conn.commit()
    assert titles(run(conn, "rust ownership"))[0] == "Clean Code"


def test_deleted_resource_leaves_index(conn):
    rid = conn.execute("SELECT id FROM resources WHERE title='Dense Passage Retrieval for Open-Domain Question Answering'").fetchone()["id"]
    assert any("Dense Passage" in t for t in titles(run(conn, "dense passage retrieval")))
    conn.execute("DELETE FROM resources WHERE id=?", (rid,))
    conn.commit()
    after = run(conn, "dense passage retrieval")
    assert after["status"] == "not_found" or not any("Dense Passage" in t for t in titles(after))
    assert conn.execute("SELECT COUNT(*) FROM embeddings WHERE resource_id=?", (rid,)).fetchone()[0] == 0


def test_sync_is_incremental_and_model_change_rebuilds(conn):
    emb = HashingEmbedder()
    first = search.sync_index(conn, emb)
    assert first["embedded"] == first["total"] > 0
    assert search.sync_index(conn, emb)["embedded"] == 0  # nothing changed
    other = HashingEmbedder(dim=512)
    other.name = "hashing-512"
    assert search.sync_index(conn, other)["embedded"] == first["total"]  # different model -> full rebuild


# --- ingestion ---------------------------------------------------------------------------------
def test_catalog_dedup_and_validation(conn):
    out = catalog.add_resources(conn, [
        {"type": "book", "title": "fluent  python!", "year": 2022},                      # dup by normalised title+year
        {"type": "book", "title": "Whatever", "isbn": "978-0-13-461099-3"},            # dup by ISBN (AIMA)
        {"type": "podcast", "title": "Not a type"},
        {"type": "book", "title": "", },
        {"type": "book", "title": "Bad level", "level": "expert"},
    ])
    assert out["added"] == [] and len(out["duplicates"]) == 2 and len(out["rejected"]) == 3


# --- audit & registration ----------------------------------------------------------------------
def test_search_is_audited_and_registered(conn):
    r = run(conn, "python")
    row = conn.execute("SELECT tool,outcome FROM audit_log WHERE trace_id=?", (r["trace_id"],)).fetchone()
    assert row["tool"] == "search_resources" and row["outcome"] == "ok"
    assert "search_resources" in tools.TOOLS
