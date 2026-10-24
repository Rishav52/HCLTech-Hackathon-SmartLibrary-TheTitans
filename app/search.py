"""Semantic search over catalog metadata.

Pipeline: embed query -> cosine similarity against stored resource vectors -> fuse with lexical coverage
-> filter -> abstain below the model's threshold. Results carry the metadata needed to cite them.

Freshness: embeddings live in the DB. On each search a cheap signature of the resources table
(count, max id, max updated_at) is compared with the cached one; any insert/update/delete triggers an
incremental sync that re-embeds only rows whose text changed. No code change or manual re-index required.
"""
import hashlib
import threading

import numpy as np

from .embedders import MODIFIER_STEMS, Embedder, get_embedder, tokens
from .tools import Context, _result, audited

W_DENSE, W_LEX = 0.75, 0.25
_lock = threading.Lock()
_cache: dict = {}          # db key -> dict(sig, model, ids, matrix, rows)
_embedder: Embedder | None = None


def set_embedder(e: Embedder | None) -> None:
    """Override the process-wide embedder (tests, config). None resets to lazy auto-selection."""
    global _embedder
    _embedder = e
    _cache.clear()


def embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def doc_text(r) -> str:
    """The text that represents a resource in embedding space."""
    return (f"{r['title']}. {r['type'].replace('_', ' ')}. {r['level']} level. By {r['authors']}. "
            f"Subjects: {r['subjects'].replace(',', ', ')}. {r['abstract']} Courses: {r['course_tags']}.")


def _db_key(conn) -> str:
    path = conn.execute("PRAGMA database_list").fetchone()["file"]
    return path or f"mem:{id(conn)}"


def _signature(conn) -> tuple:
    r = conn.execute("SELECT COUNT(*) n, COALESCE(MAX(id),0) m, COALESCE(MAX(updated_at),'') u FROM resources").fetchone()
    return (r["n"], r["m"], r["u"])


def sync_index(conn, emb: Embedder | None = None) -> dict:
    """Incrementally bring embeddings in line with resources. Returns counts of (re)embedded/removed rows."""
    emb = emb or embedder()
    rows = conn.execute("SELECT * FROM resources").fetchall()
    have = {r["resource_id"]: r for r in conn.execute("SELECT resource_id,model,text_hash FROM embeddings")}
    todo, texts = [], []
    for r in rows:
        t = doc_text(r)
        h = hashlib.sha256(t.encode()).hexdigest()
        cur = have.get(r["id"])
        if cur is None or cur["model"] != emb.name or cur["text_hash"] != h:
            todo.append((r["id"], h))
            texts.append(t)
    stale = set(have) - {r["id"] for r in rows}
    with conn:
        if texts:
            vecs = emb.encode(texts)
            for (rid, h), v in zip(todo, vecs):
                conn.execute("INSERT OR REPLACE INTO embeddings (resource_id,model,text_hash,dim,vector) VALUES (?,?,?,?,?)",
                             (rid, emb.name, h, len(v), v.astype(np.float32).tobytes()))
        for rid in stale:
            conn.execute("DELETE FROM embeddings WHERE resource_id=?", (rid,))
    return {"embedded": len(todo), "removed": len(stale), "total": len(rows)}


def _load(conn, emb: Embedder) -> dict:
    key, sig = _db_key(conn), _signature(conn)
    with _lock:
        c = _cache.get(key)
        if c and c["sig"] == sig and c["model"] == emb.name:
            return c
        sync_index(conn, emb)
        rows = conn.execute("SELECT r.*, e.vector FROM resources r JOIN embeddings e ON e.resource_id=r.id ORDER BY r.id").fetchall()
        c = {"sig": _signature(conn), "model": emb.name, "rows": rows,
             "matrix": np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1), np.float32),
             "gate": [set(tokens(doc_text(r))) for r in rows],
             "lex": [set(tokens(" ".join([r["title"], r["authors"], r["subjects"].replace(",", " ")]))) for r in rows]}
        _cache[key] = c
        return c


def _snippet(text: str, n: int = 160) -> str:
    return text if len(text) <= n else text[: n - 1].rsplit(" ", 1)[0] + "…"


def score_query(conn, query: str, min_score: float | None = None):
    """Score every resource against a query. Returns (index, fused scores, eligible mask, query tokens, embedder).

    `eligible` applies the abstention rules (score threshold; and, for lexical-only embedders, a topical-word match),
    so search and recommendations abstain identically.
    """
    emb = embedder()
    c = _load(conn, emb)
    q_toks = set(tokens(query))
    topical = q_toks - MODIFIER_STEMS
    dense = c["matrix"] @ emb.encode([query])[0]
    lex = np.array([len(q_toks & d) / len(q_toks) if q_toks else 0.0 for d in c["lex"]])
    fused = W_DENSE * dense + W_LEX * lex
    thr = emb.min_score if min_score is None else min_score
    eligible = np.array([fused[i] >= thr and not (emb.lexical_only and topical and not (topical & c["gate"][i]))
                         for i in range(len(fused))], dtype=bool)
    return c, fused, eligible, q_toks, emb


@audited("search_resources")
def search_resources(ctx: Context, trace_id: str, query: str, top_k: int = 5, types: list[str] | None = None,
                     level: str | None = None, source: str | None = None, year_min: int | None = None) -> dict:
    """Public. Natural-language search across books, journals, papers, e-books, dissertations and learning paths."""
    if not query or not query.strip():
        return _result(ctx, trace_id, "invalid", message="Empty query.")
    if not _load(ctx.conn, embedder())["rows"]:
        return _result(ctx, trace_id, "not_found", query=query, message="The catalog is empty.")
    c, fused, eligible, q_toks, emb = score_query(ctx.conn, query)
    hits = []
    for i in np.argsort(-fused):
        r = c["rows"][i]
        if not eligible[i] or (types and r["type"] not in types) or (level and r["level"] != level) or \
           (source and r["source"] != source) or (year_min and (r["year"] or 0) < year_min):
            continue
        hits.append({
            "resource_id": r["id"], "title": r["title"], "authors": r["authors"], "year": r["year"], "type": r["type"],
            "level": r["level"], "source": r["source"], "access_url": r["access_url"], "score": round(float(fused[i]), 3),
            "matched_terms": sorted(q_toks & c["lex"][i]), "snippet": _snippet(r["abstract"]),
        })
        if len(hits) >= max(1, min(top_k, 20)):
            break
    if not hits:
        return _result(ctx, trace_id, "not_found", query=query, message="No matching resource in authorized sources.")
    return _result(ctx, trace_id, "ok", query=query, retriever=emb.name, results=hits)


from . import tools as _tools  # noqa: E402
_tools.TOOLS[search_resources.tool_name] = search_resources
