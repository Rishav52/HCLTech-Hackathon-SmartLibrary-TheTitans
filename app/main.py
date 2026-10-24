"""Thin HTTP layer over the deterministic tools.

Auth: `X-User-Email` header resolves the caller. This is a prototype stand-in for SSO/JWT; the important
property is that identity is resolved here, server-side, and is never taken from model-generated text.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
config.load_env()

from . import analytics as analytics_mod, assistant as assistant_mod, llm as llm_mod, recommend as recommend_mod, search, tools
from .db import connect, init_db

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app):
    config.ensure_seeded()
    search.embedder()          # load models once at startup, not on the first user request
    get_assistant()
    yield


app = FastAPI(title="Smart Library Assistant API", version="1.0", lifespan=lifespan)
_ASSISTANT: assistant_mod.Assistant | None = None


def get_assistant() -> assistant_mod.Assistant:
    global _ASSISTANT
    if _ASSISTANT is None:
        _ASSISTANT = assistant_mod.Assistant(llm_mod.get_llm())
    return _ASSISTANT


class ChatIn(BaseModel):
    message: str
    session_id: str | None = None


def get_ctx(x_user_email: str | None = Header(default=None)):
    conn = connect()
    init_db(conn)
    user = conn.execute("SELECT * FROM users WHERE email=?", (x_user_email,)).fetchone() if x_user_email else None
    try:
        yield tools.Context(conn=conn, user=user)
    finally:
        conn.close()


@app.get("/tools/search")
def search_resources(q: str, top_k: int = 5, type: list[str] | None = Query(default=None), level: str | None = None,
                     source: str | None = None, year_min: int | None = None, ctx: tools.Context = Depends(get_ctx)):
    return search.search_resources(ctx, query=q, top_k=top_k, types=type, level=level, source=source, year_min=year_min)


@app.get("/tools/recommend")
def recommend(topic: str | None = None, level: str | None = None, top_k: int = 5, available_only: bool = False,
              ctx: tools.Context = Depends(get_ctx)):
    return recommend_mod.recommend(ctx, topic=topic, level=level, top_k=top_k, available_only=available_only)


@app.get("/analytics/{report}")
def analytics(report: str, top_n: int = 10, months: int | None = None, days: int | None = None, min_count: int = 2,
              horizon_days: int = 30, ctx: tools.Context = Depends(get_ctx)):
    """Librarian-only reports: dashboard | waitlists | overdue | circulation | utilization | catalog_gaps | forecast."""
    return analytics_mod.librarian_analytics(ctx, report=report, top_n=top_n, months=months, days=days,
                                             min_count=min_count, horizon_days=horizon_days)


@app.get("/tools/availability")
def availability(title: str | None = None, resource_id: int | None = None, ctx: tools.Context = Depends(get_ctx)):
    return tools.check_availability(ctx, title=title, resource_id=resource_id)


@app.post("/tools/reserve")
def reserve(title: str | None = None, resource_id: int | None = None, ctx: tools.Context = Depends(get_ctx)):
    return tools.reserve(ctx, title=title, resource_id=resource_id)


@app.post("/tools/renew/{loan_id}")
def renew(loan_id: int, ctx: tools.Context = Depends(get_ctx)):
    return tools.renew(ctx, loan_id=loan_id)


@app.get("/tools/loans")
def loans(user_id: int | None = None, due_within_days: int | None = None, ctx: tools.Context = Depends(get_ctx)):
    return tools.my_loans(ctx, user_id=user_id, due_within_days=due_within_days)


@app.post("/tools/return/{loan_id}")
def return_item(loan_id: int, ctx: tools.Context = Depends(get_ctx)):
    return tools.return_copy(ctx, loan_id=loan_id)


@app.post("/chat")
def chat(body: ChatIn, ctx: tools.Context = Depends(get_ctx)):
    """Conversational entry point. Identity comes from X-User-Email, never from the message body."""
    session = assistant_mod.get_session(body.session_id, ctx.user)
    return get_assistant().chat(ctx, session, body.message)


@app.get("/health")
def health():
    """Which backends are active (the UI shows this so a fallback is never mistaken for the real model)."""
    emb = search.embedder()
    return {"status": "ok", "llm": get_assistant().llm.name, "embedder": emb.name,
            "retrieval": "lexical fallback" if emb.lexical_only else "semantic"}


@app.get("/demo/personas")
def personas():
    """Demo sign-in list for the prototype UI. Disable with DEMO_PERSONAS=0; production must use real SSO."""
    if os.environ.get("DEMO_PERSONAS", "1") != "1":
        raise HTTPException(404)
    conn = connect()
    try:
        return [dict(r) for r in conn.execute("SELECT name, email, role, program FROM users ORDER BY role='librarian', id")]
    finally:
        conn.close()


app.mount("/static", StaticFiles(directory=STATIC, check_dir=False), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
