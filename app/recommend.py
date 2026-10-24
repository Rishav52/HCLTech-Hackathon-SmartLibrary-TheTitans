"""Explainable, privacy-aware recommendations.

Every recommendation is a weighted sum of named signals, and each signal that contributed produces a
human-readable reason, so "why this?" is answered from the actual computation (never invented by the LLM).

Signals (weight): requested topic (0.45) | course match (0.25) | stated interests (0.15) |
                  borrowing history (0.20, ONLY with consent) | popularity (0.05) | available now (0.05)

Privacy: identity comes from the Context; history is read only when users.history_consent = 1; popularity is an
aggregate and is shown only when >= MIN_POPULARITY_LOANS so it can't point at an individual.
"""
import re
from collections import Counter
from datetime import timedelta

import numpy as np

from . import search
from .embedders import tokens
from .tools import Context, _result, audited

W = {"topic": 0.45, "course": 0.25, "interest": 0.15, "history": 0.20, "popularity": 0.05, "availability": 0.05}
MIN_POPULARITY_LOANS = 5
POPULARITY_WINDOW_DAYS = 365


def _phrases(csv: str) -> list[str]:
    return [p.strip() for p in (csv or "").split(",") if p.strip()]


def _availability(conn) -> dict[int, dict]:
    out = {}
    for r in conn.execute("SELECT resource_id, COUNT(*) total, SUM(status='available') free FROM copies GROUP BY resource_id"):
        out[r["resource_id"]] = {"total": r["total"], "free": r["free"] or 0}
    for r in conn.execute("SELECT resource_id, COUNT(*) n FROM reservations WHERE status IN ('active','ready') GROUP BY resource_id"):
        out.setdefault(r["resource_id"], {"total": 0, "free": 0})["waiting"] = r["n"]
    return out


@audited("recommend")
def recommend(ctx: Context, trace_id: str, topic: str | None = None, level: str | None = None, top_k: int = 5,
              available_only: bool = False) -> dict:
    """Public with a topic; personalised when the caller is signed in. Returns ranked resources with reasons."""
    topic = (topic or "").strip() or None
    user = ctx.user
    if user is None and topic is None:
        return _result(ctx, trace_id, "invalid", message="Tell me a topic, or sign in so I can use your courses and interests.")

    emb = search.embedder()
    c = search._load(ctx.conn, emb)
    rows = c["rows"]
    if not rows:
        return _result(ctx, trace_id, "not_found", message="The catalog is empty.")
    n = len(rows)

    # ---- signal: topic (shares scoring + abstention with search) ----
    topic_score, topic_terms, topic_ok = np.zeros(n), [set() for _ in range(n)], np.ones(n, dtype=bool)
    if topic:
        _, fused, eligible, q_toks, _ = search.score_query(ctx.conn, topic)
        topic_score, topic_ok = np.clip(fused, 0, 1), eligible
        topic_terms = [q_toks & c["lex"][i] for i in range(n)]

    # ---- signals: user context ----
    used, courses, interests = [], set(), []
    exclude: set[int] = set()
    history_titles: dict[int, str] = {}
    history_subjects: dict[int, set] = {}
    hist_profile: Counter = Counter()
    history_used = False
    if user is not None:
        courses = set(_phrases(user["courses"]))
        interests = _phrases(user["interests"])
        if courses:
            used.append("courses")
        if interests:
            used.append("interests")
        # What the user has out or has queued right now is account state, not "history": always excluded.
        exclude |= {r[0] for r in ctx.conn.execute(
            "SELECT c.resource_id FROM loans l JOIN copies c ON c.id=l.copy_id WHERE l.user_id=? AND l.returned_at IS NULL", (user["id"],))}
        exclude |= {r[0] for r in ctx.conn.execute(
            "SELECT resource_id FROM reservations WHERE user_id=? AND status IN ('active','ready')", (user["id"],))}
        if user["history_consent"]:
            history_used = True
            used.append("borrowing_history")
            for r in ctx.conn.execute(
                    "SELECT c.resource_id, r.title, r.subjects FROM loans l JOIN copies c ON c.id=l.copy_id "
                    "JOIN resources r ON r.id=c.resource_id WHERE l.user_id=? ORDER BY l.borrowed_at", (user["id"],)):
                exclude.add(r["resource_id"])
                history_titles[r["resource_id"]] = r["title"]
                history_subjects[r["resource_id"]] = set(_phrases(r["subjects"]))
                hist_profile.update(_phrases(r["subjects"]))

    # ---- signal: popularity (aggregate only) + availability ----
    since = (ctx.today - timedelta(days=POPULARITY_WINDOW_DAYS)).isoformat()
    pop = {r["resource_id"]: r["n"] for r in ctx.conn.execute(
        "SELECT c.resource_id, COUNT(*) n FROM loans l JOIN copies c ON c.id=l.copy_id WHERE l.borrowed_at >= ? GROUP BY c.resource_id", (since,))}
    max_pop = max(pop.values(), default=0) or 1
    avail = _availability(ctx.conn)

    personal_cols = ("course", "interest", "history")
    scored = []
    for i, r in enumerate(rows):
        rid = r["id"]
        if rid in exclude or (level and r["level"] != level) or (topic and not topic_ok[i]):
            continue
        a = avail.get(rid)
        if r["access_url"]:
            state, is_free = "digital", True
        else:
            free = a["free"] if a else 0
            state, is_free = ("available" if free else "waitlist"), bool(free)
        if available_only and not is_free:
            continue

        s, reasons = {}, []
        if topic:
            s["topic"] = float(topic_score[i])
            reasons.append((W["topic"] * s["topic"], f"Matches your request \"{topic}\" (terms: {', '.join(sorted(topic_terms[i])) or 'semantic match'})"))
        overlap = sorted(courses & set(_phrases(r["course_tags"])))
        if overlap:
            s["course"] = 1.0
            reasons.append((W["course"], f"Tagged for your course{'s' if len(overlap) > 1 else ''} {', '.join(overlap)}"))
        gate = c["gate"][i]
        hit = next((p for p in interests if (set(tokens(p)) and set(tokens(p)) <= gate)), None)
        if hit:
            s["interest"] = 1.0
            reasons.append((W["interest"], f"Matches your interest in {hit}"))
        if history_used:
            shared = [p for p in _phrases(r["subjects"]) if hist_profile[p]]
            if shared:
                s["history"] = min(1.0, len(shared) / 2)
                ex = next((t for rid2, t in reversed(list(history_titles.items())) if set(shared) & history_subjects[rid2]), None)
                reasons.append((W["history"] * s["history"],
                                f"Similar subjects ({', '.join(shared[:3])}) to items you've borrowed" + (f", e.g. {ex}" if ex else "")))
        n_loans = pop.get(rid, 0)
        s["popularity"] = n_loans / max_pop
        if n_loans >= MIN_POPULARITY_LOANS and s["popularity"] >= 0.5:
            reasons.append((W["popularity"] * s["popularity"], f"Popular with readers: {n_loans} loans in the past year"))
        s["availability"] = 1.0 if is_free else 0.0
        if is_free:
            reasons.append((W["availability"], "Digital: open immediately" if state == "digital" else "A copy is available now"))

        relevance = sum(s.get(k, 0) for k in ("topic", *personal_cols))
        if relevance == 0 and (topic or user is not None and (courses or interests or history_used)):
            continue  # nothing ties this item to the request or the user: no filler
        score = sum(W[k] * v for k, v in s.items())
        if not reasons:
            continue
        reasons.sort(key=lambda x: -x[0])
        scored.append((score, i, state, [t for _, t in reasons], s, a))

    if not scored:
        return _result(ctx, trace_id, "not_found", message="No recommendations match in authorized sources.",
                       personalization={"used": used, "history_used": history_used})
    scored.sort(key=lambda x: (-x[0], rows[x[1]]["title"]))
    results = []
    for score, i, state, reasons, s, a in scored[:max(1, min(top_k, 10))]:
        r = rows[i]
        results.append({
            "resource_id": r["id"], "title": r["title"], "authors": r["authors"], "year": r["year"], "type": r["type"],
            "level": r["level"], "source": r["source"], "access_url": r["access_url"], "snippet": search._snippet(r["abstract"]),
            "score": round(score, 3), "availability": state, "waitlist": (a or {}).get("waiting", 0) if state == "waitlist" else 0,
            "reasons": reasons, "signals": {k: round(v, 2) for k, v in s.items() if v},
        })
    note = None
    if user is not None and not history_used:
        note = "Borrowing history was not used (no consent on file)."
    elif user is None:
        note = "Not personalised: sign in to use your courses and interests."
    return _result(ctx, trace_id, "ok", topic=topic, results=results, personalization={"used": used, "history_used": history_used, "note": note})


from . import tools as _tools  # noqa: E402
_tools.TOOLS[recommend.tool_name] = recommend
