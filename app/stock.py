"""Librarian stock management: add a new title (with its copies) and add copies to an existing title.

Both are role-gated here in the tool layer, validated, and audited like every other tool. They are deliberately
NOT exposed to the chat model: adding stock needs structured, exact fields (ISBN, shelf, branch), so it goes through the
form/API instead of free text. New titles are searchable on the very next query (see search.py), and copies added to a
title that has a waitlist are handed to the first waiting readers (status 'ready') so the queue stays fair.
"""
from . import catalog
from .tools import Context, _resolve_resource, _result, audited

MAX_COPIES = 50


def _int(v, default=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None if v not in (None, "") else default


@audited("add_resource")
def add_resource(ctx: Context, trace_id: str, title: str = "", type: str = "book", authors: str = "", year=None, isbn: str | None = None,
                 abstract: str = "", subjects: str = "", level: str = "intermediate", course_tags: str = "", source: str | None = None,
                 access_url: str | None = None, copies=0, branch: str | None = None, shelf: str | None = None) -> dict:
    """Librarian only. Physical items need >= 1 copy (branch + shelf); digital items need an http(s) link and no copies."""
    if not ctx.is_librarian:
        return _result(ctx, trace_id, "denied", message="Librarian role required.")
    n = _int(copies)
    if n is None or not 0 <= n <= MAX_COPIES:
        return _result(ctx, trace_id, "invalid", message=f"copies must be a number from 0 to {MAX_COPIES}.")
    digital = bool((access_url or "").strip())
    if digital and n:
        return _result(ctx, trace_id, "invalid", message="Digital resources have no physical copies; leave copies at 0.")
    if not digital and n < 1:
        return _result(ctx, trace_id, "invalid", message="Physical items need at least one copy, with a branch and shelf location.")
    if n and not ((branch or "").strip() and (shelf or "").strip()):
        return _result(ctx, trace_id, "invalid", message="Give the branch and shelf location for the new copies.")
    item = {"title": title, "type": type, "authors": authors or "", "year": year, "isbn": isbn, "abstract": abstract or "",
            "subjects": subjects or "", "level": level, "course_tags": course_tags or "", "access_url": access_url,
            "source": source or ("digital_repository" if digital else "catalog"),
            "copies": [{"branch": branch, "shelf": shelf}] * n}
    out = catalog.add_resources(ctx.conn, [item])
    if out["rejected"]:
        return _result(ctx, trace_id, "invalid", message=out["rejected"][0]["reason"])
    if out["duplicates"]:
        d = out["duplicates"][0]
        return _result(ctx, trace_id, "duplicate", existing_id=d["existing_id"],
                       message="This title is already in the catalog (matched by ISBN or title and year). Use add copies to increase its stock.")
    rid = out["added"][0]
    return _result(ctx, trace_id, "ok", resource_id=rid, title=title.strip(), copies_added=n, format="digital" if digital else "physical",
                   searchable=True, message="Added to the catalog; it is searchable immediately.")


@audited("add_copies")
def add_copies(ctx: Context, trace_id: str, resource_id: int | None = None, title: str | None = None, branch: str | None = None,
               shelf: str | None = None, count=1) -> dict:
    """Librarian only. Adds physical copies; each new copy goes to the oldest waiting reader first."""
    if not ctx.is_librarian:
        return _result(ctx, trace_id, "denied", message="Librarian role required.")
    n = _int(count)
    if n is None or not 1 <= n <= MAX_COPIES:
        return _result(ctx, trace_id, "invalid", message=f"count must be a number from 1 to {MAX_COPIES}.")
    if not ((branch or "").strip() and (shelf or "").strip()) or len(branch) > 60 or len(shelf) > 60:
        return _result(ctx, trace_id, "invalid", message="Give the branch and shelf location (max 60 characters each).")
    res, candidates = _resolve_resource(ctx.conn, resource_id, title)
    if res is None and candidates:
        return _result(ctx, trace_id, "ambiguous", candidates=candidates, message="Several titles match; pick one.")
    if res is None:
        return _result(ctx, trace_id, "not_found", message="No such resource in the catalog.")
    if res["access_url"]:
        return _result(ctx, trace_id, "invalid", title=res["title"], message="Digital resources have no physical copies.")
    promoted = 0
    with ctx.conn:
        for _ in range(n):
            cid = ctx.conn.execute("INSERT INTO copies (resource_id,branch,shelf_location) VALUES (?,?,?)",
                                   (res["id"], branch.strip(), shelf.strip())).lastrowid
            nxt = ctx.conn.execute("SELECT id FROM reservations WHERE resource_id=? AND status='active' ORDER BY id LIMIT 1",
                                   (res["id"],)).fetchone()
            if nxt:
                ctx.conn.execute("UPDATE reservations SET status='ready' WHERE id=?", (nxt["id"],))
                ctx.conn.execute("UPDATE copies SET status='held' WHERE id=?", (cid,))
                promoted += 1
    free = ctx.conn.execute("SELECT COUNT(*) FROM copies WHERE resource_id=? AND status='available'", (res["id"],)).fetchone()[0]
    total = ctx.conn.execute("SELECT COUNT(*) FROM copies WHERE resource_id=?", (res["id"],)).fetchone()[0]
    return _result(ctx, trace_id, "ok", resource_id=res["id"], title=res["title"], copies_added=n, copies_total=total, copies_available=free,
                   readers_notified=promoted, message=f"Added {n} cop{'y' if n == 1 else 'ies'}" +
                   (f"; {promoted} waiting reader{'s' if promoted != 1 else ''} now have a copy held." if promoted else "."))


def branches(conn) -> list[str]:
    known = [r[0] for r in conn.execute("SELECT DISTINCT branch FROM copies ORDER BY branch")]
    return known or ["Central Library"]


from . import tools as _tools  # noqa: E402
_tools.TOOLS[add_resource.tool_name] = add_resource
_tools.TOOLS[add_copies.tool_name] = add_copies