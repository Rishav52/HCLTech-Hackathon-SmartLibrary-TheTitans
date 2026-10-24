"""Deterministic library tools: the only code path allowed to read or change circulation state.

The LLM layer (Stage 4) may call these but never computes availability, fines or reservations itself.
Every call returns {"status", "trace_id", ...} and writes one audit_log row.

status: ok | not_found | ambiguous | denied | invalid | not_applicable
"""
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import wraps

# --- Circulation policy (single source of truth) -------------------------------------------------
LOAN_DAYS = 21
RENEWAL_DAYS = 14
MAX_RENEWALS = 2
FINE_PER_DAY = 2.00        # INR per overdue day, per item
CURRENCY = "INR"


@dataclass
class Context:
    conn: sqlite3.Connection
    user: sqlite3.Row | None = None          # authenticated user, resolved by the API layer, never by the model
    today: date = field(default_factory=date.today)

    @property
    def is_librarian(self) -> bool:
        return self.user is not None and self.user["role"] == "librarian"


def _result(ctx: Context, trace_id: str, status: str, **data) -> dict:
    return {"status": status, "trace_id": trace_id, **data}


def audited(tool_name: str):
    """Wrap a tool: assign trace_id, time it, record outcome (and errors) in audit_log."""
    def deco(fn):
        @wraps(fn)
        def wrapper(ctx: Context, **params):
            trace_id, t0 = uuid.uuid4().hex[:12], time.perf_counter()
            try:
                res = fn(ctx, trace_id, **params)
                outcome, detail = res["status"], res.get("message")
            except Exception as exc:  # noqa: BLE001 - tools must never leak a stack trace to the model
                res = {"status": "error", "trace_id": trace_id, "message": "Internal error; please try again."}
                outcome, detail = "error", f"{type(exc).__name__}: {exc}"
            ctx.conn.execute(
                "INSERT INTO audit_log (trace_id,user_id,tool,params,outcome,latency_ms,detail) VALUES (?,?,?,?,?,?,?)",
                (trace_id, ctx.user["id"] if ctx.user else None, tool_name, json.dumps(params, default=str),
                 outcome, round((time.perf_counter() - t0) * 1000, 2), detail))
            ctx.conn.commit()
            return res
        wrapper.tool_name = tool_name
        return wrapper
    return deco


# --- helpers -------------------------------------------------------------------------------------
def _resolve_resource(conn, resource_id=None, title=None):
    """Return (row, candidates). Exact title wins; otherwise a unique substring match; else ambiguous/none."""
    if resource_id is not None:
        return conn.execute("SELECT * FROM resources WHERE id=?", (resource_id,)).fetchone(), []
    if not title or not title.strip():
        return None, []
    t = title.strip()
    exact = conn.execute("SELECT * FROM resources WHERE lower(title)=lower(?)", (t,)).fetchall()
    if len(exact) == 1:
        return exact[0], []
    like = exact or conn.execute("SELECT * FROM resources WHERE lower(title) LIKE lower(?) ORDER BY title",
                                 (f"%{t}%",)).fetchall()
    if len(like) == 1:
        return like[0], []
    return None, [{"resource_id": r["id"], "title": r["title"]} for r in like]


def _queue(conn, resource_id):
    return conn.execute("SELECT * FROM reservations WHERE resource_id=? AND status IN ('active','ready') ORDER BY id",
                        (resource_id,)).fetchall()


def _fine(due_at: str, today: date, returned_at: str | None = None) -> tuple[int, float]:
    end = date.fromisoformat(returned_at) if returned_at else today
    days = max((end - date.fromisoformat(due_at)).days, 0)
    return days, round(days * FINE_PER_DAY, 2)


def _loan_view(row, today: date) -> dict:
    days_over, fine = _fine(row["due_at"], today)
    days_left = (date.fromisoformat(row["due_at"]) - today).days
    return {
        "loan_id": row["id"], "title": row["title"], "branch": row["branch"], "borrowed_at": row["borrowed_at"],
        "due_at": row["due_at"], "days_left": days_left, "overdue": days_over > 0, "days_overdue": days_over,
        "fine": fine, "renewals_used": row["renewals"], "renewals_left": max(MAX_RENEWALS - row["renewals"], 0),
    }


# --- tools ---------------------------------------------------------------------------------------
@audited("check_availability")
def check_availability(ctx: Context, trace_id: str, resource_id: int | None = None, title: str | None = None) -> dict:
    """Public: current holdings/status for a resource. Never guesses; unknown titles return not_found."""
    res, candidates = _resolve_resource(ctx.conn, resource_id, title)
    if res is None and candidates:
        return _result(ctx, trace_id, "ambiguous", candidates=candidates,
                       message="Several titles match; please pick one.")
    if res is None:
        return _result(ctx, trace_id, "not_found", message="No matching resource in authorized sources.")
    base = {"resource_id": res["id"], "title": res["title"], "type": res["type"], "source": res["source"]}
    if res["access_url"]:  # digital: no circulation
        return _result(ctx, trace_id, "ok", **base, format="digital", available=True, access_url=res["access_url"])
    copies = ctx.conn.execute("SELECT id,branch,shelf_location,status FROM copies WHERE resource_id=? ORDER BY id",
                              (res["id"],)).fetchall()
    avail = [c for c in copies if c["status"] == "available"]
    return _result(ctx, trace_id, "ok", **base, format="physical", available=bool(avail),
                   copies_total=len(copies), copies_available=len(avail), waitlist_length=len(_queue(ctx.conn, res["id"])),
                   locations=[{"copy_id": c["id"], "branch": c["branch"], "shelf": c["shelf_location"]} for c in avail])


@audited("reserve")
def reserve(ctx: Context, trace_id: str, resource_id: int | None = None, title: str | None = None) -> dict:
    """Authenticated users only. Holds a free copy ('ready') or joins the waitlist ('active')."""
    if ctx.user is None:
        return _result(ctx, trace_id, "denied", message="Authentication required.")
    res, candidates = _resolve_resource(ctx.conn, resource_id, title)
    if res is None and candidates:
        return _result(ctx, trace_id, "ambiguous", candidates=candidates, message="Several titles match; please pick one.")
    if res is None:
        return _result(ctx, trace_id, "not_found", message="No matching resource in authorized sources.")
    if res["access_url"]:
        return _result(ctx, trace_id, "not_applicable", title=res["title"], access_url=res["access_url"],
                       message="Digital resource: no reservation needed.")
    uid = ctx.user["id"]
    if ctx.conn.execute("SELECT 1 FROM reservations WHERE resource_id=? AND user_id=? AND status IN ('active','ready')",
                        (res["id"], uid)).fetchone():
        return _result(ctx, trace_id, "invalid", title=res["title"], message="You already have an active reservation for this title.")
    if ctx.conn.execute("SELECT 1 FROM loans l JOIN copies c ON c.id=l.copy_id WHERE c.resource_id=? AND l.user_id=? AND l.returned_at IS NULL",
                        (res["id"], uid)).fetchone():
        return _result(ctx, trace_id, "invalid", title=res["title"], message="You currently have this title on loan.")

    with ctx.conn:
        # A copy is free only if it is 'available'; held copies belong to earlier waitlisters.
        copy = None
        if not _queue(ctx.conn, res["id"]):
            copy = ctx.conn.execute("SELECT id,branch,shelf_location FROM copies WHERE resource_id=? AND status='available' ORDER BY id LIMIT 1",
                                    (res["id"],)).fetchone()
        status = "ready" if copy else "active"
        cur = ctx.conn.execute("INSERT INTO reservations (resource_id,user_id,created_at,status) VALUES (?,?,?,?)",
                               (res["id"], uid, ctx.today.isoformat(), status))
        if copy:
            ctx.conn.execute("UPDATE copies SET status='held' WHERE id=?", (copy["id"],))
    position = len(_queue(ctx.conn, res["id"]))
    out = dict(reservation_id=cur.lastrowid, title=res["title"], reservation_status=status, queue_position=position)
    if copy:
        out.update(branch=copy["branch"], shelf=copy["shelf_location"], message="Copy held for you at the listed branch.")
    else:
        out["message"] = f"All copies are out; you are number {position} on the waitlist."
    return _result(ctx, trace_id, "ok", **out)


@audited("renew")
def renew(ctx: Context, trace_id: str, loan_id: int) -> dict:
    """Owner only. Blocked when overdue, at the renewal cap, or when someone is waiting."""
    if ctx.user is None:
        return _result(ctx, trace_id, "denied", message="Authentication required.")
    loan = ctx.conn.execute("SELECT l.*, c.resource_id, r.title FROM loans l JOIN copies c ON c.id=l.copy_id "
                            "JOIN resources r ON r.id=c.resource_id WHERE l.id=?", (loan_id,)).fetchone()
    # Same response for "missing" and "someone else's" so loan ids cannot be probed.
    if loan is None or loan["user_id"] != ctx.user["id"]:
        return _result(ctx, trace_id, "not_found", message="No such loan on your account.")
    if loan["returned_at"]:
        return _result(ctx, trace_id, "invalid", title=loan["title"], message="This item has already been returned.")
    days_over, _ = _fine(loan["due_at"], ctx.today)
    reason = None
    if days_over > 0:
        reason = "Item is overdue; return it before renewing."
    elif loan["renewals"] >= MAX_RENEWALS:
        reason = f"Renewal limit ({MAX_RENEWALS}) reached."
    elif _queue(ctx.conn, loan["resource_id"]):
        reason = "Another reader is waiting for this title."
    if reason:
        return _result(ctx, trace_id, "invalid", title=loan["title"], due_at=loan["due_at"], message=reason)
    new_due = (date.fromisoformat(loan["due_at"]) + timedelta(days=RENEWAL_DAYS)).isoformat()
    with ctx.conn:
        ctx.conn.execute("UPDATE loans SET due_at=?, renewals=renewals+1 WHERE id=?", (new_due, loan_id))
    return _result(ctx, trace_id, "ok", loan_id=loan_id, title=loan["title"], old_due_at=loan["due_at"], new_due_at=new_due,
                   renewals_left=MAX_RENEWALS - loan["renewals"] - 1)


@audited("my_loans")
def my_loans(ctx: Context, trace_id: str, user_id: int | None = None, due_within_days: int | None = None) -> dict:
    """Open loans with due dates, renewability and fines. Users see only their own; librarians may name a user."""
    if ctx.user is None:
        return _result(ctx, trace_id, "denied", message="Authentication required.")
    target = user_id if user_id is not None else ctx.user["id"]
    if target != ctx.user["id"] and not ctx.is_librarian:
        return _result(ctx, trace_id, "denied", message="You can only view your own borrowing data.")
    rows = ctx.conn.execute("SELECT l.*, c.branch, c.resource_id, r.title FROM loans l JOIN copies c ON c.id=l.copy_id "
                            "JOIN resources r ON r.id=c.resource_id WHERE l.user_id=? AND l.returned_at IS NULL ORDER BY l.due_at",
                            (target,)).fetchall()
    loans = []
    for r in rows:
        v = _loan_view(r, ctx.today)
        v["renewable"] = not v["overdue"] and v["renewals_left"] > 0 and not _queue(ctx.conn, r["resource_id"])
        loans.append(v)
    if due_within_days is not None:
        loans = [l for l in loans if l["days_left"] <= due_within_days]  # includes overdue
    reservations = ctx.conn.execute("SELECT r.title, res.status FROM reservations res JOIN resources r ON r.id=res.resource_id "
                                    "WHERE res.user_id=? AND res.status IN ('active','ready')", (target,)).fetchall()
    return _result(ctx, trace_id, "ok", user_id=target, loans=loans, total_fine=round(sum(l["fine"] for l in loans), 2),
                   currency=CURRENCY, reservations=[dict(r) for r in reservations])


@audited("return_copy")
def return_copy(ctx: Context, trace_id: str, loan_id: int) -> dict:
    """Librarian only. Closes a loan, assesses any fine and promotes the first waitlister to 'ready'."""
    if not ctx.is_librarian:
        return _result(ctx, trace_id, "denied", message="Librarian role required.")
    loan = ctx.conn.execute("SELECT l.*, c.resource_id FROM loans l JOIN copies c ON c.id=l.copy_id WHERE l.id=?", (loan_id,)).fetchone()
    if loan is None or loan["returned_at"]:
        return _result(ctx, trace_id, "not_found", message="No open loan with that id.")
    days_over, fine = _fine(loan["due_at"], ctx.today)
    with ctx.conn:
        ctx.conn.execute("UPDATE loans SET returned_at=? WHERE id=?", (ctx.today.isoformat(), loan_id))
        nxt = ctx.conn.execute("SELECT id FROM reservations WHERE resource_id=? AND status='active' ORDER BY id LIMIT 1",
                               (loan["resource_id"],)).fetchone()
        if nxt:
            ctx.conn.execute("UPDATE reservations SET status='ready' WHERE id=?", (nxt["id"],))
            ctx.conn.execute("UPDATE copies SET status='held' WHERE id=?", (loan["copy_id"],))
        else:
            ctx.conn.execute("UPDATE copies SET status='available' WHERE id=?", (loan["copy_id"],))
    return _result(ctx, trace_id, "ok", loan_id=loan_id, days_overdue=days_over, fine=fine, currency=CURRENCY,
                   promoted_reservation_id=nxt["id"] if nxt else None)


TOOLS = {f.tool_name: f for f in (check_availability, reserve, renew, my_loans, return_copy)}
# search_resources lives in app/search.py and registers itself below (avoids a circular import).
