"""Librarian analytics: demand, waitlists, overdue trends, utilization, catalog gaps and a demand forecast.

Everything is computed by SQL/arithmetic over the authoritative tables (never by the LLM) and is librarian-only.
Output is aggregate by design: where individuals appear at all (overdue follow-up) it is by opaque user_id.
Each report returns plain lists/numbers so a dashboard (Stage 7) or the assistant can render it.
"""
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date, timedelta

from .embedders import tokens
from .tools import CURRENCY, FINE_PER_DAY, LOAN_DAYS, Context, _result, audited

WAITLIST_PER_COPY_ALERT = 1.0     # waiting readers per copy at/above which we recommend more copies
ES_ALPHA = 0.4                    # exponential-smoothing weight on the latest month
BACKTEST_MONTHS = 3
REPORTS = ("dashboard", "waitlists", "overdue", "circulation", "utilization", "catalog_gaps", "forecast")


# ------------------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------------------
def _months(today: date, n: int, include_current: bool = True) -> list[str]:
    y, m = today.year, today.month
    if not include_current:
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    out = []
    for _ in range(n):
        out.append(f"{y:04d}-{m:02d}")
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return out[::-1]


def _days(a: str, b: date) -> int:
    return (b - date.fromisoformat(a)).days


def _avg_loan_days(conn) -> float:
    v = conn.execute("SELECT AVG(julianday(returned_at) - julianday(borrowed_at)) FROM loans WHERE returned_at IS NOT NULL").fetchone()[0]
    return round(v, 1) if v else float(LOAN_DAYS)


def _es(series: list[float], alpha: float = ES_ALPHA) -> float:
    level = series[0]
    for x in series[1:]:
        level = alpha * x + (1 - alpha) * level
    return level


def _row(r, **extra) -> dict:
    return {"resource_id": r["id"], "title": r["title"], "source": r["source"], "access_url": r["access_url"], **extra}


# ------------------------------------------------------------------------------------------------------------
# reports (pure functions of the DB and `today`)
# ------------------------------------------------------------------------------------------------------------
def waitlists(conn, today: date, top_n: int = 10) -> dict:
    rows = conn.execute("""
        SELECT r.*, (SELECT COUNT(*) FROM copies c WHERE c.resource_id=r.id) copies,
               (SELECT COUNT(*) FROM copies c WHERE c.resource_id=r.id AND c.status='available') free,
               SUM(s.status='active') waiting, SUM(s.status='ready') ready, MIN(CASE WHEN s.status='active' THEN s.created_at END) oldest
        FROM reservations s JOIN resources r ON r.id=s.resource_id
        WHERE s.status IN ('active','ready') GROUP BY r.id""").fetchall()
    out = []
    for r in rows:
        per_copy = round((r["waiting"] or 0) / r["copies"], 2) if r["copies"] else None
        out.append(_row(r, copies=r["copies"], available_now=r["free"], waiting=r["waiting"] or 0, ready_for_pickup=r["ready"] or 0,
                        oldest_wait_days=_days(r["oldest"], today) if r["oldest"] else None, waiting_per_copy=per_copy,
                        action="add copies" if per_copy is not None and per_copy >= WAITLIST_PER_COPY_ALERT else "monitor"))
    out.sort(key=lambda x: (-x["waiting"], -(x["oldest_wait_days"] or 0), x["title"]))
    return {"rows": out[:top_n], "titles_with_waitlist": sum(1 for x in out if x["waiting"] > 0),
            "titles_with_ready_holds": sum(1 for x in out if x["ready_for_pickup"] > 0), "total_waiting": sum(x["waiting"] for x in out)}


def overdue(conn, today: date, months: int = 6, top_n: int = 10) -> dict:
    open_rows = conn.execute("""SELECT l.id loan_id, l.user_id, l.due_at, r.* FROM loans l JOIN copies c ON c.id=l.copy_id
                                JOIN resources r ON r.id=c.resource_id WHERE l.returned_at IS NULL""").fetchall()
    late = [(r, _days(r["due_at"], today)) for r in open_rows if r["due_at"] < today.isoformat()]
    items = sorted(({**_row(r, days_overdue=d), "loan_id": r["loan_id"], "user_id": r["user_id"], "fine": round(d * FINE_PER_DAY, 2)} for r, d in late),
                   key=lambda x: -x["days_overdue"])
    by_title = Counter(r["title"] for r, _ in late)
    keys = _months(today, months)
    trend = {k: {"month": k, "returned": 0, "late": 0, "days_late": 0} for k in keys}
    for r in conn.execute("SELECT due_at, returned_at FROM loans WHERE returned_at IS NOT NULL AND substr(returned_at,1,7) >= ?", (keys[0],)):
        t = trend.get(r["returned_at"][:7])
        if t is None:
            continue
        t["returned"] += 1
        d = _days(r["due_at"], date.fromisoformat(r["returned_at"]))
        if d > 0:
            t["late"] += 1
            t["days_late"] += d
    for t in trend.values():
        t["late_rate"] = round(t["late"] / t["returned"], 3) if t["returned"] else None
        t["fines_assessed"] = round(t.pop("days_late") * FINE_PER_DAY, 2)
    return {"summary": {"open_loans": len(open_rows), "overdue_now": len(late), "overdue_rate": round(len(late) / len(open_rows), 3) if open_rows else 0.0,
                        "fines_accruing": round(sum(i["fine"] for i in items), 2), "currency": CURRENCY},
            "items": items[:top_n], "most_overdue_titles": [{"title": t, "overdue_loans": n} for t, n in by_title.most_common(5)],
            "trend": list(trend.values())}


def circulation(conn, today: date, months: int = 12, top_n: int = 10) -> dict:
    keys = _months(today, months)
    monthly = {k: {"month": k, "loans": 0, "unique_borrowers": 0} for k in keys}
    for r in conn.execute("SELECT substr(borrowed_at,1,7) m, COUNT(*) n, COUNT(DISTINCT user_id) u FROM loans WHERE substr(borrowed_at,1,7) >= ? GROUP BY m", (keys[0],)):
        if r["m"] in monthly:
            monthly[r["m"]].update(loans=r["n"], unique_borrowers=r["u"])
    since = f"{keys[0]}-01"
    top = conn.execute("""SELECT r.*, COUNT(*) loans, COUNT(DISTINCT l.user_id) readers FROM loans l JOIN copies c ON c.id=l.copy_id
                          JOIN resources r ON r.id=c.resource_id WHERE l.borrowed_at >= ? GROUP BY r.id ORDER BY loans DESC, r.title LIMIT ?""", (since, top_n)).fetchall()
    by_type = {r["type"]: r["n"] for r in conn.execute("""SELECT r.type, COUNT(*) n FROM loans l JOIN copies c ON c.id=l.copy_id
                                                          JOIN resources r ON r.id=c.resource_id WHERE l.borrowed_at >= ? GROUP BY r.type""", (since,))}
    by_branch = {r["branch"]: r["n"] for r in conn.execute("SELECT c.branch, COUNT(*) n FROM loans l JOIN copies c ON c.id=l.copy_id WHERE l.borrowed_at >= ? GROUP BY c.branch", (since,))}
    return {"monthly": list(monthly.values()), "rows": [_row(r, loans=r["loans"], unique_readers=r["readers"]) for r in top],
            "loans_by_type": by_type, "loans_by_branch": by_branch, "total_loans": sum(m["loans"] for m in monthly.values())}


def utilization(conn, today: date, days: int = 365, top_n: int = 10) -> dict:
    since = (today - timedelta(days=days)).isoformat()
    phys = conn.execute("""SELECT r.*, COUNT(DISTINCT c.id) copies,
                                  (SELECT COUNT(*) FROM loans l JOIN copies c2 ON c2.id=l.copy_id WHERE c2.resource_id=r.id AND l.borrowed_at >= ?) loans
                           FROM resources r JOIN copies c ON c.resource_id=r.id GROUP BY r.id""", (since,)).fetchall()
    stats = [_row(r, copies=r["copies"], loans=r["loans"], loans_per_copy=round(r["loans"] / r["copies"], 2)) for r in phys]
    never = sorted((s for s in stats if s["loans"] == 0), key=lambda s: (-s["copies"], s["title"]))
    low = sorted((s for s in stats if s["loans"] > 0), key=lambda s: (s["loans_per_copy"], s["title"]))[:top_n]
    total_copies = conn.execute("SELECT COUNT(*) FROM copies").fetchone()[0]
    on_loan = conn.execute("SELECT COUNT(*) FROM copies WHERE status IN ('on_loan','held')").fetchone()[0]
    dig = conn.execute("""SELECT r.*, COUNT(u.id) events, SUM(u.kind='download') downloads, COUNT(DISTINCT u.user_id) users
                          FROM resources r LEFT JOIN usage_events u ON u.resource_id=r.id AND u.ts >= ?
                          WHERE r.access_url IS NOT NULL GROUP BY r.id""", (since,)).fetchall()
    dstats = [_row(r, events=r["events"], downloads=r["downloads"] or 0, unique_users=r["users"]) for r in dig]
    return {"summary": {"window_days": days, "physical_titles": len(stats), "copies_in_use_now_pct": round(100 * on_loan / total_copies, 1) if total_copies else 0.0,
                        "never_borrowed_titles": len(never), "digital_titles": len(dstats), "digital_unused_titles": sum(1 for d in dstats if d["events"] == 0)},
            "never_borrowed": never[:top_n], "underused": low,
            "digital_top": sorted((d for d in dstats if d["events"]), key=lambda d: -d["events"])[:top_n],
            "digital_unused": [d for d in dstats if d["events"] == 0][:top_n]}


def _norm_query(q: str) -> str:
    return " ".join(sorted(set(tokens(q))))


def catalog_gaps(conn, today: date, days: int = 90, min_count: int = 2, top_n: int = 10) -> dict:
    """Things people asked for that the catalog could not answer, grouped by normalised wording."""
    cutoff = (today - timedelta(days=days)).isoformat()
    groups: dict[str, dict] = defaultdict(lambda: {"forms": Counter(), "count": 0, "users": set(), "last": ""})
    for r in conn.execute("SELECT ts, user_id, tool, params FROM audit_log WHERE outcome='not_found' AND ts >= ? AND tool IN "
                          "('search_resources','recommend','check_availability','reserve')", (cutoff,)):
        try:
            p = json.loads(r["params"])
        except ValueError:
            continue
        q = (p.get("query") or p.get("topic") or p.get("title") or "").strip()
        key = _norm_query(q)
        if not key:
            continue
        g = groups[key]
        g["forms"][q.lower()] += 1
        g["count"] += 1
        g["last"] = max(g["last"], r["ts"][:10])
        if r["user_id"] is not None:
            g["users"].add(r["user_id"])
    rows = [{"query": g["forms"].most_common(1)[0][0], "unanswered_searches": g["count"], "distinct_users": len(g["users"]), "last_seen": g["last"],
             "action": "consider acquiring" if len(g["users"]) >= 2 else "watch"} for g in groups.values() if g["count"] >= min_count]
    rows.sort(key=lambda x: (-x["unanswered_searches"], x["query"]))
    ok = conn.execute("SELECT COUNT(*) n, SUM(outcome='not_found') nf FROM audit_log WHERE tool='search_resources' AND ts >= ?", (cutoff,)).fetchone()
    return {"rows": rows[:top_n], "window_days": days, "min_count": min_count,
            "search_success_rate": round(1 - (ok["nf"] or 0) / ok["n"], 3) if ok["n"] else None, "searches": ok["n"]}


def forecast(conn, today: date, horizon_days: int = 30, top_n: int = 10) -> dict:
    """Per-title demand forecast (exponential smoothing on monthly loans) with an honest backtest against simple baselines."""
    hist = _months(today, 12, include_current=False)      # complete months only
    per: dict[int, list[float]] = defaultdict(lambda: [0.0] * len(hist))
    idx = {k: i for i, k in enumerate(hist)}
    for r in conn.execute("SELECT c.resource_id rid, substr(l.borrowed_at,1,7) m, COUNT(*) n FROM loans l JOIN copies c ON c.id=l.copy_id "
                          "WHERE substr(l.borrowed_at,1,7) >= ? GROUP BY rid, m", (hist[0],)):
        if r["m"] in idx:
            per[r["rid"]][idx[r["m"]]] = float(r["n"])

    # backtest: forecast each of the last BACKTEST_MONTHS from the months before it
    err = {"es": 0.0, "naive": 0.0, "ma3": 0.0}
    actual = 0.0
    for series in per.values():
        for t in range(len(series) - BACKTEST_MONTHS, len(series)):
            past = series[:t]
            err["es"] += abs(_es(past) - series[t])
            err["naive"] += abs(past[-1] - series[t])
            err["ma3"] += abs(sum(past[-3:]) / 3 - series[t])
            actual += series[t]
    backtest = {"method": f"exponential smoothing (alpha={ES_ALPHA})", "months_tested": BACKTEST_MONTHS, "titles": len(per),
                **({"wape": round(err["es"] / actual, 3), "baseline_last_month_wape": round(err["naive"] / actual, 3),
                    "baseline_3mo_avg_wape": round(err["ma3"] / actual, 3)} if actual else {"wape": None})}

    avg_days = _avg_loan_days(conn)
    waiting = {r["resource_id"]: r["n"] for r in conn.execute("SELECT resource_id, COUNT(*) n FROM reservations WHERE status='active' GROUP BY resource_id")}
    out = []
    for r in conn.execute("SELECT r.*, COUNT(c.id) copies FROM resources r JOIN copies c ON c.resource_id=r.id GROUP BY r.id"):
        series = per.get(r["id"], [0.0] * len(hist))
        monthly = _es(series)
        demand = monthly * horizon_days / 30 + waiting.get(r["id"], 0)        # expected new loans + readers already queued
        capacity = r["copies"] * horizon_days / avg_days                       # loans the current copies can serve
        shortfall = demand - capacity
        extra = max(0, math.ceil(shortfall * avg_days / horizon_days - 1e-9)) if shortfall > 0 else 0
        total = int(sum(series))
        out.append(_row(r, copies=r["copies"], loans_12m=total, forecast_monthly_loans=round(monthly, 2), waiting_now=waiting.get(r["id"], 0),
                        expected_demand=round(demand, 2), capacity=round(capacity, 2), shortfall=round(shortfall, 2), suggested_extra_copies=extra,
                        confidence="low" if total < 6 else "medium" if total < 15 else "high"))
    out.sort(key=lambda x: (-x["shortfall"], x["title"]))
    return {"rows": out[:top_n], "horizon_days": horizon_days, "avg_loan_days": avg_days, "backtest": backtest,
            "notes": ["Observed loans are capped by copies on the shelf, so true demand is understated for sold-out titles; queued readers are added back.",
                      "Confidence is 'low' for titles with under 6 loans in 12 months."]}


def dashboard(conn, today: date) -> dict:
    """Headline KPIs: one number per question a librarian asks first thing."""
    since30 = (today - timedelta(days=30)).isoformat()
    wl, od, ut, gp, fc = waitlists(conn, today, 3), overdue(conn, today, 6, 3), utilization(conn, today, 365, 3), catalog_gaps(conn, today, 90, 2, 3), forecast(conn, today, 30, 3)
    top = wl["rows"][0] if wl["rows"] else None
    return {"kpis": {
        "loans_last_30_days": conn.execute("SELECT COUNT(*) FROM loans WHERE borrowed_at >= ?", (since30,)).fetchone()[0],
        "active_loans": od["summary"]["open_loans"], "overdue_now": od["summary"]["overdue_now"], "overdue_rate": od["summary"]["overdue_rate"],
        "fines_accruing": od["summary"]["fines_accruing"], "currency": CURRENCY,
        "titles_with_waitlist": wl["titles_with_waitlist"], "readers_waiting": wl["total_waiting"],
        "copies_in_use_pct": ut["summary"]["copies_in_use_now_pct"], "never_borrowed_titles_365d": ut["summary"]["never_borrowed_titles"],
        "digital_unused_titles_365d": ut["summary"]["digital_unused_titles"], "search_success_rate_90d": gp["search_success_rate"],
        "forecast_wape": fc["backtest"].get("wape")},
        "rows": ([_row_from(top)] if top else []),   # longest waitlist, citable
        "top_gaps": gp["rows"], "suggested_acquisitions": [r for r in fc["rows"] if r["suggested_extra_copies"] > 0]}


def _row_from(r: dict) -> dict:
    return {k: r[k] for k in ("resource_id", "title", "source", "access_url", "waiting")}


# ------------------------------------------------------------------------------------------------------------
# the tool
# ------------------------------------------------------------------------------------------------------------
@audited("librarian_analytics")
def librarian_analytics(ctx: Context, trace_id: str, report: str = "dashboard", top_n: int = 10, months: int | None = None,
                        days: int | None = None, min_count: int = 2, horizon_days: int = 30) -> dict:
    """Librarian only. report: dashboard | waitlists | overdue | circulation | utilization | catalog_gaps | forecast."""
    if not ctx.is_librarian:
        return _result(ctx, trace_id, "denied", message="Librarian role required.")
    if report not in REPORTS:
        return _result(ctx, trace_id, "invalid", message=f"Unknown report. Choose one of: {', '.join(REPORTS)}.")
    top_n = max(1, min(int(top_n), 50))
    c, t = ctx.conn, ctx.today
    if report == "dashboard":
        data = dashboard(c, t)
    elif report == "waitlists":
        data = waitlists(c, t, top_n)
    elif report == "overdue":
        data = overdue(c, t, max(1, min(months or 6, 24)), top_n)
    elif report == "circulation":
        data = circulation(c, t, max(1, min(months or 12, 36)), top_n)
    elif report == "utilization":
        data = utilization(c, t, max(7, min(days or 365, 1095)), top_n)
    elif report == "catalog_gaps":
        data = catalog_gaps(c, t, max(7, min(days or 90, 365)), max(1, min_count), top_n)
    else:
        data = forecast(c, t, max(7, min(horizon_days, 180)), top_n)
    return _result(ctx, trace_id, "ok", report=report, as_of=t.isoformat(), **data)


from . import tools as _tools  # noqa: E402
_tools.TOOLS[librarian_analytics.tool_name] = librarian_analytics
