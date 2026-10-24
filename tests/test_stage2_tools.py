import json

import pytest

from app import tools
from app.db import connect, init_db
from app.seed import seed


@pytest.fixture()
def conn():
    c = connect(":memory:")
    init_db(c)
    seed(c)
    yield c
    c.close()


def ctx_for(conn, email=None):
    user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone() if email else None
    return tools.Context(conn=conn, user=user)


AARAV, DIYA, LENA = "aarav@univ.example", "diya@univ.example", "lena@univ.example"


# --- availability ------------------------------------------------------------------------------
def test_availability_physical(conn):
    r = tools.check_availability(ctx_for(conn), title="Artificial Intelligence: A Modern Approach")
    assert r["status"] == "ok" and r["format"] == "physical"
    assert r["copies_total"] == 3 and r["copies_available"] == 2 and r["locations"][0]["branch"]


def test_availability_fully_out_shows_waitlist(conn):
    r = tools.check_availability(ctx_for(conn), title="python crash course")  # case-insensitive
    assert r["status"] == "ok" and r["available"] is False and r["waitlist_length"] == 3


def test_availability_digital(conn):
    r = tools.check_availability(ctx_for(conn), title="Attention Is All You Need")
    assert r["format"] == "digital" and r["access_url"].startswith("https://")


def test_not_found_never_fabricates(conn):
    r = tools.check_availability(ctx_for(conn), title="Quantum Basket Weaving for Beginners")
    assert r["status"] == "not_found" and "available" not in r


def test_ambiguous_title_returns_candidates(conn):
    r = tools.check_availability(ctx_for(conn), title="Python")
    assert r["status"] == "ambiguous" and len(r["candidates"]) > 1


# --- my_loans / privacy ------------------------------------------------------------------------
def test_my_loans_requires_auth(conn):
    assert tools.my_loans(ctx_for(conn))["status"] == "denied"


def test_my_loans_due_this_week_and_overdue_fine(conn):
    r = tools.my_loans(ctx_for(conn, AARAV), due_within_days=7)
    assert r["status"] == "ok"
    titles = {l["title"]: l for l in r["loans"]}
    assert len(titles) == 2  # due in 2 days + overdue; the one due in 9 days is excluded
    over = titles["Python for Data Analysis"]
    assert over["overdue"] and over["days_overdue"] == 4 and over["fine"] == 4 * tools.FINE_PER_DAY
    assert r["total_fine"] == over["fine"]


def test_privacy_refusal_for_other_users_data(conn):
    diya_id = conn.execute("SELECT id FROM users WHERE email=?", (DIYA,)).fetchone()["id"]
    r = tools.my_loans(ctx_for(conn, AARAV), user_id=diya_id)
    assert r["status"] == "denied" and "loans" not in r
    assert tools.my_loans(ctx_for(conn, LENA), user_id=diya_id)["status"] == "ok"  # librarian may


# --- reserve -----------------------------------------------------------------------------------
def test_reserve_requires_auth(conn):
    assert tools.reserve(ctx_for(conn), title="Fluent Python")["status"] == "denied"


def test_reserve_available_copy_is_held(conn):
    r = tools.reserve(ctx_for(conn, DIYA), title="Fluent Python")
    assert r["status"] == "ok" and r["reservation_status"] == "ready" and r["branch"] and r["queue_position"] == 1
    avail = tools.check_availability(ctx_for(conn), title="Fluent Python")
    assert avail["copies_available"] == 1  # one of two copies now held


def test_reserve_unavailable_joins_waitlist_at_end(conn):
    r = tools.reserve(ctx_for(conn, "anil@univ.example"), title="Python Crash Course")
    assert r["reservation_status"] == "active" and r["queue_position"] == 4
    # holding the title already blocks a second request
    assert tools.reserve(ctx_for(conn, "kabir@univ.example"), title="Python Crash Course")["status"] == "invalid"


def test_reserve_duplicate_and_not_found_and_digital(conn):
    assert tools.reserve(ctx_for(conn, AARAV), title="Python Crash Course")["status"] == "invalid"  # already queued
    assert tools.reserve(ctx_for(conn, AARAV), title="No Such Book Anywhere")["status"] == "not_found"
    assert tools.reserve(ctx_for(conn, AARAV), title="Attention Is All You Need")["status"] == "not_applicable"


# --- renew -------------------------------------------------------------------------------------
def loan_id(conn, email, title):
    return conn.execute("SELECT l.id FROM loans l JOIN users u ON u.id=l.user_id JOIN copies c ON c.id=l.copy_id "
                        "JOIN resources r ON r.id=c.resource_id WHERE u.email=? AND r.title=? AND l.returned_at IS NULL",
                        (email, title)).fetchone()["id"]


def test_renew_success_extends_due_date(conn):
    lid = loan_id(conn, AARAV, "Artificial Intelligence: A Modern Approach")
    r = tools.renew(ctx_for(conn, AARAV), loan_id=lid)
    assert r["status"] == "ok" and r["renewals_left"] == tools.MAX_RENEWALS - 1
    assert r["new_due_at"] > r["old_due_at"]


def test_renew_blocked_when_overdue_or_capped(conn):
    assert tools.renew(ctx_for(conn, AARAV), loan_id=loan_id(conn, AARAV, "Python for Data Analysis"))["status"] == "invalid"
    capped = tools.renew(ctx_for(conn, DIYA), loan_id=loan_id(conn, DIYA, "Introduction to Algorithms"))
    assert capped["status"] == "invalid"


def test_renew_blocked_when_someone_waiting(conn):
    r = tools.renew(ctx_for(conn, "ishita@univ.example"), loan_id=loan_id(conn, "ishita@univ.example", "Pattern Recognition and Machine Learning"))
    assert r["status"] == "invalid" and "waiting" in r["message"]


def test_renew_someone_elses_loan_indistinguishable_from_missing(conn):
    lid = loan_id(conn, DIYA, "Database System Concepts")
    other = tools.renew(ctx_for(conn, AARAV), loan_id=lid)
    missing = tools.renew(ctx_for(conn, AARAV), loan_id=999999)
    assert other["status"] == missing["status"] == "not_found" and other["message"] == missing["message"]


# --- return + waitlist promotion ---------------------------------------------------------------
def test_return_promotes_first_waitlister_and_assesses_fine(conn):
    over = loan_id(conn, AARAV, "Python for Data Analysis")
    assert tools.return_copy(ctx_for(conn, AARAV), loan_id=over)["status"] == "denied"  # librarian only
    r = tools.return_copy(ctx_for(conn, LENA), loan_id=over)
    assert r["status"] == "ok" and r["fine"] == 4 * tools.FINE_PER_DAY and r["promoted_reservation_id"] is None

    lid = loan_id(conn, "rohan@univ.example", "Python Crash Course")  # 3 people waiting
    r = tools.return_copy(ctx_for(conn, LENA), loan_id=lid)
    assert r["promoted_reservation_id"] is not None
    first = conn.execute("SELECT u.email, res.status FROM reservations res JOIN users u ON u.id=res.user_id WHERE res.id=?",
                         (r["promoted_reservation_id"],)).fetchone()
    assert first["email"] == AARAV and first["status"] == "ready"  # oldest waitlister
    assert tools.check_availability(ctx_for(conn), title="Python Crash Course")["copies_available"] == 0  # held, not free


# --- audit -------------------------------------------------------------------------------------
def test_every_call_is_audited_with_trace_id(conn):
    r = tools.reserve(ctx_for(conn, AARAV), title="No Such Book Anywhere")
    row = conn.execute("SELECT * FROM audit_log WHERE trace_id=?", (r["trace_id"],)).fetchone()
    assert row["tool"] == "reserve" and row["outcome"] == "not_found" and row["latency_ms"] is not None
    assert json.loads(row["params"]) == {"title": "No Such Book Anywhere"}
    assert row["user_id"] is not None


def test_unexpected_errors_are_audited_and_not_leaked(conn):
    r = tools.check_availability(ctx_for(conn), bogus_param=1)  # TypeError inside wrapper -> contained
    assert r["status"] == "error" and "TypeError" not in r["message"]
    assert conn.execute("SELECT outcome FROM audit_log WHERE trace_id=?", (r["trace_id"],)).fetchone()["outcome"] == "error"
