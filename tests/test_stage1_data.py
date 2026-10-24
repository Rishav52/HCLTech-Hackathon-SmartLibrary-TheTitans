from datetime import date, timedelta

import pytest

from app.db import connect, init_db
from app.seed import seed


@pytest.fixture()
def conn():
    c = connect(":memory:")
    init_db(c)
    seed(c)
    yield c
    c.close()


def test_counts(conn):
    assert conn.execute("SELECT COUNT(*) FROM resources").fetchone()[0] >= 30
    types = {r[0] for r in conn.execute("SELECT DISTINCT type FROM resources")}
    assert {"book", "ebook", "paper", "journal", "dissertation", "learning_path"} <= types


def test_digital_resources_have_url_and_no_copies(conn):
    rows = conn.execute(
        "SELECT r.title, r.access_url, (SELECT COUNT(*) FROM copies c WHERE c.resource_id=r.id) n FROM resources r"
        " WHERE r.type IN ('ebook','paper','journal','dissertation','learning_path')").fetchall()
    assert rows and all(r["access_url"] and r["n"] == 0 for r in rows)


def test_on_loan_copies_match_open_loans(conn):
    open_loans = conn.execute("SELECT COUNT(*) FROM loans WHERE returned_at IS NULL").fetchone()[0]
    on_loan = conn.execute("SELECT COUNT(*) FROM copies WHERE status='on_loan'").fetchone()[0]
    assert open_loans == on_loan


def test_demo_scenarios_present(conn):
    today = date.today()
    # due this week + overdue for the same borrower
    due = conn.execute("SELECT due_at FROM loans l JOIN users u ON u.id=l.user_id WHERE u.email='aarav@univ.example' AND returned_at IS NULL").fetchall()
    dates = [date.fromisoformat(r[0]) for r in due]
    assert any(today <= d <= today + timedelta(days=7) for d in dates)
    assert any(d < today for d in dates)
    # fully checked-out title with a waitlist
    free = conn.execute("SELECT COUNT(*) FROM copies c JOIN resources r ON r.id=c.resource_id WHERE r.title='Python Crash Course' AND c.status='available'").fetchone()[0]
    waiting = conn.execute("SELECT COUNT(*) FROM reservations res JOIN resources r ON r.id=res.resource_id WHERE r.title='Python Crash Course'").fetchone()[0]
    assert free == 0 and waiting == 3
    # a title that does not exist anywhere (not_found scenario)
    assert conn.execute("SELECT COUNT(*) FROM resources WHERE title LIKE '%Quantum Basket Weaving%'").fetchone()[0] == 0


def test_foreign_keys_enforced(conn):
    with pytest.raises(Exception):
        conn.execute("INSERT INTO copies (resource_id,branch,shelf_location) VALUES (9999,'x','y')")
