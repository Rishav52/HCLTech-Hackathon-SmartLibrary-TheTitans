"""SQLite schema and connection helpers: the authoritative library systems (catalog + circulation)."""
import os
import sqlite3

DB_PATH = os.environ.get("LIBRARY_DB", os.path.join(os.path.dirname(__file__), "..", "data", "library.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS resources (
    id           INTEGER PRIMARY KEY,
    type         TEXT NOT NULL CHECK (type IN ('book','journal','paper','ebook','dissertation','learning_path')),
    title        TEXT NOT NULL,
    authors      TEXT NOT NULL DEFAULT '',
    year         INTEGER,
    isbn         TEXT,
    abstract     TEXT NOT NULL DEFAULT '',
    subjects     TEXT NOT NULL DEFAULT '',      -- comma-separated
    level        TEXT NOT NULL DEFAULT 'intermediate' CHECK (level IN ('beginner','intermediate','advanced')),
    course_tags  TEXT NOT NULL DEFAULT '',      -- comma-separated course codes
    source       TEXT NOT NULL DEFAULT 'catalog',  -- catalog | digital_repository | journal_subscription | research_archive
    access_url   TEXT,                          -- set for digital resources
    created_at   TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f','now'))  -- bumped by trigger; drives re-indexing
);

CREATE TRIGGER IF NOT EXISTS trg_resources_touch AFTER UPDATE ON resources
BEGIN
    UPDATE resources SET updated_at = strftime('%Y-%m-%d %H:%M:%f','now') WHERE id = NEW.id;
END;

-- One embedding per resource; text_hash lets re-indexing skip unchanged rows and model mismatch force a rebuild.
CREATE TABLE IF NOT EXISTS embeddings (
    resource_id INTEGER PRIMARY KEY REFERENCES resources(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    text_hash   TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL
);

-- Physical copies only; digital resources have no copies and are always available via access_url.
CREATE TABLE IF NOT EXISTS copies (
    id             INTEGER PRIMARY KEY,
    resource_id    INTEGER NOT NULL REFERENCES resources(id),
    branch         TEXT NOT NULL,
    shelf_location TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'available' CHECK (status IN ('available','on_loan','held','lost'))
);

CREATE TABLE IF NOT EXISTS users (
    id               INTEGER PRIMARY KEY,
    name             TEXT NOT NULL,
    email            TEXT NOT NULL UNIQUE,
    role             TEXT NOT NULL CHECK (role IN ('student','faculty','researcher','librarian')),
    program          TEXT NOT NULL DEFAULT '',
    courses          TEXT NOT NULL DEFAULT '',   -- comma-separated course codes
    interests        TEXT NOT NULL DEFAULT '',
    history_consent  INTEGER NOT NULL DEFAULT 0  -- 1 = recommendations may use borrowing history
);

CREATE TABLE IF NOT EXISTS loans (
    id          INTEGER PRIMARY KEY,
    copy_id     INTEGER NOT NULL REFERENCES copies(id),
    user_id     INTEGER NOT NULL REFERENCES users(id),
    borrowed_at TEXT NOT NULL,
    due_at      TEXT NOT NULL,
    returned_at TEXT,
    renewals    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS reservations (
    id          INTEGER PRIMARY KEY,
    resource_id INTEGER NOT NULL REFERENCES resources(id),
    user_id     INTEGER NOT NULL REFERENCES users(id),
    created_at  TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active','ready','fulfilled','cancelled'))
);

-- Digital access events (the circulation system only sees physical loans).
CREATE TABLE IF NOT EXISTS usage_events (
    id          INTEGER PRIMARY KEY,
    resource_id INTEGER NOT NULL REFERENCES resources(id) ON DELETE CASCADE,
    user_id     INTEGER REFERENCES users(id),
    kind        TEXT NOT NULL CHECK (kind IN ('view','download')),
    ts          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_resource ON usage_events(resource_id, ts);

CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY,
    trace_id    TEXT NOT NULL,
    ts          TEXT NOT NULL DEFAULT (datetime('now')),
    user_id     INTEGER,
    tool        TEXT NOT NULL,
    params      TEXT NOT NULL DEFAULT '{}',
    outcome     TEXT NOT NULL,                 -- ok | not_found | denied | error
    latency_ms  REAL,
    detail      TEXT
);

CREATE INDEX IF NOT EXISTS idx_copies_resource ON copies(resource_id);
CREATE INDEX IF NOT EXISTS idx_loans_user ON loans(user_id);
CREATE INDEX IF NOT EXISTS idx_loans_copy ON loans(copy_id);
CREATE INDEX IF NOT EXISTS idx_res_resource ON reservations(resource_id, status);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    """One connection per request/task. check_same_thread=False because a web server may create the connection
    on one worker thread and use it on another; it is never shared by two requests at once."""
    target = path or DB_PATH
    conn = sqlite3.connect(target, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if target != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")   # readers don't block the writer (audit rows, reservations)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
