"""Tiny configuration helpers (no extra dependency): .env loading and first-run seeding."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_env(path: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env without overriding variables already set in the environment."""
    path = path or ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def ensure_seeded() -> bool:
    """Create and seed the demo database on first start (AUTO_SEED=0 disables). Returns True if it seeded."""
    from . import db, seed
    if os.environ.get("AUTO_SEED", "1") != "1":
        return False
    os.makedirs(os.path.dirname(os.path.abspath(db.DB_PATH)), exist_ok=True)
    conn = db.connect()
    try:
        db.init_db(conn)
        if conn.execute("SELECT COUNT(*) FROM resources").fetchone()[0]:
            return False
        seed.seed(conn)
        return True
    finally:
        conn.close()
