"""Catalog ingestion: validate, deduplicate and add resources (and optional physical copies).

New rows are picked up by the search index automatically (see search.py), so no code change or manual
re-index is needed.   CLI:  python -m app.catalog path/to/resources.json
"""
import json
import re
import sys

from .db import connect, init_db

TYPES = {"book", "journal", "paper", "ebook", "dissertation", "learning_path"}
LEVELS = {"beginner", "intermediate", "advanced"}
FIELDS = ("type", "title", "authors", "year", "isbn", "abstract", "subjects", "level", "course_tags", "source", "access_url")


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def add_resources(conn, items: list[dict]) -> dict:
    added, duplicates, rejected = [], [], []
    for it in items:
        title = (it.get("title") or "").strip()
        if not title:
            rejected.append({"title": title, "reason": "missing title"})
            continue
        if it.get("type") not in TYPES:
            rejected.append({"title": title, "reason": f"invalid type {it.get('type')!r}"})
            continue
        if it.get("level", "intermediate") not in LEVELS:
            rejected.append({"title": title, "reason": f"invalid level {it.get('level')!r}"})
            continue
        isbn = (it.get("isbn") or "").replace("-", "").strip() or None
        dup = conn.execute("SELECT id FROM resources WHERE isbn=?", (isbn,)).fetchone() if isbn else None
        if dup is None:
            dup = next((r for r in conn.execute("SELECT id,title,year FROM resources")
                        if _norm(r["title"]) == _norm(title) and r["year"] == it.get("year")), None)
        if dup:
            duplicates.append({"title": title, "existing_id": dup["id"]})
            continue
        row = {k: it.get(k) for k in FIELDS}
        row.update(title=title, isbn=isbn, level=it.get("level", "intermediate"),
                   **{k: it.get(k) or "" for k in ("authors", "abstract", "subjects", "course_tags")},
                   source=it.get("source") or "catalog")
        with conn:
            cur = conn.execute(f"INSERT INTO resources ({','.join(FIELDS)}) VALUES ({','.join('?' * len(FIELDS))})",
                               [row[k] for k in FIELDS])
            for c in it.get("copies", []):
                conn.execute("INSERT INTO copies (resource_id,branch,shelf_location) VALUES (?,?,?)",
                             (cur.lastrowid, c["branch"], c["shelf"]))
        added.append(cur.lastrowid)
    return {"added": added, "duplicates": duplicates, "rejected": rejected}


if __name__ == "__main__":
    conn = connect()
    init_db(conn)
    print(json.dumps(add_resources(conn, json.load(open(sys.argv[1]))), indent=2))
