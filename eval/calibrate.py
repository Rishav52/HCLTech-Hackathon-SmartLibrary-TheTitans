"""Calibrate the search abstention threshold for the active embedder.

    python -m eval.calibrate [--embedder auto|hashing|local]

For each threshold t it measures (a) how many answerable queries still return a relevant result in the top 3 and
(b) how many unanswerable queries are correctly refused, then reports the t that balances both. Apply the result with
SEARCH_MIN_SCORE=<t> in .env. The labeled set is small, so treat the number as a starting point and re-run it
whenever the embedder, catalog or query mix changes.
"""
import argparse

import numpy as np

from app import search
from app.embedders import get_embedder
from . import cases as C
from .metrics import mean
from .run import Env
from app.llm import RuleBasedLLM


def _scores(conn, query, filt, relevant=None):
    c, fused, elig, _, _ = search.score_query(conn, query, min_score=-1.0)   # threshold off: gate rules only
    rows = c["rows"]
    ok = [i for i in np.argsort(-fused) if elig[i] and (not filt.get("types") or rows[i]["type"] in filt["types"])
          and (not filt.get("level") or rows[i]["level"] == filt["level"])]
    return [(float(fused[i]), rows[i]["title"]) for i in ok]


def calibrate(embedder_kind=None, grid=None):
    emb = get_embedder(embedder_kind)
    search.set_embedder(emb)
    env = Env(RuleBasedLLM())
    conn = env.fresh()
    answerable = [(_scores(conn, q, f), set(rel)) for _, _, q, f, rel in C.SEARCH]
    unanswerable = [_scores(conn, q, {}) for q in C.UNANSWERABLE]
    grid = grid or [round(x, 3) for x in np.arange(0.05, 0.701, 0.025)]
    table = []
    for t in grid:
        recall = mean(any(title in rel for s, title in ranked[:3] if s >= t) for ranked, rel in answerable)
        refused = mean((not ranked) or ranked[0][0] < t for ranked in unanswerable)
        bal = 0.0 if recall + refused == 0 else 2 * recall * refused / (recall + refused)
        table.append({"threshold": t, "answerable_hit@3": round(recall, 3), "unanswerable_refused": round(refused, 3), "balanced": round(bal, 3)})
    best = max(table, key=lambda r: r["balanced"])
    plateau = [r["threshold"] for r in table if r["balanced"] == best["balanced"]]
    return {"embedder": emb.name, "current": emb.min_score, "table": table, "best": best,
            "recommended": plateau[len(plateau) // 2], "n_answerable": len(answerable), "n_unanswerable": len(unanswerable)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default=None)
    r = calibrate(ap.parse_args(argv).embedder)
    print(f"Embedder {r['embedder']} (current threshold {r['current']}); {r['n_answerable']} answerable / {r['n_unanswerable']} unanswerable queries\n")
    print("threshold  answerable hit@3  unanswerable refused  balanced")
    for x in r["table"]:
        mark = "  <-- best" if x["balanced"] == r["best"]["balanced"] else ""
        print(f"{x['threshold']:>9}  {x['answerable_hit@3']:>16}  {x['unanswerable_refused']:>20}  {x['balanced']:>8}{mark}")
    print(f"\nRecommended: SEARCH_MIN_SCORE={r['recommended']}  (middle of the best plateau; add to .env)")


if __name__ == "__main__":
    main()
