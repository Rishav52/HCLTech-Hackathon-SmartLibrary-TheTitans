import json
import math

import pytest

from app import search, tools
from eval import calibrate, cases as C, metrics as M, run as E


@pytest.fixture(autouse=True)
def _reset_embedder():
    yield
    search.set_embedder(None)


# ---- metrics (hand-computed) ---------------------------------------------------------------------------------
def test_metrics_hand_computed():
    ranked, rel = ["a", "b", "c", "d"], {"c", "z"}
    assert M.hit_at_k(ranked, rel, 2) == 0.0 and M.hit_at_k(ranked, rel, 3) == 1.0
    assert M.recall_at_k(ranked, rel, 4) == 0.5 and M.precision_at_k(ranked, rel, 4) == 0.25
    assert M.reciprocal_rank(ranked, rel) == pytest.approx(1 / 3) and M.reciprocal_rank(["x"], rel) == 0.0
    assert M.precision_at_k([], rel, 3) == 0.0 and M.recall_at_k(ranked, set(), 3) == 0.0


def test_percentiles_nearest_rank():
    xs = list(range(1, 101))
    assert M.percentile(xs, 50) == 50 and M.percentile(xs, 95) == 95 and M.percentile([7], 99) == 7 and math.isnan(M.percentile([], 50))


# ---- the labeled data must be valid before it can judge anything --------------------------------------------
def test_ground_truth_names_only_real_catalog_titles():
    env = E.Env(E.get_llm("rules"))
    titles = env.titles
    for cid, _, _, _, rel in C.SEARCH:
        assert set(rel) <= titles, cid
    assert len({c[0] for c in C.SEARCH}) == len(C.SEARCH) and all(rel for *_, rel in C.SEARCH)
    assert set(C.DIYA_PRIVATE_TITLES) <= titles


# ---- full run -----------------------------------------------------------------------------------------------
def test_full_run_passes_offline_and_writes_reports(tmp_path):
    res = E.run("rules", "hashing", str(tmp_path))
    assert res["passed"], [g for g in res["gates"] if g["gated"] and not g["passed"]]
    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["meta"]["retrieval"] == "lexical fallback" and len(saved["gates"]) >= 12
    md = (tmp_path / "report.md").read_text()
    assert "**Overall: PASS**" in md and "Search relevance" in md and "Latency" in md
    # the paraphrase gap of the lexical fallback is reported, not hidden
    para = next(g for g in saved["gates"] if "paraphrase" in g["name"])
    assert para["gated"] is False
    assert res["grounding"]["responses"] > 40 and res["grounding"]["ungrounded_rate"] == 0.0


def test_harness_catches_policy_drift_and_removed_guardrails(tmp_path, monkeypatch):
    """The evaluation must be able to fail. These sabotages mirror real regressions."""
    monkeypatch.setattr(tools, "RENEWAL_DAYS", 7)
    r = E.run("rules", "hashing", str(tmp_path / "a"))
    assert not r["passed"] and any(not c["passed"] and "renewal period" in c["name"] for c in r["transactions"]["checks"])
    monkeypatch.setattr(tools, "RENEWAL_DAYS", 14)

    from app import assistant as A
    monkeypatch.setattr(A, "asks_for_other_users_data", lambda ctx, m: False)
    r = E.run("rules", "hashing", str(tmp_path / "b"))
    assert not r["passed"] and r["privacy"]["pass_rate"] < 1.0


def test_oracles_do_not_import_policy_from_the_system_under_test():
    src = open(E.__file__).read()
    assert "tools.RENEWAL_DAYS" not in src and "tools.FINE_PER_DAY" not in src


# ---- calibration ------------------------------------------------------------------------------------------
def test_calibration_table_is_well_formed():
    r = calibrate.calibrate("hashing")
    ts = [x["threshold"] for x in r["table"]]
    assert ts == sorted(ts) and r["recommended"] in ts
    recalls = [x["answerable_hit@3"] for x in r["table"]]
    assert recalls == sorted(recalls, reverse=True)             # raising the threshold can only lose answerable hits
    refused = [x["unanswerable_refused"] for x in r["table"]]
    assert refused == sorted(refused)                           # ...and can only refuse more unanswerable queries
    assert r["best"]["balanced"] >= 0.9 and r["n_unanswerable"] == len(C.UNANSWERABLE)


def test_min_score_override_from_environment(monkeypatch):
    from app.embedders import get_embedder
    monkeypatch.setenv("SEARCH_MIN_SCORE", "0.42")
    assert get_embedder("hashing").min_score == 0.42
