"""tests/test_api.py — 3 contract tests for the L4 FastAPI."""
import json

import pytest
from fastapi.testclient import TestClient

from disha.api.main import app

client = TestClient(app)


def test_health_reports_all_frozen_artifacts():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # All seven frozen artifacts that the API depends on must exist
    for key in ("monthly_panel", "cate_frozen", "uplift_real", "uplift_synth",
                "signals", "ope", "business_case", "dgp_gate"):
        assert body["frozen_artifacts"][key] is True, f"missing artifact: {key}"


def test_dual_arm_plans_round_trip():
    # Synthetic arm exists and has populated rep_day; real arm too
    r_synth = client.get("/sim/run?arm=synthetic&date=2025-11-17")
    r_real  = client.get("/sim/run?arm=real&date=2025-11-17")
    for r, arm in [(r_synth, "synthetic"), (r_real, "real")]:
        assert r.status_code == 200, f"{arm} arm failed: {r.text}"
        b = r.json()
        assert b["arm"] == arm
        assert b["summary"]["n_stops"] > 0, f"{arm} arm produced zero stops"


def test_sim_run_summary_exposes_honest_coverage_metric():
    """Change-B contract: dashboard 'Coverage efficiency' tile must read
    coverage_efficiency_pct (visited / feasible), not the older
    capacity_used_pct relabelled.  Value bounded to [0, 100]."""
    for arm in ("real", "synthetic"):
        b = client.get(f"/sim/run?arm={arm}&date=2025-11-17").json()
        s = b["summary"]
        assert "coverage_efficiency_pct" in s, \
            f"[{arm}] /sim/run summary missing coverage_efficiency_pct"
        assert 0.0 <= float(s["coverage_efficiency_pct"]) <= 100.0, \
            f"[{arm}] coverage_efficiency_pct out of bounds: {s['coverage_efficiency_pct']}"


def test_no_unlabelled_revenue_per_field_day_rupee_field():
    """Change-A contract: a raw 'revenue_per_field_day' rupee value (unlabelled
    as OPE) is exactly the credibility trap we just removed.  An OPE-framed key
    is fine; an unqualified rupee key is not."""
    forbidden = {"revenue_per_field_day", "revenue_per_field_day_inr"}
    for path in ("/sim/run?arm=real&date=2025-11-17",
                 "/sim/run?arm=synthetic&date=2025-11-17",
                 "/qini", "/business_case", "/learning"):
        body = client.get(path).json()
        flat_keys = _all_keys(body)
        leaks = forbidden & flat_keys
        assert not leaks, f"{path} exposes unlabelled rupee field(s): {leaks}"


def _all_keys(obj) -> set:
    """Recursively collect every dict key in a JSON-like blob."""
    out: set = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            out |= _all_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _all_keys(v)
    return out


def test_business_case_and_qini_endpoints():
    bc = client.get("/business_case")
    assert bc.status_code == 200
    b = bc.json()
    assert "annual_lift_inr_low" in b
    assert "annual_lift_inr_high" in b
    assert b["annual_lift_inr_high"] >= b["annual_lift_inr_low"]

    q = client.get("/qini")
    assert q.status_code == 200
    qb = q.json()
    assert qb["demo_path"] == "dual_synthetic_led"
    assert "uplift_eval" in qb


def test_no_biased_inr_leaks_in_api_responses():
    """Change-6 contract: real-data CF absolute CATE is biased downward; the
    API must never expose captured_prize / total_captured_prize in any
    rep-facing or simulator response."""
    for arm in ("real", "synthetic"):
        plan = client.get(f"/plan/REP_0100?arm={arm}").json()
        assert "total_captured_prize" not in plan["rep_day"], \
            f"[{arm}] /plan leaks total_captured_prize"
        for stop in plan["rep_day"]["stops"]:
            assert "captured_prize" not in stop, \
                f"[{arm}] /plan stop leaks captured_prize: {stop}"

        sim = client.get(f"/sim/run?arm={arm}").json()
        assert "total_prize" not in sim["summary"], \
            f"[{arm}] /sim/run summary leaks total_prize"
        for rd in sim["rep_days"]:
            assert "total_captured_prize" not in rd, \
                f"[{arm}] /sim/run rep_day leaks total_captured_prize"
            for stop in rd["stops"]:
                assert "captured_prize" not in stop, \
                    f"[{arm}] /sim/run stop leaks captured_prize"


def test_on_disk_plans_dont_leak_inr():
    """Companion to the API check: the saved data/processed/plans/*.json
    must also be free of absolute INR fields."""
    import json, pathlib
    plans_dir = pathlib.Path("data/processed/plans")
    for p in plans_dir.glob("plan_*.json"):
        data = json.loads(p.read_text())
        assert "total_captured_prize" not in data.get("rep_days", [{}])[0], \
            f"{p.name} leaks total_captured_prize"
        for rd in data.get("rep_days", []):
            assert "total_captured_prize" not in rd, f"{p.name} → rd"
            for stop in rd.get("stops", []):
                assert "captured_prize" not in stop, f"{p.name} → stop"

def test_plan_stops_include_actionable_intelligence():
    plan = client.get("/plan/REP_0100?arm=real&date=2025-11-17")
    assert plan.status_code == 200
    stop = plan.json()["rep_day"]["stops"][0]
    assert stop["evidence"], "stop must include explainable evidence"
    assert "next_best_action" in stop
    nba = stop["next_best_action"]
    assert nba["primary_product"] == stop["product"]
    assert nba["agronomic_advice"]
    assert nba["promotional_mechanic"]


def test_anomalies_learning_and_outcomes_endpoints():
    anomalies = client.get("/anomalies?date=2025-11-17&limit=5")
    assert anomalies.status_code == 200
    assert "items" in anomalies.json()

    payload = {
        "outcomes": [{
            "rep_id": "REP_TEST",
            "retailer_id": "RTL_TEST",
            "tehsil": "Patna_T038",
            "product": "Actara 25 WG",
            "seq": 1,
            "outcome": "order_placed",
            "captured_at": "2025-11-17T09:00:00Z",
            "order_value": 1000,
        }]
    }
    posted = client.post("/outcomes", json=payload)
    assert posted.status_code == 200
    body = posted.json()
    assert body["accepted"] == 1
    assert body["learning"]["total_outcomes"] >= 1
    assert body["recalibration"]["cadence"] == "weekly"

    learning = client.get("/learning")
    assert learning.status_code == 200
    assert learning.json()["learning"]["recommendation_acceptance_rate"] >= 0
