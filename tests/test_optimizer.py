"""tests/test_optimizer.py — single feasibility test for the L3 router.

Per the final-sprint cut list: no bias analysis, no adversarial suite.
The test confirms that every rep-day plan:
  (a) respects the per-rep capacity budget
  (b) every stop is in a tehsil within the rep's assigned territory
  (c) every stop is on a window-open cell at the requested date
  (d) dual-arm interface honored (run_dual_arm returns both 'real' and 'synthetic')
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture(scope="module")
def both_arms():
    from disha.optimizer.router import run_dual_arm
    return run_dual_arm("2025-11-17", rep_ids=["REP_0001", "REP_0050", "REP_0100"])


def test_dual_arm_returns_both(both_arms):
    assert set(both_arms.keys()) == {"real", "synthetic"}
    assert both_arms["real"].arm == "real"
    assert both_arms["synthetic"].arm == "synthetic"


def test_each_rep_day_respects_capacity(both_arms):
    for arm, plan in both_arms.items():
        for rd in plan.rep_days:
            assert rd.capacity_used_min <= rd.capacity_budget_min + 1e-6, (
                f"[{arm}] rep {rd.rep_id} used {rd.capacity_used_min:.0f} min "
                f"> budget {rd.capacity_budget_min:.0f} min"
            )


def test_every_stop_is_window_open(both_arms):
    """Every selected (tehsil, product) must have window_decay > 0 in the
    month of the requested date — hard constraint of the locked architecture."""
    import pandas as pd
    cate_real = pd.read_parquet("data/processed/uplift_real_cate.parquet")
    cate_synth = pd.read_parquet("data/processed/uplift_synthetic_cate.parquet")
    for arm, plan in both_arms.items():
        cate = cate_real if arm == "real" else cate_synth
        month_start = pd.Timestamp(plan.date).replace(day=1)
        wmap = (
            cate[cate["month_start"] == month_start]
            .set_index(["tehsil", "product"])["window_decay_this_product"]
            .to_dict()
        )
        for rd in plan.rep_days:
            for stop in rd.stops:
                key = (stop.tehsil, stop.product)
                assert wmap.get(key, 0.0) > 0, (
                    f"[{arm}] rep {rd.rep_id} stop {stop.seq}: "
                    f"({stop.tehsil}, {stop.product}) is NOT window-open"
                )


def test_stops_only_in_rep_territory(both_arms):
    import pandas as pd, json
    reps = pd.read_csv("data/raw/reps_territory.csv")
    reps["tehsils"] = reps["tehsil_list"].map(json.loads)
    rep_tehsils = dict(zip(reps["rep_id"], reps["tehsils"]))
    for arm, plan in both_arms.items():
        for rd in plan.rep_days:
            allowed = set(rep_tehsils.get(rd.rep_id, []))
            for stop in rd.stops:
                assert stop.tehsil in allowed, (
                    f"[{arm}] rep {rd.rep_id} stop {stop.seq} tehsil "
                    f"{stop.tehsil} not in their territory list"
                )


def test_stopwhy_exposes_no_absolute_inr_only_rank(both_arms):
    """Change-1 contract: the rep-facing WHY must not expose absolute CATE INR.
    Real-data CF CATE is biased downward (~−₹24k mean), so a per-stop ₹ would
    be misleading.  We render priority_rank + priority_pct + driver_human
    instead — all honest on real data."""
    for arm, plan in both_arms.items():
        for rd in plan.rep_days:
            for stop in rd.stops:
                w = stop.why
                # Forbidden field — would have been the inflated/biased ₹
                assert not hasattr(w, "incremental_inr"), (
                    f"[{arm}] StopWhy must NOT have incremental_inr; "
                    "real-data CF absolute CATE is biased — use rank fields."
                )
                # Required honest fields
                assert isinstance(w.priority_rank, int) and w.priority_rank >= 1, (
                    f"[{arm}] priority_rank must be a positive int; got {w.priority_rank!r}"
                )
                assert 0 < w.priority_pct <= 100.0, (
                    f"[{arm}] priority_pct must be in (0, 100]; got {w.priority_pct!r}"
                )
                assert w.driver_human and not w.driver_human.startswith("pct_"), (
                    f"[{arm}] driver_human must be plain text, not raw feature "
                    f"name; got {w.driver_human!r}"
                )
                # Plain text must not leak rupee figures into the rep-facing layer
                assert "₹" not in w.plain_text, (
                    f"[{arm}] plain_text leaks a rupee figure: {w.plain_text!r}"
                )
