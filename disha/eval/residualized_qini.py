"""
disha.eval.residualized_qini — single source of truth for the
scale-residualized window-Qini headline number.

Why this exists
---------------
The headline metric ("Causal Forest residualized window-Qini") was being
asserted in prose across every narrative surface (deck, README, SOLUTION.md,
PITCH.md, dgp_gate banner) without ever being computed and serialized.  The
only place the residualization ran was a pytest fixture in
`tests/test_independence.py` — and even there, only the >= 0.15 guard was
recorded, not the actual value.

This module lifts that exact fixture into a reusable function so that the
test suite, the API, and every doc/UI surface read the SAME computed
number.  No more prose-only headlines.

Computation (must match `tests/test_independence.py::residualized_qini`
EXACTLY — both call this function now):
  1. Join uplift_real_cate.parquet to monthly_panel on (tehsil, month_start, product)
     to align CATE estimates with the four scale features.
  2. Standardize the scale features.
  3. For each estimator (t/s/r-learner, causal_forest): fit a Ridge(alpha=1)
     from standardized scale features to the estimator's CATE; residualize.
  4. Compute `qini_window_constrained(cate_raw)` and
     `qini_window_constrained(cate_resid)` over Y_revenue / T with
     window_open = (window_decay_this_product > 0).
  5. Return {estimator: {"orig": float, "resid": float}}.

Output: when run as a script, merges the result into
`data/processed/uplift_eval.json` under top-level key `qini_window_residualized`
without modifying any existing field.  The API serves it via /qini.
"""
from __future__ import annotations

import io
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from disha.eval.qini import qini_window_constrained
from disha.signals.correlations import SCALE_FEATURES

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _ROOT / "data" / "processed"

ESTIMATORS = ("t_learner", "s_learner", "r_learner", "causal_forest")


def compute_residualized_window_qini(
    real_panel: pd.DataFrame | None = None,
    real_cate: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    """Compute raw + scale-residualized window-Qini for all four learners.

    Matches `tests/test_independence.py::TestQiniScaleArtifactAdversarial.residualized_qini`
    line-for-line so that the test and the serialized number are provably
    the same computation.

    Returns
    -------
    {estimator: {"orig": float, "resid": float}} for every estimator whose
    `cate_<estimator>` column is present in `uplift_real_cate.parquet`.
    """
    if real_panel is None:
        real_panel = pd.read_parquet(_PROCESSED / "monthly_panel.parquet")
    if real_cate is None:
        real_cate = pd.read_parquet(_PROCESSED / "uplift_real_cate.parquet")

    keys = ["tehsil", "month_start", "product"]
    m = real_cate.merge(
        real_panel[keys + list(SCALE_FEATURES)], on=keys, how="inner"
    )
    S = StandardScaler().fit_transform(
        m[SCALE_FEATURES].astype(float).fillna(0.0).values
    )
    window_open = (m["window_decay_this_product"].astype(float).values > 0).astype(int)
    Y = m["Y_revenue"].astype(float).values
    T = m["T"].astype(int).values

    out: dict[str, dict[str, float]] = {}
    for est in ESTIMATORS:
        col = f"cate_{est}"
        if col not in m.columns:
            continue
        cate = m[col].astype(float).values
        if not np.isfinite(cate).all():
            continue
        mdl = Ridge(alpha=1.0).fit(S, cate)
        cate_resid = cate - mdl.predict(S)
        q_orig = qini_window_constrained(cate, Y, T, window_open).qini_coefficient
        q_resid = qini_window_constrained(cate_resid, Y, T, window_open).qini_coefficient
        out[est] = {"orig": float(q_orig), "resid": float(q_resid)}
    return out


def serialize_to_uplift_eval(
    result: dict[str, dict[str, float]] | None = None,
    path: Path | None = None,
) -> Path:
    """Merge the computed dict into uplift_eval.json under
    `qini_window_residualized` WITHOUT touching any existing field."""
    if result is None:
        result = compute_residualized_window_qini()
    path = path or (_PROCESSED / "uplift_eval.json")
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["qini_window_residualized"] = result
    path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    result = compute_residualized_window_qini()
    print(json.dumps(result, indent=2))
    path = serialize_to_uplift_eval(result)
    cf_resid = result.get("causal_forest", {}).get("resid")
    log.info("Serialized → %s (causal_forest.resid = %.4f)", path, cf_resid)
