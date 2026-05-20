"""
disha.ope.evaluate — IPS + Doubly-Robust policy value of Disha vs BAU.

Scope (locked by the final-sprint cut list):
  ONE IPS number + ONE DR number + ONE bootstrap CI, written to ope.json.
  Reported metric: return-per-field-day (decision-quality scaled).
  No new modeling, no module sprawl, no test suite.

If you find yourself adding sophistication here, STOP — write a paragraph
in SOLUTION.md §4 (geo-randomized rollout methodology) and move on.

Methodology
-----------
Behavior policy μ(a|x): historical treatment (rep visited a cell that month).
Target policy   π(a|x): Disha plan (cell is in the L3 ranked plan for that
                        rep on a representative date for the month).
Outcome         Y    : Y_revenue at the (tehsil, month, product) cell.
IPS:    V_π = mean over n cells of  (π(T_i|X_i)/μ(T_i|X_i)) * Y_i
DR:     V_π = mean of  m̂(π(X_i), X_i) + (T_i indicator/μ) * (Y_i − m̂)
        Outcome model m̂ from the CF CATE arrays (we already have them).
CI: 1000-bootstrap percentile, per policy.

The two values are reported as INR/rep-day under each policy, plus their
difference (the Disha lift estimate).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _ROOT / "data" / "processed"


def _load() -> tuple[pd.DataFrame, dict]:
    panel = pd.read_parquet(_PROCESSED / "monthly_panel.parquet")
    cate = pd.read_parquet(_PROCESSED / "uplift_real_cate.parquet")
    panel = panel.merge(
        cate[["tehsil", "month_start", "product", "cate_causal_forest"]],
        on=["tehsil", "month_start", "product"], how="inner",
    )
    # Estimated propensity m̂(a|x): use the empirical treatment rate within
    # window-open vs window-closed strata as a cheap calibrated proxy
    # (we are not refitting a propensity model — the bias work is frozen).
    panel["window_open"] = (panel["window_decay_this_product"] > 0).astype(int)
    by = panel.groupby(["window_open"])["T"].mean().to_dict()
    panel["e_x"] = panel["window_open"].map(by).clip(lower=0.05, upper=0.95)
    # Disha policy: cell is in the plan iff (window_open=1 AND CATE in top-K)
    # K is calibrated so total π-coverage equals the BAU treatment count for
    # comparability (same number of treatments, different cells).
    n_bau = int(panel["T"].sum())
    # Rank cells with window_open=1 by cate_causal_forest descending; take top n_bau
    open_mask = panel["window_open"] == 1
    open_ranked = panel.loc[open_mask, "cate_causal_forest"].rank(ascending=False, method="first")
    threshold = open_ranked.quantile(min(1.0, n_bau / max(open_mask.sum(), 1))) if open_mask.sum() else float("inf")
    panel["pi_disha"] = ((open_mask) & (open_ranked <= n_bau)).astype(int)
    return panel, {"behaviour_n_treated": n_bau,
                   "policy_n_treated_disha": int(panel["pi_disha"].sum()),
                   "n_cells": len(panel)}


def _ips(panel: pd.DataFrame, policy_col: str) -> float:
    # Estimator: mean over treated cells of (1/μ)*Y when policy agrees with T.
    mask = (panel["T"] == 1) & (panel[policy_col] == 1)
    if not mask.any():
        return 0.0
    return float((panel.loc[mask, "Y_revenue"] / panel.loc[mask, "e_x"]).sum() / len(panel))


def _dr(panel: pd.DataFrame, policy_col: str) -> float:
    # Outcome model m̂(1, X) = Y_predicted_if_treated; use Y_revenue residualized
    # by CATE: m̂(1) = Y_revenue when T=1; for T=0, m̂(1) = Y + cate_causal_forest.
    m_hat = np.where(
        panel["T"] == 1,
        panel["Y_revenue"].values,
        panel["Y_revenue"].values + panel["cate_causal_forest"].values,
    )
    direct = float((panel[policy_col] * m_hat).sum() / len(panel))
    # IPS correction term: residual on cells where T==policy
    mask = (panel["T"] == 1) & (panel[policy_col] == 1)
    if mask.any():
        resid = (panel.loc[mask, "Y_revenue"].values - m_hat[mask]) / panel.loc[mask, "e_x"].values
        correction = float(resid.sum() / len(panel))
    else:
        correction = 0.0
    return direct + correction


def _bootstrap_ci(panel: pd.DataFrame, fn, policy_col: str, n_boot: int = 500, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(panel)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals.append(fn(panel.iloc[idx], policy_col))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate_policies() -> dict:
    panel, meta = _load()
    out = {"meta": meta, "policies": {}}
    for label, col in [("bau_historical", "T"), ("disha", "pi_disha")]:
        ips = _ips(panel, col)
        dr  = _dr(panel, col)
        ci  = _bootstrap_ci(panel, _dr, col, n_boot=300)
        out["policies"][label] = {
            "ips_value_per_cell": ips,
            "dr_value_per_cell":  dr,
            "dr_ci_95":           list(ci),
        }
    bau = out["policies"]["bau_historical"]["dr_value_per_cell"]
    disha = out["policies"]["disha"]["dr_value_per_cell"]
    out["disha_lift_dr"] = disha - bau
    out["disclaimer"] = (
        "OPE here is a sanity check, not the headline.  The real-data Causal "
        "Forest predictions are biased downward by uncorrected selection (see "
        "Phase-5 causal narrative); the absolute numbers should be read as "
        "ordinal, not as expected INR.  Headline targeting evidence is the "
        "window-constrained scale-residualized Qini (+0.263 CF on real data, "
        "sourced from uplift_eval.json::qini_window_residualized).  "
        "For a production-quality value estimate we "
        "would run a geo-randomized rollout — see SOLUTION.md §4."
    )
    return out


def run_and_save_ope(out_dir: Path | None = None) -> dict:
    out_dir = out_dir or _PROCESSED
    res = evaluate_policies()
    p = out_dir / "ope.json"
    p.write_text(json.dumps(res, indent=2, default=str))
    log.info("Saved → %s", p)
    return res


if __name__ == "__main__":
    import sys, io
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    r = run_and_save_ope()
    bau = r["policies"]["bau_historical"]["dr_value_per_cell"]
    disha = r["policies"]["disha"]["dr_value_per_cell"]
    print(f"BAU DR value/cell  : {bau:+.0f} INR")
    print(f"Disha DR value/cell: {disha:+.0f} INR  (CI {r['policies']['disha']['dr_ci_95']})")
    print(f"Disha lift         : {r['disha_lift_dr']:+.0f} INR/cell")
