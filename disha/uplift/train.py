"""
disha.uplift.train — L2 driver.

Runs all four estimators on TWO panels:
  (1) synthetic — pre-registered DGP, ground-truth CATE known; recovery is
      the PRIMARY technical evidence for the engine.
  (2) real      — monthly_panel.parquet; CATE r ≈ 0.41 is a conservative
      lower bound, reported with structural-artifact gap as proactive caveat.

Identification choice
---------------------
Meta-learners run on raw Y_revenue + binary T.  Confounding is handled by
cross-fit Stage-1 residualization on the confound set
(lag_revenue_1m + n_retailers_in_tehsil + n_growers + effect modifiers).
The two-way-FE DML ATE in disha.twin.lift_probe remains the authoritative
ATE source; this module focuses on CATE recovery and decision quality.

Outputs (data/processed/):
  uplift_synthetic_cate.parquet     per-row CATE from each estimator + tau_true
  uplift_real_cate.parquet          per-row CATE from each estimator (real Y)
  uplift_eval.json                  recovery + Qini metrics for both panels
"""
from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from disha.twin.dgp_gate import AGRONOMIC_CATE_FEATURES
from disha.uplift.dgp import DGP_SPEC_V1, build_synthetic_panel, true_ate
from disha.uplift.learners import ESTIMATORS
from disha.eval.qini import (
    evaluate_cate_recovery,
    qini_curve,
    qini_window_constrained,
)

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _ROOT / "data" / "processed"

# Confounders used ONLY for Stage-1 residualization (NEVER as effect modifiers).
# These are baseline-wealth / capacity proxies that predict Y and T but were
# excluded from AGRONOMIC_CATE_FEATURES by the pre-L2 corrections.
CONFOUND_FEATURES = [
    "lag_revenue_1m",
    "n_retailers_in_tehsil",
    "n_growers",
]


# ── runner ────────────────────────────────────────────────────────────────────

def _run_all_estimators(
    panel: pd.DataFrame,
    label: str,
    seed: int = 42,
) -> dict:
    out: dict[str, np.ndarray] = {}
    for name, fn in ESTIMATORS.items():
        t0 = time.time()
        log.info("[%s] fitting %s …", label, name)
        try:
            cate = fn(
                panel,
                x_effect_cols=AGRONOMIC_CATE_FEATURES,
                x_confound_cols=CONFOUND_FEATURES,
                y_col="Y_revenue",
                t_col="T",
                seed=seed,
            )
        except Exception as e:
            log.warning("[%s] %s FAILED: %s", label, name, e)
            cate = np.full(len(panel), np.nan)
        finite = np.isfinite(cate)
        if finite.any():
            log.info(
                "[%s] %s done in %.1fs  mean=%.1f std=%.1f  (finite: %d/%d)",
                label, name, time.time() - t0,
                float(np.mean(cate[finite])), float(np.std(cate[finite])),
                int(finite.sum()), len(cate),
            )
        else:
            log.info("[%s] %s done in %.1fs  (no finite values)",
                     label, name, time.time() - t0)
        out[name] = cate
    return out


def run_l2(seed: int = 42) -> dict:
    panel = pd.read_parquet(_PROCESSED / "monthly_panel.parquet")
    log.info("Loaded real panel: %d rows × %d cols", len(panel), len(panel.columns))

    # Pre-registered synthetic DGP
    synth = build_synthetic_panel(panel, spec=DGP_SPEC_V1)
    log.info(
        "Synthetic panel built. true_ate=%.2f  σ(tau_true)=%.1f  (DGP=%s)",
        true_ate(synth), synth["tau_true"].std(), DGP_SPEC_V1["name"],
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        synth_cates = _run_all_estimators(synth, label="synth", seed=seed)
        real_cates = _run_all_estimators(panel, label="real", seed=seed)

    # ── Per-row CATE parquets ────────────────────────────────────────────────
    keys = ["tehsil", "month_start", "month_index", "product"]
    synth_out = synth[keys + ["tau_true", "Y_revenue", "T",
                              "window_decay_this_product"]].copy()
    for name, c in synth_cates.items():
        synth_out[f"cate_{name}"] = c
    synth_out.to_parquet(_PROCESSED / "uplift_synthetic_cate.parquet", index=False)

    real_out = panel[keys + ["Y_revenue", "T",
                             "window_decay_this_product"]].copy()
    for name, c in real_cates.items():
        real_out[f"cate_{name}"] = c
    real_out.to_parquet(_PROCESSED / "uplift_real_cate.parquet", index=False)

    # ── Recovery metrics (synthetic only) ────────────────────────────────────
    from scipy.stats import spearmanr
    recovery: dict[str, dict] = {}
    for name, cate in synth_cates.items():
        if not np.isfinite(cate).any():
            recovery[name] = {"error": "no finite predictions"}
            continue
        rec = evaluate_cate_recovery(cate, synth["tau_true"].values)
        rho_w, _ = spearmanr(cate, synth["window_decay_this_product"].values)
        rec["window_cate_spearman"] = (
            float(rho_w) if rho_w is not None and not np.isnan(rho_w) else 0.0
        )
        recovery[name] = rec
        log.info(
            "[recovery synth/%s] r=%+.3f  ATE_bias=%+0.1f  MSE=%.0f  win_rho=%+.3f",
            name, rec["spearman_r"], rec["ate_bias"], rec["mse"], rec["window_cate_spearman"],
        )

    # ── Qini metrics (both panels) ───────────────────────────────────────────
    qini: dict[str, dict] = {"synthetic": {}, "real": {}}
    window_open_real = (panel["window_decay_this_product"].values > 0).astype(int)
    window_open_synth = (synth["window_decay_this_product"].values > 0).astype(int)
    for name in ESTIMATORS.keys():
        if not np.isfinite(synth_cates[name]).any() or not np.isfinite(real_cates[name]).any():
            continue
        try:
            qs_all = qini_curve(synth_cates[name], synth["Y_revenue"].values, synth["T"].values)
            qs_win = qini_window_constrained(
                synth_cates[name], synth["Y_revenue"].values, synth["T"].values,
                window_open_synth,
            )
            qr_all = qini_curve(real_cates[name], panel["Y_revenue"].values, panel["T"].values)
            qr_win = qini_window_constrained(
                real_cates[name], panel["Y_revenue"].values, panel["T"].values,
                window_open_real,
            )
            qini["synthetic"][name] = {
                "qini_all": round(qs_all.qini_coefficient, 4),
                "qini_window_only": round(qs_win.qini_coefficient, 4),
            }
            qini["real"][name] = {
                "qini_all": round(qr_all.qini_coefficient, 4),
                "qini_window_only": round(qr_win.qini_coefficient, 4),
            }
            log.info(
                "[qini %s] synth all=%+.3f win=%+.3f | real all=%+.3f win=%+.3f",
                name,
                qini["synthetic"][name]["qini_all"],
                qini["synthetic"][name]["qini_window_only"],
                qini["real"][name]["qini_all"],
                qini["real"][name]["qini_window_only"],
            )
        except Exception as e:
            log.warning("[qini %s] failed: %s", name, e)

    summary = {
        "seed": seed,
        "dgp_spec": DGP_SPEC_V1,
        "feature_set_effect": list(AGRONOMIC_CATE_FEATURES),
        "feature_set_confound": list(CONFOUND_FEATURES),
        "n_rows": len(panel),
        "synthetic_recovery": recovery,
        "qini": qini,
    }
    out_path = _PROCESSED / "uplift_eval.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Saved → %s", out_path)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
    run_l2()
