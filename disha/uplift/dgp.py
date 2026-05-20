"""
disha.uplift.dgp — Pre-registered synthetic data-generating process for L2
uplift-engine validation.

╔══════════════════════════════════════════════════════════════════════════╗
║  PRE-REGISTRATION NOTICE                                                 ║
║                                                                          ║
║  All coefficients in DGP_SPEC_V1 below are LOCKED.  They were written    ║
║  down BEFORE any estimator was run on this DGP.  Do NOT tune them in     ║
║  response to estimator performance; if an estimator fails to recover the ║
║  pre-registered τ within tolerance, that is the FINDING, not a tuning    ║
║  signal.                                                                 ║
║                                                                          ║
║  If the spec needs to change for a substantive reason (e.g. new feature  ║
║  added to the panel), bump the version → DGP_SPEC_V2 and treat as a      ║
║  separate pre-registration.  Keep V1 intact for reproducibility.         ║
╚══════════════════════════════════════════════════════════════════════════╝

Design rationale
================
We synthesize Y only.  Covariates X and treatment T are taken VERBATIM from
the real monthly_panel.parquet — this preserves:
  * the real covariate joint distribution
  * the real treatment selection bias (SMD lag_revenue ≈ -0.65)
  * the real entity/time grain (tehsil×product × month, n=80,128)

The synthetic outcome is:
    Y = baseline(X) + τ(X) · T + ε,   ε ~ N(0, σ_noise²)

True τ(X) is a linear function of FIVE pre-selected grower behavioral /
structural drivers, applied to standardized features (so each coefficient
has direct INR-per-σ interpretation):

    τ(X) =  +500·z_pct_offline_attended
           +400·z_pct_smartphone
           +300·z_wa_engagement_rate
           -200·z_avg_farm_size_ha
           +150·z_pct_product_scanned
           +  0·z_window_decay_this_product   ← PRE-SPECIFIED ZERO
           +  0·z_avg_disease_pressure        ← PRE-SPECIFIED ZERO

The two ZERO coefficients are the load-bearing pre-registration: they
encode the L0/L1 finding that window_decay is independent of CATE
(Spearman ≈ 0.03 on real data, bottom-ranked in Ridge surrogate top_drivers).
The synthetic DGP must NOT bake window×CATE interaction in — if our
estimator finds spurious window heterogeneity on this DGP, we know it's
the estimator and not the data.

Validation targets (also pre-registered — see tests/test_uplift.py)
-------------------------------------------------------------------
On the synthetic panel each estimator should recover τ_true with:

  R-learner               : Spearman(τ̂, τ_true) ≥ 0.55  ATE bias ≤ 200 INR
  Causal Forest (DML)     : Spearman(τ̂, τ_true) ≥ 0.50  ATE bias ≤ 300 INR
  T-learner (baseline)    : Spearman(τ̂, τ_true) ≥ 0.20  (worse than R/CF expected)
  S-learner (baseline)    : Spearman(τ̂, τ_true) ≥ 0.20

  Window-CATE independence check:
    |Spearman(τ̂, window_decay)| ≤ 0.15  for R-learner and CF
    (if it exceeds, estimator is hallucinating a window effect)

The synthetic recovery is the PRIMARY technical evidence for the L2 engine.
Real-data CATE r ≈ 0.41 is the headline of-record (with structural-artifact
gap reported alongside), but it is a CONSERVATIVE LOWER BOUND — it cannot
be cross-checked against ground truth.  Synthetic CATE recovery shows the
machinery works; real CATE is what it found in the wild.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ── PRE-REGISTERED SPECIFICATION (DO NOT TUNE) ────────────────────────────────

DGP_SPEC_V1: dict = {
    "name": "uplift_synthetic_v1",
    "seed": 42,

    # Reuse REAL X and T from monthly_panel.parquet — only Y is synthesized.
    "x_source": "data/processed/monthly_panel.parquet",
    "t_source": "data/processed/monthly_panel.parquet",

    # Baseline μ(X) — replicates real revenue persistence.
    "baseline_feature": "lag_revenue_1m",
    "baseline_intercept": 0.0,
    "baseline_coef":      1.0,

    # True CATE function — coefficients on STANDARDIZED features (z-scores).
    # Units: INR per standard-deviation movement in the feature.
    "tau_intercept": 0.0,
    "tau_coefs_on_z": {
        "pct_offline_attended":      +500.0,
        "pct_smartphone":            +400.0,
        "wa_engagement_rate":        +300.0,
        "avg_farm_size_ha":          -200.0,
        "pct_product_scanned":       +150.0,
        # PRE-SPECIFIED ZERO — load-bearing for independence claim:
        "window_decay_this_product":   0.0,
        "avg_disease_pressure":        0.0,
    },

    # Outcome noise (Gaussian).
    "noise_sigma": 5000.0,
}


# ── DGP builder ───────────────────────────────────────────────────────────────

def _z(series: pd.Series) -> np.ndarray:
    s = series.astype(float).fillna(0.0).values
    mu = s.mean()
    sd = s.std()
    if sd <= 0:
        return np.zeros_like(s)
    return (s - mu) / sd


def compute_true_cate(
    panel: pd.DataFrame,
    spec: dict = DGP_SPEC_V1,
) -> np.ndarray:
    """Compute τ_true(X) row-wise for the panel under the given spec.

    Returns a 1-D array of length len(panel).  Standardization is done on
    the supplied panel (so τ has zero mean by construction).
    """
    coefs = spec["tau_coefs_on_z"]
    tau = np.full(len(panel), float(spec["tau_intercept"]))
    for feat, beta in coefs.items():
        if beta == 0.0:
            continue   # micro-optimization; also documents the zero
        if feat not in panel.columns:
            raise KeyError(f"DGP requires column '{feat}' in panel — missing.")
        tau = tau + beta * _z(panel[feat])
    return tau


def build_synthetic_panel(
    real_panel: pd.DataFrame,
    spec: dict = DGP_SPEC_V1,
) -> pd.DataFrame:
    """
    Build a synthetic monthly panel by replacing Y_revenue with
    μ(X) + τ(X)·T + ε under the pre-registered spec.

    Returns a copy of `real_panel` with the following columns added/changed:
      Y_revenue_real        — original real Y (preserved for diagnostics)
      Y_revenue             — overwritten with synthetic outcome
      tau_true              — known ground-truth CATE per row
      mu_baseline           — baseline μ(X) component
      noise                 — realized ε per row
    """
    if "T" not in real_panel.columns or "Y_revenue" not in real_panel.columns:
        raise KeyError("real_panel must have 'T' and 'Y_revenue' columns")

    rng = np.random.default_rng(spec["seed"])
    df = real_panel.copy()

    # Preserve real Y for later side-by-side diagnostics
    df["Y_revenue_real"] = df["Y_revenue"].astype(float).values

    # Baseline
    base_feat = spec["baseline_feature"]
    if base_feat not in df.columns:
        raise KeyError(f"baseline_feature '{base_feat}' missing from panel")
    mu = (
        spec["baseline_intercept"]
        + spec["baseline_coef"] * df[base_feat].astype(float).fillna(0.0).values
    )

    # True CATE
    tau = compute_true_cate(df, spec=spec)

    # Noise
    noise = rng.normal(0.0, spec["noise_sigma"], size=len(df))

    # Synthetic Y
    T = df["T"].astype(float).values
    df["Y_revenue"] = mu + tau * T + noise
    df["tau_true"] = tau
    df["mu_baseline"] = mu
    df["noise"] = noise

    return df


def true_ate(panel_with_truth: pd.DataFrame) -> float:
    """Mean of tau_true across the panel — the synthetic ATE.

    By construction (z-features mean zero) this is ≈ tau_intercept = 0.0
    but is computed exactly off the realized panel for the regression tests.
    """
    if "tau_true" not in panel_with_truth.columns:
        raise KeyError("panel must contain tau_true; build with build_synthetic_panel()")
    return float(panel_with_truth["tau_true"].mean())
