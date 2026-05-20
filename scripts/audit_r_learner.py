"""
Audit: compare our R-learner Stage-2 against econml.LinearDML on the
pre-registered synthetic DGP.

Math reference (Nie & Wager 2021, R-learner formulation):
  Stage 1: cross-fit  m̂(X) = Ê[Y|X],  ê(X) = Ê[T|X]
           Y_tilde = Y − m̂(X),  T_tilde = T − ê(X)
  Stage 2: τ̂ = argmin_g  Σ (Y_tilde_i − T_tilde_i · g(X_i))²
  For linear g(X) = X·β:
           argmin_β  Σ (Y_tilde_i − (T_tilde_i · X_i)·β)²
                  = argmin_β  Σ (Y_tilde_i − W_i · β)²    where W = T_tilde · X
  Solution: β̂ = (W'W + λI)⁻¹ W' Y_tilde
  Prediction at new X:  τ̂(X) = X · β̂           ← function value
                NOT:    τ̂(X) = (T_tilde · X) · β̂  ← Y_tilde value (the original bug)

This script:
  1. Builds the synthetic panel from the pre-registered DGP
  2. Runs our fixed r_learner
  3. Runs econml.LinearDML (which is the standard implementation of the same
     identification strategy)
  4. Reports Spearman recovery vs τ_true for both
  5. Reports Spearman(ours, econml) cross-implementation agreement
"""
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression

from disha.twin.dgp_gate import AGRONOMIC_CATE_FEATURES
from disha.uplift.dgp import DGP_SPEC_V1, build_synthetic_panel
from disha.uplift.learners import r_learner
from disha.uplift.train import CONFOUND_FEATURES

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

panel = pd.read_parquet("data/processed/monthly_panel.parquet")
synth = build_synthetic_panel(panel, spec=DGP_SPEC_V1)
tau_true = synth["tau_true"].values

# Our fixed R-learner
cate_ours = r_learner(
    synth,
    x_effect_cols=AGRONOMIC_CATE_FEATURES,
    x_confound_cols=CONFOUND_FEATURES,
    y_col="Y_revenue",
    t_col="T",
    seed=42,
)

# econml LinearDML — same R-learner identification, library implementation.
# Use the same X (effect) and W (confound-only).
from econml.dml import LinearDML

X_eff = synth[AGRONOMIC_CATE_FEATURES].astype(float).fillna(0.0).values
confound_only = [c for c in CONFOUND_FEATURES if c not in set(AGRONOMIC_CATE_FEATURES)]
W = synth[confound_only].astype(float).fillna(0.0).values
Y = synth["Y_revenue"].astype(float).values
T = synth["T"].astype(int).values

est = LinearDML(
    model_y=Ridge(alpha=1.0),
    model_t=LogisticRegression(max_iter=500),
    discrete_treatment=True,
    cv=5,
    random_state=42,
)
est.fit(Y=Y, T=T, X=X_eff, W=W)
cate_econml = np.asarray(est.effect(X_eff)).reshape(-1)

# Diagnostics
def summarize(name, cate):
    rho, _ = spearmanr(cate, tau_true)
    bias = cate.mean() - tau_true.mean()
    return f"{name:25s}  r={rho:+.4f}  ATE_bias={bias:+8.1f}  std={cate.std():.1f}"

print()
print("=" * 70)
print("R-LEARNER AUDIT - synthetic DGP (tau_true known, pre-registered)")
print("=" * 70)
print(f"n rows: {len(synth)}")
print(f"std(tau_true) = {tau_true.std():.1f}")
print(f"ATE_true      = {tau_true.mean():+.2f}")
print()
print(summarize("Our fixed r_learner",   cate_ours))
print(summarize("econml.LinearDML",      cate_econml))
print()
rho_cross, _ = spearmanr(cate_ours, cate_econml)
print(f"Cross-implementation agreement: Spearman(ours, econml) = {rho_cross:+.4f}")
print()
print("Interpretation:")
print("  - Both should recover tau_true with similar r (>= 0.5 on linear DGP).")
print("  - Cross-implementation Spearman should be >= 0.7 if same identification.")
print("  - The ORIGINAL buggy version gave r = -0.22 - clearly inconsistent")
print("    with econml's standard implementation.")
