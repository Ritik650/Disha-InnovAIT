"""
disha.uplift.learners — T-, S-, R-learner + Causal-Forest CATE estimators.

Contract
--------
Each estimator takes:
  panel         : pd.DataFrame
  x_effect_cols : effect-modifier features (the LOCKED 7 drivers; what
                  τ(X) is allowed to depend on)
  x_confound_cols (optional, default = x_effect_cols)
                : Stage-1 residualization controls (baseline predictors of Y
                  and T that we want to partial out before identifying τ).
                  This is where lag_revenue_1m / n_retailers belong — they
                  are confounders of T,Y, NOT legitimate effect modifiers.
  y_col, t_col  : raw outcome and binary treatment column names
  seed, n_folds : cross-fit controls

Returns: (n,) np.ndarray of held-out CATE predictions.

Identification choice
---------------------
We do NOT pre-demean Y, T by two-way FE in this layer.  Each estimator does
its own cross-fit Stage-1 residualization on x_confound_cols.  This is the
standard meta-learner setup and is what is required for valid CATE
recovery on the synthetic DGP.

The two-way-FE ATE estimator in disha.twin.lift_probe (DML+FE) remains the
authoritative source for the ATE narrative; this module is for CATE
identification specifically.
"""
from __future__ import annotations

import logging
import warnings
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _arr(panel: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    return panel[list(cols)].astype(float).fillna(0.0).values


def _yt(panel: pd.DataFrame, y_col: str, t_col: str) -> tuple[np.ndarray, np.ndarray]:
    y = panel[y_col].astype(float).values
    t = panel[t_col].astype(int).values
    return y, t


def _resolve_confound(
    x_effect_cols: Iterable[str],
    x_confound_cols: Optional[Iterable[str]],
) -> list[str]:
    if x_confound_cols is None:
        return list(x_effect_cols)
    out = list(x_confound_cols)
    # Always include effect-modifier columns so Stage 1 captures any signal
    # they carry about Y or T; the user-supplied confounders are additive.
    for c in x_effect_cols:
        if c not in out:
            out.append(c)
    return out


# ── T-learner (baseline) ──────────────────────────────────────────────────────

def t_learner(
    panel: pd.DataFrame,
    x_effect_cols: Iterable[str],
    x_confound_cols: Optional[Iterable[str]] = None,
    y_col: str = "Y_revenue",
    t_col: str = "T",
    seed: int = 42,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> np.ndarray:
    """T-learner: separate Ridge(μ̂₀, μ̂₁); τ̂(x) = μ̂₁ − μ̂₀.  Cross-fit.

    Uses the UNION of confound + effect columns for both arms (so both arms
    see the same covariates and τ̂ is well-defined on x_effect_cols).
    """
    feat_cols = _resolve_confound(x_effect_cols, x_confound_cols)
    X = _arr(panel, feat_cols)
    Y, T = _yt(panel, y_col, t_col)
    n = len(Y)
    cate = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, val in kf.split(np.arange(n)):
        tr_t = T[tr]
        m1_tr = tr[tr_t == 1]
        m0_tr = tr[tr_t == 0]
        if len(m1_tr) < 10 or len(m0_tr) < 10:
            continue
        sc = StandardScaler().fit(X[tr])
        Xv = sc.transform(X[val])
        mu1 = Ridge(alpha=alpha).fit(sc.transform(X[m1_tr]), Y[m1_tr])
        mu0 = Ridge(alpha=alpha).fit(sc.transform(X[m0_tr]), Y[m0_tr])
        cate[val] = mu1.predict(Xv) - mu0.predict(Xv)
    return cate


# ── S-learner (baseline) ──────────────────────────────────────────────────────

def s_learner(
    panel: pd.DataFrame,
    x_effect_cols: Iterable[str],
    x_confound_cols: Optional[Iterable[str]] = None,
    y_col: str = "Y_revenue",
    t_col: str = "T",
    seed: int = 42,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> np.ndarray:
    """S-learner: single Ridge on [X, T]; τ̂(x) = ŷ(x, 1) − ŷ(x, 0)."""
    feat_cols = _resolve_confound(x_effect_cols, x_confound_cols)
    X = _arr(panel, feat_cols)
    Y, T = _yt(panel, y_col, t_col)
    n = len(Y)
    cate = np.zeros(n)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, val in kf.split(np.arange(n)):
        sc = StandardScaler().fit(X[tr])
        Xt = sc.transform(X[tr])
        Xv = sc.transform(X[val])
        XT_tr = np.hstack([Xt, T[tr].reshape(-1, 1)])
        m = Ridge(alpha=alpha).fit(XT_tr, Y[tr])
        XT_v1 = np.hstack([Xv, np.ones((len(val), 1))])
        XT_v0 = np.hstack([Xv, np.zeros((len(val), 1))])
        cate[val] = m.predict(XT_v1) - m.predict(XT_v0)
    return cate


# ── R-learner ─────────────────────────────────────────────────────────────────

def r_learner(
    panel: pd.DataFrame,
    x_effect_cols: Iterable[str],
    x_confound_cols: Optional[Iterable[str]] = None,
    y_col: str = "Y_revenue",
    t_col: str = "T",
    seed: int = 42,
    n_folds: int = 5,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    R-learner with explicit Stage-1 confound set + Stage-2 effect-modifier set.

    Stage 1: cross-fit
      m̂(X_confound) := Ê[Y | X_confound]
      ê(X_confound) := Ê[T | X_confound]
      Y_tilde = Y − m̂;  T_tilde = T − ê
    Stage 2: cross-fit
      regress Y_tilde on (T_tilde · X_effect) → β̂
      τ̂(x) = X_effect(x) · β̂      (predicted via held-out fold)
    """
    confound_cols = _resolve_confound(x_effect_cols, x_confound_cols)
    X_conf = _arr(panel, confound_cols)
    X_eff = _arr(panel, x_effect_cols)
    Y, T = _yt(panel, y_col, t_col)
    n = len(Y)

    # Stage 1
    Y_tilde = np.zeros(n)
    T_tilde = np.zeros(n)
    kf1 = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, val in kf1.split(np.arange(n)):
        sc = StandardScaler().fit(X_conf[tr])
        Xt = sc.transform(X_conf[tr])
        Xv = sc.transform(X_conf[val])
        Y_tilde[val] = Y[val] - Ridge(alpha=alpha).fit(Xt, Y[tr]).predict(Xv)
        T_tilde[val] = T[val].astype(float) - Ridge(alpha=alpha).fit(Xt, T[tr]).predict(Xv)

    # Stage 2 — Robinson loss:
    #     min_β  Σ (Y_tilde_i − T_tilde_i · X_eff_i · β)²
    # → regress Y_tilde on W = T_tilde · X_eff (no intercept) to get β,
    # then τ̂(X_val) = X_val · β   (NOT W_val · β; W_val rescales by T_tilde
    # which corrupts the τ ranking — that was the original bug).
    W = T_tilde.reshape(-1, 1) * X_eff
    col_mask = W.std(axis=0) > 0
    if not col_mask.any():
        return np.zeros(n)
    W = W[:, col_mask]
    X_eff_kept = X_eff[:, col_mask]
    cate = np.zeros(n)
    kf2 = KFold(n_splits=n_folds, shuffle=True, random_state=seed + 500)
    for tr, val in kf2.split(np.arange(n)):
        mdl = Ridge(alpha=alpha, fit_intercept=False).fit(W[tr], Y_tilde[tr])
        cate[val] = X_eff_kept[val] @ mdl.coef_
    return cate


# ── Causal Forest (econml.CausalForestDML) ────────────────────────────────────

def causal_forest(
    panel: pd.DataFrame,
    x_effect_cols: Iterable[str],
    x_confound_cols: Optional[Iterable[str]] = None,
    y_col: str = "Y_revenue",
    t_col: str = "T",
    seed: int = 42,
    n_estimators: int = 200,
    min_samples_leaf: int = 50,
) -> np.ndarray:
    """
    CausalForestDML with X = effect modifiers, W = confound-only controls.

    econml partials out (Y, T) ~ W using model_y/model_t (default Ridge here),
    then the honest forest is grown on X.  T must be binary {0,1}.
    """
    from econml.dml import CausalForestDML
    from sklearn.linear_model import LogisticRegression

    X = _arr(panel, x_effect_cols)
    # W = confound-only (excluding effect modifiers) — these are partialled out.
    confound_only = [c for c in (x_confound_cols or []) if c not in set(x_effect_cols)]
    W = _arr(panel, confound_only) if confound_only else None
    Y, T = _yt(panel, y_col, t_col)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="econml")
        warnings.filterwarnings("ignore", category=FutureWarning)
        est = CausalForestDML(
            model_y=Ridge(alpha=1.0),
            model_t=LogisticRegression(max_iter=500),
            discrete_treatment=True,
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            cv=5,
            random_state=seed,
        )
        est.fit(Y=Y, T=T, X=X, W=W)
        cate = est.effect(X)

    return np.asarray(cate).reshape(-1)


# ── Estimator registry ────────────────────────────────────────────────────────

ESTIMATORS = {
    "t_learner": t_learner,
    "s_learner": s_learner,
    "r_learner": r_learner,
    "causal_forest": causal_forest,
}
