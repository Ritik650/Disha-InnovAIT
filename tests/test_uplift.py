"""
tests/test_uplift.py

L2 pinned tests:
  (1) Synthetic DGP recovery — each estimator must recover the PRE-REGISTERED
      τ_true within the tolerance documented in disha/uplift/dgp.py.
  (2) Window-CATE independence on synthetic — since β_window = 0 in the spec,
      estimators should not hallucinate a strong window effect.
  (3) R-learner Stage-2 correctness — cross-implementation Spearman against
      econml.LinearDML must remain ≥ 0.7 (defends against the Y_tilde-bug
      regression).
  (4) Qini-curve sanity — uplift curves return finite values and the
      window-constrained variant excludes correct rows.

These tests run the actual estimators, so they're slow (~30s total).  They
are the LOAD-BEARING evidence that the L2 engine works as designed.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

warnings.filterwarnings("ignore")

from disha.twin.dgp_gate import AGRONOMIC_CATE_FEATURES
from disha.uplift.dgp import DGP_SPEC_V1, build_synthetic_panel
from disha.uplift.learners import (
    causal_forest,
    r_learner,
    s_learner,
    t_learner,
)
from disha.uplift.train import CONFOUND_FEATURES
from disha.eval.qini import qini_curve, qini_window_constrained


_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED = _ROOT / "data" / "processed"


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures (heavy I/O; session-scoped)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def real_panel():
    p = _PROCESSED / "monthly_panel.parquet"
    if not p.exists():
        pytest.skip("monthly_panel.parquet not built")
    return pd.read_parquet(p)


@pytest.fixture(scope="session")
def synthetic_panel(real_panel):
    return build_synthetic_panel(real_panel, spec=DGP_SPEC_V1)


@pytest.fixture(scope="session")
def synth_r_learner_cate(synthetic_panel):
    return r_learner(
        synthetic_panel,
        x_effect_cols=AGRONOMIC_CATE_FEATURES,
        x_confound_cols=CONFOUND_FEATURES,
        y_col="Y_revenue", t_col="T", seed=42,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pre-registration integrity
# ──────────────────────────────────────────────────────────────────────────────

class TestDgpSpecIntegrity:
    """The pre-registered DGP spec is the load-bearing artifact for L2 claims;
    do not tune coefficients in response to estimator performance."""

    def test_window_coef_is_pre_specified_zero(self):
        assert DGP_SPEC_V1["tau_coefs_on_z"]["window_decay_this_product"] == 0.0
        assert DGP_SPEC_V1["tau_coefs_on_z"]["avg_disease_pressure"] == 0.0

    def test_spec_version_locked(self):
        # If you bump to V2, add a separate spec; do not mutate V1.
        assert DGP_SPEC_V1["name"] == "uplift_synthetic_v1"

    def test_grower_behavioral_coefs_locked(self):
        """The five non-zero CATE coefficients are pre-registered."""
        expected = {
            "pct_offline_attended":      +500.0,
            "pct_smartphone":            +400.0,
            "wa_engagement_rate":        +300.0,
            "avg_farm_size_ha":          -200.0,
            "pct_product_scanned":       +150.0,
        }
        actual = DGP_SPEC_V1["tau_coefs_on_z"]
        for feat, coef in expected.items():
            assert actual[feat] == coef, f"{feat} coefficient mutated: {actual[feat]} != {coef}"


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic τ_true recovery (LOAD-BEARING)
# ──────────────────────────────────────────────────────────────────────────────

class TestSyntheticRecovery:
    """Each estimator must recover the pre-registered τ_true within tolerance.

    Tolerances are looser than the latest seed-42 numbers to allow seed
    variation; they are still tight enough to catch any estimator regression.
    """

    def test_r_learner_recovers_tau(self, synthetic_panel, synth_r_learner_cate):
        tau_true = synthetic_panel["tau_true"].values
        rho, _ = spearmanr(synth_r_learner_cate, tau_true)
        assert rho >= 0.70, (
            f"R-learner Spearman(τ̂, τ_true) = {rho:.3f} < 0.70. "
            "Most likely the Stage-2 prediction bug regression "
            "(predicting Y_tilde = W·β instead of τ̂ = X·β)."
        )
        # Window-CATE: pre-registered as zero, finite-sample noise ≈ -0.15
        rho_w, _ = spearmanr(synth_r_learner_cate,
                              synthetic_panel["window_decay_this_product"].values)
        assert abs(rho_w) <= 0.25, (
            f"R-learner window_decay rho = {rho_w:+.3f} too large; "
            "DGP has β_window = 0, estimator should not hallucinate window effect."
        )

    def test_causal_forest_recovers_tau(self, synthetic_panel):
        cate = causal_forest(
            synthetic_panel,
            x_effect_cols=AGRONOMIC_CATE_FEATURES,
            x_confound_cols=CONFOUND_FEATURES,
            y_col="Y_revenue", t_col="T", seed=42,
            n_estimators=100,  # smaller for test speed; still recovers >= 0.7
            min_samples_leaf=50,
        )
        tau_true = synthetic_panel["tau_true"].values
        rho, _ = spearmanr(cate, tau_true)
        assert rho >= 0.70, (
            f"Causal-Forest Spearman(τ̂, τ_true) = {rho:.3f} < 0.70. "
            "Engine regressed or econml is broken."
        )

    def test_t_learner_works_on_linear_dgp(self, synthetic_panel):
        """T-learner is the optimal estimator on this linear-Gaussian DGP;
        expect r ≥ 0.90."""
        cate = t_learner(
            synthetic_panel,
            x_effect_cols=AGRONOMIC_CATE_FEATURES,
            x_confound_cols=CONFOUND_FEATURES,
            y_col="Y_revenue", t_col="T", seed=42,
        )
        tau_true = synthetic_panel["tau_true"].values
        rho, _ = spearmanr(cate, tau_true)
        assert rho >= 0.90, (
            f"T-learner Spearman(τ̂, τ_true) = {rho:.3f} < 0.90 on linear DGP. "
            "Expected r ≈ 0.97."
        )

    def test_s_learner_known_weakness(self, synthetic_panel):
        """S-learner is expected to UNDER-recover τ on this DGP because Ridge
        shrinks the T coefficient (the 'drowning out' phenomenon).  We test
        that it doesn't go NEGATIVE — that would indicate a logic bug, not
        the documented weakness."""
        cate = s_learner(
            synthetic_panel,
            x_effect_cols=AGRONOMIC_CATE_FEATURES,
            x_confound_cols=CONFOUND_FEATURES,
            y_col="Y_revenue", t_col="T", seed=42,
        )
        tau_true = synthetic_panel["tau_true"].values
        rho, _ = spearmanr(cate, tau_true)
        # Allow weak positive; just guard against sign-flip bug.
        assert rho > -0.20, f"S-learner went negative ({rho:.3f}); logic bug"


# ──────────────────────────────────────────────────────────────────────────────
# R-learner Stage-2 correctness (cross-implementation check)
# ──────────────────────────────────────────────────────────────────────────────

class TestRLearnerStage2Correctness:
    """Defend against the Y_tilde-prediction bug returning by comparing our
    R-learner CATE rankings against econml.LinearDML on the same synthetic
    panel.  If our prediction logic regresses to W·β, cross-impl Spearman
    will collapse far below 0.7."""

    def test_cross_implementation_agreement(self, synthetic_panel, synth_r_learner_cate):
        from econml.dml import LinearDML

        X = synthetic_panel[AGRONOMIC_CATE_FEATURES].astype(float).fillna(0.0).values
        confound_only = [c for c in CONFOUND_FEATURES
                         if c not in set(AGRONOMIC_CATE_FEATURES)]
        W = synthetic_panel[confound_only].astype(float).fillna(0.0).values
        Y = synthetic_panel["Y_revenue"].astype(float).values
        T = synthetic_panel["T"].astype(int).values

        est = LinearDML(
            model_y=Ridge(alpha=1.0),
            model_t=LogisticRegression(max_iter=500),
            discrete_treatment=True,
            cv=5,
            random_state=42,
        )
        est.fit(Y=Y, T=T, X=X, W=W)
        cate_econml = np.asarray(est.effect(X)).reshape(-1)

        rho_cross, _ = spearmanr(synth_r_learner_cate, cate_econml)
        assert rho_cross >= 0.70, (
            f"R-learner cross-impl Spearman vs econml.LinearDML = {rho_cross:.3f} "
            "< 0.70.  Most likely cause: Stage-2 prediction bug regression "
            "(predicting W·β instead of X·β)."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Qini sanity
# ──────────────────────────────────────────────────────────────────────────────

class TestQini:
    def test_qini_returns_finite(self, synthetic_panel, synth_r_learner_cate):
        res = qini_curve(
            cate=synth_r_learner_cate,
            y=synthetic_panel["Y_revenue"].values,
            t=synthetic_panel["T"].values,
        )
        assert np.isfinite(res.qini_coefficient)
        assert np.isfinite(res.auuc)
        assert res.curve_x.shape == res.curve_y.shape
        assert res.n == len(synthetic_panel)

    def test_window_constrained_excludes_closed(self, synthetic_panel, synth_r_learner_cate):
        window_open = (synthetic_panel["window_decay_this_product"].values > 0).astype(int)
        res = qini_window_constrained(
            cate=synth_r_learner_cate,
            y=synthetic_panel["Y_revenue"].values,
            t=synthetic_panel["T"].values,
            window_open=window_open,
        )
        assert res.n == int(window_open.sum()), (
            "Constrained Qini must operate on exactly the in-window subset"
        )
