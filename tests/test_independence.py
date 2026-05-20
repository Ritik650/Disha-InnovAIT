"""
tests/test_independence.py

ADVERSARIAL TESTS — not pinning tests.  These exist to catch circular
evidence:  "the R-learner says CATE is independent of window, therefore
CATE is independent of window" is invalid if we know the R-learner returns
that same number even when the truth is zero.

The template (use for every layer after L2):
  1. Characterize the estimator's bias on a known-truth synthetic.
     (test_synthetic_r_learner_window_rho_within_known_bias_band)
  2. Re-prove the substantive claim using a method that does NOT share
     that estimator's bias.
     (test_real_window_independence_via_model_free_method)
  3. Quantify how much of any signal-vs-CATE pattern survives partialling
     out scale, so "signal X modifies CATE" doesn't sneak in as a scale
     correlation.
     (test_signal_role_landscape_post_residualization)

If a future change reintroduces the circularity, one of these tests fails.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from disha.signals.correlations import (
    SCALE_FEATURES,
    compute_signal_cate_correlations,
)
from disha.twin.dgp_gate import AGRONOMIC_CATE_FEATURES
from disha.uplift.dgp import DGP_SPEC_V1, build_synthetic_panel
from disha.uplift.learners import r_learner
from disha.uplift.train import CONFOUND_FEATURES


_ROOT = Path(__file__).resolve().parents[1]
_PROCESSED = _ROOT / "data" / "processed"


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_panel():
    p = _PROCESSED / "monthly_panel.parquet"
    if not p.exists():
        pytest.skip("monthly_panel.parquet not built")
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def cate_frozen():
    p = _PROCESSED / "cate_frozen.parquet"
    if not p.exists():
        pytest.skip("cate_frozen.parquet not built")
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def signals_panel():
    p = _PROCESSED / "signals_panel.parquet"
    if not p.exists():
        pytest.skip("signals_panel.parquet not built")
    return pd.read_parquet(p)


@pytest.fixture(scope="module")
def synthetic_panel(real_panel):
    return build_synthetic_panel(real_panel, spec=DGP_SPEC_V1)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: two-way FE demeaning (entity = tehsil×product, time = month_index)
# ──────────────────────────────────────────────────────────────────────────────

def _two_way_demean(values, entity_grp, time_grp, n_iter=20):
    s = pd.Series(np.asarray(values, dtype=float))
    eg = pd.Series(entity_grp)
    tg = pd.Series(time_grp)
    for _ in range(n_iter):
        s = s - s.groupby(eg).transform("mean")
        s = s - s.groupby(tg).transform("mean")
    return s.values


def _model_free_window_cate_spearman(panel: pd.DataFrame) -> tuple[float, list[float], list[float]]:
    """FE-stratified, R-learner-free measurement of window × CATE.

    Strata: window-closed cell, then 4 rank-quartiles within window-open cells.
    Stratum ATE = Cov(Y_dm, T_dm) / Var(T_dm) within each stratum.
    Returns (Spearman across strata, list of stratum ATEs, list of stratum window means).
    """
    entity = (panel["tehsil"].astype(str) + "___" + panel["product"].astype(str)).values
    time = panel["month_index"].values
    Y_dm = _two_way_demean(panel["Y_revenue"].astype(float).values, entity, time)
    T_dm = _two_way_demean(panel["T"].astype(float).values, entity, time)
    window = panel["window_decay_this_product"].astype(float).values

    def stratum_ate(mask):
        Y_d = Y_dm[mask]; T_d = T_dm[mask]
        if T_d.var() == 0:
            return float("nan")
        cov = np.mean(Y_d * T_d) - Y_d.mean() * T_d.mean()
        return cov / T_d.var()

    strata_masks = []
    strata_window_means = []
    # Closed
    m0 = window == 0
    strata_masks.append(m0)
    strata_window_means.append(float(window[m0].mean() if m0.sum() else 0.0))
    # Open Q1..Q4 by rank
    open_mask = window > 0
    open_idx = np.where(open_mask)[0]
    if len(open_idx) > 0:
        open_rank = pd.Series(window[open_mask]).rank(method="first").values
        quart = (open_rank / (len(open_idx) + 1) * 4).astype(int).clip(0, 3)
        for q in range(4):
            mk = np.zeros(len(window), dtype=bool)
            mk[open_idx[quart == q]] = True
            if mk.sum() > 0:
                strata_masks.append(mk)
                strata_window_means.append(float(window[mk].mean()))

    strata_ates = [stratum_ate(m) for m in strata_masks]
    rho, _ = spearmanr(strata_ates, strata_window_means)
    return float(rho if not np.isnan(rho) else 0.0), strata_ates, strata_window_means


# ──────────────────────────────────────────────────────────────────────────────
# THE adversarial test class — three load-bearing checks
# ──────────────────────────────────────────────────────────────────────────────

class TestWindowIndependenceAdversarial:
    """Defend the 'CATE ⊥ window' claim against circular evidence.

    Step 1 — characterize the R-learner's bias on a known-truth synthetic.
    Step 2 — re-prove independence on real data using a method that does NOT
             share that bias.

    If either step fails, the entire window-independence narrative (and
    therefore the L3 architecture choice of window-as-hard-constraint vs
    multiplicative-prize) needs revisiting.
    """

    def test_synthetic_r_learner_window_rho_within_known_bias_band(
        self, synthetic_panel,
    ):
        """STEP 1: The R-learner returns a non-zero window_cate_spearman even
        when β_window is PRE-REGISTERED ZERO in the DGP.  Document the bias
        band so nobody mistakes the real-data −0.16 for evidence of
        dependence.

        Adversarial intent: if a future estimator change makes R-learner
        unbiased here (rho ≈ 0), GREAT — then we COULD use win_rho as
        real-data evidence again.  The test would still pass; just update
        the band.  If the bias gets WORSE, the disclaimer needs sharpening
        and this test catches it.
        """
        cate = r_learner(
            synthetic_panel,
            x_effect_cols=AGRONOMIC_CATE_FEATURES,
            x_confound_cols=CONFOUND_FEATURES,
            y_col="Y_revenue", t_col="T", seed=42,
        )
        rho, _ = spearmanr(
            cate, synthetic_panel["window_decay_this_product"].values,
        )
        rho = float(rho if not np.isnan(rho) else 0.0)
        assert -0.30 <= rho <= 0.05, (
            f"R-learner win_rho on β_window=0 synthetic = {rho:+.3f}; expected "
            "in characterized bias band [-0.30, +0.05].  If outside, the bias "
            "has shifted and the dgp_gate disclaimer ('on synthetic with "
            "β_window=0 our R-learner gives ≈ -0.16, known bias') must be "
            "updated.  Do NOT silently widen this band — it would mask "
            "estimator regressions."
        )

    def test_real_window_independence_via_model_free_method(self, real_panel):
        """STEP 2: Prove window independence on real data using an estimator
        that does NOT share the R-learner's bias.

        Method: stratify cells (closed + open quartiles), compute FE-demeaned
        stratum ATEs, Spearman across strata.  This is FWL-style, no
        machine learning involved.  Expected: |rho| < 0.50 (non-monotonic
        pattern within sampling noise of small strata).
        """
        rho, strata_ates, strata_window_means = _model_free_window_cate_spearman(real_panel)
        assert abs(rho) < 0.50, (
            f"Model-free FE-stratified window-CATE Spearman = {rho:+.3f}; "
            "expected |rho| < 0.50.  Independence claim fails — window may "
            "genuinely modify CATE on real data, which would invalidate the "
            "window-as-hard-constraint L3 architecture.\n"
            f"  Stratum ATEs:    {[round(a, 0) for a in strata_ates]}\n"
            f"  Stratum windows: {[round(w, 3) for w in strata_window_means]}"
        )

    def test_independence_evidence_is_not_r_learner_circular(
        self, synthetic_panel, real_panel,
    ):
        """STEP 3 (meta): the model-free measurement on real data must give a
        DIFFERENT number than the R-learner's biased win_rho on the SAME
        real data.  If they collapse to the same value, the two evidences
        aren't independent and we're back to circular reasoning.
        """
        rho_model_free, _, _ = _model_free_window_cate_spearman(real_panel)
        # R-learner win_rho on real CATE (from cate_frozen, the dgp_gate output)
        cate_frozen = pd.read_parquet(_PROCESSED / "cate_frozen.parquet")
        merged = real_panel.merge(
            cate_frozen[["tehsil", "month_start", "product", "cate_mean"]],
            on=["tehsil", "month_start", "product"], how="inner",
        )
        rho_r_learner, _ = spearmanr(
            merged["cate_mean"].values,
            merged["window_decay_this_product"].values,
        )
        rho_r_learner = float(rho_r_learner if not np.isnan(rho_r_learner) else 0.0)
        # The two evidences are independent if they don't collapse.  Use a
        # generous gap (>=0.05) since both are noisy.
        gap = abs(rho_model_free - rho_r_learner)
        assert gap >= 0.05, (
            f"Model-free Spearman ({rho_model_free:+.3f}) and R-learner "
            f"Spearman ({rho_r_learner:+.3f}) differ by only {gap:.3f}.  "
            "The 'two independent evidences' framing relies on them giving "
            "distinct numbers; if they're identical, we may be measuring the "
            "same biased quantity twice."
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-signal role landscape after scale residualization (replaces the previous
# "4 effect-modifier + 1 context" pinning, which mistook 2 scale artifacts
# for effect-modifiers)
# ──────────────────────────────────────────────────────────────────────────────

class TestSignalScaleArtifactAdversarial:
    """Re-pin the signal role landscape against the residualized delta, not
    the raw delta.  Adversarial intent: if a future signal change makes a
    "scale artifact" signal actually become economic (or vice versa), this
    test fails and forces the L1/L4 narrative to be revisited."""

    @pytest.fixture(scope="class")
    def results(self, signals_panel, cate_frozen, real_panel):
        return compute_signal_cate_correlations(signals_panel, cate_frozen, real_panel)

    def test_role_landscape_is_2_artifact_2_economic_1_context(self, results):
        roles = {n: r["role"] for n, r in results["signals"].items()}
        artifacts = [n for n, r in roles.items() if r == "scale_artifact"]
        economics = [n for n, r in roles.items() if r == "effect_modifier"]
        contexts = [n for n, r in roles.items() if r == "context_only"]
        assert set(artifacts) == {"demand_spike", "oos_opportunity"}, (
            f"Expected scale_artifact = {{demand_spike, oos_opportunity}}; got {artifacts}.  "
            "These two signals' negative ΔCATE shrinks ≥70% under scale "
            "residualization — they fire in big tehsils, that's why CATE looks lower."
        )
        assert set(economics) == {"disease_pressure_alert", "window_urgency"}, (
            f"Expected effect_modifier = {{disease, window_urgency}}; got {economics}.  "
            "These two signals' ΔCATE survives scale residualization with "
            "|delta_resid| ≥ ₹100 and shrink < 70%."
        )
        assert set(contexts) == {"digital_demand_signal"}, (
            f"Expected context_only = {{digital_demand_signal}}; got {contexts}.  "
            "digital is correctly demoted by the prevalence guard (0.13% < 0.5%)."
        )

    def test_artifact_signals_shrink_at_least_70pct(self, results):
        for sig in ("demand_spike", "oos_opportunity"):
            shrink = results["signals"][sig]["shrink_after_residualization"]
            assert shrink >= 0.70, (
                f"{sig} shrink-after-residualization = {shrink:.2f} < 0.70.  "
                "If this signal's delta now survives residualization, it's not "
                "a scale artifact anymore — update L1 narrative."
            )

    def test_economic_signals_survive_residualization(self, results):
        for sig in ("disease_pressure_alert", "window_urgency"):
            r = results["signals"][sig]
            assert abs(r["delta_mean_cate_residualized"]) >= 100.0, (
                f"{sig} residualized delta = {r['delta_mean_cate_residualized']:+.0f} "
                "INR/month; below ₹100 threshold.  Economic signal claim weakens."
            )
            assert r["shrink_after_residualization"] < 0.50, (
                f"{sig} shrink = {r['shrink_after_residualization']:.2f} >= 0.50; "
                "more than half the delta evaporates under residualization — "
                "the economic story is suspicious."
            )


# ──────────────────────────────────────────────────────────────────────────────
# L2 analogue of the signal scale-artifact check: does the window-constrained
# Qini lift survive when we strip scale features from the CATE used for ranking?
#
# This is the load-bearing test for the entire L2 headline claim ("Disha's
# CATE-tail targeting is exploitable on real data").  If residualized Qini
# collapses, the apparent uplift was scale routing — a strategy that needs
# no causal machinery.
# ──────────────────────────────────────────────────────────────────────────────

class TestQiniScaleArtifactAdversarial:
    """L2 analogue of TestSignalScaleArtifactAdversarial.

    Three load-bearing checks (template for every future targeting metric):
      1. T-learner is a documented scale router — its Qini lift must shrink
         substantially under residualization (>= 60%).  This characterizes
         what "naive estimator = scale routing" looks like.
      2. Causal Forest's residualized window-Qini must remain >= 0.15 — the
         primary economic uplift evidence used in the headline.
      3. AT LEAST ONE estimator must have residualized window-Qini >= 0.15.
         If all four estimators collapse under residualization, Disha provides
         no economic value over scale routing and the architecture is invalid.
    """

    @pytest.fixture(scope="class")
    def residualized_qini(self, real_panel):
        """Compute orig and residualized window-constrained Qini for all 4 learners.

        Delegates to disha.eval.residualized_qini so that the test guard and
        the serialized headline number are the SAME computation.  If you
        change one, you change both — this is the structural fix for the
        prior 'prose +0.263 vs serialized +0.155' gap.
        """
        from disha.eval.residualized_qini import compute_residualized_window_qini
        real_cate = pd.read_parquet(_PROCESSED / "uplift_real_cate.parquet")
        return compute_residualized_window_qini(real_panel=real_panel, real_cate=real_cate)

    def test_t_learner_is_documented_scale_router(self, residualized_qini):
        """T-learner's apparent Qini lift must shrink substantially under
        residualization.  This is the SCALE ROUTING BASELINE — it documents
        what the field sees when an estimator just learns 'big tehsils =
        higher outcome'."""
        r = residualized_qini["t_learner"]
        shrink = 1.0 - abs(r["resid"]) / abs(r["orig"]) if abs(r["orig"]) > 1e-9 else 0
        assert r["orig"] >= 0.40, (
            f"T-learner raw window-Qini = {r['orig']:+.3f} < 0.40; "
            "the documented scale-routing baseline is gone."
        )
        assert shrink >= 0.60, (
            f"T-learner shrink under residualization = {shrink:.2f} < 0.60; "
            "T-learner used to be a documented scale router (76% shrink).  "
            "If shrink dropped, either T-learner got cleverer or the scale "
            "feature set changed — re-examine the L1/L2 claims."
        )

    def test_causal_forest_residualized_qini_holds(self, residualized_qini):
        """Causal Forest is the L3 CATE source.  Its residualized
        window-Qini must remain >= 0.15 — this IS the economic uplift
        evidence used in the headline."""
        r = residualized_qini["causal_forest"]
        assert r["resid"] >= 0.15, (
            f"Causal-Forest residualized window-Qini = {r['resid']:+.3f} < 0.15.  "
            "The economic uplift evidence used in the headline ('Disha's "
            "CATE-targeting beats scale routing') no longer holds.  If you "
            "cannot restore this, the demo_path must flip to synthetic_dgp_headline."
        )

    def test_some_estimator_has_economic_qini(self, residualized_qini):
        """Adversarial: if EVERY estimator's residualized window-Qini is
        below 0.15, the L2 claim ('Disha provides economic value beyond
        scale routing') is dead and the architecture is invalid."""
        max_resid = max(r["resid"] for r in residualized_qini.values())
        assert max_resid >= 0.15, (
            f"Max residualized window-Qini across all 4 estimators = "
            f"{max_resid:+.3f} < 0.15.  No estimator delivers economic "
            "value after scale routing is removed.  L2 architecture is invalid."
        )
