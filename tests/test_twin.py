"""
tests/test_twin.py

Unit tests for L0 Territory Digital Twin components:
  - 3-tier window_status fallback paths + source_tier tagging
  - Monthly causal panel properties (grain, T/Y integrity)
  - Weather module (synthetic fallback)
  - Vectorised window features (smoke test)
  - Lift probe (statistical sanity + is_flat logic)
  - Balance module (smoke test)
  - TwinStore API (without twin.parquet — graceful degradation)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from disha.common.data import (
    WindowStatus,
    build_tehsil_dominant_crops,
    build_tehsil_grower_features,
    build_treatment_panel_monthly,
    get_tehsil_district_map,
    get_tehsil_dominant_crop,
    load_growers,
    window_status,
)
from disha.twin.lift_probe import (
    _cohens_d,
    _welch_ttest,
    run_lift_probe,
)
from disha.twin.weather import (
    _synthetic_weather,
    aggregate_to_weekly,
)


# ── 3-tier window_status fallback ─────────────────────────────────────────────

class TestWindowStatusTiers:
    """Each tier of the 3-tier fallback must set source_tier correctly."""

    def test_tier_direct_when_crop_provided(self):
        result = window_status(
            tehsil="X_fake",
            product="Tilt 250 EC",
            query_date=date(2025, 12, 28),
            tehsil_dominant_crop="wheat",
        )
        assert all(s.source_tier == "direct" for s in result)

    def test_tier_tehsil_from_growers(self):
        grw = load_growers()
        # Pick a tehsil that has growers
        tehsil_with_growers = grw.dropna(subset=["crop"])["tehsil"].iloc[0]
        result = window_status(
            tehsil=tehsil_with_growers,
            product="Tilt 250 EC",
            query_date=date(2025, 12, 28),
            growers_df=grw,
        )
        if result:
            assert all(s.source_tier == "tehsil" for s in result), \
                f"Expected 'tehsil' tier, got: {[s.source_tier for s in result]}"

    def test_tier_district_fallback(self):
        """A tehsil with no growers but whose district has growers → district tier."""
        grw = load_growers()
        # Construct a fake tehsil in a district that HAS growers
        real_district = grw.dropna(subset=["crop"])["district"].iloc[0]
        fake_tehsil = f"FAKE_TEHSIL_FOR_DISTRICT_{real_district}"

        result = window_status(
            tehsil=fake_tehsil,
            product="Tilt 250 EC",
            query_date=date(2025, 12, 28),
            growers_df=grw,
            district=real_district,
        )
        if result:
            assert all(s.source_tier == "district" for s in result), \
                f"Expected 'district' tier, got: {[s.source_tier for s in result]}"

    def test_tier_default_when_no_data(self):
        """Completely unknown tehsil + no growers_df → default (wheat)."""
        result = window_status(
            tehsil="COMPLETELY_UNKNOWN",
            product="Tilt 250 EC",
            query_date=date(2025, 12, 28),
        )
        # Should return wheat windows at 'default' tier
        if result:
            assert all(s.source_tier == "default" for s in result)
            assert all(s.crop == "wheat" for s in result)

    def test_source_tier_field_present_on_all_results(self):
        """Every WindowStatus must have a source_tier string (backward-compat check)."""
        result = window_status(
            tehsil="any",
            product="Tilt 250 EC",
            query_date=date(2026, 1, 5),
            tehsil_dominant_crop="wheat",
        )
        for ws in result:
            assert isinstance(ws.source_tier, str)
            assert ws.source_tier in ("direct", "tehsil", "district", "default")

    def test_no_growers_df_no_dominant_crop_defaults_to_wheat(self):
        """Without growers_df or tehsil_dominant_crop, crop must be wheat."""
        result = window_status(
            tehsil="UNKNOWN_TEHSIL",
            product="Tilt 250 EC",
            query_date=date(2026, 1, 5),
        )
        for ws in result:
            assert ws.crop == "wheat"


# ── Tehsil dominant crop helpers ──────────────────────────────────────────────

class TestDominantCropHelpers:
    def test_get_tehsil_district_map_returns_dict(self):
        m = get_tehsil_district_map()
        assert isinstance(m, dict)
        assert len(m) > 0

    def test_get_tehsil_dominant_crop_known_tehsil(self):
        grw = load_growers()
        tehsil = grw.dropna(subset=["crop"])["tehsil"].iloc[0]
        crop, tier = get_tehsil_dominant_crop(tehsil, grw)
        assert tier == "tehsil"
        assert isinstance(crop, str) and len(crop) > 0

    def test_get_tehsil_dominant_crop_unknown_defaults(self):
        grw = load_growers()
        crop, tier = get_tehsil_dominant_crop("NONEXISTENT_TEHSIL", grw, {})
        assert tier == "default"
        assert crop == "wheat"

    def test_build_tehsil_dominant_crops_covers_retailers(self):
        from disha.common.data import load_retailers
        ret = load_retailers()
        dominant = build_tehsil_dominant_crops()
        ret_tehsils = set(ret["tehsil"].unique())
        dom_tehsils = set(dominant["tehsil"].unique())
        assert ret_tehsils.issubset(dom_tehsils), \
            f"Retailer tehsils not covered: {ret_tehsils - dom_tehsils}"

    def test_build_tehsil_dominant_crops_all_tiers_valid(self):
        dominant = build_tehsil_dominant_crops()
        valid_tiers = {"tehsil", "district", "default"}
        assert set(dominant["crop_tier"].unique()).issubset(valid_tiers)


# ── Per-tehsil grower features ─────────────────────────────────────────────────

class TestGrowerFeatures:
    def test_grower_features_has_required_columns(self):
        feats = build_tehsil_grower_features()
        for col in ["tehsil", "n_growers", "avg_farm_size_ha", "pct_smartphone",
                    "pct_offline_attended", "pct_product_scanned"]:
            assert col in feats.columns

    def test_grower_features_rates_in_unit_interval(self):
        feats = build_tehsil_grower_features()
        for col in ["pct_smartphone", "pct_offline_attended", "pct_product_scanned"]:
            assert feats[col].between(0, 1).all(), f"{col} out of [0,1]"

    def test_grower_features_no_null_n_growers(self):
        feats = build_tehsil_grower_features()
        assert feats["n_growers"].notna().all()


# ── Monthly causal panel ───────────────────────────────────────────────────────

class TestMonthlyPanel:
    @pytest.fixture(scope="class")
    def panel(self):
        return build_treatment_panel_monthly()

    def test_required_columns(self, panel):
        required = {"tehsil", "month_start", "product", "T", "Y_revenue",
                    "lag_revenue_1m", "n_retailers_in_tehsil", "month_index"}
        assert required.issubset(panel.columns)

    def test_grain_unique(self, panel):
        """Each (tehsil, month_start, product) must appear at most once."""
        dups = panel.duplicated(subset=["tehsil", "month_start", "product"])
        assert not dups.any(), f"{dups.sum()} duplicate (tehsil, month, product) rows"

    def test_treatment_binary(self, panel):
        assert set(panel["T"].unique()).issubset({0, 1})

    def test_non_negative_outcomes(self, panel):
        assert (panel["Y_revenue"] >= 0).all()
        assert (panel["lag_revenue_1m"] >= 0).all()

    def test_month_start_is_first_of_month(self, panel):
        assert (panel["month_start"].dt.day == 1).all()

    def test_month_index_range(self, panel):
        # Rabi season: Oct(10) through Mar(3)
        valid_months = {10, 11, 12, 1, 2, 3}
        assert set(panel["month_index"].unique()).issubset(valid_months)

    def test_treatment_prevalence_reasonable(self, panel):
        rate = panel["T"].mean()
        assert 0.05 < rate < 0.80, f"Treatment rate {rate:.2%} out of expected range"

    def test_wa_engagement_rate_present(self, panel):
        assert "wa_engagement_rate" in panel.columns
        assert (panel["wa_engagement_rate"] >= 0).all()
        assert (panel["wa_engagement_rate"] <= 1).all()

    def test_grower_features_joined(self, panel):
        assert "n_growers" in panel.columns
        assert "pct_smartphone" in panel.columns


# ── Weather module ─────────────────────────────────────────────────────────────

class TestWeather:
    def test_synthetic_returns_expected_columns(self):
        df = _synthetic_weather("TestDistrict", "2025-10-06", "2026-03-29")
        for col in ["date", "temp_max", "temp_min", "precip_mm", "rh_max", "rh_min"]:
            assert col in df.columns

    def test_synthetic_date_range(self):
        df = _synthetic_weather("X", "2025-10-06", "2026-03-29")
        assert df["date"].min() == pd.Timestamp("2025-10-06")
        assert df["date"].max() == pd.Timestamp("2026-03-29")

    def test_synthetic_deterministic(self):
        df1 = _synthetic_weather("PatnaDistrict", "2025-10-06", "2026-03-29")
        df2 = _synthetic_weather("PatnaDistrict", "2025-10-06", "2026-03-29")
        pd.testing.assert_frame_equal(df1, df2)

    def test_synthetic_different_districts_differ(self):
        df1 = _synthetic_weather("DistrictA", "2025-10-06", "2026-03-29")
        df2 = _synthetic_weather("DistrictB", "2025-10-06", "2026-03-29")
        assert not df1["precip_mm"].equals(df2["precip_mm"])

    def test_aggregate_to_weekly_columns(self):
        df = _synthetic_weather("X", "2025-10-06", "2026-03-29")
        wk = aggregate_to_weekly(df)
        for col in ["week_start", "rainfall_mm_7d", "rh_max_7d",
                    "disease_pressure_days", "disease_pressure_flag"]:
            assert col in wk.columns

    def test_aggregate_flag_is_binary(self):
        df = _synthetic_weather("X", "2025-10-06", "2026-03-29")
        wk = aggregate_to_weekly(df)
        assert set(wk["disease_pressure_flag"].unique()).issubset({0, 1})

    def test_aggregate_season_has_approx_26_weeks(self):
        # Season Oct 06 – Mar 29: 24–26 ISO weeks depending on Monday alignment
        df = _synthetic_weather("X", "2025-10-06", "2026-03-29")
        wk = aggregate_to_weekly(df)
        assert 24 <= len(wk) <= 27, f"Expected ~26 ISO weeks, got {len(wk)}"


# ── Lift probe ─────────────────────────────────────────────────────────────────

class TestLiftProbe:
    def _make_panel(self, mean_treated, mean_control, n=500, seed=42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n_t = n // 2
        treated = pd.DataFrame({
            "T": 1,
            "Y_revenue": np.maximum(0, rng.normal(mean_treated, 1000, n_t)),
            "lag_revenue_1m": np.maximum(0, rng.normal(mean_treated * 0.9, mean_treated * 0.1, n_t)),
        })
        control = pd.DataFrame({
            "T": 0,
            "Y_revenue": np.maximum(0, rng.normal(mean_control, 1000, n - n_t)),
            "lag_revenue_1m": np.maximum(0, rng.normal(mean_control * 0.9, mean_control * 0.1, n - n_t)),
        })
        return pd.concat([treated, control], ignore_index=True)

    def test_clear_lift_not_flat(self):
        # Large effect: mean_treated >> mean_control; DR-ATE CI should not straddle zero
        panel = self._make_panel(mean_treated=10000, mean_control=1000, n=1000)
        result = run_lift_probe(panel)
        assert result["raw_lift"] > 0
        assert not result["ate_is_flat"]

    def test_flat_when_no_effect(self):
        panel = self._make_panel(mean_treated=5000, mean_control=5000, n=200)
        result = run_lift_probe(panel)
        assert result["ate_is_flat"]

    def test_result_has_required_keys(self):
        panel = self._make_panel(5000, 4000)
        result = run_lift_probe(panel)
        for key in ["raw_lift", "t_stat_naive", "p_value_naive", "cohens_d_naive",
                    "ate_is_flat", "cate_is_heterogeneous", "dr_ate",
                    "dr_ci_lower", "dr_ci_upper", "n_treated", "n_control"]:
            assert key in result

    def test_ate_is_flat_is_bool(self):
        panel = self._make_panel(5000, 4000)
        result = run_lift_probe(panel)
        assert isinstance(result["ate_is_flat"], bool)

    def test_cate_is_heterogeneous_is_none(self):
        """cate_is_heterogeneous is always None from lift_probe; set by dgp_gate."""
        panel = self._make_panel(5000, 4000)
        result = run_lift_probe(panel)
        assert result["cate_is_heterogeneous"] is None

    def test_dr_ci_is_ordered(self):
        panel = self._make_panel(5000, 4000, n=400)
        result = run_lift_probe(panel)
        assert result["dr_ci_lower"] <= result["dr_ate"] <= result["dr_ci_upper"]

    def test_welch_ttest_known_values(self):
        rng = np.random.default_rng(0)
        a = rng.normal(10, 1, 1000)
        b = rng.normal(0, 1, 1000)
        t, p = _welch_ttest(a, b)
        assert t > 100   # enormous effect
        assert p < 0.001

    def test_welch_ttest_null(self):
        rng = np.random.default_rng(1)
        a = rng.normal(5, 1, 500)
        b = rng.normal(5, 1, 500)
        t, p = _welch_ttest(a, b)
        assert p > 0.05  # should not reject null

    def test_cohens_d_symmetric(self):
        a = np.array([10.0, 11.0, 9.0])
        b = np.array([5.0, 6.0, 4.0])
        assert _cohens_d(a, b) == pytest.approx(-_cohens_d(b, a), abs=1e-9)

    def test_run_on_real_monthly_panel(self):
        """Smoke test: lift probe must complete without error on real data."""
        panel = build_treatment_panel_monthly()
        result = run_lift_probe(panel)
        assert "ate_is_flat" in result
        assert isinstance(result["ate_is_flat"], bool)


# ── TwinStore (graceful degradation without twin.parquet) ────────────────────

class TestTwinStore:
    def test_store_instantiates_without_twin(self, tmp_path):
        """Store must not crash when twin.parquet is absent."""
        from disha.twin.store import TwinStore
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store = TwinStore(twin_path=tmp_path / "nonexistent.parquet")
        assert store._twin is None

    def test_state_returns_empty_dict_when_no_twin(self, tmp_path):
        from disha.twin.store import TwinStore
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store = TwinStore(twin_path=tmp_path / "nonexistent.parquet")
        s = store.state("TER_0001", "Patna_T001", date(2026, 1, 10))
        assert s == {}

    def test_window_status_works_without_twin(self, tmp_path):
        """window_status must work even without twin.parquet (uses data module directly)."""
        from disha.twin.store import TwinStore
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store = TwinStore(twin_path=tmp_path / "nonexistent.parquet")
        result = store.window_status("Patna_T001", "Tilt 250 EC", date(2025, 12, 28))
        assert isinstance(result, list)

    def test_demo_path_fallback_without_gate(self, tmp_path):
        """demo_path property must not crash when dgp_gate.json is absent."""
        from disha.twin.store import TwinStore
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            store = TwinStore(twin_path=tmp_path / "nonexistent.parquet")
        assert store.demo_path in (
            "real_data_headline", "synthetic_dgp_headline",
            "pending_l2_assessment", "dual_synthetic_led",
        )


# ── DGP gate ─────────────────────────────────────────────────────────────────

class TestDgpGate:
    def test_pending_on_insufficient_features(self):
        """Panel with no CATE driver features → pending stability."""
        from disha.twin.dgp_gate import crossfit_tlearner_stability
        panel = pd.DataFrame({
            "T": [1, 0, 1, 0, 1, 0],
            "Y_revenue": [100.0, 90.0, 110.0, 85.0, 105.0, 95.0],
        })
        result = crossfit_tlearner_stability(panel, n_folds=2)
        assert result["cate_is_heterogeneous"] is None
        assert result["stability_score"] is None

    def test_ate_not_flat_always_real(self):
        from disha.twin.dgp_gate import DEMO_PATH_REAL, determine_demo_path
        gate = determine_demo_path({"ate_is_flat": False})
        assert gate["demo_path"] == DEMO_PATH_REAL
        assert gate["ate_is_flat"] is False

    def test_flat_stable_uses_dual_synthetic_led(self):
        """Phase-6 lock: when ate is flat AND cate is heterogeneous, demo
        path is dual_synthetic_led (lead synthetic engine-validation,
        follow with real-data honest result).  Replaces the brief
        real_data_headline lock (Phase-5) which was a hasty strategic
        call; the engineering numbers underneath are unchanged."""
        from disha.twin.dgp_gate import DEMO_PATH_DUAL_SYNTHETIC_LED, determine_demo_path
        gate = determine_demo_path(
            {"ate_is_flat": True},
            {"cate_is_heterogeneous": True, "stability_score": 0.55},
        )
        assert gate["demo_path"] == DEMO_PATH_DUAL_SYNTHETIC_LED, (
            f"Expected dual_synthetic_led (Phase-6 strategic lock); "
            f"got {gate['demo_path']}.  If you genuinely want to revert to "
            "real_data_headline, document a Phase-7 reopening in PROGRESS.md."
        )

    def test_flat_unstable_uses_synthetic(self):
        from disha.twin.dgp_gate import DEMO_PATH_SYNTHETIC, determine_demo_path
        gate = determine_demo_path(
            {"ate_is_flat": True},
            {"cate_is_heterogeneous": False, "stability_score": 0.10},
        )
        assert gate["demo_path"] == DEMO_PATH_SYNTHETIC

    def test_flat_pending_uses_pending(self):
        from disha.twin.dgp_gate import DEMO_PATH_PENDING, determine_demo_path
        gate = determine_demo_path({"ate_is_flat": True}, None)
        assert gate["demo_path"] == DEMO_PATH_PENDING

    def test_gate_result_has_l1_design_note(self):
        """l1_design_note must always be present to guide L1 feature engineering."""
        from disha.twin.dgp_gate import determine_demo_path
        gate = determine_demo_path({"ate_is_flat": True}, None)
        assert "l1_design_note" in gate
        assert "window_decay_this_product" in gate["l1_design_note"]


# ── Real-panel regression tests (pin numeric inferential values) ───────────────

class TestRealPanelRegression:
    """
    Regression tests that pin the numeric values of causal estimates on the
    real Syngenta panel.  These tests exist because 89 green structural tests
    gave false confidence — execution was validated but not inferential correctness.

    Ranges are derived independently (FWL OLS two-way FE diagnostic script) and
    should be treated as load-bearing: if they break, the estimator has regressed.

    DML ATE band  : [–4000, +1500] INR/month
      Expected ≈ –₹1.5k to –₹1.8k; –₹10k indicates cross-sectional AIPW bias.
    CATE cross-seed r : [0.80, 0.95]  (post both pre-L2 corrections)
      Expected ≈ 0.87; > 0.95 indicates a leakage feature crept back in;
      < 0.80 indicates the R-learner Stage-2 prediction bug has returned
      (predicting Y_tilde at val instead of τ̂ = X·β).
    Structural-dummy artifact: r_structural > r_agronomic must hold (artifact direction).

    Note: DML ATE runs on the raw monthly panel (has entity/time structure).
    CATE stability runs on the enriched monthly_panel.parquet (has agronomic features).
    """

    @pytest.fixture(scope="class")
    def raw_panel(self):
        return build_treatment_panel_monthly()

    @pytest.fixture(scope="class")
    def enriched_panel(self):
        """Load monthly_panel.parquet which includes agronomic window features."""
        import yaml
        root = Path(__file__).resolve().parents[1]
        with open(root / "config" / "settings.yaml") as f:
            cfg = yaml.safe_load(f)
        parquet_path = root / cfg["paths"]["processed"] / "monthly_panel.parquet"
        if not parquet_path.exists():
            pytest.skip(
                "monthly_panel.parquet not found — "
                "run `python -m disha.twin.build` first"
            )
        return pd.read_parquet(parquet_path)

    def test_dr_ate_within_sane_band(self, enriched_panel):
        """
        DML ATE on the enriched monthly panel must fall in [–4000, +1500] INR/month
        and the 95% CI must straddle zero (ate_is_flat=True).

        Enriched panel (monthly_panel.parquet) is used because it contains the
        agronomic X features (window_decay, disease_pressure, etc.) needed for
        proper two-way FE + partialling.  Expected: ATE ≈ –₹1.6k, CI straddles zero.
        Values near –₹10k indicate cross-sectional AIPW extrapolation bias.
        """
        from disha.twin.lift_probe import compute_dr_ate
        result = compute_dr_ate(enriched_panel, n_folds=5, seed=42)
        ate = result["dr_ate"]
        ci_lower = result["ci_lower"]
        ci_upper = result["ci_upper"]
        assert -4000 <= ate <= 1500, (
            f"DML ATE = {ate:.0f} INR/month outside sane band [-4000, +1500]. "
            "Values near -10k indicate cross-sectional AIPW bias. "
            "Fix: ensure DML uses two-way FE absorption (entity + month demeaning)."
        )
        assert ci_lower <= 0.0 <= ci_upper, (
            f"DML CI = [{ci_lower:.0f}, {ci_upper:.0f}] does not straddle zero. "
            "Expected: ATE near-zero after FE absorption. "
            "Straddling CI confirms ate_is_flat=True on the real panel."
        )

    def test_cate_cross_seed_r_in_range(self, enriched_panel):
        """R-learner cross-seed Spearman r on the CORRECTED 7-driver set with
        the CORRECTED Stage-2 prediction must be in [0.80, 0.95].
        Default seed 42 → r ≈ 0.88.

        If r > 0.95: a leakage feature crept back into AGRONOMIC_CATE_FEATURES.
        If r < 0.80: the R-learner Stage-2 prediction bug is back (predicting
            Y_tilde at val = T_tilde·X·β instead of τ̂ = X·β).
        """
        from disha.twin.dgp_gate import crossfit_tlearner_stability
        result = crossfit_tlearner_stability(enriched_panel, seed=42)
        r = result["stability_score"]
        assert r is not None, (
            "stability_score is None — insufficient agronomic features or "
            "missing entity/time structure in panel."
        )
        assert 0.80 <= r <= 0.95, (
            f"CATE cross-seed r = {r:.3f} outside expected range [0.80, 0.95]. "
            "r > 0.95 → leakage feature back in AGRONOMIC_CATE_FEATURES. "
            "r < 0.80 → R-learner Stage-2 prediction bug regression "
            "(check dgp_gate._r_learner_held_out_cate predicts X·β not W·β)."
        )

    def test_threshold_is_immutable_50(self):
        """The CATE-stability gate is locked at 0.50 — moving it is goalpost-shifting."""
        from disha.twin.dgp_gate import CROSSFIT_STABILITY_THRESHOLD
        assert CROSSFIT_STABILITY_THRESHOLD == 0.50, (
            "CROSSFIT_STABILITY_THRESHOLD must remain 0.50 per L2 scope contract. "
            f"Current value: {CROSSFIT_STABILITY_THRESHOLD}.  If a finding requires "
            "a different gate, document it explicitly — do not silently retune."
        )

    def test_structural_dummy_r_exceeds_agronomic_r(self, enriched_panel):
        """Structural-dummy-only R-learner must give r > agronomic R-learner r.

        This is the LOAD-BEARING artifact test required by the pre-L2 corrections:
        it documents that structural calendar/product slots produce SPURIOUSLY
        higher cross-seed stability than genuine effect-modifier features.
        If the gap ever flips negative, either:
          (a) a leaked feature is back in the agronomic set, OR
          (b) the structural-dummy run is broken (degenerate X, all-NaN, etc.)
        Either case warrants investigation before proceeding.
        """
        from disha.twin.dgp_gate import crossfit_tlearner_stability
        result = crossfit_tlearner_stability(enriched_panel, seed=42)
        agronomic_r = result["stability_score"]
        structural_r = result["structural_dummy_stability_score"]
        gap = result["structural_minus_agronomic_gap"]
        assert structural_r is not None, (
            "structural_dummy_stability_score is None — the structural-dummy "
            "artifact run was skipped or failed; check _structural_dummies_X()."
        )
        assert structural_r > agronomic_r, (
            f"Structural-dummy r ({structural_r:.3f}) ≤ agronomic r ({agronomic_r:.3f}); "
            f"gap = {gap:+.3f}.  Expected POSITIVE gap (artifact direction). "
            "If this fails, a leaked feature may be back in the agronomic set."
        )
