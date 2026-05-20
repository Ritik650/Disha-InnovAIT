"""
tests/test_data_contract.py

Data contract tests — written BEFORE implementation (TDD).
Validates: schemas, data quality invariants, the tehsil-week-product bridge,
and the agronomic window function.
"""

import json
from datetime import date

import pandas as pd
import pytest

from disha.common.data import (
    SKU_TO_PRODUCT,
    WindowStatus,
    best_window_status,
    build_treatment_panel,
    load_agronomy,
    load_digital_funnel,
    load_growers,
    load_inventory,
    load_pos,
    load_reps,
    load_retailers,
    load_settings,
    load_visit_log,
    load_whatsapp,
    window_status,
)


# ── Schema / load tests ────────────────────────────────────────────────────────

class TestLoaders:
    def test_reps_schema(self):
        df = load_reps()
        assert set(["rep_id", "territory_id", "state", "district", "tehsil_list"]).issubset(df.columns)
        assert len(df) == 500
        assert df["rep_id"].nunique() == 500

    def test_reps_tehsil_list_parsed(self):
        df = load_reps()
        for val in df["tehsil_list"]:
            assert isinstance(val, list), "tehsil_list must be parsed as Python list"

    def test_retailers_schema(self):
        df = load_retailers()
        assert set(["retailer_id", "territory_id", "tehsil"]).issubset(df.columns)
        assert len(df) == 4000
        assert df["retailer_id"].nunique() == 4000

    def test_visit_log_schema(self):
        df = load_visit_log()
        assert set(["rep_id", "visit_date", "territory_id", "visit_tehsil",
                    "visit_type", "product_recommended"]).issubset(df.columns)
        assert len(df) == 30000
        assert pd.api.types.is_datetime64_any_dtype(df["visit_date"])
        assert "visit_week" in df.columns

    def test_visit_types_known(self):
        df = load_visit_log()
        known = {"retailer meeting", "campaign_conducted", "grower meeting"}
        assert set(df["visit_type"].unique()).issubset(known)

    def test_pos_schema(self):
        df = load_pos()
        assert set(["retailer_id", "transaction_id", "sku_id", "sku_name",
                    "sku_qty", "sku_price", "transaction_date"]).issubset(df.columns)
        assert len(df) == 235042
        assert "revenue" in df.columns
        assert (df["revenue"] >= 0).all()

    def test_pos_revenue_ballpark(self):
        df = load_pos()
        total = df["revenue"].sum()
        # Known: ~₹2.8B — check within 10%
        assert 2.5e9 < total < 3.1e9, f"Total revenue out of expected range: {total:.0f}"

    def test_inventory_schema(self):
        df = load_inventory()
        assert set(["retailer_id", "sku_id", "sku_name", "sku_qty", "week_end_date"]).issubset(df.columns)
        assert len(df) == 310544
        assert "is_oos" in df.columns
        oos_rate = df["is_oos"].mean()
        # Known OOS rate ~3%
        assert 0.01 < oos_rate < 0.06, f"OOS rate out of expected range: {oos_rate:.4f}"

    def test_growers_schema(self):
        df = load_growers()
        assert set(["grower_id", "state", "district", "tehsil",
                    "crop_calendar", "crop"]).issubset(df.columns)
        assert len(df) == 6000
        # crop_calendar must be a dict (possibly empty), never a string
        for val in df["crop_calendar"]:
            assert isinstance(val, dict)

    def test_growers_null_calendar_handled(self):
        df = load_growers()
        # ~450 growers have null calendar — they must have empty dict, not NaN
        null_count = df["grower_crop_calendar"].isna().sum()
        # crop_calendar (parsed) should have 0 non-dict entries
        non_dict = sum(1 for v in df["crop_calendar"] if not isinstance(v, dict))
        assert non_dict == 0

    def test_growers_crop_distribution(self):
        df = load_growers()
        crops = df["crop"].dropna().value_counts()
        assert "wheat" in crops.index, "Wheat must be the dominant crop"
        assert crops["wheat"] > 2000

    def test_digital_funnel_schema(self):
        df = load_digital_funnel()
        assert set(["campaign_id", "week_start_date", "campaign_crop",
                    "campaign_product"]).issubset(df.columns)
        assert len(df) == 104

    def test_whatsapp_schema(self):
        df = load_whatsapp()
        assert set(["id", "campaign_product", "grower_id",
                    "delivered_status", "opened_status"]).issubset(df.columns)
        assert len(df) == 4479


# ── SKU ↔ Product bridge ───────────────────────────────────────────────────────

class TestSkuBridge:
    def test_all_pos_skus_in_bridge(self):
        pos = load_pos()
        pos_sku_names = set(pos["sku_name"].unique())
        assert pos_sku_names == set(SKU_TO_PRODUCT.keys()), (
            f"POS SKUs not covered: {pos_sku_names - set(SKU_TO_PRODUCT.keys())}"
        )

    def test_all_visit_products_in_bridge(self):
        vis = load_visit_log()
        visit_products = set(vis["product_recommended"].dropna().unique())
        bridge_products = set(SKU_TO_PRODUCT.values())
        assert visit_products == bridge_products, (
            f"Visit products not in bridge: {visit_products - bridge_products}"
        )

    def test_bridge_is_bijective(self):
        assert len(SKU_TO_PRODUCT) == len(set(SKU_TO_PRODUCT.values())), \
            "SKU→product map must be 1:1"


# ── Geography join integrity ───────────────────────────────────────────────────

class TestGeography:
    def test_territory_tehsil_overlap(self):
        reps = load_reps()
        retailers = load_retailers()
        # All retailer territory_ids must be in reps
        rep_territories = set(reps["territory_id"].unique())
        ret_territories = set(retailers["territory_id"].dropna().unique())
        orphan_territories = ret_territories - rep_territories
        assert len(orphan_territories) == 0, (
            f"Retailers reference territory_ids not in reps: {orphan_territories}"
        )

    def test_visit_tehsil_overlap_with_retailers(self):
        vis = load_visit_log()
        retailers = load_retailers()
        vis_tehsils = set(vis["visit_tehsil"].unique())
        ret_tehsils = set(retailers["tehsil"].unique())
        overlap = vis_tehsils & ret_tehsils
        # Need at least 50% overlap to ensure bridge works
        overlap_ratio = len(overlap) / len(vis_tehsils)
        assert overlap_ratio > 0.5, (
            f"Only {overlap_ratio:.1%} visit tehsils have matching retailers"
        )

    def test_grower_tehsil_overlap_with_retailers(self):
        grw = load_growers()
        ret = load_retailers()
        grw_tehsils = set(grw["tehsil"].unique())
        ret_tehsils = set(ret["tehsil"].unique())
        overlap = len(grw_tehsils & ret_tehsils)
        assert overlap > 2000, f"Only {overlap} grower-retailer tehsil matches"


# ── Treatment-outcome panel ────────────────────────────────────────────────────

class TestTreatmentPanel:
    @pytest.fixture(scope="class")
    def panel(self):
        return build_treatment_panel(horizon_weeks=3)

    def test_panel_has_required_columns(self, panel):
        required = {"tehsil", "week_start", "product", "T", "Y_revenue", "n_visits"}
        assert required.issubset(panel.columns)

    def test_treatment_values_binary(self, panel):
        assert set(panel["T"].unique()).issubset({0, 1})

    def test_treated_units_exist(self, panel):
        assert (panel["T"] == 1).sum() > 0, "Must have treated observations"

    def test_control_units_exist(self, panel):
        assert (panel["T"] == 0).sum() > 0, "Must have control observations"

    def test_no_negative_revenue(self, panel):
        assert (panel["Y_revenue"] >= 0).all()

    def test_products_match_bridge(self, panel):
        panel_products = set(panel["product"].unique())
        bridge_products = set(SKU_TO_PRODUCT.values())
        assert panel_products.issubset(bridge_products)

    def test_treatment_prevalence_reasonable(self, panel):
        rate = panel["T"].mean()
        # Expect 5–60% treatment rate (not all tehsil-week-product combos are visited)
        assert 0.05 < rate < 0.60, f"Treatment rate {rate:.2%} out of expected range"

    def test_week_start_is_datetime(self, panel):
        assert pd.api.types.is_datetime64_any_dtype(panel["week_start"])


# ── Agronomic window function ──────────────────────────────────────────────────

class TestAgronomicWindow:
    def test_wheat_tilt_in_window(self):
        """Tilt 250 EC applied ~14 days before wheat tillering (Jan 10) should be open."""
        result = window_status(
            tehsil="Patna_T001",
            product="Tilt 250 EC",
            query_date=date(2025, 12, 28),
            tehsil_dominant_crop="wheat",
        )
        assert any(s.open for s in result), "Window should be open 14d before stage"
        open_window = next(s for s in result if s.open)
        assert open_window.decay_factor == 1.0

    def test_wheat_tilt_after_window_closed(self):
        """Tilt 250 EC applied 10 days after wheat tillering (Jan 10) is closed."""
        result = window_status(
            tehsil="Patna_T001",
            product="Tilt 250 EC",
            query_date=date(2026, 1, 20),  # 10 days after Jan 10
            tehsil_dominant_crop="wheat",
        )
        for s in result:
            if s.crop == "wheat" and s.stage == "tillering":
                assert not s.open, "Window should be closed 10d after stage"
                assert s.decay_factor == 0.0

    def test_decay_at_stage_approx(self):
        """Decay factor must be 1.0 exactly at stage approx date."""
        result = window_status(
            tehsil="x",
            product="Tilt 250 EC",
            query_date=date(2026, 1, 10),  # exactly at wheat tillering approx
            tehsil_dominant_crop="wheat",
        )
        for s in result:
            if s.crop == "wheat" and s.stage == "tillering":
                assert s.decay_factor == 1.0

    def test_decay_linearly_after_stage(self):
        """Decay must be strictly between 0 and 1 on a day between stage approx and close."""
        result = window_status(
            tehsil="x",
            product="Tilt 250 EC",
            query_date=date(2026, 1, 13),  # 3 days after Jan 10 (close_after=7)
            tehsil_dominant_crop="wheat",
        )
        for s in result:
            if s.crop == "wheat" and s.stage == "tillering":
                assert 0.0 < s.decay_factor < 1.0

    def test_unknown_product_returns_empty(self):
        result = window_status("x", "NonExistentProduct", date(2026, 1, 1))
        assert result == []

    def test_best_window_returns_most_urgent(self):
        """best_window_status should return the window with smallest days_left."""
        ws = best_window_status(
            tehsil="x",
            product="Tilt 250 EC",
            query_date=date(2025, 12, 28),
            tehsil_dominant_crop="wheat",
        )
        assert ws is not None
        assert ws.open

    def test_best_window_none_when_all_closed(self):
        ws = best_window_status(
            tehsil="x",
            product="Tilt 250 EC",
            query_date=date(2026, 4, 1),  # season over
            tehsil_dominant_crop="wheat",
        )
        assert ws is None

    def test_all_products_have_agronomy_config(self):
        agro = load_agronomy()
        configured = set(agro["products"].keys())
        bridge_products = set(SKU_TO_PRODUCT.values())
        missing = bridge_products - configured
        assert not missing, f"Products missing from agronomy.yaml: {missing}"

    def test_fallback_calendars_cover_all_crops(self):
        agro = load_agronomy()
        grw = load_growers()
        data_crops = set(grw["crop"].dropna().unique())
        fallback_crops = set(agro.get("fallback_calendars", {}).keys())
        missing = data_crops - fallback_crops
        assert not missing, f"Crops in data missing from fallback_calendars: {missing}"


# ── Settings & agronomy config ─────────────────────────────────────────────────

class TestConfig:
    def test_settings_loads(self):
        cfg = load_settings()
        assert "seed" in cfg
        assert cfg["seed"] == 42

    def test_agronomy_loads(self):
        agro = load_agronomy()
        assert "products" in agro
        assert "fallback_calendars" in agro
        assert "defaults" in agro

    def test_agronomy_defaults_present(self):
        agro = load_agronomy()
        assert "window_open_days_before" in agro["defaults"]
        assert "window_close_days_after" in agro["defaults"]
