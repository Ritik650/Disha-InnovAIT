"""
disha.api.main — thin FastAPI for the rep app + manager dashboard.

Design rules (locked by final-sprint scope):
  - NO new modeling.  Every endpoint serves precomputed artifacts.
  - Plans come from disha.optimizer (computed on-demand and cached per date).
  - WHY text comes from the Stop dataclass (template-rendered upstream).
  - 7 endpoints total; CORS open so the static rep app + dashboard can call it.

Run:    uvicorn disha.api.main:app --reload --port 8000
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from disha.optimizer.router import RoutePlan, _to_dict, _to_public_dict, run_dual_arm

# ── public serializers (single source of truth in disha.optimizer.router) ────
# Real-data CF CATE absolute values are biased downward by uncorrected
# selection.  We use them ONLY for ranking; absolute INR fields are stripped
# at both the disk-JSON and API-response layers via `_to_public_dict`.

def _public_stop_dict(stop) -> dict:
    return _to_public_dict(stop)


def _public_rep_day_dict(rd) -> dict:
    return _to_public_dict(rd)


def _public_plan_summary(plan: RoutePlan) -> dict:
    """sim_run summary minus any absolute INR."""
    rep_days = plan.rep_days
    n_active = sum(1 for rd in rep_days if rd.stops)
    n_stops = sum(len(rd.stops) for rd in rep_days)
    cap_used = sum(rd.capacity_used_min for rd in rep_days)
    cap_budget = sum(rd.capacity_budget_min for rd in rep_days)
    # coverage_efficiency_pct = feasible in-window cells visited /
    # feasible in-window cells available.  Capped at 100 to absorb the
    # rare edge case where multiple reps share overlapping tehsils.
    cov_pct = round(100 * n_stops / max(plan.n_cells_feasible, 1), 1)
    cov_pct = min(cov_pct, 100.0)
    return {
        "n_reps_total":             len(rep_days),
        "n_reps_active":            n_active,
        "n_stops":                  n_stops,
        "avg_stops_per_active":     round(n_stops / max(n_active, 1), 2),
        "capacity_used_min":        cap_used,
        "capacity_budget_min":      cap_budget,
        "capacity_used_pct":        round(100 * cap_used / max(cap_budget, 1), 1),
        "coverage_efficiency_pct":  cov_pct,
        "n_cells_considered":       plan.n_cells_considered,
        "n_cells_feasible":         plan.n_cells_feasible,
    }

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[2]
_PROCESSED = _ROOT / "data" / "processed"
_RAW = _ROOT / "data" / "raw"
_OUTCOMES = _PROCESSED / "outcomes.jsonl"

app = FastAPI(
    title="Disha API",
    description="Deadline-aware field-force routing under agronomic windows",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


# ── helpers ────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=8)
def _plans_for_date(date_iso: str) -> dict:
    """Build both arms for a date (greedy is fast; cached for repeats)."""
    return run_dual_arm(date_iso)


def _arm_plan(date_iso: str, arm: str) -> RoutePlan:
    if arm not in ("real", "synthetic"):
        raise HTTPException(400, f"arm must be 'real' or 'synthetic'; got {arm!r}")
    return _plans_for_date(date_iso)[arm]


@lru_cache(maxsize=1)
def _retailers() -> pd.DataFrame:
    return pd.read_csv(_RAW / "retailers.csv")


@lru_cache(maxsize=1)
def _signals() -> pd.DataFrame:
    return pd.read_parquet(_PROCESSED / "signals_panel.parquet")


@lru_cache(maxsize=1)
def _monthly_panel() -> pd.DataFrame:
    return pd.read_parquet(_PROCESSED / "monthly_panel.parquet")


@lru_cache(maxsize=1)
def _pos() -> pd.DataFrame:
    df = pd.read_csv(_RAW / "retailer_pos.csv", parse_dates=["transaction_date"])
    return df


@lru_cache(maxsize=1)
def _inventory() -> pd.DataFrame:
    df = pd.read_csv(_RAW / "retailer_inventory_weekly.csv", parse_dates=["week_end_date"])
    return df


@lru_cache(maxsize=1)
def _visits() -> pd.DataFrame:
    df = pd.read_csv(_RAW / "retailer_visit_log.csv", parse_dates=["visit_date"])
    return df


@lru_cache(maxsize=1)
def _growers() -> pd.DataFrame:
    return pd.read_csv(_RAW / "growers.csv")


class OutcomeIn(BaseModel):
    rep_id: str
    retailer_id: str
    tehsil: str
    product: str
    seq: int
    outcome: Literal["accepted", "completed", "order_placed", "sale_made", "no_purchase", "skipped"]
    captured_at: str | None = None
    order_value: float | None = Field(default=None, ge=0)
    notes: str | None = None


class OutcomesBatch(BaseModel):
    outcomes: list[OutcomeIn]


def _date_parts(date_iso: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    date = pd.Timestamp(date_iso)
    return date, date.replace(day=1)


def _signal_row(tehsil: str, product: str, date_iso: str) -> dict:
    _, month_start = _date_parts(date_iso)
    sig = _signals()
    row = sig[
        (sig["tehsil"] == tehsil)
        & (sig["product"] == product)
        & (sig["month_start"] == month_start)
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _monthly_row(tehsil: str, product: str, date_iso: str) -> dict:
    _, month_start = _date_parts(date_iso)
    panel = _monthly_panel()
    row = panel[
        (panel["tehsil"] == tehsil)
        & (panel["product"] == product)
        & (panel["month_start"] == month_start)
    ]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()


def _retailer_product_context(retailer_id: str, product: str, date_iso: str) -> dict:
    date, _ = _date_parts(date_iso)
    pos = _pos()
    inv = _inventory()

    product_pos = pos[(pos["retailer_id"] == retailer_id) & (pos["sku_name"] == product)]
    recent = product_pos[
        (product_pos["transaction_date"] <= date)
        & (product_pos["transaction_date"] > date - pd.Timedelta(days=28))
    ]
    previous = product_pos[
        (product_pos["transaction_date"] <= date - pd.Timedelta(days=28))
        & (product_pos["transaction_date"] > date - pd.Timedelta(days=56))
    ]
    recent_qty = float(recent["sku_qty"].sum())
    previous_qty = float(previous["sku_qty"].sum())
    velocity_delta_pct = round(100 * (recent_qty - previous_qty) / max(previous_qty, 1.0), 1)

    product_inv = inv[
        (inv["retailer_id"] == retailer_id)
        & (inv["sku_name"] == product)
        & (inv["week_end_date"] <= date)
    ].sort_values("week_end_date")
    stock_qty = float(product_inv.iloc[-1]["sku_qty"]) if not product_inv.empty else 0.0
    daily_velocity = recent_qty / 28.0
    days_of_stock = round(stock_qty / max(daily_velocity, 0.25), 1)

    all_recent = pos[
        (pos["retailer_id"] == retailer_id)
        & (pos["transaction_date"] <= date)
        & (pos["transaction_date"] > date - pd.Timedelta(days=28))
    ]
    non_focus_qty = float(all_recent[all_recent["sku_name"] != product]["sku_qty"].sum())
    competitive_pressure = "high" if non_focus_qty > recent_qty * 1.5 and non_focus_qty > 0 else "normal"

    return {
        "recent_qty_28d": round(recent_qty, 1),
        "previous_qty_28d": round(previous_qty, 1),
        "pos_velocity_delta_pct": velocity_delta_pct,
        "stock_qty": round(stock_qty, 1),
        "days_of_stock": days_of_stock,
        "inventory_status": "stock-out" if stock_qty <= 0 else "low" if days_of_stock <= 7 else "healthy",
        "competitive_pressure": competitive_pressure,
        "competitive_proxy_qty_28d": round(non_focus_qty, 1),
    }


def _grower_context(tehsil: str, date_iso: str) -> dict:
    date, _ = _date_parts(date_iso)
    growers = _growers()
    sub = growers[growers["tehsil"] == tehsil].copy()
    if sub.empty:
        return {"n_growers": 0, "crop": "unknown", "growth_stage": "unknown", "smartphone_pct": 0}
    crop = "unknown"
    stage = "sowing"
    try:
        parsed = sub["grower_crop_calendar"].dropna().map(json.loads)
        if len(parsed):
            calendar = parsed.iloc[0]
            crop = calendar.get("crop", "unknown")
            stages = calendar.get("stages", [])
            future = [s for s in stages if pd.Timestamp(s.get("approx")) >= date]
            stage = future[0]["stage"] if future else "late season"
    except Exception:
        pass
    smartphone_pct = round(100 * (sub["device_type"] == "smartphone").mean(), 1)
    return {
        "n_growers": int(len(sub)),
        "crop": crop,
        "growth_stage": stage,
        "smartphone_pct": smartphone_pct,
        "offline_attendance_pct": round(100 * sub["offline_campaign_attended"].fillna(False).mean(), 1),
    }


def _stop_evidence(stop: dict, date_iso: str) -> dict:
    sig = _signal_row(stop["tehsil"], stop["product"], date_iso)
    panel = _monthly_row(stop["tehsil"], stop["product"], date_iso)
    retail = _retailer_product_context(stop["retailer_id"], stop["product"], date_iso)
    growers = _grower_context(stop["tehsil"], date_iso)

    evidence = [
        {
            "key": "weather",
            "label": "Weather",
            "value": f"{float(panel.get('avg_rainfall_mm', 0)):.1f} mm rain / RH {float(panel.get('avg_rh_max', 0)):.0f}%",
            "reason": sig.get("disease_alert_reason") or "Weather and humidity are monitored for disease-pressure timing.",
            "severity": "high" if int(sig.get("disease_alert_flag", 0) or 0) else "normal",
        },
        {
            "key": "ndvi",
            "label": "Crop health (proxy)",
            "value": f"index {max(0.0, min(1.0, 1.0 - float(panel.get('avg_disease_pressure', 0) or 0))):.2f}",
            "reason": "Derived proxy from disease pressure and weather context — live NDVI/satellite feed is roadmap.",
            "severity": "high" if float(panel.get("avg_disease_pressure", 0) or 0) > 0.65 else "normal",
        },
        {
            "key": "pest_disease",
            "label": "Pest/disease (proxy)",
            "value": f"score {float(sig.get('disease_alert_score', 0)):.2f}",
            "reason": sig.get("disease_alert_reason") or "No acute disease alert, but window timing is still active. Note: disease pressure is a weather-driven proxy; government pest-surveillance-bulletin ingestion is roadmap via the same feed-agnostic connector contract.",
            "severity": "high" if int(sig.get("disease_alert_flag", 0) or 0) else "normal",
        },
        {
            "key": "inventory",
            "label": "Inventory",
            "value": f"{retail['inventory_status']} / {retail['days_of_stock']} days",
            "reason": f"{retail['stock_qty']} units on hand with {retail['recent_qty_28d']} units sold in the last 28 days.",
            "severity": "high" if retail["inventory_status"] in ("stock-out", "low") else "normal",
        },
        {
            "key": "pos",
            "label": "POS velocity",
            "value": f"{retail['pos_velocity_delta_pct']:+.1f}%",
            "reason": sig.get("demand_spike_reason") or "Recent POS velocity is compared with the previous 28-day baseline.",
            "severity": "high" if int(sig.get("demand_spike_flag", 0) or 0) or retail["pos_velocity_delta_pct"] > 25 else "normal",
        },
        {
            "key": "growth_stage",
            "label": "Crop stage",
            "value": f"{growers['crop']} / {growers['growth_stage']}",
            "reason": f"{growers['n_growers']} known growers in tehsil; {growers['smartphone_pct']}% smartphone reach.",
            "severity": "normal",
        },
        {
            "key": "competition",
            "label": "Competitive activity (proxy)",
            "value": retail["competitive_pressure"],
            "reason": f"Derived proxy: {retail['competitive_proxy_qty_28d']} units of non-focus SKUs sold at this outlet in 28 days — live competitor feed is roadmap.",
            "severity": "high" if retail["competitive_pressure"] == "high" else "normal",
        },
    ]
    return {"signals": sig, "panel": panel, "retail": retail, "growers": growers, "evidence": evidence}


def _next_best_action(stop: dict, date_iso: str) -> dict:
    ctx = _stop_evidence(stop, date_iso)
    retail = ctx["retail"]
    growers = ctx["growers"]
    window_days = int(stop["why"]["window_days_left"])
    product = stop["product"]

    if retail["inventory_status"] == "stock-out":
        promo = "Trigger urgent replenishment and capture lost-demand notes."
    elif retail["inventory_status"] == "low":
        promo = "Offer a small replenishment bundle before the local window closes."
    elif retail["competitive_pressure"] == "high":
        promo = "Use defensive retailer incentive and farmer proof points against competing SKU momentum."
    else:
        promo = "Deploy WhatsApp follow-up plus retailer counter-card for the active grower segment."

    advice = (
        f"Advise {growers['crop']} growers at {growers['growth_stage']} stage to complete protection within "
        f"{window_days} days; delay can reduce fit with the agronomic window."
    )
    return {
        "primary_product": product,
        "product_to_discuss": product,
        "agronomic_advice": advice,
        "promotional_mechanic": promo,
        "visit_objective": "secure order" if retail["inventory_status"] != "stock-out" else "recover stock-out",
        "talk_track": [
            f"Open with the {product} timing window.",
            f"Show POS/inventory context: {retail['recent_qty_28d']} units sold and {retail['days_of_stock']} days of stock.",
            "Record sale/order/no-purchase outcome before leaving the outlet.",
        ],
    }


def _enrich_stop(stop: dict, date_iso: str) -> dict:
    enriched = dict(stop)
    ctx = _stop_evidence(stop, date_iso)
    enriched["evidence"] = ctx["evidence"]
    enriched["next_best_action"] = _next_best_action(stop, date_iso)
    enriched["local_context"] = {
        "retailer": ctx["retail"],
        "growers": ctx["growers"],
        "signal_flags": {
            "demand_spike": bool(ctx["signals"].get("demand_spike_flag", 0)),
            "oos_opportunity": bool(ctx["signals"].get("oos_opportunity_flag", 0)),
            "disease_alert": bool(ctx["signals"].get("disease_alert_flag", 0)),
            "window_urgency": bool(ctx["signals"].get("window_urgency_flag", 0)),
            "digital_demand": bool(ctx["signals"].get("digital_demand_flag", 0)),
        },
    }
    return enriched


def _enrich_rep_day(rep_day: dict, date_iso: str) -> dict:
    enriched = dict(rep_day)
    enriched["stops"] = [_enrich_stop(stop, date_iso) for stop in rep_day.get("stops", [])]
    return enriched


def _priority_stops(plan: RoutePlan, date_iso: str, limit: int = 12) -> list[dict]:
    rows: list[dict] = []
    for rd in plan.rep_days:
        public_rd = _public_rep_day_dict(rd)
        for stop in public_rd.get("stops", []):
            rows.append({**stop, "rep_id": public_rd["rep_id"]})
    rows.sort(key=lambda s: (s["why"]["priority_pct"], s["why"]["window_days_left"]))
    return [_enrich_stop(stop, date_iso) for stop in rows[:limit]]


def _read_outcomes() -> list[dict]:
    if not _OUTCOMES.exists():
        return []
    rows = []
    for line in _OUTCOMES.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_outcomes(outcomes: list[OutcomeIn]) -> list[dict]:
    _PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = [o.model_dump() if hasattr(o, "model_dump") else o.dict() for o in outcomes]
    with _OUTCOMES.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
    return rows


def _learning_metrics() -> dict:
    rows = _read_outcomes()
    total = len(rows)
    positive = {"accepted", "completed", "order_placed", "sale_made"}
    accepted = sum(1 for r in rows if r.get("outcome") in positive)
    orders = sum(1 for r in rows if r.get("outcome") in {"order_placed", "sale_made"})
    return {
        "total_outcomes": total,
        "accepted_outcomes": accepted,
        "orders_or_sales": orders,
        "recommendation_acceptance_rate": round(100 * accepted / max(total, 1), 1) if total else 0.0,
        "recent": rows[-20:],
    }


def _recalibration_status(date_iso: str = "2025-11-17") -> dict:
    date, _ = _date_parts(date_iso)
    weekday = int(date.weekday())
    last_recalibrated = date - pd.Timedelta(days=weekday)
    next_recalibration = last_recalibrated + pd.Timedelta(days=7)
    metrics = _learning_metrics()
    return {
        "cadence": "weekly",
        "daily_planning_date": date.strftime("%Y-%m-%d"),
        "last_recalibrated": last_recalibrated.strftime("%Y-%m-%d"),
        "next_recalibration": next_recalibration.strftime("%Y-%m-%d"),
        "status": "ready" if metrics["total_outcomes"] >= 25 else "collecting_outcomes",
        "outcomes_since_recalibration": metrics["total_outcomes"],
        "rule": "New outcomes update acceptance priors daily; model artifacts are eligible for weekly refresh.",
    }


def _anomaly_feed(date_iso: str, limit: int = 20) -> list[dict]:
    _, month_start = _date_parts(date_iso)
    sig = _signals()
    sub = sig[sig["month_start"] == month_start].copy()
    items: list[dict] = []
    specs = [
        ("disease_alert", "Early pest/disease emergence (proxy)", "disease_alert_score", "disease_alert_reason", "Prioritize protection products and agronomic advice this week. Note: disease pressure is a weather-driven proxy; government pest-surveillance-bulletin ingestion is roadmap via the same feed-agnostic connector contract."),
        ("demand_spike", "Sudden demand spike", "demand_spike_z", "demand_spike_reason", "Visit outlets with rising POS before competitors capture demand."),
        ("oos_opportunity", "Competitor/stock-out opportunity", "oos_rate_max", "oos_opportunity_reason", "Replenish and defend retailer shelf availability."),
        ("digital_demand", "Digital demand signal", "digital_demand_score", "digital_demand_reason", "Follow up with campaign-aware retailer and grower messaging."),
        ("window_urgency", "Agronomic window urgency", "window_urgency_decay", "window_urgency_reason", "Move planned visits forward before the crop window closes."),
    ]
    for key, title, score_col, reason_col, action in specs:
        flag_col = f"{key}_flag"
        if flag_col not in sub.columns:
            continue
        flagged = sub[sub[flag_col] == 1].sort_values(score_col, ascending=False).head(6)
        for _, row in flagged.iterrows():
            items.append({
                "type": key,
                "title": title,
                "tehsil": row["tehsil"],
                "product": row["product"],
                "score": round(float(row.get(score_col, 0) or 0), 3),
                "reason": row.get(reason_col) or title,
                "recommended_action": action,
                "severity": "high" if key in {"disease_alert", "window_urgency", "oos_opportunity"} else "medium",
            })
    items.sort(key=lambda x: (x["severity"] != "high", -x["score"]))
    return items[:limit]


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "disha-api",
        "frozen_artifacts": {
            "monthly_panel":    (_PROCESSED / "monthly_panel.parquet").exists(),
            "cate_frozen":      (_PROCESSED / "cate_frozen.parquet").exists(),
            "uplift_real":      (_PROCESSED / "uplift_real_cate.parquet").exists(),
            "uplift_synth":     (_PROCESSED / "uplift_synthetic_cate.parquet").exists(),
            "signals":          (_PROCESSED / "signals_panel.parquet").exists(),
            "ope":              (_PROCESSED / "ope.json").exists(),
            "business_case":    (_PROCESSED / "business_case.json").exists(),
            "dgp_gate":         (_PROCESSED / "dgp_gate.json").exists(),
        },
    }


@app.get("/plan/{rep_id}")
def get_rep_plan(
    rep_id: str,
    date: str = Query(default="2025-11-17", description="ISO YYYY-MM-DD"),
    arm: Literal["real", "synthetic"] = "real",
):
    """Get a rep's ordered route for a specific date and arm.

    Default arm=real — the rep app sources the honest, real-data CATE
    RANKING (the same +0.263 residualized-Qini signal as the dashboard,
    sourced from uplift_eval.json::qini_window_residualized via /qini).
    The synthetic arm is dashboard-only, labelled for engine validation."""
    plan = _arm_plan(date, arm)
    rep_day = next((rd for rd in plan.rep_days if rd.rep_id == rep_id), None)
    if rep_day is None:
        raise HTTPException(404, f"rep {rep_id!r} not in {arm} plan for {date}")
    return {
        "arm": plan.arm,
        "date": plan.date,
        "solver_used": plan.solver_used,
        "rep_day": _enrich_rep_day(_public_rep_day_dict(rep_day), date),
        "learning": _learning_metrics(),
        "recalibration": _recalibration_status(date),
    }


@app.get("/retailer/{retailer_id}/why")
def get_retailer_why(
    retailer_id: str,
    date: str = Query(default="2025-11-17"),
    arm: Literal["real", "synthetic"] = "real",
):
    """Get the WHY explanation for visiting a specific retailer on a date."""
    plan = _arm_plan(date, arm)
    for rd in plan.rep_days:
        for stop in rd.stops:
            if stop.retailer_id == retailer_id:
                return {
                    "retailer_id": retailer_id,
                    "rep_id": rd.rep_id,
                    "tehsil": stop.tehsil,
                    "product": stop.product,
                    "why": _to_dict(stop.why),
                    "evidence": _enrich_stop(_public_stop_dict(stop), date).get("evidence", []),
                    "next_best_action": _enrich_stop(_public_stop_dict(stop), date).get("next_best_action", {}),
                    "service_min": stop.service_min,
                }
    # Retailer not in any plan that day — return base info from retailers table
    retailers = _retailers()
    r = retailers[retailers["retailer_id"] == retailer_id]
    if r.empty:
        raise HTTPException(404, f"retailer {retailer_id!r} unknown")
    return {
        "retailer_id": retailer_id,
        "tehsil": str(r.iloc[0]["tehsil"]),
        "in_todays_plan": False,
        "reason": "Not in today's optimized route — no in-window high-CATE cell.",
    }


@app.get("/territory/{rep_id}/signals")
def get_territory_signals(
    rep_id: str,
    date: str = Query(default="2025-11-17"),
):
    """L1 signal flags for tehsils in a rep's territory at a date."""
    reps = pd.read_csv(_ROOT / "data" / "raw" / "reps_territory.csv")
    rep = reps[reps["rep_id"] == rep_id]
    if rep.empty:
        raise HTTPException(404, f"rep {rep_id!r} unknown")
    tehsils = json.loads(rep.iloc[0]["tehsil_list"])
    month_start = pd.Timestamp(date).replace(day=1)
    sig = _signals()
    sub = sig[
        sig["tehsil"].isin(tehsils)
        & (sig["month_start"] == month_start)
    ]
    flag_cols = [c for c in sub.columns if c.endswith("_flag")]
    out = sub.groupby("tehsil")[flag_cols].max().reset_index()
    return {
        "rep_id": rep_id,
        "date": date,
        "tehsils": out.to_dict(orient="records"),
    }


@app.get("/sim/run")
def sim_run(
    arm: Literal["real", "synthetic"] = "real",
    date: str = Query(default="2025-11-17"),
):
    """Whole-territory simulator endpoint: returns the full plan for an arm
    plus aggregate metrics for the dashboard's split-screen view."""
    plan = _arm_plan(date, arm)
    return {
        "arm": arm,
        "date": date,
        "summary": _public_plan_summary(plan),
        "rep_days": [_public_rep_day_dict(rd) for rd in plan.rep_days],
        "priority_stops": _priority_stops(plan, date),
        "anomalies": _anomaly_feed(date),
        "learning": _learning_metrics(),
        "recalibration": _recalibration_status(date),
    }


@app.get("/business_case")
def get_business_case():
    p = _PROCESSED / "business_case.json"
    if not p.exists():
        raise HTTPException(503, "business_case.json not built")
    return json.loads(p.read_text())


@app.get("/qini")
def get_qini():
    """Headline metric: residualized window-Qini per estimator + dgp_gate facts."""
    eval_p = _PROCESSED / "uplift_eval.json"
    gate_p = _PROCESSED / "dgp_gate.json"
    ope_p = _PROCESSED / "ope.json"
    if not (eval_p.exists() and gate_p.exists()):
        raise HTTPException(503, "frozen eval / gate artifacts not built")
    return {
        "demo_path":      json.loads(gate_p.read_text()).get("demo_path"),
        "uplift_eval":    json.loads(eval_p.read_text()),
        "ope":            json.loads(ope_p.read_text()) if ope_p.exists() else None,
    }


@app.get("/anomalies")
def get_anomalies(
    date: str = Query(default="2025-11-17"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """Actionable anomaly/opportunity feed derived from signal artifacts."""
    return {"date": date, "items": _anomaly_feed(date, limit)}


@app.post("/outcomes")
def post_outcomes(batch: OutcomesBatch):
    """Persist visit outcomes for the daily learning loop."""
    rows = _append_outcomes(batch.outcomes)
    return {
        "accepted": len(rows),
        "learning": _learning_metrics(),
        "recalibration": _recalibration_status(),
    }


@app.get("/learning")
def get_learning():
    """Current outcome-learning counters used by manager and rep UIs."""
    return {
        "learning": _learning_metrics(),
        "recalibration": _recalibration_status(),
    }
