"""
disha.signals.base — SignalEvent dataclass and default thresholds.

Each signal is a CATE effect-modifier: it explains *who responds differently*
to a rep visit, not just *who buys more*.  Signals that correlate with baseline
revenue but not CATE are flagged as distractors and excluded from the uplift
feature set (kept as route-context only).

Signal types:
  demand_spike          — z-score POS revenue surge above 4-week baseline
  oos_opportunity       — product OOS + rising demand (supply-gap opportunity)
  disease_pressure_alert— weather disease pressure active while crop in-window
  window_urgency        — agronomic protection window in peak decay zone
  digital_demand_signal — WhatsApp-active tehsil with high product scan interest
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


@dataclass
class SignalEvent:
    tehsil: str
    month_start: pd.Timestamp
    product: str              # product name, or "all" for tehsil-level signals
    signal_type: str          # one of the five keys below
    flag: int                 # 1 = signal active, 0 = not
    score: float              # raw magnitude (z-score, oos_rate, decay, etc.)
    reason_template: str      # plain-language WHY for L4 rep client


# Default detection thresholds — override via SIGNAL_CONFIG in settings or tests
SIGNAL_CONFIG: dict = {
    "demand_spike_z_threshold": 1.5,      # z-score above 4-wk rolling baseline
    "oos_rate_threshold": 0.10,            # ≥10% retailers OOS for this product
    "disease_window_decay_threshold": 0.20, # window must be open to alert
    "disease_pressure_threshold": 0.20,    # avg disease_pressure_flag in month
    "window_urgency_decay_threshold": 0.65, # top decay zone (window closing)
    "digital_wa_threshold": 0.0,           # any WA engagement (binary in data)
    "digital_scan_threshold": 0.25,        # ≥25% growers scanned this product
}

SIGNAL_TYPES = [
    "demand_spike",
    "oos_opportunity",
    "disease_pressure_alert",
    "window_urgency",
    "digital_demand_signal",
]
