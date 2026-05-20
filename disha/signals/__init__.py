"""disha.signals — L1 signal detectors (CATE effect-modifiers)."""

from disha.signals.base import SIGNAL_CONFIG, SIGNAL_TYPES, SignalEvent
from disha.signals.correlations import (
    compute_signal_cate_correlations,
    run_and_save_correlations,
)
from disha.signals.demand import compute_demand_spike
from disha.signals.digital import compute_digital_demand
from disha.signals.disease import compute_disease_pressure_alert
from disha.signals.oos import compute_oos_opportunity
from disha.signals.run import run_all_detectors, run_and_save_signals
from disha.signals.window import compute_window_urgency

__all__ = [
    "SIGNAL_CONFIG",
    "SIGNAL_TYPES",
    "SignalEvent",
    "compute_demand_spike",
    "compute_oos_opportunity",
    "compute_disease_pressure_alert",
    "compute_window_urgency",
    "compute_digital_demand",
    "run_all_detectors",
    "run_and_save_signals",
    "compute_signal_cate_correlations",
    "run_and_save_correlations",
]
