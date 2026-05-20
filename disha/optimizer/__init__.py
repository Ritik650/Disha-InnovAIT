"""disha.optimizer — L3 deadline-aware route optimizer.

Dual-arm: takes either the real or synthetic CATE source, produces an
ordered per-rep daily route plan under window + capacity constraints.
"""
from disha.optimizer.router import (
    RoutePlan,
    RepDayPlan,
    Stop,
    StopWhy,
    build_route_plan,
    run_dual_arm,
)

__all__ = [
    "RoutePlan", "RepDayPlan", "Stop", "StopWhy",
    "build_route_plan", "run_dual_arm",
]
