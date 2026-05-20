"""
disha.twin.store — TwinStore: the runtime interface to the territory digital twin.

Usage
-----
    store = TwinStore()
    state = store.state("TER_0001", "Patna_T001", date(2026, 1, 10))
    windows = store.window_status("Patna_T001", "Tilt 250 EC", date(2026, 1, 10))
"""
from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]


class TwinStore:
    """
    Runtime facade over twin.parquet.

    .state()          → raw feature dict for one (territory, tehsil, week)
    .window_status()  → list[WindowStatus] with 3-tier crop fallback
    .lift_probe_result → dict loaded from lift_probe.json (set after L0 build)
    .use_synthetic_dgp → bool flag for L2 trainer
    """

    def __init__(self, twin_path: Optional[Path] = None):
        import yaml
        with open(_ROOT / "config" / "settings.yaml") as f:
            cfg = yaml.safe_load(f)

        if twin_path is None:
            twin_path = _ROOT / cfg["paths"]["twin_parquet"]

        if not Path(twin_path).exists():
            warnings.warn(
                f"twin.parquet not found at {twin_path}. "
                "Run `python -m disha.twin.build` first. TwinStore will return empty states.",
                stacklevel=2,
            )
            self._twin: Optional[pd.DataFrame] = None
        else:
            self._twin = pd.read_parquet(twin_path)
            # Ensure week_start is datetime for fast comparisons
            self._twin["week_start"] = pd.to_datetime(self._twin["week_start"])

        # Load growers + district map once for window_status lookups
        from disha.common.data import get_tehsil_district_map, load_growers
        self._growers = load_growers()
        self._tehsil_district_map = get_tehsil_district_map()

        import json

        # Load lift probe result if available
        probe_path = _ROOT / cfg["paths"]["processed"] / "lift_probe.json"
        self.lift_probe_result: Optional[dict] = None
        self.use_synthetic_dgp: bool = False
        if probe_path.exists():
            with open(probe_path) as f:
                self.lift_probe_result = json.load(f)
            self.use_synthetic_dgp = self.lift_probe_result.get("ate_is_flat", False)

        # Load DGP gate result if available (written by dgp_gate.run_and_save_dgp_gate)
        gate_path = _ROOT / cfg["paths"]["processed"] / "dgp_gate.json"
        self._dgp_gate: Optional[dict] = None
        if gate_path.exists():
            with open(gate_path) as f:
                self._dgp_gate = json.load(f)

    # ── Demo path ─────────────────────────────────────────────────────────────

    @property
    def demo_path(self) -> str:
        """
        Headline demo path from dgp_gate.json.
        Falls back to lift_probe.json ate_is_flat flag if gate not built yet.
        """
        if self._dgp_gate is not None:
            return self._dgp_gate.get("demo_path", "pending_l2_assessment")
        return "synthetic_dgp_headline" if self.use_synthetic_dgp else "real_data_headline"

    # ── Core state lookup ──────────────────────────────────────────────────────

    def state(
        self,
        territory_id: str,
        tehsil: str,
        week: date | datetime | str,
    ) -> dict:
        """
        Return the full feature vector for (territory, tehsil, week) as a dict.
        Returns {} if the twin has not been built or the key does not exist.
        """
        if self._twin is None:
            return {}

        week_ts = pd.Timestamp(week)
        mask = (
            (self._twin["territory_id"] == territory_id) &
            (self._twin["tehsil"] == tehsil) &
            (self._twin["week_start"] == week_ts)
        )
        rows = self._twin.loc[mask]
        if rows.empty:
            return {}
        return rows.iloc[0].to_dict()

    def state_by_tehsil(self, tehsil: str, week: date | datetime | str) -> dict:
        """State lookup without territory_id (useful for optimizer hot-path)."""
        if self._twin is None:
            return {}
        week_ts = pd.Timestamp(week)
        mask = (self._twin["tehsil"] == tehsil) & (self._twin["week_start"] == week_ts)
        rows = self._twin.loc[mask]
        if rows.empty:
            return {}
        return rows.iloc[0].to_dict()

    # ── Window status (delegates to data module with district-level fallback) ──

    def window_status(
        self,
        tehsil: str,
        product: str,
        query_date: date | datetime | str,
    ):
        """3-tier window_status call using stored growers + district map."""
        from disha.common.data import window_status as _ws
        district = self._tehsil_district_map.get(tehsil)
        return _ws(
            tehsil=tehsil,
            product=product,
            query_date=query_date,
            growers_df=self._growers,
            district=district,
        )

    def best_window_status(self, tehsil: str, product: str, query_date):
        """Return the single most urgent open WindowStatus or None."""
        from disha.common.data import best_window_status as _bws
        district = self._tehsil_district_map.get(tehsil)
        from disha.common.data import window_status as _ws
        statuses = _ws(
            tehsil=tehsil,
            product=product,
            query_date=query_date,
            growers_df=self._growers,
            district=district,
        )
        open_ws = [s for s in statuses if s.open]
        return min(open_ws, key=lambda s: s.days_left) if open_ws else None

    # ── Convenience accessors ──────────────────────────────────────────────────

    def available_weeks(self) -> list[pd.Timestamp]:
        if self._twin is None:
            return []
        return sorted(self._twin["week_start"].unique())

    def available_tehsils(self, territory_id: Optional[str] = None) -> list[str]:
        if self._twin is None:
            return []
        df = self._twin
        if territory_id:
            df = df[df["territory_id"] == territory_id]
        return sorted(df["tehsil"].unique())

    def territory_tehsils(self, territory_id: str) -> list[str]:
        return self.available_tehsils(territory_id)

    def dominant_crop(self, tehsil: str) -> Optional[str]:
        if self._twin is None:
            return None
        row = self._twin.loc[self._twin["tehsil"] == tehsil].head(1)
        return row["dominant_crop"].iloc[0] if not row.empty else None

    def win_decay(self, tehsil: str, product: str, week: date | datetime | str) -> float:
        """Shortcut: window decay factor for a specific (tehsil, product, week)."""
        s = self.state_by_tehsil(tehsil, week)
        from disha.twin.build import _safe_col
        col = f"win_decay_{_safe_col(product)}"
        return float(s.get(col, 0.0))
