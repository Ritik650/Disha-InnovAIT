"""Diagnostic: characterize seed-sensitivity of agronomic CATE cross-seed r."""
import logging
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

from disha.twin.dgp_gate import crossfit_tlearner_stability

panel = pd.read_parquet("data/processed/monthly_panel.parquet")

print(f"{'seed':>6} {'agronomic_r':>12} {'structural_r':>13} {'gap':>8}")
print("-" * 45)
for s in [42, 7, 99, 2024, 1234]:
    res = crossfit_tlearner_stability(panel, seed=s)
    agr = res["stability_score"]
    struct = res["structural_dummy_stability_score"]
    gap = res["structural_minus_agronomic_gap"]
    print(f"{s:>6} {agr:>12.4f} {struct:>13.4f} {gap:>+8.4f}")
