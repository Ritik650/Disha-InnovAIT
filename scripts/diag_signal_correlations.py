"""Re-run L1 signal-CATE correlations on the corrected cate_frozen.parquet."""
import pandas as pd
from disha.signals.correlations import run_and_save_correlations

sig = pd.read_parquet("data/processed/signals_panel.parquet")
res = run_and_save_correlations(sig)
print(f"\n{'signal':25s} {'prev':>7s} {'r':>9s} {'dCATE':>10s} {'dResid':>10s} {'shrink':>7s} {'role':>15s}")
print("-" * 95)
for name, r in res["signals"].items():
    print(f"{name:25s} {r['prevalence']*100:6.2f}% "
          f"{r['spearman_r_magnitude_vs_cate']:+9.4f} "
          f"{r['delta_mean_cate']:+10.1f} "
          f"{r['delta_mean_cate_residualized']:+10.1f} "
          f"{r['shrink_after_residualization']*100:6.0f}% "
          f"{r['role']:>15s}")
