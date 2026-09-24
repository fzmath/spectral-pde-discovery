"""Final validation of SPD v6 with optimized parameters."""
import numpy as np, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'H:\2026科研\Spectral-PDE-Discovery\code')
from spd_v6_weak import spd_v6, load_1d, TRUE_PDES
from pathlib import Path
import json

# Optimized params from sweep
OPT = dict(n_test=100, width_ratio=0.15, penalty=2.5, cutoff=0.7)

datasets = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d']
noises = [None, 'snr_40', 'snr_30', 'snr_20', 'snr_10']
results = {}

print("=" * 80)
print("SPD v6 FINAL (optimized: n_test=100, width=0.15, pen=2.5, cut=0.7)")
print("=" * 80)

for ds in datasets:
    print(f"\n--- {ds} ---")
    results[ds] = {}
    tt = TRUE_PDES[ds]
    for noise in noises:
        label = 'clean' if noise is None else noise
        nf = f"{ds}_{noise}.npz" if noise else None
        u, du, x, t = load_1d(ds, nf)
        r = spd_v6(u, du, x, t, tt, **OPT)
        results[ds][label] = r
        status = "EXACT" if r['exact'] else f"{r['true_found']}/{r['total_true']}+{len(r['false_positives'])}FP"
        print(f"  {label:8s}: {status:14s} ({r['time_s']:.1f}s)")

out = Path(r"H:\2026科研\Spectral-PDE-Discovery\code\spd_v6_final_results.json")
with open(out, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
print(f"\nSaved to {out}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY: True terms recovered (EXACT count)")
print("=" * 80)
exact_count = 0
total = 0
for ds in datasets:
    for noise in ['clean', 'snr_40', 'snr_30', 'snr_20', 'snr_10']:
        total += 1
        if results[ds][noise]['exact']:
            exact_count += 1
print(f"EXACT recovery: {exact_count}/{total} cases")
