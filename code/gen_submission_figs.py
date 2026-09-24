"""
Regenerate figures as both PDF and EPS for submission.
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

PAPER_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\paper")
CODE_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\code")

with open(CODE_DIR / "spd_v6_final_results.json") as f:
    v6 = json.load(f)

baselines = {
    'burgers': {
        'clean': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 0}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 0}},
        'snr_40': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 6}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 0}},
        'snr_30': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 6}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 2}},
        'snr_20': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 6}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 1}},
        'snr_10': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 6}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 3}},
    },
    'kdv': {
        'clean': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 3}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 0}},
        'snr_40': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 3}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 1}},
        'snr_30': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 7}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 1}},
        'snr_20': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 9}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 0}},
        'snr_10': {'SR3_FD': {'true_found': 2, 'total': 2, 'false_pos': 9}, 'WeakSINDy': {'true_found': 2, 'total': 2, 'false_pos': 1}},
    },
    'kuramoto_sivishinky': {
        'clean': {'SR3_FD': {'true_found': 3, 'total': 3, 'false_pos': 0}, 'WeakSINDy': {'true_found': 3, 'total': 3, 'false_pos': 0}},
        'snr_40': {'SR3_FD': {'true_found': 3, 'total': 3, 'false_pos': 9}, 'WeakSINDy': {'true_found': 3, 'total': 3, 'false_pos': 0}},
        'snr_30': {'SR3_FD': {'true_found': 3, 'total': 3, 'false_pos': 9}, 'WeakSINDy': {'true_found': 3, 'total': 3, 'false_pos': 0}},
        'snr_20': {'SR3_FD': {'true_found': 2, 'total': 3, 'false_pos': 6}, 'WeakSINDy': {'true_found': 3, 'total': 3, 'false_pos': 0}},
        'snr_10': {'SR3_FD': {'true_found': 2, 'total': 3, 'false_pos': 6}, 'WeakSINDy': {'true_found': 3, 'total': 3, 'false_pos': 0}},
    },
    'advection1d': {
        'clean': {'SR3_FD': {'true_found': 0, 'total': 1, 'false_pos': 0}, 'WeakSINDy': {'true_found': 1, 'total': 1, 'false_pos': 0}},
        'snr_40': {'SR3_FD': {'true_found': 1, 'total': 1, 'false_pos': 2}, 'WeakSINDy': {'true_found': 1, 'total': 1, 'false_pos': 2}},
        'snr_30': {'SR3_FD': {'true_found': 1, 'total': 1, 'false_pos': 3}, 'WeakSINDy': {'true_found': 1, 'total': 1, 'false_pos': 0}},
        'snr_20': {'SR3_FD': {'true_found': 1, 'total': 1, 'false_pos': 4}, 'WeakSINDy': {'true_found': 1, 'total': 1, 'false_pos': 2}},
        'snr_10': {'SR3_FD': {'true_found': 1, 'total': 1, 'false_pos': 3}, 'WeakSINDy': {'true_found': 1, 'total': 1, 'false_pos': 2}},
    },
}

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150})

# Figure 1
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
pdes = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d']
labels = ['Burgers', 'KdV', 'KS', 'Advection']
noise_labels = ['clean', 'snr_40', 'snr_30', 'snr_20', 'snr_10']
noise_display = ['clean', '40', '30', '20', '10']

for idx, (pde, label) in enumerate(zip(pdes, labels)):
    ax = axes[idx]
    x = np.arange(5)
    width = 0.25
    spd_rates = [v6[pde][nl]['true_found']/v6[pde][nl]['total_true']*100 for nl in noise_labels]
    sr3_rates = [baselines[pde][nl]['SR3_FD']['true_found']/baselines[pde][nl]['SR3_FD']['total']*100 for nl in noise_labels]
    weak_rates = [baselines[pde][nl]['WeakSINDy']['true_found']/baselines[pde][nl]['WeakSINDy']['total']*100 for nl in noise_labels]
    ax.bar(x - width, sr3_rates, width, label='SR3 (FD)', color='#4C72B0', alpha=0.8)
    ax.bar(x, weak_rates, width, label='Weak SINDy', color='#55A868', alpha=0.8)
    ax.bar(x + width, spd_rates, width, label='SPD (ours)', color='#C44E52', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(noise_display, fontsize=9)
    ax.set_title(label, fontsize=12); ax.set_ylim(0, 110)
    if idx == 0: ax.set_ylabel('True term recovery (%)')
    if idx == 0: ax.legend(fontsize=8, loc='lower left')

plt.tight_layout()
plt.savefig(PAPER_DIR / "fig1_recovery_rate.pdf", bbox_inches='tight')
plt.savefig(PAPER_DIR / "fig1_recovery_rate.eps", bbox_inches='tight', format='eps')
plt.close()

# Figure 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
pde = 'advection1d'
x = np.arange(5); width = 0.25
spd_rates = [v6[pde][nl]['true_found']/v6[pde][nl]['total_true']*100 for nl in noise_labels]
sr3_rates = [baselines[pde][nl]['SR3_FD']['true_found']/baselines[pde][nl]['SR3_FD']['total']*100 for nl in noise_labels]
weak_rates = [baselines[pde][nl]['WeakSINDy']['true_found']/baselines[pde][nl]['WeakSINDy']['total']*100 for nl in noise_labels]
ax1.bar(x - width, sr3_rates, width, label='SR3 (FD)', color='#4C72B0', alpha=0.8)
ax1.bar(x, weak_rates, width, label='Weak SINDy', color='#55A868', alpha=0.8)
ax1.bar(x + width, spd_rates, width, label='SPD (ours)', color='#C44E52', alpha=0.8)
ax1.set_xticks(x); ax1.set_xticklabels(noise_display)
ax1.set_ylabel('Recovery rate (%)'); ax1.set_title('True term recovery'); ax1.legend(fontsize=9)

spd_fp = [len(v6[pde][nl]['false_positives']) for nl in noise_labels]
sr3_fp = [baselines[pde][nl]['SR3_FD']['false_pos'] for nl in noise_labels]
weak_fp = [baselines[pde][nl]['WeakSINDy']['false_pos'] for nl in noise_labels]
ax2.bar(x - width, sr3_fp, width, label='SR3 (FD)', color='#4C72B0', alpha=0.8)
ax2.bar(x, weak_fp, width, label='Weak SINDy', color='#55A868', alpha=0.8)
ax2.bar(x + width, spd_fp, width, label='SPD (ours)', color='#C44E52', alpha=0.8)
ax2.set_xticks(x); ax2.set_xticklabels(noise_display)
ax2.set_ylabel('False positives'); ax2.set_title('False positives'); ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig(PAPER_DIR / "fig2_advection.pdf", bbox_inches='tight')
plt.savefig(PAPER_DIR / "fig2_advection.eps", bbox_inches='tight', format='eps')
plt.close()

# Figure 3
fig, ax = plt.subplots(figsize=(8, 4))
pdes_2d = ['Adv-Diff', 'React-Diff', 'Heat (non-per)']
x = np.arange(3); width = 0.25
spd_2d = [100, 100, 83.3]
fd_2d = [83.3, 88.9, 0]
ax.bar(x - width/2, fd_2d, width, label='FD+STLSQ', color='#4C72B0', alpha=0.8)
ax.bar(x + width/2, spd_2d, width, label='SPD (ours)', color='#C44E52', alpha=0.8)
ax.set_xticks(x); ax.set_xticklabels(pdes_2d)
ax.set_ylabel('Avg recovery rate (%)'); ax.set_title('2D PDE recovery (pointwise variant)')
ax.legend(); ax.set_ylim(0, 110)
plt.tight_layout()
plt.savefig(PAPER_DIR / "fig3_2d_results.pdf", bbox_inches='tight')
plt.savefig(PAPER_DIR / "fig3_2d_results.eps", bbox_inches='tight', format='eps')
plt.close()

# Figure 4
fig, ax = plt.subplots(figsize=(7, 4))
methods = ['SR3 (FD)\n+ grid search', 'Weak SINDy', 'SPD\n(ours)']
times_1d = [3.0, 120.0, 5.0]
colors = ['#4C72B0', '#55A868', '#C44E52']
bars = ax.bar(methods, times_1d, color=colors, alpha=0.8, width=0.5)
ax.set_ylabel('Time per run (s)'); ax.set_title('Computational cost (1D PDEs)')
ax.set_yscale('log')
for bar, t in zip(bars, times_1d):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1, f'{t:.0f}s', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(PAPER_DIR / "fig4_timing.pdf", bbox_inches='tight')
plt.savefig(PAPER_DIR / "fig4_timing.eps", bbox_inches='tight', format='eps')
plt.close()

print("All figures (PDF + EPS) generated.")
