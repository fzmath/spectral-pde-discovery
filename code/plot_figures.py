"""Generate figures for the paper."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

CODE = Path(r"H:\2026科研\Spectral-PDE-Discovery\code")
PAPER = Path(r"H:\2026科研\Spectral-PDE-Discovery\paper")
PAPER.mkdir(exist_ok=True)

with open(CODE / "spd_v3_extended_results.json") as f:
    R1 = json.load(f)
with open(CODE / "spd_2d_extended.json") as f:
    R2 = json.load(f)

M1 = ['FD+STLSQ', 'Spectral+STLSQ', 'SPD v3 (ours)']
M2 = ['FD+STLSQ', 'Spectral+STLSQ', 'SPD (ours)']
LAB = ['FD+STLSQ', 'Spectral+STLSQ', 'SPD (ours)']
C = ['#4C72B0', '#55A868', '#C44E52']
MK = ['o', 's', '^']
noises = ['clean', 'snr_40', 'snr_30', 'snr_20', 'snr_10']
nlab = ['Clean', 'SNR 40', 'SNR 30', 'SNR 20', 'SNR 10']

def save(fig, name):
    fig.savefig(PAPER / f"{name}.pdf")
    try: fig.savefig(PAPER / f"{name}.eps")
    except: pass
    plt.close(fig)

# Fig 1: 1D recovery rates
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
pdes = ['burgers','kdv','kuramoto_sivishinky','advection1d','nls']
titles = ['Burgers','KdV','KS','Advection','NLS']
for idx,(pde,title) in enumerate(zip(pdes,titles)):
    ax = axes[idx]; x = np.arange(5); w = 0.25
    for mi,m in enumerate(M1):
        rates = []
        for ns in noises:
            if ns in R1['1d'][pde] and 'error' not in R1['1d'][pde][ns]:
                rates.append(R1['1d'][pde][ns][m]['true_found']/R1['1d'][pde][ns][m]['total_true']*100)
            else: rates.append(0)
        ax.bar(x+mi*w, rates, w, label=LAB[mi], color=C[mi], alpha=0.85)
    ax.set_xticks(x+w); ax.set_xticklabels(nlab, rotation=45, ha='right')
    ax.set_ylim(0,110); ax.set_title(title)
    if idx==0: ax.set_ylabel('Recovery (%)')
    if idx==2: ax.legend(loc='upper right', framealpha=0.9)
fig.suptitle('True Term Recovery Rate (1D PDEs)', y=1.02, fontsize=14)
fig.tight_layout(); save(fig, "fig1_recovery_rate"); print("Fig1 done")

# Fig 2: Advection highlight
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,4))
pde='advection1d'; x=np.arange(5); w=0.25
for mi,m in enumerate(M1):
    rates=[R1['1d'][pde][ns][m]['true_found']/R1['1d'][pde][ns][m]['total_true']*100 for ns in noises]
    ax1.bar(x+mi*w,rates,w,label=LAB[mi],color=C[mi],alpha=0.85)
ax1.set_xticks(x+w); ax1.set_xticklabels(nlab,rotation=45,ha='right')
ax1.set_ylim(0,110); ax1.set_ylabel('Recovery (%)'); ax1.set_title('Advection (non-periodic)'); ax1.legend()
for mi,m in enumerate(M1):
    fps=[len(R1['1d'][pde][ns][m]['false_positives']) for ns in noises]
    ax2.plot(x,fps,marker=MK[mi],label=LAB[mi],color=C[mi],lw=2,ms=8)
ax2.set_xticks(x); ax2.set_xticklabels(nlab,rotation=45,ha='right')
ax2.set_ylabel('False positives'); ax2.set_title('False positives (Advection)'); ax2.legend(); ax2.grid(alpha=0.3)
fig.tight_layout(); save(fig,"fig2_advection"); print("Fig2 done")

# Fig 3: 2D results
fig,axes = plt.subplots(1,3,figsize=(14,4))
pdes2 = ['advection_diffusion_2d','reaction_diffusion_2d','heat_soil_uniform_2d_p1']
t2 = ['2D Adv-Diff','2D React-Diff','2D Heat']
ns2 = ['clean','snr_40','snr_20']
for idx,(pde,title) in enumerate(zip(pdes2,t2)):
    ax=axes[idx]
    data = R1['2d'].get(pde, R2.get(pde, {}))
    mlist = M1 if pde in R1['2d'] else M2
    x=np.arange(3); w=0.25
    for mi,m in enumerate(mlist):
        rates=[]
        for ns in ns2:
            if ns in data and m in data[ns]:
                rates.append(data[ns][m]['true_found']/data[ns][m]['total_true']*100)
            else: rates.append(0)
        ax.bar(x+mi*w,rates,w,label=LAB[mi],color=C[mi],alpha=0.85)
    ax.set_xticks(x+w); ax.set_xticklabels(['Clean','SNR 40','SNR 20'])
    ax.set_ylim(0,110); ax.set_title(title)
    if idx==0: ax.set_ylabel('Recovery (%)')
    if idx==2: ax.legend()
fig.suptitle('Two-Dimensional PDE Discovery', y=1.02, fontsize=14)
fig.tight_layout(); save(fig,"fig3_2d_results"); print("Fig3 done")

# Fig 4: timing
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
t1={m:[] for m in M1}
for pde in pdes:
    for ns in noises:
        if ns in R1['1d'][pde] and 'error' not in R1['1d'][pde][ns]:
            for m in M1: t1[m].append(R1['1d'][pde][ns][m]['time_s'])
x=np.arange(3); avg=[np.mean(t1[m]) for m in M1]
ax1.bar(x,avg,color=C,alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(['FD+\nSTLSQ','Spec+\nSTLSQ','SPD\n(ours)'])
ax1.set_ylabel('Time (s)'); ax1.set_title('1D (avg)')
for i,v in enumerate(avg): ax1.text(i,v+0.05,f'{v:.2f}s',ha='center',fontsize=10)

t2d={m:[] for m in M2}
for pde in pdes2:
    data=R1['2d'].get(pde,R2.get(pde,{}))
    for ns in ns2:
        if ns in data:
            for m in M2:
                if m in data[ns]: t2d[m].append(data[ns][m]['time_s'])
avg2=[np.mean(t2d[m]) if t2d[m] else 0 for m in M2]
ax2.bar(x,avg2,color=C,alpha=0.85)
ax2.set_xticks(x); ax2.set_xticklabels(['FD+\nSTLSQ','Spec+\nSTLSQ','SPD\n(ours)'])
ax2.set_ylabel('Time (s)'); ax2.set_title('2D (avg)')
for i,v in enumerate(avg2): ax2.text(i,v+0.1,f'{v:.2f}s',ha='center',fontsize=10)
fig.suptitle('Computational Cost', y=1.02, fontsize=14)
fig.tight_layout(); save(fig,"fig4_timing"); print("Fig4 done")
print("\nAll figures generated!")
