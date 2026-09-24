"""Generate spectrum visualization figure for advection equation."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

data = np.load(r"H:\2026科研\Spectral-PDE-Discovery\code\spectrum_data.npz")
spec_st = data['spec_st']  # spatiotemporal spectrum
spec_filt = data['spec_filt']  # filtered
KX = data['KX']; KT = data['KT']
adv_x = data['adv_x']; adv_t = data['adv_t']

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif',
    'axes.linewidth': 0.8, 'xtick.direction': 'out', 'ytick.direction': 'out'
})

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

# Panel a: raw spatiotemporal spectrum
ax = axes[0]
im = ax.pcolormesh(KX, KT, np.log10(spec_st+1e-10), cmap='viridis', shading='auto')
ax.set_xlabel('$k_x$'); ax.set_ylabel('$k_t$')
ax.set_title('(a) Raw spatiotemporal spectrum')
ax.set_xlim([-30, 30]); ax.set_ylim([-3, 3])
# Draw characteristic line k_t = -c*k_x, c=0.1
kx_line = np.linspace(-30, 30, 100)
ax.plot(kx_line, -0.1*kx_line, 'r--', lw=1.2, label='$k_t=-ck_x$')
ax.legend(fontsize=8, loc='upper right')

# Panel b: after filter
ax = axes[1]
im = ax.pcolormesh(KX, KT, np.log10(spec_filt+1e-10), cmap='viridis', shading='auto')
ax.set_xlabel('$k_x$'); ax.set_ylabel('$k_t$')
ax.set_title('(b) After exponential filter')
ax.set_xlim([-30, 30]); ax.set_ylim([-3, 3])
ax.plot(kx_line, -0.1*kx_line, 'r--', lw=1.2)

# Panel c: spatial-only spectrum (averaged over time)
ax = axes[2]
kx = 2*np.pi*np.fft.fftfreq(len(adv_x), d=(adv_x[1]-adv_x[0]))
u_field = np.load(Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde\advection1d\advection1d.npz"))['u']
if u_field.ndim == 3: u_field = u_field[:,:,0]
spec_sp = np.fft.fft(u_field, axis=0)
spec_sp_avg = np.mean(np.abs(spec_sp), axis=1)
spec_sp_shifted = np.fft.fftshift(spec_sp_avg)
kx_shifted = np.fft.fftshift(kx)
ax.semilogy(kx_shifted, spec_sp_shifted+1e-10, 'b-', lw=1)
ax.set_xlabel('$k_x$'); ax.set_ylabel('$|\\hat{u}(k_x)|$ (avg over t)')
ax.set_title('(c) Spatial-only spectrum')
ax.set_xlim([-30, 30])
ax.axvline(x=0.6*np.max(np.abs(kx)), color='r', ls='--', lw=1, label='$k_c$')
ax.legend(fontsize=8)

plt.tight_layout()
out = Path(r"H:\2026科研\Spectral-PDE-Discovery\submission\figs\spectrum_viz.pdf")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, bbox_inches='tight')
plt.savefig(Path(r"H:\2026科研\Spectral-PDE-Discovery\submission\figs\spectrum_viz.eps"), bbox_inches='tight')
print(f"Saved {out}")
