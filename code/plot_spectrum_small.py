"""Regenerate spectrum visualization at lower DPI to reduce file size."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

data = np.load(r"H:\2026科研\Spectral-PDE-Discovery\code\spectrum_data.npz")
spec_st = data['spec_st']
spec_filt = data['spec_filt']
KX = data['KX']; KT = data['KT']

plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif',
    'axes.linewidth': 0.6,
})

fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))

ax = axes[0]
ax.pcolormesh(KX, KT, np.log10(spec_st+1e-10), cmap='viridis', shading='auto', rasterized=True)
ax.set_xlabel('$k_x$'); ax.set_ylabel('$k_t$')
ax.set_title('(a) Raw ST spectrum', fontsize=9)
ax.set_xlim([-30, 30]); ax.set_ylim([-3, 3])
kx_line = np.linspace(-30, 30, 100)
ax.plot(kx_line, -0.1*kx_line, 'r--', lw=0.8)

ax = axes[1]
ax.pcolormesh(KX, KT, np.log10(spec_filt+1e-10), cmap='viridis', shading='auto', rasterized=True)
ax.set_xlabel('$k_x$'); ax.set_ylabel('$k_t$')
ax.set_title('(b) After filter', fontsize=9)
ax.set_xlim([-30, 30]); ax.set_ylim([-3, 3])
ax.plot(kx_line, -0.1*kx_line, 'r--', lw=0.8)

ax = axes[2]
kx = 2*np.pi*np.fft.fftfreq(1024, d=10.0/1024)
u_field = np.load(Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde\advection1d\advection1d.npz"))['u']
if u_field.ndim == 3: u_field = u_field[:,:,0]
spec_sp = np.fft.fft(u_field, axis=0)
spec_sp_avg = np.mean(np.abs(spec_sp), axis=1)
spec_sp_shifted = np.fft.fftshift(spec_sp_avg)
kx_shifted = np.fft.fftshift(kx)
ax.semilogy(kx_shifted, spec_sp_shifted+1e-10, 'b-', lw=0.8)
ax.set_xlabel('$k_x$'); ax.set_ylabel('$|\\hat{u}(k_x)|$')
ax.set_title('(c) Spatial-only', fontsize=9)
ax.set_xlim([-30, 30])

plt.tight_layout()
out = Path(r"H:\2026科研\Spectral-PDE-Discovery\submission\figs\spectrum_viz.png")
plt.savefig(out, bbox_inches='tight', dpi=300)
print(f"Saved {out}, size = {out.stat().st_size/1024:.0f} KB")
