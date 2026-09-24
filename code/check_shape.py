import numpy as np
d = np.load(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde\advection_diffusion_2d\advection_diffusion_2d.npz", allow_pickle=True)
print('keys:', list(d.keys()))
print('u shape:', d['u'].shape)
print('x shape:', d['x'].shape)
print('y shape:', d['y'].shape)
print('t shape:', d['t'].shape)
print('du shape:', d['du'].shape)
