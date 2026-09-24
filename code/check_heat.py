import numpy as np
d = np.load(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde\heat_soil_uniform_2d_p1\heat_soil_uniform_2d_p1.npz", allow_pickle=True)
u = np.squeeze(d['u'])
ut = np.squeeze(d['du'])
print('u shape:', u.shape, 'range:', u.min(), u.max())
print('ut shape:', ut.shape, 'range:', ut.min(), ut.max())
print('u mean:', u.mean(), 'ut mean:', ut.mean())
# Check if ut is zero or very small
print('ut abs max:', np.abs(ut).max())
print('ut std:', ut.std())
