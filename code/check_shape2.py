import numpy as np
for name in ['reaction_diffusion_2d', 'heat_soil_uniform_2d_p1', 'heat_soil_uniform_3d_p1']:
    import glob
    files = glob.glob(rf"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde\{name}\*.npz")
    if files:
        d = np.load(files[0], allow_pickle=True)
        print(f"{name}: u={d['u'].shape}, keys={list(d.keys())}")
