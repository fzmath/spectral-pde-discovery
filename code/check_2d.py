import numpy as np
from pathlib import Path

d = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")
for n in ['reaction_diffusion_2d', 'heat_soil_uniform_2d_p1', 'heat_soil_uniform_3d_p1']:
    p = d / n / f"{n}.npz"
    if p.exists():
        data = np.load(p)
        u = data['u']
        du = data['du'] if 'du' in data else None
        keys = list(data.keys())
        print(f"{n}: u={u.shape}, du={du.shape if du is not None else 'N/A'}, keys={keys}")
        if u.ndim >= 3:
            print(f"  u range: [{u.min():.4f}, {u.max():.4f}]")
    else:
        print(f"{n}: NOT FOUND")
