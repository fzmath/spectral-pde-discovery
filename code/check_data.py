"""Check MDBench dataset formats and true PDE forms."""
import numpy as np
from pathlib import Path

DATA_DIR = Path(r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde")

for name in ['nls', 'heat_laser', 'heat_soil_uniform_1d_p1', 'advection_diffusion_2d', 'reaction_diffusion_2d']:
    p = DATA_DIR / name
    if not p.exists():
        print(f"{name}: NOT FOUND")
        continue
    files = list(p.glob("*.npz"))
    if not files:
        print(f"{name}: no npz files")
        continue
    data = np.load(files[0])
    print(f"\n{name}:")
    print(f"  keys: {list(data.keys())}")
    for k in data.keys():
        arr = data[k]
        print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min():.4f}, {arr.max():.4f}]")
