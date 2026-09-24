import numpy as np
import os

base = r"H:\2026科研\Spectral-PDE-Discovery\mdbench\data\processed\data\pde"
folders = ['burgers', 'kdv', 'kuramoto_sivishinky', 'advection1d', 
           'advection_diffusion_2d', 'reaction_diffusion_2d', 
           'heat_soil_uniform_2d_p1', 'heat_soil_uniform_3d_p1']

for folder in folders:
    fpath = os.path.join(base, folder)
    if os.path.isdir(fpath):
        files = [f for f in os.listdir(fpath) if f.endswith('.npz') and 'clean' in f]
        if files:
            data = np.load(os.path.join(fpath, files[0]))
            print(f"{folder}: {files[0]}, u shape = {data['u'].shape}")
        else:
            files = [f for f in os.listdir(fpath) if f.endswith('.npz')]
            if files:
                data = np.load(os.path.join(fpath, files[0]))
                print(f"{folder}: {files[0]}, u shape = {data['u'].shape}")
