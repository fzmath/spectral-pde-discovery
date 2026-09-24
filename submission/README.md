# Spectral PDE Discovery (SPD)

This repository contains the code and data for the paper:

**"Spatiotemporal spectral derivatives with forward stepwise BIC for robust partial differential equation discovery"**

by Fuchang Wang and Huirong Cao.

## Overview

We propose a robust PDE discovery framework with three components:

1. **Spatiotemporal spectral derivatives**: A 2D Fourier transform over the space-time domain computes derivatives simultaneously, handling non-periodic traveling-wave solutions where spatial-only spectral methods fail.
2. **Spectral weak formulation**: The PDE residual is integrated against localized Gaussian test functions, suppressing high-frequency noise in high-order derivatives.
3. **Normalized forward stepwise BIC**: Candidate terms are normalized to unit L2 norm, and forward stepwise selection with BIC determines sparsity automatically—no threshold tuning required.

## Repository Structure

```
submission/
├── main.tex              # LaTeX source
├── refs.bib              # Bibliography
├── fig1-4.pdf/eps        # Figures
├── code/
│   ├── spd_v6_weak.py           # Main implementation (spectral weak form)
│   ├── spd_v6_final.py          # Final 1D experiments
│   ├── run_spatial_spectral_baseline.py  # Spatial-only spectral baseline
│   ├── spd_2d_extended.py       # 2D/3D scalability experiments
│   ├── gen_submission_figs_v2.py # Figure generation
│   └── results/                  # Experimental results (JSON)
└── README.md
```

## Requirements

- Python 3.11
- NumPy >= 1.26
- SciPy >= 1.15
- Matplotlib >= 3.6
- pysindy >= 1.7 (for baseline comparisons)

## Usage

### 1D Experiments

```bash
python spd_v6_final.py
```

### Spatial Spectral Baseline

```bash
python run_spatial_spectral_baseline.py
```

### 2D/3D Experiments

```bash
python spd_2d_extended.py
```

### Regenerate Figures

```bash
python gen_submission_figs_v2.py
```

## Data

The experiments use the MDBench benchmark dataset. Please download it from:
https://github.com/gryaklab/mdbench

Place the data in `mdbench/data/processed/data/pde/`.

## Key Results

- **Non-periodic advection**: Spatial-only spectral methods fail completely (0/1), while our method achieves exact recovery at clean.
- **Burgers equation**: Recovers both true terms at all noise levels, including the small diffusion coefficient at snr 10.
- **Speed**: 10-20x faster than Weak SINDy with comparable accuracy.

## Citation

If you use this code, please cite:

```
@article{wang2026spectral,
  title={Spatiotemporal spectral derivatives with forward stepwise BIC for robust partial differential equation discovery},
  author={Wang, Fuchang and Cao, Huirong},
  journal={Applied Mathematics and Computation},
  year={2026}
}
```

## Contact

Fuchang Wang (corresponding author): wangfuchang@cidp.edu.cn
