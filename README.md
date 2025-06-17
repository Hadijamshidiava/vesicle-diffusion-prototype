### Prerequisites

This project requires Python 3.8+ and the following packages:

- `numpy`
- `vispy`
- `dataclasses` *(for Python versions < 3.7, where it's not built-in)*

You can install the required packages using pip:

```bash
pip install numpy vispy dataclasses
```

### Vesicle Diffusion Prototype

This project is a fast and modular prototype for simulating vesicle diffusion within a 2D triangular mesh — a simplified abstraction of vesicle transport along biological membranes.

The simulation was developed in response to a scientific paper shared as part of a PhD interview preparation, aiming to demonstrate understanding of spatial diffusion, geometric sampling, and efficient real-time computation.

### Key Features

- Vesicles modeled as circular particles with membrane sample points
- Triangle occupancy detection using a fully vectorized 2D side-test
- Brownian motion (diffusion) with rejection sampling for spatial exclusion
- Per-vesicle diffusion coefficients
- Real-time interactive visualization using Vispy
- Ready for GPU acceleration using CuPy or PyTorch
- Modular design, easily extendable to tetrahedral 3D meshes and reaction models

### Reference

This prototype was inspired by:

> **Iain Hepburn et al., "Vesicle and reaction-diffusion hybrid modeling with STEPS"**  
> *Communications Biology, 2024*  
> [https://doi.org/10.1038/s42003-024-06276-5](https://doi.org/10.1038/s42003-024-06276-5)

The current code implements a simplified 2D equivalent of the tetrahedral diffusion model described in the paper, using triangle meshes and point-based surface sampling for efficiency.


