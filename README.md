![PyPI](https://img.shields.io/pypi/v/pypfc)
![License](https://img.shields.io/github/license/HHallb/pyPFC)

# pyPFC: An Open-Source Python Package for Phase Field Crystal Simulations
A Python software package for setting up, running and processing Phase Field Crystal (PFC) simulations. The code uses PyTorch to allow execution on both CPUs and GPUs, depending on the available hardware. The PFC implementation in pyPFC relies on heavy use of FFTs and is set on a regular Cartesian 3D grid. Periodic boundary conditions are assumed along all coordinate axes. pyPFC evolves the density field in non-dimensional time and spatial dimensions are expressed in units of lattice parameters.

![PFC atoms](https://github.com/HHallb/pyPFC/raw/main/images/PFC_atoms.png)

## Table of Contents
<!--ts-->
- [pyPFC: An Open-Source Python Package for Phase Field Crystal Simulations](#pypfc-an-open-source-python-package-for-phase-field-crystal-simulations)
  - [Configuration Parameters](#configuration-parameters)
  - [Quick Start Example](#quick-start-example)
  - [Description of Source Files](#description-of-source-files)
    - [Class `pypfc_grid`](#class-pypfc_grid)
    - [Class `pypfc_base`](#class-pypfc_base)
    - [Class `pypfc_pre`](#class-pypfc_pre)
    - [Class `pypfc_io`](#class-pypfc_io)
    - [Class `pypfc`](#class-pypfc)
    - [Class `pypfc_ovito`](#class-pypfc_ovito)
  - [Package dependencies](#package-dependencies)  
  - [Installation and Usage](#installation-and-usage)
  - [Troubleshooting](#troubleshooting)
  - [Licencing](#licensing)
  - [References](#references)
<!--te-->

## Configuration Parameters
The general behavior of pyPFC is controlled by a set of configuration parameters, collected in a Python dictionary. The parameters are described in the table below.

| Parameter name         | Defaults to                       | Description                                                                      
| ---------------------- | --------------------------------- | ---------------------------------------------------------------------------------
| alat                   | 1.0                               | Lattice parameter (non-dimensional)
| alpha                  | [1, 1]                            | C2 Gaussian peak widths, excluding the zero-mode peak
| C20_amplitude          | 0.0                               | Amplitude of the zero-mode Gaussian peak in C2
| C20_alpha              | 1.0                               | Width of the zero-mode Gaussian peak in C2
| density_interp_order   | 2                                 | Interpolation order for density maxima localization
| density_merge_distance | 0.1                               | Distance for merging density maxima (in units of lattice parameters)
| density_threshold      | 0.5                               | Threshold for density maxima detection
| device_number          | 0                                 | GPU device number (if multiple GPUs are available)
| device_type            | 'gpu'                             | PyTorch device ('cpu' or 'gpu')
| dtime                  | 1.0e-3                            | Non-dimensional time increment
| dtype_cpu              | np.double                         | Floating-point precision of numpy arrays
| dtype_gpu              | torch.float64                     | Floating-point precision of PyTorch tensors
| evaluate_phase_field   | True                              | Evaluate phase field (or not)
| normalize_pf           | True                              | Normalize the phase fields to [0,1], or not
| npeaks                 | 2                                 | Number of Gaussian peaks, excluding the zero-mode peak, to use in C2
| pf_gauss_var           | 0.1                               | Variance (sigma) of the Gaussian smoothing kernel used in phase field evaluations
| pf_iso_level           | 0.5                               | Iso-level for phase field contouring
| sigma                  | 0.0                               | Temperature-like parameter (non-dimensional)
| struct                 | 'FCC'                             | Crystal structure
| torch_threads          | 8                                 | Number of CPU threads to use if device_type is 'cpu'
| torch_threads_interop  | 8                                 | Number of interop threads to use if device_type is 'cpu'
| update_scheme          | '1st_order'                       | Time integration scheme ('1st_order', '2nd_order' or 'exponential')
| update_scheme_params   | [1.0, 1.0, 1.0, None, None, None] | Parameters in the time integration scheme: [g1, g2, g3, alpha, beta, gamma]
| verbose                | True                              | Verbose output (or not)

## Quick Start Example
This is a quick start example for using the pyPFC package to perform a simple phase field crystal (PFC) simulation. The simulation traces the growth of a spherical crystal, centered in a 3D periodic domain. The example can be found as `./examples/ex04_quick_start.py`, where it is commented in more detail. The example demonstrates how to set up a simulation, generate an initial density field, evolve the density field over time and save the results to VTK files for visualization.

It can be noted that the only strictly required input to pyPFC is `domain_size`, defining the size of the simulation domain in units of lattice parameters along each coordinate axis. All other [configuration parameters](#configuration-parameters) are set to default values, which can be adjusted as needed.

Before running this script, ensure that you have the pyPFC package and its [dependencies](#package-dependencies) installed.

```python
import pypfc
import numpy as np

# Simulation parameters
nstep       = 4000
nout        = 1000
output_path = './examples/ex04_output/'

# Computational domain
domain_size = [20, 20, 20]

# Create simulation object
sim = pypfc.setup_simulation(domain_size)

# Generate initial density field
den = sim.do_single_crystal(params=[domain_size[0]*0.25])
sim.set_density(den)

# Evolve density field
for step in range(nstep):
    sim.do_step_update()
    if np.mod(step+1, nout) == 0:
        print('Step:', step+1)
        den, _ = sim.get_density()
        atom_coord, atom_data = sim.interpolate_density_maxima(den)
        filename = output_path + 'pfc_data_' + str(step+1)
        sim.write_vtk_points(filename, atom_coord, [atom_data[:,0]], ['den'])
```

## Description of Source Files
The software is built on classes, contained in separate modules/files, with an inheritance chain (from top to bottom) comprising:

| File (*.py)     | Description             
| --------------- | ----------------------
| **pypfc_grid**  | Grid methods            
| **pypfc_base**  | Base methods            
| **pypfc_pre**   | Pre-processing methods  
| **pypfc_io**    | Data IO methods         
| **pypfc**       | Main class               

In addition, **pypfc_ovito.py** provides custom interfaces to selected functionalities in [OVITO](https://www.ovito.org/), useful for post-processing of pyPFC simulation output.

Methods in the different classes are described in individual subsections below.

### Class `pypfc_grid`

| Method            | Description                                   
| ----------------- | ----------------------------------------------
| `set_ndiv(ndiv)`  | Sets the number of grid points `[nx,ny,nz]`
| `get_ndiv`        | Returns `ndiv`
| `set_ddiv(ddiv)`  | Sets the grid spacing `[dx,dy,dz]`
| `get_ddiv`        | Returns `ddiv`
| `get_domain_size` | Returns the grid size `domain_size=ndiv*ddiv`
| `copy_from(grid)` | Makes a duplicate of a grid object

### Class `pypfc_base`

| Method                                                        | Description                                   
| ------------------------------------------------------------- | ---------------------------------------------- 
| `set_alpha(alpha)`                                            | Set C2 peak widths
| `get_alpha`                                                   | Get C2 peak widths
| `set_verbose(verbose)`                                        | Set verbose output flag
| `get_verbose`                                                 | Get verbose output flag
| `set_dtype_cpu(dtype)`                                        | Set the cpu floating point precision
| `get_dtype_cpu`                                               | Get the cpu floating point precision
| `set_dtype_gpu(dtype)`                                        | Set the gpu floating point precision
| `get_dtype_gpu`                                               | Get the gpu floating point precision
| `set_device_type(device_type)`                                | Set the device type (cpu or gpu)
| `get_device_type`                                             | Get the device type (cpu or gpu)
| `set_device_number(device_number)`                            | Set the GPU device number
| `get_device_number`                                           | Get the GPU device number
| `set_k2_d(k2_d)`                                              | Set sum of squared wave vector (on the device)
| `get_k2_d`                                                    | Get sum of squared wave vector (on the device)
| `set_torch_threads`                                           | Set torch.get_num_threads() and torch.get_num_interop_threads()
| `get_torch_threads`                                           | Get torch.get_num_threads() and torch.get_num_interop_threads()
| `get_time_stamp`                                              | Get time stamp string: YYYY-MM-DD H:M
| `get_k(npoints, dspacing)`                                    | Define a 1D wave vector at npoint points with spacing dspacing
| `evaluate_k2_d`                                               | Evaluate the sum of squared wave vectors
| `get_integrated_field_in_volume(field, limits)`               | Integrate a field variable within a volume defined by lower and upper coordinate limits
| `get_field_average_along_axis(field, axis)`                   | Evaluate the mean value of a field variable along a certain axis ('x', 'y' or 'z')
| `get_integrated_field_along_axis(field, axis)`                | Integrate a field variable along a certain axis ('x', 'y' or 'z')
| `interpolate_atoms(intrpPos, pos, values, num_nnb, power)`    | Interpolate values at given positions
| `interpolate_density_maxima(self, den, ene, pf)`              | Interpolate density field maxima, plus (optionally) energy and phase field(s)
| `get_phase_field_contour(self, pf, pf_zoom, evaluate_volume)` | Find the iso-contour surface of a 3D phase field using marching cubes and (optionally) the enclosed volume
| `get_rlv(struct, alat)`                                       | Get the reciprocal lattice vectors for a particular crystal structure
| `evaluate_reciprocal_planes`                                  | Establish the reciprocal vectors/planes for a particular crystal structure
| `evaluate_C2_d`                                               | Establish the two-point correlation function (on the device) for a particular crystal structure
| `evaluate_directional_correlation_kernel(self, H0, Rot)`      | Establish the directional correlation kernel for a particular crystal structure
| `get_xtal_nearest_neighbors`                                  | Provides the number of nearest neighbors and neighbor distances for different crystal structures
| `get_csp(pos, normalize_csp)`                                 | Evaluate the centro-symmetry parameter (CSP) for a set of atoms

### Class `pypfc_pre`

| Method                                                        | Description                                   
| ------------------------------------------------------------- | ---------------------------------------------- 
| `set_struct(struct)`                                          | Set crystal structure
| `get_struct`                                                  | Get crystal structure
| `set_density(den)`                                            | Set PFC density field
| `get_density`                                                 | Get PFC density field
| `set_energy(ene)`                                             | Set PFC energy field
| `get_energy(ene)`                                             | Get PFC energy field
| `set_ampl`                                                    | Set amplitudes of the density field approximation
| `get_ampl`                                                    | Get amplitudes of the density field approximation
| `set_nlns`                                                    | Set liquid/solid densities in the density field approximation
| `get_nlns`                                                    | Get liquid/solid densities in the density field approximation
| `set_sigma(sigma)`                                            | Set sigma
| `get_sigma`                                                   | Get sigma
| `set_npeaks(npeaks)`                                          | Set the number of Gaussian peaks in the pair correlation function
| `get_npeaks`                                                  | Set the number of Gaussian peaks in the pair correlation function
| `do_single_crystal(xtalRot, params, model)`                   | Define a centered crystal in a periodic 3D domain
| `do_bicrystal(xtalRot, params, liq_width, model)`             | Define a centered crystal, embedded inside a matrix crystal, in a periodic 3D domain
| `do_polycrystal(xtalRot, params, liq_width, model)`           | Define a polycrystal in a periodic 3D domain
| `generate_density_field(crd, g)`                              | Define a 3D PFC density field
| `evaluate_ampl_dens`                                          | Get the amplitudes and densities for different density field expansions

### Class `pypfc_io`

| Method                                                                                                                  |Description                                   
| ----------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- 
| `write_extended_xyz(filename, coord, atom_data, atom_data_labels, simulation_time, gz)`                                 | Save PFC data in extended XYZ format
| `read_extended_xyz(filename, nfields)`                                                                                  | Read PFC data in extended XYZ format from file
| `write_vtk_points(filename, coord, scalarData, scalarDataName, vectorData, vectorDataName, tensorData, tensorDataName)` | Save 3D point data to a VTK file
| `write_vtk_structured_grid(filename, arrayData, arrayName)`                                                             | Save 3D field data to a VTK file
| `save_pickle(filename, data)`                                                                                           | Save a list of data objects to a binary pickle file
| `load_pickle(filename, ndata)`                                                                                          | Load data objetcs from a binary pickle file
| `write_info_file(filename=, output_path)`                                                                               | Write simulation setup information to a file
| `append_to_info_file(info, filename=, output_path)`                                                                     | Append linea to a text file

### Class `pypfc`

The class is initiated by supplying the arguments `domain_size=[Lx,Ly,Lz]` and `ndiv = [nx,ny,nz]`, where `[Lx,Ly,Lz]` is the domain size and `[nx,ny,nz]` define the number of grid points along each coordinate direction. The values in `[nx,ny,nz]` must all be even numbers, an error will be raised otherwise. The only required input is `domain_size`. All other parameters, including `ndiv` are set by defaults, but can be modified as desired. An optional dictionary of configuration parameters can be supplied through the argument `config`.

#### Class Methods and their Arguments

| Method                                                                                       | Description                                   
| -------------------------------------------------------------------------------------------- | ---------------------------------------------- 
| `set_alat(alat)`                                                                             | Set lattice parameter (non-dimensional)
| `get_alat`                                                                                   | Get lattice parameter (non-dimensional)
| `set_dtime(dtime)`                                                                           | Set time increment (non-dimensional)
| `get_dtime`                                                                                  | Get time increment (non-dimensional)
| `set_alpha(alpha)`                                                                           | Set peak widths in C2
| `get_alpha`                                                                                  | Get peak widths in C2
| `set_C2_d(C2_d)`                                                                             | Set C2 (on the device)
| `get_C2_d`                                                                                   | Get C2 (on the device)
| `set_H2_d(H0, Rot)`                                                                          | Set H2 (in Fourier space, on the device)
| `set_update_scheme(update_scheme)`                                                           | Set time integration scheme
| `get_update_scheme`                                                                          | Get time integration scheme
| `set_update_scheme_params(params)`                                                           | Set time integration scheme parameters
| `get_update_scheme_params`                                                                   | Get time integration scheme parameters
| `get_energy`                                                                                 | Get the PFC energy field and its mean value
| `set_density(density)`                                                                       | Set the PFC density field
| `get_density`                                                                                | Get the PFC density field and its mean value
| `set_phase_field_kernel(H0, Rot)`                                                            | Set phase field kernel
| `set_phase_field_smoothing_kernel(pf_gauss_var)`                                             | Set phase field smoothing kernel
| `cleanup`                                                                                    | Clean up variables
| `do_step_update`                                                                             | Update the (X)PFC density field using the chosen time integration scheme
| `evaluate_energy`                                                                            | Evaluate the PFC energy field
| `get_phase_field`                                                                            | Evaluate the phase field using wavelet filtering

### Class `pypfc_ovito`
This class provides pyPFC interfaces to a limited subset of the functionalities in [OVITO](https://www.ovito.org/).

#### Class Methods and their Arguments

| Method                                                                                       | Description                                   
| -------------------------------------------------------------------------------------------- | ---------------------------------------------- 
| `do_ovito_csp(num_neighbors)`                                                                | Evaluate Centro-Symmetry Parameter (CSP) using OVITO's CentroSymmetryModifier
| `do_ovito_dxa(rep, tol)`                                                                     | Perform dislocation analysis (DXA) using OVITO
| `do_ovito_ptm(refRot, outputEulerAng, outputStrain)`                                         | Evaluate crystal structure, orientation and elastic Green-Lagrange strain using OVITO's Polyhedral Template Matching (PTM)

## Package Dependencies
The following Python packages are required:

* numpy
* ovito (if pypfc_ovito is to be used)
* scikit-image
* scipy
* torch
* vtk

Note that PyPI only installs torch with CPU support. To add GPU support, refer to [Installation and Usage](#installation-and-usage).

## Installation and Usage
The simplest way to install pyPFC is via pip, which should ensure that [package dependencies](#package-dependencies) are met automatically. Note, however, that **PyTorch is only installed with CPU support** since PyPI only provides the CPU version of torch. For GPU support, see the tip a couple of lines down in this section.

Install from PyPI using:
```bash
pip install pypfc
```
or
```bash
sudo pip install pypfc
```
Alternatively, install from source by:
```bash
git clone https://github.com/HHallb/pyPFC.git
cd pyPFC
pip install .
```
Import pyPFC into your Python code by `import pypfc` and, optionally, `import pypfc_ovito`. See the [quick start example](#quick-start-example) or the examples provided in `./examples/`.

> [!IMPORTANT]
> To install torch with CUDA 12.1 enabled, run: `pip install torch --index-url https://download.pytorch.org/whl/cu121`. For other CUDA versions and further information, please refer to the official [PyTorch documentation](https://pytorch.org/get-started/locally/).

## Troubleshooting

#### Q: I have a Nvidia GPU installed but pyPFC tells me I don't.
**A:** Ensure that PyTorch is installed with CUDA enabled. To be sure, you can also check your setup by running:
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
```
If `is_available()` is `False` or `device_count()` is `0`, PyTorch cannot see your GPU. See [Installation and Usage](#installation-and-usage) on how to enable CUDA. 

---
#### Q: The solid crystal phase fails to stabilize and/or appears to "melt" away.
**A:** This is most likely due to `domain_size` not being set correctly to accommodate the current lattice periodicity.

---

#### Q: The atom positions obtained by `interpolate_density_maxima` do not seem to coincide with the density field maxima.
**A:** This is likely related to either insufficient grid resolution or too low interpolation order. The former issue is mitigated by reducing the values in `ddiv` and in the latter case `density_interp_order` should be increased. Usually `density_interp_order=2` is fine and increasing the number will also increase the time spent on interpolation.

---

## Licencing
This software is released under a [GNU GPLv3 license](https://www.gnu.org/licenses/).

## References
Further details on PFC modeling and example applications can be found in:

1. [K.H. Blixt and H. Hallberg, **Phase Field Crystal Modeling of Grain Boundary Migration: Mobility, Energy and Structural Variability**, *Acta Materialia*, 297:121318, 2025](https://doi.org/10.1016/j.actamat.2025.121318)
2. [H. Hallberg and K.H. Blixt, **Assessing grain boundary variability through phase field crystal simulations**, *Physical Review Materials*, 8(3):113605, 2024](https://doi.org/10.1103/PhysRevMaterials.8.113605)
3. [K.H. Blixt and H. Hallberg, **Phase field crystal modeling of grain boundary structures in diamond cubic systems**, *Physical Review Materials*, 8(3):033606, 2024](https://doi.org/10.1103/PhysRevMaterials.8.033606)
4. [H. Hallberg and K.H. Blixt, **Multiplicity of grain boundary structures and related energy variations**, *Materials Today Communications*, 38:107724, 2024](https://doi.org/10.1016/j.mtcomm.2023.107724)
5. [H. Hallberg and K.H. Blixt, **Evaluation of Nanoscale Deformation Fields from Phase Field Crystal Simulations**, *Metals*, 12(10):1630, 2022](https://doi.org/10.3390/met12101630)
6. [K.H. Blixt and H. Hallberg, **Grain boundary and particle interaction: Enveloping and pass-through mechanisms studied by 3D phase field crystal simulations**, *Materials & Design*, 220:110845, 2022](https://doi.org/10.1016/j.matdes.2022.110845)
7. [K.H. Blixt and H. Hallberg, **Grain boundary stiffness based on phase field crystal simulations**, *Materials Letters*, 318:132178, 2022](https://doi.org/10.1016/j.matlet.2022.132178)
8. [K.H. Blixt and H. Hallberg, **Evaluation of grain boundary energy, structure and stiffness from phase field crystal simulations**, *Modelling and Simulation in Materials Science and Engineering*, 30(1):014002, 2022](https://doi.org/10.1088/1361-651X/ac3ca1)
