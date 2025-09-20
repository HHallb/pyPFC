
# pyPFC: An Open-Source Python Package for Phase Field Crystal Simulations
A Python software package for setting up, running and processing Phase Field Crystal (PFC) simulations. The code uses PyTorch to allow execution on both CPUs and GPUs, depending on the available hardware. The PFC implementation in pyPFC relies on heavy use of FFTs and is set on a regular Cartesian 3D grid. Periodic boundary conditions are assumed along all coordinate axes.

![PFC atoms](https://github.com/HHallb/pyPFC/raw/main/images/PFC_atoms.png)

## Control Parameters
The general behavior of pyPFC is controlled by a set of parameters, collected in a Python dictionary. The parameters are described in the table below.

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

Before running this script, ensure that you have the pyPFC package and its dependencies installed.

```python
import pypfc
import numpy as np

# Simulation parameters
nstep       = 4000
nout        = 1000
output_path = './examples/ex04_output/'

# Computational grid
dSize = np.array([20, 20, 20], dtype=float)
ndiv  = 10 * np.array([20, 20, 20], dtype=int)
ddiv  = dSize / ndiv

# Create simulation object
pypfc = pypfc.setup_simulation(ndiv, ddiv)

# Generate initial density field
den = pypfc.do_single_crystal(params=[dSize[0]*0.25])
pypfc.set_density(den)

# Evolve density field
for step in range(nstep):
    pypfc.do_step_update()
    if np.mod(step+1, nout) == 0 or step+1 == nstep:
        print('Step:', step+1)
        den, _ = pypfc.get_density()
        atom_coord, atom_data = pypfc.interpolate_density_maxima(den)
        filename = output_path + 'pfc_data_' + str(step+1)
        pypfc.write_vtk_points(filename, atom_coord, [atom_data[:,0]], ['den'])
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

### The Class `pypfc_grid`
The class is initiated by supplying the arguments `ndiv = [nx,ny,nz]` and `ddiv=[dx,dy,dz]`, where `[nx,ny,nz]` are the number of grid points and `[dx,dy,dz]` the grid spacing along each coordinate direction.

#### Class Methods and their Arguments

| Method            | Description                                   
| ----------------- | ----------------------------------------------
| `set_ndiv(ndiv)`  | Sets the number of grid points `nx,ny,nz`
| `get_ndiv`        | Returns `ndiv`
| `set_ddiv(ddiv)`  | Sets the grid spacing `dx,dy,dz`
| `get_ddiv`        | Returns `ddiv`
| `get_dSize`       | Returns the grid size `dSize=ndiv*ddiv`
| `copy_from(grid)` | Makes a duplicate of a grid class object

### The Class `pypfc_base`
The class is initiated by supplying the arguments `ndiv` and `ddiv`. A dictionary of configuration parameters can also be supplied through `config`.

#### Class Methods and their Arguments

| Method                                                        | Description                                   
| ------------------------------------------------------------- | ---------------------------------------------- 
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

### The Class `pypfc_pre`
The class is initiated by supplying the arguments `ndiv` and `ddiv`. A dictionary of configuration parameters can also be supplied through `config`.

#### Class Methods and their Arguments

| Method                                                        | Description                                   
| ------------------------------------------------------------- | ---------------------------------------------- 
| `set_struct(struct)`                                          | Set crystal structure
| `get_struct`                                                  | Get crystal structure
| `set_density(den)`                                            | Set PFC density field
| `get_density`                                                 | Get PFC density field
| `set_energy(ene)`                                             | Set PFC energy field
| `get_energy(ene)`                                             | Get PFC energy field
| `get_ampl`                                                    | Get amplitudes of the density field approximation
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

### The Class `pypfc_io`
The class is initiated by supplying the arguments `ndiv` and `ddiv`. A dictionary of configuration parameters can also be supplied through `config`.

#### Class Methods and their Arguments

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

### The Class `pypfc`
The class is initiated by supplying the arguments `ndiv` and `ddiv`. The pypfc class defines the default values of the parameters that control the PFC simulation. A dictionary of configuration parameters can also be supplied through the argument `config`.

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

## Package Dependencies
The following Python packages are required:

* numpy
* ovito (if pypfc_ovito is to be used)
* scikit-image
* scipy
* torch
* vtk

## Installation and Usage
Install using `pip install pypfc` and import with `import pypfc` and (optionally) `import pypfc_ovito`.

## Licencing and Acknowledgment
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