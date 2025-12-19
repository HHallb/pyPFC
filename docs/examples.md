![pyPFC logo](images/pyPFC_logo_transparent.png)

# Examples

This page provides an overview of examples included with pyPFC. The examples demonstrates different aspects of the simulation framework. Additional useful information is also provided:

- [File Formats](#file-formats)
- [Indicative GPU Memory Requirements](#indicative-gpu-memory-requirements-depending-on-floating-point-precision)
- [Performance Notes](#performance-notes)
- [Recommended Configurations](#recommended-configurations)
- [Customization](#customization)

## Running Examples

```bash
# Navigate to examples directory
cd examples/

# Run specific example
python ex04_quick_start.py
```

## Example 0

**Source Code**: [ex00_density_field.py](https://github.com/HHallb/pyPFC/blob/main/examples/ex00_density_field.py)

**Description**: Investigate solid/liquid coexistence, interpolation of density field maxima and phase field evaluation. Output is written to VTK files and saved in pickle format for subsequent post-processing, for example, using Matplotlib.

## Example 1

**Source Code**: [ex01_grain_growth.py](https://github.com/HHallb/pyPFC/blob/main/examples/ex01_grain_growth.py)

**Description**: Growth/solidification from a spherical nucleus. Data is saved to VTK and pickle formats. The example is suitable for performance benchmarking of different pyPFC parameter settings and of alternative hardware configurations.

## Example 2

**Source Code**:  [ex02_grain_boundary_migration.py](https://github.com/HHallb/pyPFC/blob/main/examples/ex02_grain_boundary_migration.py)

**Description**: Simulation of selective grain growth in a bicrystal under an artificial driving pressure, controlled via a directional convolution kernel. Field averages are evluated and data is written to VTK and pickle files.

## Example 3

**Source Code**:  [ex03_polycrystal_solidification.py](https://github.com/HHallb/pyPFC/blob/main/examples/ex03_polycrystal_solidification.py)

**Description**: A polycrystal is seeded in a large liquid domain and the simulation shows the subsequent polycrystal solidification process. Grain boundaries are formed and evolve during the process. Data is written to Extended XYZ format for post-processing.

## Example 4

**Source code**: [ex04_quick_start.py](https://github.com/HHallb/pyPFC/blob/main/examples/ex04_quick_start.py)

**Description**: Basic single crystal simulation, output is written in VTK format. The example is also found in the [Quick Start Guide](quick_start.md).

## Example 5

**Source code**: [ex05_structure_analysis.py](https://github.com/HHallb/pyPFC/blob/main/examples/ex05_structure_analysis.py)

**Description**: This is an example to illustrate the use of structure analysis by the centro-symmetry parameter (CSP) method in the pyPFC package and use of the class `pypfc_ovito`. The example demonstrates how to set up a simulation, generate an initial density field, evolve the density field over time, and perform structure analysis to identify defects in the crystal structure. The results are saved to VTK files for visualization.

---

## File Formats

pyPFC works with different file formats for data I/O:

| Extension | Description | Viewer
|-----------|-------------|--------
| `.pickle` | Binary Python pickle format                 | Custom analysis using `pypfc.save_pickle()` and `pypfc.load_pickle()`
| `.txt`    | Standard ASCII text files                   | Text editor
| `.vtp`    | VTK point data, binary XML format           | Can be opened in, for example, [ParaView](https://www.paraview.org/)
| `.vts`    | VTK structured grid data, binary XML format | Can be opened in, for example, [ParaView](https://www.paraview.org/)
| `.xyz`    | Extended XYZ format                         | Can be opened in, for example, [OVITO](https://www.ovito.org/)

## Performance Notes

### Indicative GPU Memory Requirements Depending on Floating-Point Precision

| Grid Size | Memory (Single) | Memory (Double)
|-----------|-----------------|-----------------
| 64³ | ~1 GB | ~2 GB
| 128³ | ~8 GB | ~16 GB
| 256³ | ~64 GB | ~128 GB

### Recommended Configurations

**For Testing** (fast execution):

```python
config = {
    'device_type': 'GPU',
    'dtype_gpu': 'single',
}
```

**For Production** (high accuracy):

```python
config = {
    'device_type': 'GPU',
    'dtype_gpu': 'double',
}
```

## Customization

All examples can be customized by modifying parameters:

```python
# Modify any example
domain_size = [custom_x, custom_y, custom_z]
ndiv        = [custom_nx, custom_ny, custom_nz]
params['your_parameter'] = your_value
```

For more advanced customization, see the [API documentation](api/core.md).
