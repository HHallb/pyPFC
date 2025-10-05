# Examples

This page provides an overview of all examples included with pyPFC. Each example demonstrates different aspects of the simulation framework.

## Example Categories

### Basic Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [Quick Start](#ex04-quick-start) | Basic single crystal simulation | ⭐ |
| [Density Field](#ex00-density-field) | Understanding density fields | ⭐ |

### Intermediate Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [Grain Growth](#ex01-grain-growth) | Polycrystal grain boundary dynamics | ⭐⭐ |
| [Grain Boundary Migration](#ex02-grain-boundary-migration) | Controlled boundary movement | ⭐⭐ |

### Advanced Examples

| Example | Description | Complexity |
|---------|-------------|------------|
| [Polycrystal Solidification](#ex03-polycrystal-solidification) | Phase transformation modeling | ⭐⭐⭐ |
| [Structure Analysis](#ex05-structure-analysis) | OVITO integration and analysis | ⭐⭐⭐ |

## Example Details

### ex00: Density Field

**File**: `examples/ex00_density_field.py`

This example introduces the fundamental concept of density fields in PFC simulations.

```python
import pypfc
import matplotlib.pyplot as plt
import numpy as np

# Basic density field visualization
pfc = pypfc.PyPFC()
domain_size = [20.0, 20.0, 4.0]
ndiv = [40, 40, 8]

pfc.setup_simulation(domain_size, ndiv, {'device_type': 'CPU'})
pfc.do_single_crystal()

# Get and visualize density field
density = pfc.get_density_field()
plt.figure(figsize=(12, 4))

# Show different slices
for i, z in enumerate([0, ndiv[2]//2, ndiv[2]-1]):
    plt.subplot(1, 3, i+1)
    plt.imshow(density[:, :, z], cmap='viridis')
    plt.title(f'z-slice {z}')
    plt.colorbar()

plt.tight_layout()
plt.show()
```

**Learning Goals**:
- Understanding density field representation
- Basic visualization techniques
- Grid indexing and slicing

---

### ex01: Grain Growth

**File**: `examples/ex01_grain_growth.py`

Demonstrates polycrystal systems and grain boundary dynamics.

```python
import pypfc
import numpy as np

# Setup polycrystal simulation
pfc = pypfc.PyPFC()
domain_size = [64.0, 64.0, 16.0]
ndiv = [128, 128, 32]

config = {
    'device_type': 'GPU',
    'dtype_gpu': 'single',
    'grain_seeds': [4, 4],  # 4x4 grain structure
    'update_scheme': 'exponential',
    'update_scheme_params': [1.0, 0.01]
}

pfc.setup_simulation(domain_size, ndiv, config)
pfc.do_polycrystal()

# Monitor grain evolution
grain_sizes = []
simulation_steps = 500

for step in range(simulation_steps):
    pfc.do_step_update()
    
    if step % 50 == 0:
        # Analyze grain structure
        positions = pfc.interpolate_density_maxima()
        energy = pfc.get_energy()
        
        print(f"Step {step}: Energy = {energy:.6f}, Atoms = {len(positions)}")
        
        # Save snapshots
        if step % 100 == 0:
            pfc.write_vtk_points(f'grain_growth_step_{step:04d}.vtu')

print("Grain growth simulation complete")
```

**Key Features**:
- Polycrystal initialization
- Long-term evolution tracking
- Energy minimization monitoring
- Grain boundary migration

**Output Files**:
- `grain_growth_step_*.vtu` - Atomic positions at different times
- Analysis data for grain size evolution

---

### ex02: Grain Boundary Migration

**File**: `examples/ex02_grain_boundary_migration.py`

Focuses on controlled grain boundary dynamics with external driving forces.

```python
import pypfc
import numpy as np

# Specialized grain boundary setup
pfc = pypfc.PyPFC()
domain_size = [32.0, 64.0, 8.0]
ndiv = [64, 128, 16]

# Configuration for boundary migration study
config = {
    'device_type': 'GPU',
    'dtype_gpu': 'double',
    'boundary_type': 'tilt',
    'misorientation_angle': 15.0,  # degrees
    'driving_force': 0.001,
    'update_scheme': '2nd_order',
    'update_scheme_params': [0.5, 0.001]
}

pfc.setup_simulation(domain_size, ndiv, config)

# Create bicrystal with controlled boundary
pfc.do_bicrystal_tilt_boundary()

# Track boundary position
boundary_positions = []
times = []

for step in range(1000):
    pfc.do_step_update()
    
    if step % 10 == 0:
        # Measure boundary position
        density = pfc.get_density_field()
        boundary_pos = pfc.analyze_grain_boundary(density)
        
        boundary_positions.append(boundary_pos)
        times.append(step * config['update_scheme_params'][0])
        
        if step % 100 == 0:
            print(f"Step {step}: Boundary at y = {boundary_pos:.2f}")

# Calculate migration velocity
velocity = np.gradient(boundary_positions, times)
print(f"Average migration velocity: {np.mean(velocity):.6f} units/time")
```

**Analysis Features**:
- Boundary position tracking
- Migration velocity calculation
- Crystallographic orientation analysis
- Driving force effects

---

### ex03: Polycrystal Solidification

**File**: `examples/ex03_polycrystal_solidification.py`

Advanced example showing liquid-solid phase transformation.

```python
import pypfc
import numpy as np
import matplotlib.pyplot as plt

# Large-scale solidification simulation
pfc = pypfc.PyPFC()
domain_size = [100.0, 100.0, 20.0]
ndiv = [200, 200, 40]

config = {
    'device_type': 'GPU',
    'dtype_gpu': 'single',
    'initial_phase': 'liquid',
    'nucleation_sites': 25,
    'temperature_gradient': True,
    'cooling_rate': 0.001,
    'update_scheme': 'exponential'
}

pfc.setup_simulation(domain_size, ndiv, config)

# Initialize with liquid phase
pfc.do_liquid_phase()

# Add nucleation sites
nucleation_positions = pfc.add_random_nuclei(config['nucleation_sites'])

# Solidification process
solid_fraction = []
temperatures = []

for step in range(2000):
    pfc.do_step_update()
    
    # Apply cooling
    if step % 10 == 0:
        current_temp = pfc.reduce_temperature(config['cooling_rate'])
        temperatures.append(current_temp)
        
        # Measure solidification
        density = pfc.get_density_field()
        solid_frac = pfc.calculate_solid_fraction(density)
        solid_fraction.append(solid_frac)
        
        print(f"Step {step}: T = {current_temp:.4f}, Solid = {solid_frac:.3f}")
        
        if step % 200 == 0:
            pfc.write_vtk_points(f'solidification_{step:04d}.vtu')

# Plot solidification curve
plt.figure(figsize=(10, 6))
plt.plot(temperatures, solid_fraction, 'b-', linewidth=2)
plt.xlabel('Temperature')
plt.ylabel('Solid Fraction')
plt.title('Solidification Curve')
plt.grid(True)
plt.show()
```

**Advanced Features**:
- Phase transformation modeling
- Temperature control
- Nucleation and growth
- Solidification kinetics analysis

---

### ex04: Quick Start

**File**: `examples/ex04_quick_start.py`

The simplest example to get started with pyPFC.

```python
import pypfc

# Minimal working example
pfc = pypfc.PyPFC()
pfc.setup_simulation([16.0, 16.0, 4.0], [32, 32, 8])
pfc.do_single_crystal()

# Run 50 steps
for step in range(50):
    pfc.do_step_update()
    
    if step % 10 == 0:
        energy = pfc.get_energy()
        print(f"Step {step}: Energy = {energy:.6f}")

# Save results
pfc.write_extended_xyz('quickstart_result.xyz')
print("Quick start complete!")
```

**Perfect For**:
- First-time users
- Installation verification
- Basic functionality testing

---

### ex05: Structure Analysis

**File**: `examples/ex05_structure_analysis.py`

Integration with OVITO for advanced structural analysis.

!!! note "Requirements"
    This example requires OVITO to be installed:
    ```bash
    pip install ovito
    ```

```python
import pypfc
from pypfc_ovito import OvitoAnalyzer
import numpy as np

# Create test structure
pfc = pypfc.PyPFC()
domain_size = [40.0, 40.0, 10.0]
ndiv = [80, 80, 20]

pfc.setup_simulation(domain_size, ndiv, {'device_type': 'GPU'})
pfc.do_single_crystal()

# Run simulation to develop defects
for step in range(200):
    pfc.do_step_update()

# Extract atomic positions
positions = pfc.interpolate_density_maxima()
pfc.write_extended_xyz('structure_for_analysis.xyz')

# Initialize OVITO analyzer
analyzer = OvitoAnalyzer('structure_for_analysis.xyz')

# Perform various analyses
results = analyzer.analyze_structure({
    'common_neighbor_analysis': True,
    'polyhedral_template_matching': True,
    'dislocation_analysis': True,
    'elastic_strain': True
})

print("Structure Analysis Results:")
print(f"Total atoms: {results['n_atoms']}")
print(f"Crystal structure: {results['crystal_structure']}")
print(f"Defect atoms: {results['defect_count']}")

if 'dislocations' in results:
    print(f"Dislocation lines: {len(results['dislocations'])}")
    for i, disl in enumerate(results['dislocations']):
        print(f"  Dislocation {i}: {disl['burgers_vector']}")

# Generate visualization
analyzer.create_visualization({
    'color_by': 'structure_type',
    'show_dislocations': True,
    'export_image': 'structure_analysis.png'
})

print("Analysis complete. Check 'structure_analysis.png' for visualization.")
```

**Analysis Capabilities**:
- Crystal structure identification
- Defect detection and classification
- Dislocation analysis
- Elastic strain calculation
- Automated visualization

## Running Examples

### Command Line

```bash
# Navigate to examples directory
cd examples/

# Run specific example
python ex04_quick_start.py

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 python ex01_grain_growth.py
```

### Jupyter Notebooks

Interactive versions of all examples are available as Jupyter notebooks:

```bash
# Start Jupyter
jupyter notebook

# Open any .ipynb file in the examples directory
```

### Batch Processing

For parameter studies:

```bash
# Run multiple configurations
for size in 32 64 128; do
    python ex01_grain_growth.py --grid_size $size --output_dir results_$size
done
```

## Output Files

Examples generate various output files:

| Extension | Description | Viewer |
|-----------|-------------|--------|
| `.vtu` | VTK unstructured grid | ParaView, VisIt |
| `.xyz` | Extended XYZ format | OVITO, VMD |
| `.pkl` | Python pickle | Custom analysis |
| `.png/.jpg` | Images | Any image viewer |
| `.txt` | Data files | Text editor, plotting |

## Performance Notes

### GPU Memory Requirements

| Grid Size | Memory (Single) | Memory (Double) |
|-----------|-----------------|-----------------|
| 64³ | ~1 GB | ~2 GB |
| 128³ | ~8 GB | ~16 GB |
| 256³ | ~64 GB | ~128 GB |

### Recommended Configurations

**For Testing** (fast execution):
```python
config = {
    'device_type': 'GPU',
    'dtype_gpu': 'single',
    'update_scheme': 'exponential',
    'update_scheme_params': [2.0, 0.1]
}
```

**For Production** (high accuracy):
```python
config = {
    'device_type': 'GPU',
    'dtype_gpu': 'double',
    'update_scheme': '2nd_order',
    'update_scheme_params': [0.5, 0.001]
}
```

## Customization

All examples can be customized by modifying parameters:

```python
# Modify any example
domain_size = [custom_x, custom_y, custom_z]
ndiv = [custom_nx, custom_ny, custom_nz]
config['your_parameter'] = your_value
```

For more advanced customization, see the [API documentation](api/core.md).

---

**Next Steps**: Try running the examples and explore the [API reference](api/core.md) to understand the underlying methods.