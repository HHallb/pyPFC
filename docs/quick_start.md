# Quick Start Guide

This guide will get you running your first pyPFC simulation in minutes.

## Your First Simulation

Let's create a simple single crystal simulation:

```python
import numpy as np
import matplotlib.pyplot as plt
import pypfc

# 1. Initialize the PFC simulation
pfc = pypfc.PyPFC()

# 2. Define simulation parameters
domain_size = [32.0, 32.0, 8.0]  # Domain size in lattice units
ndiv = [64, 64, 16]               # Grid divisions (must be even)

# Configuration
config = {
    'device_type': 'GPU',         # Use 'CPU' if no GPU available
    'dtype_gpu': 'double',        # Precision: 'single' or 'double'
    'update_scheme': 'exponential',
    'update_scheme_params': [1.0, 0.01]  # [dt, tolerance]
}

# 3. Setup the simulation
pfc.setup_simulation(domain_size, ndiv, config)

# 4. Create initial crystal structure
pfc.do_single_crystal()

print("Initial setup complete!")
print(f"Domain: {domain_size}")
print(f"Grid: {ndiv}")
print(f"Device: {pfc.get_device_type()}")
```

## Running the Simulation

```python
# Run simulation for 100 steps
energies = []
steps = []

for step in range(100):
    # Update the system
    pfc.do_step_update()
    
    # Monitor energy every 10 steps
    if step % 10 == 0:
        energy = pfc.get_energy()
        energies.append(energy)
        steps.append(step)
        print(f"Step {step:3d}: Energy = {energy:.6f}")

# Plot energy evolution
plt.figure(figsize=(8, 5))
plt.plot(steps, energies, 'b-o', linewidth=2, markersize=4)
plt.xlabel('Simulation Step')
plt.ylabel('Free Energy')
plt.title('Energy Evolution')
plt.grid(True, alpha=0.3)
plt.show()
```

## Analyzing Results

### Extract Atomic Positions

```python
# Get density field and find atom positions
density = pfc.get_density_field()
positions = pfc.interpolate_density_maxima()

print(f"Found {len(positions)} atoms")
print(f"Density range: {density.min():.3f} to {density.max():.3f}")

# Visualize atomic positions (2D slice)
plt.figure(figsize=(10, 8))
z_slice = density.shape[2] // 2  # Middle slice
plt.imshow(density[:, :, z_slice], cmap='viridis', origin='lower')
plt.colorbar(label='Density')
plt.title(f'Density Field (z-slice {z_slice})')

# Overlay atom positions in this slice
slice_atoms = positions[np.abs(positions[:, 2] - z_slice) < 1]
if len(slice_atoms) > 0:
    plt.scatter(slice_atoms[:, 1], slice_atoms[:, 0], 
               c='red', s=20, alpha=0.7, label='Atoms')
    plt.legend()

plt.xlabel('Y Grid Points')
plt.ylabel('X Grid Points')
plt.show()
```

### Save Results

```python
# Save density field and atomic positions
pfc.write_vtk_points('single_crystal_atoms.vtu')
pfc.write_extended_xyz('single_crystal_atoms.xyz')

print("Results saved:")
print("- single_crystal_atoms.vtu (for ParaView)")
print("- single_crystal_atoms.xyz (for OVITO/VMD)")
```

## Advanced Example: Polycrystal

```python
# Create a polycrystal system
pfc_poly = pypfc.PyPFC()

# Setup larger domain for polycrystal
domain_size = [64.0, 64.0, 16.0]
ndiv = [128, 128, 32]

config_poly = {
    'device_type': 'GPU',
    'dtype_gpu': 'single',  # Use single precision for larger systems
    'polycrystal': True,
    'grain_seeds': [3, 3],  # 3x3 grains
}

pfc_poly.setup_simulation(domain_size, ndiv, config_poly)

# Generate polycrystal structure
pfc_poly.do_polycrystal()

# Run shorter simulation (polycrystals take more time)
for step in range(50):
    pfc_poly.do_step_update()
    
    if step % 10 == 0:
        energy = pfc_poly.get_energy()
        print(f"Polycrystal Step {step:2d}: Energy = {energy:.6f}")

# Analyze grain structure
positions = pfc_poly.interpolate_density_maxima()
pfc_poly.write_vtk_points('polycrystal_atoms.vtu')

print(f"Polycrystal complete with {len(positions)} atoms")
```

## Configuration Options

### Essential Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `device_type` | Computation device | `'GPU'` | `'GPU'`, `'CPU'` |
| `dtype_gpu` | Precision | `'double'` | `'single'`, `'double'` |
| `update_scheme` | Time integration | `'exponential'` | `'1st_order'`, `'2nd_order'`, `'exponential'` |

### Performance Tips

```python
# For large simulations
config_large = {
    'device_type': 'GPU',
    'dtype_gpu': 'single',      # Faster, less memory
    'update_scheme': 'exponential',
    'update_scheme_params': [1.0, 0.1],  # Larger time step
    'density_threshold': 0.5,   # Lower threshold for atom detection
}

# For high accuracy
config_precise = {
    'device_type': 'GPU',
    'dtype_gpu': 'double',      # Higher precision
    'update_scheme': '2nd_order',
    'update_scheme_params': [0.5, 0.001],  # Smaller time step
    'density_interp_order': 3,  # Higher interpolation order
}
```

## Common Patterns

### Monitoring Convergence

```python
def check_convergence(pfc, tolerance=1e-6, window=10):
    """Check if simulation has converged based on energy stability"""
    energies = []
    
    for step in range(window):
        pfc.do_step_update()
        energies.append(pfc.get_energy())
    
    energy_std = np.std(energies)
    return energy_std < tolerance, energy_std

# Usage
converged, stability = check_convergence(pfc)
print(f"Converged: {converged}, Stability: {stability:.2e}")
```

### Memory Management

```python
import torch

def run_with_memory_check(pfc, max_steps=1000):
    """Run simulation with memory monitoring"""
    
    for step in range(max_steps):
        pfc.do_step_update()
        
        # Check memory every 100 steps
        if step % 100 == 0 and torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e9  # GB
            print(f"Step {step}: Memory used = {memory_used:.2f} GB")
            
            # Clear cache if memory is high
            if memory_used > 8.0:  # 8 GB threshold
                torch.cuda.empty_cache()
```

## Next Steps

Now that you've run your first simulation:

1. **Explore Examples**: Check out the [examples gallery](examples.md)
2. **Learn the API**: Read the [API documentation](api/core.md)
3. **Advanced Usage**: Explore more complex simulation setups

## Troubleshooting

### Common Issues

**Simulation "melts" (atoms disappear)**:
- Check that `domain_size` matches crystal periodicity
- Reduce time step in `update_scheme_params`

**GPU out of memory**:
- Use `'dtype_gpu': 'single'`
- Reduce `ndiv` values
- Call `torch.cuda.empty_cache()`

**Slow performance**:
- Verify GPU is being used: `pfc.get_device_type()`
- Check PyTorch CUDA installation
- Ensure `ndiv` values are even numbers

---

**Ready for more?** Explore the [comprehensive examples](examples.md) to see pyPFC's full capabilities!