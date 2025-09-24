# pyPFC: AI Coding Guidelines

pyPFC is a Python package for Phase Field Crystal (PFC) simulations using PyTorch for CPU/GPU acceleration. Understanding its architecture and patterns is essential for effective development.

## Core Architecture

### Class Inheritance Chain
The package follows a strict inheritance hierarchy (from base to main):
```
pypfc_grid -> pypfc_base -> pypfc_pre -> pypfc_io -> pypfc (main)
```
- **pypfc_grid**: Grid setup and domain discretization
- **pypfc_base**: Device management, FFT operations, and core mathematical functions  
- **pypfc_pre**: Density field generation and crystal structure setup
- **pypfc_io**: File I/O operations (VTK, XYZ, pickle formats)
- **pypfc**: Main simulation class with time integration and energy evaluation

### Key Design Patterns

**Configuration-Driven Architecture**: All simulation parameters are managed through a `DEFAULTS` dictionary in the main class. User configurations merge with defaults, with unknown keys ignored but logged.

**Dual Precision Support**: The codebase maintains both CPU (`numpy`) and GPU (`PyTorch`) representations:
- `_dtype_cpu`: numpy precision (typically `np.double`)  
- `_dtype_gpu`: PyTorch precision (typically `torch.float64`)
- Complex types automatically derived: `torch.cfloat`/`torch.cdouble`

**Device Abstraction**: GPU/CPU switching handled via `device_type` parameter with automatic fallback and memory management.

## Critical Development Workflows

### Device Management
Always check GPU availability and handle fallbacks:
```python
nGPU = torch.cuda.device_count()
if nGPU > 0 and self._device_type.upper() == 'GPU':
    self._device = torch.device('cuda')
    torch.cuda.set_device(self._device_number)
else:
    self._device = torch.device('cpu')
```

### Memory Layout Requirements
All PyTorch tensors must be contiguous for FFT operations:
```python
self._den_d = torch.zeros((nx, ny, nz), dtype=dtype_gpu, device=device).contiguous()
```

### Configuration Parameter Handling
New parameters must be added to `DEFAULTS` dictionary and handled in `__init__`:
```python
DEFAULTS = {
    'new_param': default_value,
    # ... other defaults
}
```

## Data Flow Patterns

### Simulation Lifecycle
1. **Setup**: `setup_simulation(domain_size, ndiv, config)`
2. **Initialization**: `do_single_crystal()` or `do_polycrystal()` 
3. **Evolution**: `do_step_update()` in main loop
4. **Analysis**: `interpolate_density_maxima()`, `get_energy()`, `get_phase_field()`
5. **Output**: `write_vtk_points()`, `write_extended_xyz()`
6. **Cleanup**: `cleanup()` to free GPU memory

### Fourier Space Operations
The package heavily uses FFTs with specific naming conventions:
- `_d` suffix: device tensors (GPU/CPU)
- `_f_` prefix: Fourier space quantities
- `k2_d`: Sum of squared wave vectors
- `C2_d`: Two-point correlation function

## Project-Specific Conventions

### Naming Conventions
- **Functions and variables**: Use underscore separation (snake_case)
  - Examples: `test_variable`, `this_is_a_function()`, `density_threshold`
  - Consistent throughout codebase: `do_step_update()`, `get_energy()`, `set_device_type()`

### Performance and Memory Management
- **Computational efficiency** and **minimal memory usage** are primary concerns
- **In-place tensor operations** preferred whenever possible:
  ```python
  # Preferred: in-place operations
  tensor.add_(other_tensor)
  tensor.mul_(scalar)
  
  # Avoid: creating new tensors unnecessarily
  tensor = tensor + other_tensor  # Creates new memory
  ```
- Contiguous memory layout essential for FFT performance

### Grid Requirements
- All grid divisions (`ndiv`) must be even numbers
- Domain size specified in lattice parameters
- Periodic boundary conditions assumed

### File Naming Patterns
- Examples: `ex##_description.py` (## = zero-padded number)
- Output files: `pfc_data_` + step number
- Configuration: `pypfc_setup.txt` for simulation parameters

### Time Integration Schemes
Support multiple schemes via `update_scheme` parameter:
- `'1st_order'`, `'2nd_order'`, `'exponential'`
- Parameters controlled via `update_scheme_params` array

## Testing and Validation

### Example-Based Testing
Run examples as integration tests:
```bash
cd examples/
python ex04_quick_start.py  # Basic functionality
python ex01_grain_growth.py # Complex simulation
```

### GPU Validation
Always test GPU memory management:
```python
torch.cuda.empty_cache()  # Clear cache between runs
torch.cuda.memory_allocated()  # Monitor usage
```

## Common Pitfalls

### Grid Setup
- **Issue**: Simulation instability or "melting"
- **Cause**: `domain_size` not matching lattice periodicity
- **Fix**: Ensure domain accommodates crystal structure

### Interpolation Accuracy  
- **Issue**: Atom positions don't match density maxima
- **Cause**: Low grid resolution or interpolation order
- **Fix**: Reduce `ddiv` values or increase `density_interp_order`

### Memory Management
- **Issue**: GPU memory errors in long simulations
- **Cause**: Tensor accumulation without cleanup
- **Fix**: Call `cleanup()` method and `torch.cuda.empty_cache()`

### Performance Optimization
- **Issue**: Slow tensor operations or excessive memory usage
- **Cause**: Non-contiguous tensors or out-of-place operations
- **Fix**: Use `.contiguous()` and in-place operations (`add_()`, `mul_()`, etc.)

## Integration Points

### OVITO Interface (`pypfc_ovito.py`)
Optional module providing crystal structure analysis:
- Polyhedral Template Matching (PTM)
- Dislocation analysis (DXA)  
- Requires separate OVITO package installation

### VTK Output Pipeline
Structured for visualization in ParaView/VisIt:
- Point data: atom positions and properties
- Grid data: continuous density/energy fields
- Time series support for animations