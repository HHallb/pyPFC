![pyPFC logo](images/pyPFC_logo_transparent.png)

# Troubleshooting

If you encounter installation issues or errors when trying to run pyPFC, answers can hopefully be found below:

## I have a Nvidia GPU installed but pyPFC tells me I don't

Ensure that PyTorch is installed with CUDA enabled. To be sure, you can also check your setup by running:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
```

If `is_available()` is `False` or `device_count()` is `0`, PyTorch cannot see your GPU. See the section on GPU support in the [Installation Guide](installation.md) on how to enable CUDA.

## The solid crystal phase fails to stabilize and/or appears to "melt" away

This is most likely due to `domain_size` not being set correctly to accommodate the current lattice periodicity.

## The atom positions obtained by `interpolate_density_maxima` do not seem to coincide with the density field maxima

This is likely related to either insufficient grid resolution or too low interpolation order. The former issue is mitigated by reducing the values in `ddiv` and in the latter case `density_interp_order` should be increased. Usually `density_interp_order=2` is fine and increasing the number will also increase the time spent on interpolation.

## ImportError: No module named 'pypfc'

Make sure you're in the correct environment and pyPFC is installed:

```bash
pip list | grep pypfc
```

## CUDA Out of Memory

Reduce simulation size or use CPU:

```python
params = {'device_type': 'CPU'}
domain_size = [size_x, size_y, size_z] # Consider reducing these numbers
pfc.setup_simulation(domain_size, config=params)
```

## FFT Errors

Ensure grid dimensions are even numbers (pyPFC should issue a warning if this is not the case):

```python
# Good: even dimensions
ndiv = [64, 64, 32]

# Bad: odd dimensions
ndiv = [63, 65, 31]
```

## Performance Issues

### Slow Simulations

1. **Enable GPU**: Verify CUDA installation and GPU detection
2. **Grid Size**: Start with smaller grids for testing

### Memory Issues

1. **Reduce Grid Size**: Use a smaller simulation domain or reduce the number of grid points `ndiv`
2. **Use Single Precision**: Set `dtype_gpu: 'torch.float32'` and `dtype_cpu': 'np.single'`
3. **Clear Device Cache**: Call `torch.cuda.empty_cache()`
