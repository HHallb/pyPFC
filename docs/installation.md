# Installation Guide

This guide will help you install pyPFC and its dependencies on your system.

## Prerequisites

Before installing pyPFC, ensure you have the following prerequisites:

### Python Requirements
- Python 3.8 or higher
- pip package manager

### System Requirements
- **Minimum**: 4 GB RAM, 2 GB disk space
- **Recommended**: 16 GB RAM, GPU with CUDA support
- **Operating Systems**: Linux, macOS, Windows

## Installation Methods

### Method 1: Development Installation (Recommended)

For development or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/HHallb/pyPFC.git
cd pyPFC

# Install in development mode
pip install -e .
```

### Method 2: Direct Installation from GitHub

```bash
pip install git+https://github.com/HHallb/pyPFC.git
```

### Method 3: Local Installation

If you have downloaded the source code:

```bash
# Navigate to the pyPFC directory
cd path/to/pyPFC

# Install the package
pip install .
```

## Dependencies

pyPFC automatically installs the following required dependencies:

### Core Dependencies
- **PyTorch** (≥1.9.0): GPU/CPU tensor operations and FFT
- **NumPy** (≥1.20.0): Numerical computations
- **SciPy** (≥1.7.0): Scientific computing utilities

### Optional Dependencies

For enhanced functionality, you may want to install:

```bash
# For visualization and analysis
pip install matplotlib vtk

# For OVITO integration (optional)
pip install ovito

# For Jupyter notebook examples
pip install jupyter ipywidgets
```

## GPU Support

### CUDA Installation

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8 (check your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (if no GPU available)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Verify GPU Support

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.get_device_name()}")
```

## Virtual Environment (Recommended)

Use a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv pypfc_env

# Activate (Linux/macOS)
source pypfc_env/bin/activate

# Activate (Windows)
pypfc_env\Scripts\activate

# Install pyPFC
pip install -e .
```

## Verification

Verify your installation:

```python
import pypfc
import torch
import numpy as np

# Test basic functionality
pfc = pypfc.PyPFC()
print("pyPFC successfully imported!")

# Check versions
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'pypfc'

**Solution**: Make sure you're in the correct environment and pyPFC is installed:

```bash
pip list | grep pypfc
```

#### CUDA Out of Memory

**Solution**: Reduce simulation size or use CPU:

```python
config = {'device_type': 'CPU'}
pfc.setup_simulation(domain_size, ndiv, config)
```

#### FFT Errors

**Solution**: Ensure grid dimensions are even numbers:

```python
# Good: even dimensions
ndiv = [64, 64, 32]

# Bad: odd dimensions
ndiv = [63, 65, 31]
```

### Performance Issues

#### Slow Simulations

1. **Enable GPU**: Verify CUDA installation and GPU detection
2. **Memory Layout**: Ensure tensors are contiguous
3. **Grid Size**: Start with smaller grids for testing

#### Memory Issues

1. **Reduce Grid Size**: Use smaller `ndiv` values
2. **Use Single Precision**: Set `dtype_gpu: 'single'`
3. **Clear Cache**: Call `torch.cuda.empty_cache()`

## Development Setup

For contributing to pyPFC:

```bash
# Clone with development tools
git clone https://github.com/HHallb/pyPFC.git
cd pyPFC

# Install development dependencies
pip install -e ".[dev]"

# Run tests
python -m pytest tests/

# Check code style
flake8 src/
```

## Docker Installation

For containerized deployment:

```dockerfile
FROM pytorch/pytorch:latest

WORKDIR /app
COPY . /app

RUN pip install -e .

CMD ["python", "-c", "import pypfc; print('pyPFC ready!')"]
```

## Next Steps

After successful installation:

1. [Quick Start Guide](quick_start.md) - Run your first simulation
2. [Examples](examples.md) - Explore example notebooks
3. [API Documentation](api/core.md) - Learn the API

---

**Need Help?** 

- Check the [troubleshooting section](#troubleshooting)
- Review [GitHub Issues](https://github.com/HHallb/pyPFC/issues)
- Join the community discussions