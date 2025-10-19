![pyPFC logo](images/pyPFC_logo_transparent.png)

# Installation Guide

The following installation methods can be considered:

## Method 1: Local Installation

The simplest way to install pyPFC is via pip, which should ensure that the package dependencies are met automatically. Note, however, that **PyTorch is only installed with CPU support** since PyPI only provides the CPU version of torch. GPU support needs to be added manually.

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

Import pyPFC into your Python code by `import pypfc` and, optionally, `import pypfc_ovito`. See the [Quick Start Tutorial](quick_start.md) or the examples provided in `./examples/`.

## Method 2: Development Installation

For development or if you want to modify the code:

```bash
# Clone the repository
git clone https://github.com/HHallb/pyPFC.git
cd pyPFC

# Install in development mode
pip install -e .
```

## Method 3: Direct Installation from GitHub

```bash
pip install git+https://github.com/HHallb/pyPFC.git
```

## Dependencies

pyPFC automatically installs the following required packages:

* numpy
* scikit-image
* scipy
* torch
* vtk

Note that PyPI only installs torch with CPU support. To add GPU support, refer to [GPU support](#gpu-support).

### Optional Dependencies

For enhanced functionality, you may want to install:

```bash
# For visualization and analysis
pip install matplotlib

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

For other CUDA versions and further information, please refer to the official [PyTorch documentation](https://pytorch.org/get-started/locally/).

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
sim = pypfc.setup_simulation([2,2,2])
print("pyPFC successfully imported!")

# Check versions
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Next Steps

After successful installation:

1. [Quick Start Guide](quick_start.md) - Run your first simulation
2. [Examples](examples.md) - Explore pyPFC example
3. [API Documentation](api/core.md) - Learn the API

## Need Help?

Check the [troubleshooting section](troubleshooting.md).
