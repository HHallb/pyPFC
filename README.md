# pyPFC: An Open-Source Python Package for Phase Field Crystal Simulations
A Python software package for setting up, running and processing Phase Field Crystal (PFC) simulations. The code uses PyTorch to allow execution on both CPUs and GPUs, depending on the available hardware.

## Description of Source Files
The software is built on classes, contained in separate modules/files, with an inheritance chain (from bottom to top) comprising:

| No | File (*.py)     | Description             |
| -- | --------------- | ------------------------|
| 4  | **pypfc_main**  | Main class              | 
| 3  | **pypfc_io**    | Data IO methods         |
| 2  | **pypfc_aux**   | Auxiliary methods       |
| 1  | **pypfc_base**  | Base methods            |
| 0  | **pypfc_grid**  | Grid methods            |

In addition, **pypfc_ovito.py** provides custom interfaces to selected functionalities in [OVITO](https://www.ovito.org/), useful for post-processing of pyPFC simulation output.

## Requirements
The following Python packages are required:

* datetime
* gzip
* matplotlib (only used in the examples)
* numpy
* os
* ovito (if pypfc_ovito is to be used)
* pickle
* re
* scikit-image
* scipy
* sys
* time
* torch
* vtk

## Installation and Usage
Import with `import pypfc_main` and (optionally) `import pypfc_ovito`.

## Licencing and Acknowledgment
This software is released under a [GNU GPLv3 license](https://www.gnu.org/licenses/).
