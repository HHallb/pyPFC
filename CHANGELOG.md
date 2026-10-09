# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-10-09

### Added

- Added a primitive pyPFC GUI in `pypfc_gui.py` to inspect and work on data fiels saved in the new HDF5-based file format.
- Added the total, integrated, energy as an optional output from `evaluate_energy()` and `get_energy()` in `pypfc` class
- Added `evaluate_grand_potential_energy()` and `get_grand_potential_energy()` in `pypfc` class to evaluate and retrieve the grand potential energy density
- Added `evaluate_chemical_potential()` and `get_chemical_potential()` in `pypfc` class to evaluate and retrieve the chemical potential
- Added the possibility to define a strained density field in `generate_density_field()` and `do_polycrystal()` in `pypfc_pre` class
- Added more model alternatives in `do_polycrystal()` in `pypfc_pre` class
- Added `save_hdf5()` and `load_hdf5()` in `pypfc_io` to handle saving and loading files in HDF5 format
- Added `get_atom_bond_data()` in `pypfc_base` class to evaluate min/max atom bond angles and bond lengths
- Added `interpolate_gb_from_phase_field()` in `pypfc_base` class to interpolate phase field iso-contours representing bicrystal GBs
- Added `minimum_periodic_domain()` in `pypfc_base` class to find the minimum periodic 3D domain for a given crystal structure and GB configuration
- Added `get_csl_config()` in `pypfc_base` class to generate CSL configurations for symmetric tilt or twist GBs
- Added `expand_minimum_domain()` in `pypfc_base` class to repeat a minimum domain size to match a target domain size

### Changed

- Changed internally in `interpolate_density_maxima()` in `pypfc_base` class to highlight that additional input arguments beyond the density can be arbitrary fields, not just phase fields

### Fixed

- Added the missing parameters `g1`, `g2` and `g3` to `evaluate_energy()` in `pypfc` class

## [Unreleased] - 2025-12-01

### Added

- Added SECURIY.md with info on security-related matters
- Function `get_xtal_nearest_neighbors()` in `pypfc_base` class to define the number of nearest neighbors and neighbor distances for different crystal structures
- Function `get_csp()` in `pypfc_base` class for evaluation of the centro-symmetry parameter (CSP)
- Function `do_ovito_csp()` in `pypfc_ovito` class for evaluation of the centro-symmetry parameter (CSP) using Ovito
- Example `ex05_structure_analysis.py`, demonstrating the new `get_csp()` and `do_ovito_csp()` methods
- Added check that all entries in ndiv are even numbers in `pypfc_grid`
- Added the option of generating a cylindrical single crystal in `do_single_crystal()` in `pypfc_pre`.

### Changed

- Changed the documentation into MkDocs format, using docstrings for automatized documentation
- Updated README.md with documentation notes
- Updated README.md with a Table of Contents
- Improved performance of `interpolate_density_maxima()` method by ~5%
- `evaluate_k2_d` is now done directly on the device by torch functionality in `pypfc_base`

### Fixed

- Added declaration and typing of class variable `alpha` in `pypfc_base` class
- Corrected typos in README.md
- Fixed missing closing quote in pyproject.toml
- Fixed the arguments used to initialize `pypfc_grid` in class `pypfc_ovito`

### Deprecated

- [Add any deprecated features here]

### Removed

- [Add any removed features here]

### Security

- [Add any security updates here]

## [0.0.3] - 2024-09-23

### Added

- Third release of pyPFC on PyPI, considered to be the initial public version.
