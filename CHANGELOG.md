# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2025-10-19

### Added

- Function `get_xtal_nearest_neighbors()` in `pypfc_base` class to define the number of nearest neighbors and neighbor distances for different crystal structures
- Function `get_csp()` in `pypfc_base` class for evaluation of the centro-symmetry parameter (CSP)
- Function `do_ovito_csp()` in `pypfc_ovito` class for evaluation of the centro-symmetry parameter (CSP) using Ovito
- Example `ex05_structure_analysis.py`, demonstrating the new `get_csp()` and `do_ovito_csp()` methods
- Added check that all entries in ndiv are even numbers in `pypfc_grid`

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
