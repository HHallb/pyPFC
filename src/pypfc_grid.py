# Copyright (C) 2025 Håkan Hallberg
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE file for full license text

import numpy as np
from typing import Union, List

class setup_grid:

    def __init__(self, domain_size: Union[List[float], np.ndarray], ndiv: Union[List[int], np.ndarray]) -> None:
        """
        Initialize the grid setup with domain size and grid divisions.
        
        Parameters
        ----------
        domain_size : ndarray of float, shape (3,)
            Physical size of the simulation domain in each direction [Lx, Ly, Lz].
            Specified in lattice parameter units for crystal simulations.
        ndiv : ndarray of int, shape (3,)
            Number of grid divisions along each coordinate axis [nx, ny, nz].
            All values must be even numbers for FFT compatibility.
            
        Raises
        ------
        ValueError
            If any element in `ndiv` is not an even number.
        """

        # Check that all grid divisions are even numbers
        if not all(np.mod(n, 2) == 0 for n in ndiv):
            raise ValueError(f"All grid divisions must be even numbers, got ndiv={ndiv}")
        
        self._ndiv        = ndiv
        self._ddiv        = domain_size / ndiv
        self._dx          = self._ddiv[0]
        self._dy          = self._ddiv[1]
        self._dz          = self._ddiv[2]
        self._nx          = ndiv[0]
        self._ny          = ndiv[1]
        self._nz          = ndiv[2]
        self._domain_size = domain_size
        self._Lx          = self._domain_size[0]
        self._Ly          = self._domain_size[1]
        self._Lz          = self._domain_size[2]
        self._nz_half     = self._nz // 2 + 1

# =====================================================================================

    def set_ndiv(self, ndiv: Union[List[int], np.ndarray]) -> None:
        """
        Set the number of grid divisions in each direction.
        
        Updates the grid division parameters and related grid point counts.
        All divisions must be even numbers for FFT compatibility.
        
        Parameters
        ----------
        ndiv : array_like of int, shape (3,)
            Number of grid divisions in each direction [nx, ny, nz]. 
            Must be even numbers.
            
        Raises
        ------
        ValueError
            If any value in ndiv is not an even number.
        """
        # Check that all grid divisions are even numbers
        if not all(np.mod(n, 2) == 0 for n in ndiv):
            raise ValueError(f"All grid divisions must be even numbers, got ndiv={ndiv}")
        self._ndiv = ndiv
        self._nx = ndiv[0]
        self._ny = ndiv[1]
        self._nz = ndiv[2]

# =====================================================================================

    def get_ndiv(self) -> np.ndarray:
        """
        Get the number of grid divisions in each direction.
        
        Returns
        -------
        numpy.ndarray
            Number of grid divisions [nx, ny, nz] along each axis.
        """
        return self._ndiv
    
# =====================================================================================

    def set_ddiv(self, ddiv: Union[List[float], np.ndarray]) -> None:
        """
        Set the grid spacing in each direction.
        
        Parameters
        ----------
        ddiv : array_like of float, shape (3,)
            Grid spacing in each direction [dx, dy, dz].
        """
        self._ddiv = ddiv
        self._dx = ddiv[0]
        self._dy = ddiv[1]
        self._dz = ddiv[2]

# =====================================================================================

    def get_ddiv(self) -> np.ndarray:
        """
        Get the grid spacing in each direction.
        
        Returns
        -------
        numpy.ndarray
            Grid spacing [dx, dy, dz] for each coordinate axis.
        """
        return self._ddiv

# =====================================================================================

    def get_domain_size(self) -> np.ndarray:
        """
        Get the physical domain size in each direction.
        
        Returns
        -------
        numpy.ndarray
            Physical domain size [Lx, Ly, Lz] in lattice parameters.
        """
        return self._domain_size

# =====================================================================================

    def copy_from(self, grid: 'setup_grid') -> None:
        """
        Copy grid parameters from another grid object.
        
        This method copies all grid configuration parameters from another 
        setup_grid instance, including domain size, grid divisions, and
        derived parameters.
        
        Parameters
        ----------
        grid : setup_grid
            Another setup_grid instance to copy parameters from.
        """
        self._ndiv        = grid.get_ndiv()
        self._ddiv        = grid.get_ddiv()
        self._domain_size = grid.get_domain_size()
        self._dx          = self._ddiv[0]
        self._dy          = self._ddiv[1]
        self._dz          = self._ddiv[2]
        self._nx          = self._ndiv[0]
        self._ny          = self._ndiv[1]
        self._nz          = self._ndiv[2]
        self._Lx          = self._domain_size[0]
        self._Ly          = self._domain_size[1]
        self._Lz          = self._domain_size[2]

# =====================================================================================
