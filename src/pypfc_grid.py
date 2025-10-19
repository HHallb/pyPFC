'''
pyPFC: An Open-Source Python Package for Phase Field Crystal Simulations
Copyright (C) 2025 Håkan Hallberg

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import numpy as np

class setup_grid:

    def __init__(self, domain_size, ndiv):
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

    def set_ndiv(self, ndiv):
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

    def get_ndiv(self):
        """
        Get the number of grid divisions in each direction.
        
        Returns
        -------
        numpy.ndarray
            Number of grid divisions [nx, ny, nz] along each axis.
        """
        return self._ndiv
    
# =====================================================================================

    def set_ddiv(self, ddiv):
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

    def get_ddiv(self):
        """
        Get the grid spacing in each direction.
        
        Returns
        -------
        numpy.ndarray
            Grid spacing [dx, dy, dz] for each coordinate axis.
        """
        return self._ddiv

# =====================================================================================

    def get_domain_size(self):
        """
        Get the physical domain size in each direction.
        
        Returns
        -------
        numpy.ndarray
            Physical domain size [Lx, Ly, Lz] in lattice parameters.
        """
        return self._domain_size

# =====================================================================================

    def copy_from(self, grid):
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
