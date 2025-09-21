'''
pyPFC - An Open-Source Python implementation of the Phase Field Crystal method.
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

class setup_grid:

    def __init__(self, domain_size, ndiv):
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

    def set_ndiv(self, ndiv):
        self._ndiv = ndiv
        self._nx = ndiv[0]
        self._ny = ndiv[1]
        self._nz = ndiv[2]

    def get_ndiv(self):
        return self._ndiv
    
    def set_ddiv(self, ddiv):
        self._ddiv = ddiv
        self._dx = ddiv[0]
        self._dy = ddiv[1]
        self._dz = ddiv[2]

    def get_ddiv(self):
        return self._ddiv

    def get_domain_size(self):
        return self._domain_size

    def copy_from(self, grid):
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
