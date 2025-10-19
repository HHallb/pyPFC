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
import time
import torch
from pypfc_base import setup_base
class setup_pre(setup_base):

    def __init__(self, domain_size, ndiv, config):
        """
        Initialize the class.

        Parameters
        ----------
        domain_size : array_like of int, shape (3,)
            Number of grid divisions along each coordinate axis [nx, ny, nz].
        ndiv : array_like of int, shape (3,)
            Number of grid divisions along each coordinate axis [nx, ny, nz].
        config : dict, optional
            Configuration parameters as key-value pairs.
            See the [pyPFC overview](core.md) for a complete list of the configuration parameters.
        """

        # Initiate the inherited class
        # ============================
        super().__init__(domain_size, ndiv, config=config)

        # Handle input arguments
        # ======================
        nx,ny,nz = self.get_ndiv()

        self._den    = np.zeros((nx, ny, nz), dtype=config['dtype_cpu'])
        self._ene    = np.zeros((nx, ny, nz), dtype=config['dtype_cpu'])
        self._struct = config['struct']
        self._alat   = config['alat']
        self._sigma  = config['sigma']
        self._npeaks = config['npeaks']

        # Get density field amplitudes and densitites
        # ===========================================
        if self._verbose: tstart = time.time()
        self._ampl, self._nlns = self.evaluate_ampl_dens()
        self._ampl_d = torch.from_numpy(self._ampl).to(self._device)
        self._nlns_d = torch.from_numpy(self._nlns).to(self._device)
        if self._verbose:
            tend = time.time()
            print(f'Time to evaluate amplitudes and densities: {tend-tstart:.3f} s')

# =====================================================================================

    def set_struct(self, struct):
        """
        Set the crystal structure.
        
        Parameters
        ----------
        struct : {'FCC', 'BCC'}
            Crystal structure type: `'FCC'`, `'BCC'`.
        """
        self._struct = struct

# =====================================================================================

    def get_struct(self):
        """
        Get the crystal structure.
        
        Returns
        -------
        struct : str
            Crystal structure type: `'FCC'`, `'BCC'`.
        """
        return self._struct

# =====================================================================================

    def set_density(self, den):
        """
        Set the density field.
        
        Parameters
        ----------
        den : ndarray of float, shape (nx,ny,nz)
            Density field.
        """
        self._den = den

# =====================================================================================

    def get_density(self):
        """
        Get the density field.
        
        Returns
        -------
        den : ndarray of float, shape (nx,ny,nz)
            Density field.
        """
        return self._den

# =====================================================================================

    def set_energy(self, ene):
        """
        Set the PFC energy field.
        
        Parameters
        ----------
        ene : ndarray of float, shape (nx,ny,nz)
            PFC energy field.
        """
        self._ene = ene

# =====================================================================================

    def set_ampl(self, ampl):
        """
        Set the amplitudes in the density approximation.
        
        Parameters
        ----------
        ampl : array_like of float, shape (N,)
            Amplitudes.
        """
        ampl         = np.array(ampl, dtype=self._dtype_cpu)
        self._ampl   = ampl
        self._ampl_d = torch.from_numpy(ampl).to(self._device)

# =====================================================================================

    def get_ampl(self):
        """
        Get the amplitudes in the density approximation.
        
        Returns
        -------
        ampl : ndarray of float, shape (N,)
            Amplitudes.
        """
        return self._ampl

# =====================================================================================

    def set_nlns(self, nlns):
        """
        Set the liquid and solid phase densities.
        
        Parameters
        ----------
        nlns : array_like of float, shape (2,)
            $[n_{l},n_{s}]$ where $n_{l}$ is liquid phase density 
            and $n_{s}$ is solid phase density.
        """
        nlns         = np.array(nlns, dtype=self._dtype_cpu)
        self._nlns   = nlns
        self._nlns_d = torch.from_numpy(nlns).to(self._device)

# =====================================================================================

    def get_nlns(self):
        """
        Get the liquid and solid phase densities.
        
        Returns
        -------
        nlns : ndarray of float, shape (2,)
            $[n_{l},n_{s}]$ where $n_{l}$ is liquid phase density 
            and $n_{s}$ is solid phase density.
        """
        return self._nlns

# =====================================================================================

    def set_sigma(self, sigma):
        """
        Set the temperature-like parameter sigma.
        
        Parameters
        ----------
        sigma : float
            Temperature-like parameter sigma.
        """
        self._sigma = sigma

# =====================================================================================

    def get_sigma(self):
        """
        Get the temperature-like parameter sigma.
        
        Returns
        -------
        sigma : float
            Temperature-like parameter sigma
        """
        return self._sigma

# =====================================================================================

    def set_npeaks(self, npeaks):
        """
        Set the number of peaks in the density field approximation.
        
        Parameters
        ----------
        npeaks : int
            Number of peaks in the density field approximation.
        """
        self._npeaks = npeaks

# =====================================================================================

    def get_npeaks(self):
        """
        Get the number of peaks in the density field approximation.
        
        Returns
        -------
        npeaks : int
            Number of peaks in the density field approximation.
        """
        return self._npeaks

# =====================================================================================

    def do_single_crystal(self, xtal_rot=None, params=None, model=0):
        """
        Define a single crystal in a periodic 3D domain.
        
        Parameters
        ----------
        xtal_rot : ndarray of float, shape (3,3), optional
            Crystal orientation (rotation matrix). Default is an identity matrix.
        params : list, optional
            List containing parameters for the single crystal model:
            
            - `model=0`: [radius] - spherical crystal radius
            - `model=1`: [start_x, end_x] - crystal extent in x direction
        model : int, optional  
            Density field layout.
            
            - 0: Spherical crystal
            - 1: Crystal extending throughout y and z, covering interval in x
        
        Returns
        -------
        density : ndarray of float, shape (nx,ny,nz)
            Density field.
            
        Raises
        ------
        ValueError
            If the value of `model` is not supported (should be 0 or 1).
        """

        # Default orientation
        if xtal_rot is None:
            xtal_rot = np.eye(3, dtype=self._dtype_cpu)

        # Grid
        nx,ny,nz = self._ndiv
        dx,dy,dz = self._ddiv
        Lx,Ly,Lz = self._ndiv*self._ddiv

        # Allocate output array
        density = np.full((nx, ny, nz), self._nlns[1], dtype=self._dtype_cpu)

        # Crystal orientations (passive rotations)
        Rot = xtal_rot[:,:].T

        xc = np.linspace(0, (nx-1)*dx, nx)
        yc = np.linspace(0, (ny-1)*dy, ny)
        zc = np.linspace(0, (nz-1)*dz, nz)

        # Generate crystal
        Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
        
        if model==0:
            xtal_radius = params[0]
            condition  = (np.sqrt((Xc-Lx/2)**2 + (Yc-Ly/2)**2 + (Zc-Lz/2)**2) <= xtal_radius)

        elif model==1:
            start_x   = params[0]
            end_x     = params[1]
            condition = (Xc >= start_x) & (Xc <= end_x)

        else:
            raise ValueError(f'Unsupported value: model={model}')

        crd = np.array([Xc[condition], Yc[condition], Zc[condition]])
        density[condition] = self.generate_density_field(crd, Rot)

        return density

# =====================================================================================

    def do_bicrystal(self, xtal_rot, params=None, liq_width=0.0, model=0):
        """
        Define a bicrystal with two different crystal orientations.
        
        Parameters
        ----------
        xtal_rot : ndarray of float, shape (3,3,2)
            Crystal orientations (rotation matrices) for the two grains.
        params : list, optional
            List containing parameters for the bicrystal model.
        liq_width : float, optional
            Width of the liquid band along the grain boundary.
        model : int, optional
            Density field layout.
            
            - 0: Cylindrical crystal, extending through z
            - 1: Spherical crystal  
            - 2: Bicrystal with two planar grain boundaries, normal to x
        
        Returns
        -------
        density : ndarray of float, shape (nx,ny,nz)
            Density field.

        Raises
        ------
        ValueError
            If the value of `model` is not supported (should be 0, 1 or 2).
        """

        # Grid
        nx,ny,nz = self._ndiv
        dx,dy,dz = self._ddiv
        Lx,Ly,Lz = self._ndiv*self._ddiv

        # Allocate output array
        density = np.full((nx, ny, nz), self._nlns[1], dtype=self._dtype_cpu)

        # Crystal orientations (passive rotations)
        Rot0 = xtal_rot[:,:,0].T
        Rot1 = xtal_rot[:,:,1].T

        xc = np.linspace(0, (nx-1)*dx, nx)
        yc = np.linspace(0, (ny-1)*dy, ny)
        zc = np.linspace(0, (nz-1)*dz, nz)

        # Generate bicrystal
        Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
        
        if model==0:
            xtalRadius = params[0]
            condition0 = (np.sqrt((Xc-Lx/2)**2 + (Yc-Ly/2)**2) >  (xtalRadius+liq_width/2))
            condition1 = (np.sqrt((Xc-Lx/2)**2 + (Yc-Ly/2)**2) <= (xtalRadius-liq_width/2))

        elif model==1:
            xtalRadius = params[0]
            condition0 = (np.sqrt((Xc-Lx/2)**2 + (Yc-Ly/2)**2 + (Zc-Lz/2)**2) >  (xtalRadius+liq_width/2))
            condition1 = (np.sqrt((Xc-Lx/2)**2 + (Yc-Ly/2)**2 + (Zc-Lz/2)**2) <= (xtalRadius-liq_width/2))

        elif model==2:
            gb_x1      = params[0]
            gb_x2      = params[1]
            condition0 = (Xc <=  (gb_x1-liq_width/2)) | (Xc >= (gb_x2+liq_width/2))
            condition1 = (Xc >= (gb_x1+liq_width/2)) & (Xc <= (gb_x2-liq_width/2))

        else:
            raise ValueError(f'Unsupported value: model={model}')

        crd = np.array([Xc[condition0], Yc[condition0], Zc[condition0]])
        density[condition0] = self.generate_density_field(crd, Rot0)
        crd = np.array([Xc[condition1], Yc[condition1], Zc[condition1]])
        density[condition1] = self.generate_density_field(crd, Rot1)

        return density

# =====================================================================================

    def do_polycrystal(self, xtal_rot, params=None, liq_width=0.0, model=0):
        """
        Define a polycrystal in a periodic 3D domain.
        
        Parameters
        ----------
        xtal_rot : ndarray of float, shape (3,3,n_xtal)
            Crystal orientations (rotation matrices) for n_xtal crystals.
        params : list, optional
            List containing parameters for the polycrystal model.
        liq_width : float, optional
            Width of the liquid band along the grain boundaries.
        model : int, optional
            Density field layout.
            
            - 0: A row of cylindrical seeds along y, with cylinders extending through z
        
        Returns
        -------
        density : ndarray of float, shape (nx,ny,nz)
            Polycrystal density field.
            
        Raises
        ------
        ValueError
            If the value of `model` is not supported (should be 0).
        """

        # Grid
        nx,ny,nz = self._ndiv
        dx,dy,dz = self._ddiv
        Lx,Ly,Lz = self._ndiv*self._ddiv

        # Allocate output array
        density = np.full((nx, ny, nz), self._nlns[1], dtype=self._dtype_cpu)
        
        # Number of crystals
        n_xtal = xtal_rot.shape[2]

        # Generate grid coordinates
        xc = np.linspace(0, (nx-1)*dx, nx)
        yc = np.linspace(0, (ny-1)*dy, ny)
        zc = np.linspace(0, (nz-1)*dz, nz)
        Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')

        # Generate polycrystal        
        if model==0:
            xtal_radius = (Ly - n_xtal*liq_width) / n_xtal / 2
            xcrd       = Lx / 2
            for i in range(n_xtal+1):
                ycrd      = i*liq_width + i*2*xtal_radius
                condition = (np.sqrt((Xc-xcrd)**2 + (Yc-ycrd)**2) <= xtal_radius)
                crd       = np.array([Xc[condition], Yc[condition], Zc[condition]])
                if i<n_xtal:
                    density[condition] = self.generate_density_field(crd, xtal_rot[:,:,i].T)
                else:
                    density[condition] = self.generate_density_field(crd, xtal_rot[:,:,0].T)
        else:
            raise ValueError(f'Unsupported value: model={model}')

        return density

# =====================================================================================

    def generate_density_field(self, crd, g):
        """
        Define a 3D density field for (X)PFC modeling.

        Parameters
        ----------
        crd : ndarray of float, shape (3,...)
            Grid point coordinates [x,y,z].
        g : ndarray of float, shape (3,3)
            Rotation matrix for crystal orientation.
    
        Returns
        -------
        density : ndarray of float
            Density field for the specified crystal structure with appropriate
            Fourier modes and amplitudes.
            
        Raises
        ------
        ValueError
            If `struct` is not one of the supported crystal structures 
            ('SC', 'BCC', 'FCC', 'DC').
            
        Notes
        -----
        The density field is generated based on the current crystal structure 
        (`struct`) and density field amplitudes (`ampl`) settings.
        """

        q    = 2*np.pi
        nAmp = len(self._ampl) # Number of density field modes/amplitudes
        n0   = self._nlns[1]   # Reference density (liquid)

        crdRot   = np.dot(g,crd)
        xc,yc,zc = crdRot

        match self._struct.upper():
            case 'SC':
                nA = self._ampl[0]*(np.cos(q*xc)*np.cos(q*yc)+np.cos(q*xc)*np.cos(q*zc)+np.cos(q*yc)*np.cos(q*zc))
                density = n0 + nA
            case 'BCC':
                nA = 4*self._ampl[0]*(np.cos(q*xc)*np.cos(q*yc)+np.cos(q*xc)*np.cos(q*zc)+np.cos(q*yc)*np.cos(q*zc)) # [110]
                nB = 2*self._ampl[1]*(np.cos(2*q*xc)+np.cos(2*q*yc)+np.cos(2*q*zc))                                  # [200]
                density = n0 + nA + nB
            case 'FCC':
                nA = 8*self._ampl[0]*(np.cos(q*xc)*np.cos(q*yc)*np.cos(q*zc))                                        # [111]
                nB = 2*self._ampl[1]*(np.cos(2*q*xc)+np.cos(2*q*yc)+np.cos(2*q*zc))                                  # [200]
                if nAmp==3:
                    nC = 4*self._ampl[2]*(np.cos(2*q*xc)*np.cos(2*q*zc) + np.cos(2*q*yc)*np.cos(2*q*zc) + np.cos(2*q*xc)*np.cos(2*q*yc))
                else:
                    nC = 0
                density = n0 + nA + nB + nC
            case 'DC': # Defined by two superposed FCC lattices, shifted with respect to each other
                nA = self._ampl[0]*8*(np.cos(q*xc)*np.cos(q*yc)*np.cos(q*zc) - np.sin(q*xc)*np.sin(q*yc)*np.sin(q*zc))
                nB = self._ampl[1]*8*(np.cos(2*q*xc)*np.cos(2*q*yc) + np.cos(2*q*xc)*np.cos(2*q*zc) + np.cos(2*q*yc)*np.cos(2*q*zc))
                if nAmp==3:
                    nC = self._ampl[2]*8*(np.cos(q*xc)*np.cos(q*yc)*np.cos(3*q*zc) + np.cos(q*xc)*np.cos(3*q*yc)*np.cos(q*zc) +
                                np.cos(3*q*xc)*np.cos(q*yc)*np.cos(q*zc) + np.sin(q*xc)*np.sin(q*yc)*np.sin(3*q*zc) +
                                np.sin(q*xc)*np.sin(3*q*yc)*np.sin(q*zc) + np.sin(3*q*xc)*np.sin(q*yc)*np.sin(q*zc))
                else:
                    nC = 0
                density = n0 + nA + nB + nC
            case _:
                raise ValueError(f'Unsupported value of struct: {self._struct.upper()}')

        return density

# =====================================================================================

    def evaluate_ampl_dens(self):
        """
        Get density field amplitudes and phase densities for XPFC simulations.

        Returns
        -------
        ampl : ndarray of float, shape (npeaks,)
            Density field amplitudes for the specified crystal structure and 
            number of peaks.
        nLnS : ndarray of float, shape (2,)
            Densities in the liquid (nL) and solid (nS) phases.
            
        Raises
        ------
        ValueError
            If `npeaks` is not supported for the current crystal structure.
        ValueError
            If `sigma` value is not supported for the current configuration.
        ValueError
            If `struct` is not 'BCC' or 'FCC', or if amplitudes and densities 
            are not available for the specified structure.
            
        Notes
        -----
        This method provides pre-calculated density field amplitudes and phase
        densities for different crystal structures (BCC, FCC) and numbers of 
        Fourier peaks in the two-point correlation function. The values depend
        on the effective temperature (sigma) in the Debye-Waller factor.
        
        The method uses lookup tables of pre-computed values for common 
        parameter combinations used in (X)PFC modeling.
        """

        if self._struct.upper()=='BCC':
            if self._sigma==0:
                if self._npeaks==2:
                    # Including [110], [200]
                    ampl = np.array([ 0.116548193580713,  0.058162568591367], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.151035610711215, -0.094238426687741], dtype=self._dtype_cpu)
                elif self._npeaks==3:
                    # Including [110], [200], [211]
                    ampl = np.array([ 0.111291217521458,  0.056111205274590, 0.005813371421170], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.158574317081128, -0.108067574994277], dtype=self._dtype_cpu)
                else:
                    raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
            elif self._sigma==0.1:
                if self._npeaks==2:
                    # Including [110], [200]
                    ampl = np.array([ 0.113205280767407,  0.042599977405133], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.106228213129645, -0.055509415103115], dtype=self._dtype_cpu)
                else:
                    raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
            else:
                raise ValueError(f'Unsupported value of sigma={self._sigma}')
        elif self._struct.upper()=='FCC':
            if self._sigma==0:
                if self._npeaks==2:
                    # Including [111], [200]
                    ampl = np.array([ 0.127697395147358,  0.097486643368977], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.127233738562750, -0.065826817872435], dtype=self._dtype_cpu)
                elif self._npeaks==3:
                    # Including [111], [200], [220]
                    ampl = np.array([ 0.125151338544038,  0.097120295466816, 0.009505792832995], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.138357209505865, -0.081227380909546], dtype=self._dtype_cpu)
                else:
                    raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
            else:
                raise ValueError(f'Unsupported value of sigma={self._sigma}')
        else:
            raise ValueError(f'Amplitudes and densities are not set. Unsupported value of struct={self._struct}')

        return ampl, nLnS

# =====================================================================================
