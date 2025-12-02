# Copyright (C) 2025 Håkan Hallberg
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE file for full license text

import numpy as np
import time
import torch
import warnings
import scipy.interpolate as spi
from pypfc_base import setup_base
from typing import Union, List, Optional, Tuple, Dict, Any

class setup_pre(setup_base):

    def __init__(self, domain_size: Union[List[float], np.ndarray], ndiv: Union[List[int], np.ndarray], config: Dict[str, Any]) -> None:
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

    def set_struct(self, struct: str) -> None:
        """
        Set the crystal structure.
        
        Parameters
        ----------
        struct : {'FCC', 'BCC'}
            Crystal structure type: `'FCC'`, `'BCC'`.
        """
        self._struct = struct

# =====================================================================================

    def get_struct(self) -> str:
        """
        Get the crystal structure.
        
        Returns
        -------
        struct : str
            Crystal structure type: `'FCC'`, `'BCC'`.
        """
        return self._struct

# =====================================================================================

    def set_density(self, den: np.ndarray) -> None:
        """
        Set the density field.
        
        Parameters
        ----------
        den : ndarray of float, shape (nx,ny,nz)
            Density field.
        """
        self._den = den

# =====================================================================================

    def get_density(self) -> np.ndarray:
        """
        Get the density field.
        
        Returns
        -------
        den : ndarray of float, shape (nx,ny,nz)
            Density field.
        """
        return self._den

# =====================================================================================

    def set_energy(self, ene: np.ndarray) -> None:
        """
        Set the PFC energy field.
        
        Parameters
        ----------
        ene : ndarray of float, shape (nx,ny,nz)
            PFC energy field.
        """
        self._ene = ene

# =====================================================================================

    def set_ampl(self, ampl: Union[List[float], np.ndarray]) -> None:
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

    def get_ampl(self) -> np.ndarray:
        """
        Get the amplitudes in the density approximation.
        
        Returns
        -------
        ampl : ndarray of float, shape (N,)
            Amplitudes.
        """
        return self._ampl

# =====================================================================================

    def set_nlns(self, nlns: Union[List[float], np.ndarray]) -> None:
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

    def get_nlns(self) -> np.ndarray:
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

    def set_sigma(self, sigma: float) -> None:
        """
        Set the temperature-like parameter sigma.
        
        Parameters
        ----------
        sigma : float
            Temperature-like parameter sigma.
        """
        self._sigma = sigma

# =====================================================================================

    def get_sigma(self) -> float:
        """
        Get the temperature-like parameter sigma.
        
        Returns
        -------
        sigma : float
            Temperature-like parameter sigma
        """
        return self._sigma

# =====================================================================================

    def set_npeaks(self, npeaks: int) -> None:
        """
        Set the number of peaks in the density field approximation.
        
        Parameters
        ----------
        npeaks : int
            Number of peaks in the density field approximation.
        """
        self._npeaks = npeaks

# =====================================================================================

    def get_npeaks(self) -> int:
        """
        Get the number of peaks in the density field approximation.
        
        Returns
        -------
        npeaks : int
            Number of peaks in the density field approximation.
        """
        return self._npeaks

# =====================================================================================

    def do_single_crystal(self, xtal_rot: Optional[np.ndarray] = None, params: Optional[List[float]] = None, model: int = 0) -> np.ndarray:
        """
        Define a single crystal in a periodic 3D domain.
        
        ![Single Crystal Example](../images/do_single_crystal.png)
        
        Parameters
        ----------
        xtal_rot : ndarray of float, shape (3,3), optional
            Crystal orientation (rotation matrix). Default is an identity matrix.
        params : list, optional
            List containing parameters for the single crystal model:
            
            - `model=0`: [r] - spherical crystal radius
            - `model=1`: [x1, x2] - crystal extent in x direction
            - `model=2`: [r] - cylindrical crystal radius

        model : int, optional  
            Density field layout.
            
            - 0: Spherical crystal
            - 1: Crystal extending throughout y and z, covering interval in x
            - 2: Cylindrical crystal, extending through z
        
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
            radius    = params[0]
            condition = (np.sqrt((Xc-Lx/2)**2 + (Yc-Ly/2)**2 + (Zc-Lz/2)**2) <= radius)

        elif model==1:
            x1 = params[0]
            x2 = params[1]
            condition = (Xc >= x1) & (Xc <= x2)

        elif model==2:
            radius    = params[0]
            condition = (np.sqrt((Xc-Lx/2)**2 + (Yc-Ly/2)**2) <= radius)

        else:
            raise ValueError(f'Unsupported value: model={model}')

        crd = np.array([Xc[condition], Yc[condition], Zc[condition]])
        density[condition] = self.generate_density_field(crd, Rot)

        return density

# =====================================================================================

    def do_bicrystal(self, xtal_rot: np.ndarray, params: Optional[List[float]] = None, liq_width: float = 0.0, model: int = 0) -> np.ndarray:
        """
        Define a bicrystal with two different crystal orientations.
        
        ![Bicrystal Example](../images/do_bicrystal.png)

        Parameters
        ----------
        xtal_rot : ndarray of float, shape (3,3,2)
            Crystal orientations (rotation matrices) for the two grains.
        params : list, optional
            List containing parameters for the bicrystal model.
            
            - `model=0`: [r] - cylindrical crystal radius
            - `model=1`: [r] - spherical crystal radius
            - `model=2`: [gb_x1, gb_x2] - grain boundary positions along x

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

    def do_polycrystal(self, xtal_rot: np.ndarray, params: Optional[List[float]] = None, liq_width: float = 0.0, model: int = 0) -> np.ndarray:
        """
        Define a polycrystal in a periodic 3D domain.
        
        ![Polycrystal Example](../images/do_polycrystal.png)

        Parameters
        ----------
        xtal_rot : ndarray of float, shape (3,3,n_xtal)
            Crystal orientations (rotation matrices) for n_xtal crystals.
        params : list, optional
            List containing parameters for the polycrystal model.
            
            - `model=0`: No parameter needed. The number of crystal seeds is determined
              from the number of provided orientations.

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

    def generate_density_field(self, crd: np.ndarray, g: np.ndarray) -> np.ndarray:
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

    # def evaluate_ampl_dens(self):
    #     """
    #     Get density field amplitudes and phase densities for PFC simulations.

    #     Returns
    #     -------
    #     ampl : ndarray of float, shape (npeaks,)
    #         Density field amplitudes for the specified crystal structure and 
    #         number of peaks.
    #     nLnS : ndarray of float, shape (2,)
    #         Densities in the liquid (nL) and solid (nS) phases.
            
    #     Raises
    #     ------
    #     ValueError
    #         If `npeaks` is not supported for the current crystal structure.
    #     ValueError
    #         If `sigma` value is not supported for the current configuration.
    #     ValueError
    #         If `struct` is not 'BCC' or 'FCC', or if amplitudes and densities 
    #         are not available for the specified structure.
            
    #     Notes
    #     -----
    #     This method provides pre-calculated density field amplitudes and phase
    #     densities for different crystal structures (BCC, FCC) and numbers of 
    #     Fourier peaks in the two-point correlation function. The values depend
    #     on the effective temperature (sigma) in the Debye-Waller factor.
        
    #     The method uses lookup tables of pre-computed values for common 
    #     parameter combinations used.
    #     """

    #     if self._struct.upper()=='BCC':
    #         if self._sigma==0:
    #             if self._npeaks==2:
    #                 # Including [110], [200]
    #                 ampl = np.array([ 0.116548193580713,  0.058162568591367], dtype=self._dtype_cpu)
    #                 nLnS = np.array([-0.151035610711215, -0.094238426687741], dtype=self._dtype_cpu)
    #             elif self._npeaks==3:
    #                 # Including [110], [200], [211]
    #                 ampl = np.array([ 0.111291217521458,  0.056111205274590, 0.005813371421170], dtype=self._dtype_cpu)
    #                 nLnS = np.array([-0.158574317081128, -0.108067574994277], dtype=self._dtype_cpu)
    #             else:
    #                 raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
    #         elif self._sigma==0.1:
    #             if self._npeaks==2:
    #                 # Including [110], [200]
    #                 ampl = np.array([ 0.113205280767407,  0.042599977405133], dtype=self._dtype_cpu)
    #                 nLnS = np.array([-0.106228213129645, -0.055509415103115], dtype=self._dtype_cpu)
    #             else:
    #                 raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
    #         else:
    #             raise ValueError(f'Unsupported value of sigma={self._sigma}')
    #     elif self._struct.upper()=='FCC':
    #         if self._sigma==0:
    #             if self._npeaks==2:
    #                 # Including [111], [200]
    #                 ampl = np.array([ 0.127697395147358,  0.097486643368977], dtype=self._dtype_cpu)
    #                 nLnS = np.array([-0.127233738562750, -0.065826817872435], dtype=self._dtype_cpu)
    #             elif self._npeaks==3:
    #                 # Including [111], [200], [220]
    #                 ampl = np.array([ 0.125151338544038,  0.097120295466816, 0.009505792832995], dtype=self._dtype_cpu)
    #                 nLnS = np.array([-0.138357209505865, -0.081227380909546], dtype=self._dtype_cpu)
    #             else:
    #                 raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
    #         else:
    #             raise ValueError(f'Unsupported value of sigma={self._sigma}')
    #     else:
    #         raise ValueError(f'Amplitudes and densities are not set. Unsupported value of struct={self._struct}')

    #     return ampl, nLnS

# =====================================================================================

    def evaluate_ampl_dens(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get density field amplitudes and phase densities for PFC simulations.

        ![Amplitude and Density Example](../images/evaluate_ampl_dens.png)

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
            If `struct` is not 'SC', 'BCC' or 'FCC' or if amplitudes and densities 
            are not available for the specified structure.
            
        Notes
        -----
        This method provides pre-calculated density field amplitudes and phase
        densities for different crystal structures (SC, BCC, FCC) and numbers of 
        Fourier peaks in the two-point correlation function. The values depend
        on the temperature-like parameter `sigma`.
        
        Fore efficiency, the method uses lookup tables of pre-computed values.

        Supported configurations:

        | `struct` | `npeaks` | `sigma` range
        |----------|----------|--------------
        | SC       | 1        | [0.0, 0.25]
        |          | 2        | [0.0, 0.22]
        |          | 3        | [0.0, 0.22]
        | BCC      | 1        | [0.0, 0.33]
        |          | 2        | [0.0, 0.30]
        |          | 3        | [0.0, 0.25]
        | FCC      | 1        | [0.0, 0.25]
        |          | 2        | [0.0, 0.25]
        |          | 3        | [0.0, 0.25]

        """

        interp_kind='cubic'# 'linear'

        SC_0_1 = np.array([
            [  0.000000000000,   0.000111577614,   0.000446089169,   0.010555829201],
            [  0.006410256410,   0.000236982525,   0.000541836921,   0.010077966126],
            [  0.012820512821,   0.000641402131,   0.000941401903,   0.010000061607],
            [  0.019230769231,   0.001318335164,   0.001618169525,   0.010000077418],
            [  0.025641025641,   0.002282617389,   0.002628585696,   0.010746643204],
            [  0.032051282051,   0.003498620536,   0.003829627386,   0.010518265617],
            [  0.038461538462,   0.004982010224,   0.005280816639,   0.010001430136],
            [  0.044871794872,   0.006842462432,   0.007401644516,   0.013690256280],
            [  0.051282051282,   0.008805630895,   0.009130431731,   0.010447622310],
            [  0.057692307692,   0.011118984522,   0.011415826873,   0.010000198025],
            [  0.064102564103,   0.013730674205,   0.014049905800,   0.010384603572],
            [  0.070512820513,   0.016695054269,   0.017221633534,   0.013354488522],
            [  0.076923076923,   0.019782842410,   0.020087973632,   0.010186048547],
            [  0.083333333333,   0.023238087623,   0.023531709901,   0.010011427952],
            [  0.089743589744,   0.026990635859,   0.027286450079,   0.010069936269],
            [  0.096153846154,   0.031036519431,   0.031328120776,   0.010021190805],
            [  0.102564102564,   0.035383390598,   0.035672285932,   0.010000018023],
            [  0.108974358974,   0.040081716325,   0.040470969658,   0.011638495062],
            [  0.115384615385,   0.045000225385,   0.045286431493,   0.010011524429],
            [  0.121794871795,   0.050319246162,   0.050682418197,   0.011313713506],
            [  0.128205128205,   0.055951554482,   0.056363103427,   0.012086546584],
            [  0.134615384615,   0.061826558828,   0.062110003699,   0.010072065368],
            [  0.141025641026,   0.068103752967,   0.068380774005,   0.010000014094],
            [  0.147435897436,   0.074734566060,   0.075008997020,   0.010000000000],
            [  0.153846153846,   0.081873321143,   0.082386108537,   0.013733061456],
            [  0.160256410256,   0.089102505637,   0.089378025465,   0.010126730218],
            [  0.166666666667,   0.096861317985,   0.097130570815,   0.010071809842],
            [  0.173076923077,   0.105027102201,   0.105289049673,   0.010000238680],
            [  0.179487179487,   0.113622669299,   0.113880887340,   0.010001093206],
            [  0.185897435897,   0.122673294409,   0.122934168436,   0.010132697015],
            [  0.192307692308,   0.132214993113,   0.132495012036,   0.010589596096],
            [  0.198717948718,   0.142226456632,   0.142482377619,   0.010222313754],
            [  0.205128205128,   0.152968122534,   0.153389361756,   0.013250543047],
            [  0.211538461538,   0.163950715851,   0.164193289379,   0.010180124019],
            [  0.217948717949,   0.175743644187,   0.175979625619,   0.010174802831],
            [  0.224358974359,   0.188435371331,   0.188815129474,   0.013096115840],
            [  0.230769230769,   0.201502657830,   0.201718920724,   0.010055792526],
            [  0.237179487179,   0.215861282639,   0.216196816866,   0.012765149293],
            [  0.243589743590,   0.230842445922,   0.231060217278,   0.010518461250],
            [  0.250000000000,   0.247138627136,   0.247328457630,   0.010079900173],
        ], dtype=self._dtype_cpu)

        SC_0_1_2 = np.array([
            [  0.000000000000,  -0.132858846558,  -0.079836896681,   0.127959885427,   0.073379702903],
            [  0.005641025641,  -0.132735341195,  -0.079739076815,   0.127922253334,   0.073370589703],
            [  0.011282051282,  -0.132364791685,  -0.079445561236,   0.127809318817,   0.073343259548],
            [  0.016923076923,  -0.131747097675,  -0.078956181784,   0.127620966675,   0.073297740986],
            [  0.022564102564,  -0.130882091642,  -0.078270657506,   0.127357003188,   0.073234082305],
            [  0.028205128205,  -0.129769538479,  -0.077388593621,   0.127017153477,   0.073152352621],
            [  0.033846153846,  -0.128409134955,  -0.076309480101,   0.126601057693,   0.073052643452],
            [  0.039487179487,  -0.126800509064,  -0.075032689844,   0.126108265939,   0.072935070831],
            [  0.045128205128,  -0.124943219306,  -0.073557476491,   0.125538231770,   0.072799778033],
            [  0.050769230769,  -0.122836753946,  -0.071882971878,   0.124890304072,   0.072646939027],
            [  0.056410256410,  -0.120480530336,  -0.070008183178,   0.124163717063,   0.072476762763],
            [  0.062051282051,  -0.117873886215,  -0.067931997549,   0.123357544366,   0.072289496184],
            [  0.067692307692,  -0.115016108127,  -0.065653151071,   0.122470802559,   0.072085439044],
            [  0.073333333333,  -0.111906393055,  -0.063170262694,   0.121502274030,   0.071864940754],
            [  0.078974358974,  -0.108543872480,  -0.060481808008,   0.120450580916,   0.071628417196],
            [  0.084615384615,  -0.104927608773,  -0.057586121050,   0.119314136178,   0.071376361520],
            [  0.090256410256,  -0.101056598206,  -0.054481391207,   0.118091110486,   0.071109359551],
            [  0.095897435897,  -0.096929775732,  -0.051165660689,   0.116779391303,   0.070828109116],
            [  0.101538461538,  -0.092546022203,  -0.047636822910,   0.115376531933,   0.070533444469],
            [  0.107179487179,  -0.087904174908,  -0.043892622231,   0.113879687499,   0.070226367429],
            [  0.112820512821,  -0.083003042594,  -0.039930655665,   0.112285533629,   0.069908087496],
            [  0.118461538462,  -0.077841426624,  -0.035748377332,   0.110590161919,   0.069580074143],
            [  0.124102564103,  -0.072418150466,  -0.031343106701,   0.108788943671,   0.069244125903],
            [  0.129743589744,  -0.066732100625,  -0.026712041968,   0.106876349395,   0.068902463056],
            [  0.135384615385,  -0.060782283385,  -0.021852280286,   0.104845705327,   0.068557854214],
            [  0.141025641026,  -0.054567903599,  -0.016760846990,   0.102688857896,   0.068213792786],
            [  0.146666666667,  -0.048088474631,  -0.011434736365,   0.100395699753,   0.067874748951],
            [  0.152307692308,  -0.041343972834,  -0.005870966549,   0.097953479337,   0.067546540020],
            [  0.157948717949,  -0.034335058407,  -0.000066650346,   0.095345764715,   0.067236892808],
            [  0.163589743590,  -0.027063393380,   0.005980920068,   0.092550805619,   0.066956336198],
            [  0.169230769231,  -0.019532114434,   0.012274188374,   0.089538823336,   0.066719690257],
            [  0.174871794872,  -0.011746555752,   0.018815298693,   0.086267189622,   0.066548724927],
            [  0.180512820513,  -0.003715409072,   0.025606271013,   0.082671040675,   0.066477347355],
            [  0.186153846154,   0.004547275348,   0.032649942722,   0.078642513705,   0.066563071162],
            [  0.191794871795,   0.013018199697,   0.039953931162,   0.073974972974,   0.066917715581],
            [  0.197435897436,   0.021655359689,   0.047550691933,   0.068151167605,   0.067823190224],
            [  0.203076923077,   0.030355022542,   0.055800953753,   0.057811288899,   0.071096572993],
            [  0.208717948718,   0.038952723262,   0.063717426540,   0.052213884260,   0.072159610987],
            [  0.214358974359,   0.047700495077,   0.071424128628,   0.049090256322,   0.071773767476],
            [  0.220000000000,   0.056493926359,   0.079262119713,   0.045007351882,   0.071714002788],
        ], dtype=self._dtype_cpu)

        SC_0_1_2_3 = np.array([
            [  0.000000000000,  -0.147575298165,  -0.097980999695,   0.125927911738,   0.074175050598,   0.013483813673],
            [  0.005641025641,  -0.147447152874,  -0.097875116654,   0.125894799327,   0.074164781613,   0.013478792652],
            [  0.011282051282,  -0.147062676952,  -0.097557413906,   0.125795429337,   0.074133979081,   0.013463728290],
            [  0.016923076923,  -0.146421750113,  -0.097027730348,   0.125629703140,   0.074082656384,   0.013438616677],
            [  0.022564102564,  -0.145524171418,  -0.096285796693,   0.125397455158,   0.074010836212,   0.013403451276],
            [  0.028205128205,  -0.144369658578,  -0.095331234315,   0.125098451018,   0.073918551157,   0.013358222893],
            [  0.033846153846,  -0.142957846994,  -0.094163553614,   0.124732384919,   0.073805844564,   0.013302919624],
            [  0.039487179487,  -0.141288288552,  -0.092782151937,   0.124298876141,   0.073672771678,   0.013237526794],
            [  0.045128205128,  -0.139360450182,  -0.091186311022,   0.123797464613,   0.073519401119,   0.013162026870],
            [  0.050769230769,  -0.137173712222,  -0.089375194011,   0.123227605436,   0.073345816736,   0.013076399354],
            [  0.056410256410,  -0.134727366609,  -0.087347842010,   0.122588662219,   0.073152119899,   0.012980620645],
            [  0.062051282051,  -0.132020614975,  -0.085103170236,   0.121879899049,   0.072938432327,   0.012874663867],
            [  0.067692307692,  -0.129052566687,  -0.082639963765,   0.121100470879,   0.072704899540,   0.012758498653],
            [  0.073333333333,  -0.125822227342,  -0.079956877149,   0.120249382726,   0.072451690986,   0.012632089023],
            [  0.078974358974,  -0.122328531475,  -0.077052414111,   0.119325581062,   0.072179020015,   0.012495399648],
            [  0.084615384615,  -0.118570294154,  -0.073924943641,   0.118327794050,   0.071887129882,   0.012348386409],
            [  0.090256410256,  -0.114546238868,  -0.070572681685,   0.117254599578,   0.071576313975,   0.012191001344],
            [  0.095897435897,  -0.110254990845,  -0.066993689000,   0.116104381334,   0.071246922489,   0.012023190661],
            [  0.101538461538,  -0.105695078221,  -0.063185866230,   0.114875300194,   0.070899374280,   0.011844893817],
            [  0.107179487179,  -0.100864934927,  -0.059146949537,   0.113565259425,   0.070534171641,   0.011656042327],
            [  0.112820512821,  -0.095762905972,  -0.054874507172,   0.112171862035,   0.070151918860,   0.011456558207],
            [  0.118461538462,  -0.090387256038,  -0.050365937551,   0.110692358077,   0.069753345686,   0.011246351914],
            [  0.124102564103,  -0.084736182624,  -0.045618469544,   0.109123578915,   0.069339337291,   0.011025319624],
            [  0.129743589744,  -0.078807835432,  -0.040629165997,   0.107461854351,   0.068910972891,   0.010793339564],
            [  0.135384615385,  -0.072600344322,  -0.035394931834,   0.105702906876,   0.068469576086,   0.010550267037],
            [  0.141025641026,  -0.066111859028,  -0.029912528571,   0.103841714883,   0.068016781337,   0.010295927594],
            [  0.146666666667,  -0.059340605148,  -0.024178597751,   0.101872332976,   0.067554623012,   0.010030107523],
            [  0.152307692308,  -0.052284962725,  -0.018189696631,   0.099787651736,   0.067085656672,   0.009752540388],
            [  0.157948717949,  -0.044943576520,  -0.011942350658,   0.097579069956,   0.066613127490,   0.009462887658],
            [  0.163589743590,  -0.037315511163,  -0.005433128686,   0.095236036840,   0.066141209404,   0.009160710213],
            [  0.169230769231,  -0.029400470778,   0.001341251398,   0.092745394611,   0.065675353879,   0.008845425382],
            [  0.174871794872,  -0.021199112302,   0.008383777536,   0.090090401540,   0.065222815069,   0.008516240105],
            [  0.180512820513,  -0.012713500811,   0.015696948557,   0.087249228583,   0.064793470966,   0.008172043708],
            [  0.186153846154,  -0.003947777360,   0.023282526882,   0.084192507015,   0.064401172770,   0.007811226046],
            [  0.191794871795,   0.005090827816,   0.031141255729,   0.080879097768,   0.064066099560,   0.007431352919],
            [  0.197435897436,   0.014390407211,   0.039272644705,   0.077248129970,   0.063819215940,   0.007028536824],
            [  0.203076923077,   0.023931739822,   0.047675231479,   0.073202205189,   0.063711707836,   0.006596076198],
            [  0.208717948718,   0.033683488713,   0.056349006985,   0.068565386280,   0.063838562701,   0.006120983582],
            [  0.214358974359,   0.043591796275,   0.065308889903,   0.062943429214,   0.064416537258,   0.005572256254],
            [  0.220000000000,   0.053546288741,   0.074703616025,   0.054817316639,   0.066283110896,   0.004823815970],
        ], dtype=self._dtype_cpu)

        BCC_0_1 = np.array([
            [  0.000000000000,  -0.100757472156,  -0.038555033678,   0.121675755831],
            [  0.008461538462,  -0.100543348093,  -0.038393328265,   0.121628227645],
            [  0.016923076923,  -0.099900816692,  -0.037907974575,   0.121485622088],
            [  0.025384615385,  -0.098829385702,  -0.037098292640,   0.121247831776],
            [  0.033846153846,  -0.097328230033,  -0.035963134749,   0.120914684073],
            [  0.042307692308,  -0.095396180155,  -0.034500875977,   0.120485935017],
            [  0.050769230769,  -0.093031707323,  -0.032709395653,   0.119961266967],
            [  0.059230769231,  -0.090232904016,  -0.030586052895,   0.119340285511],
            [  0.067692307692,  -0.086997450935,  -0.028127683073,   0.118622482429],
            [  0.076153846154,  -0.083322613845,  -0.025330468025,   0.117807341316],
            [  0.084615384615,  -0.079205171214,  -0.022190008828,   0.116894189395],
            [  0.093076923077,  -0.074641390313,  -0.018701201401,   0.115882273852],
            [  0.101538461538,  -0.069626969515,  -0.014858186894,   0.114770735098],
            [  0.110000000000,  -0.064156975166,  -0.010654274693,   0.113558597102],
            [  0.118461538462,  -0.058225766609,  -0.006081851576,   0.112244756023],
            [  0.126923076923,  -0.051826907135,  -0.001132274517,   0.110827966815],
            [  0.135384615385,  -0.044953057989,   0.004204255963,   0.109306827415],
            [  0.143846153846,  -0.037595851844,   0.009938845795,   0.107679760021],
            [  0.152307692308,  -0.029745741136,   0.016084085676,   0.105944988798],
            [  0.160769230769,  -0.021391815355,   0.022654263992,   0.104100513190],
            [  0.169230769231,  -0.012521579626,   0.029665622492,   0.102144075725],
            [  0.177692307692,  -0.003120684538,   0.037136665292,   0.100073122868],
            [  0.186153846154,   0.006827406086,   0.045088535131,   0.097884756955],
            [  0.194615384615,   0.017341827282,   0.053545475466,   0.095575676580],
            [  0.203076923077,   0.028444829137,   0.062535403458,   0.093142101749],
            [  0.211538461538,   0.040162430446,   0.072090628335,   0.090579678730],
            [  0.220000000000,   0.052525256627,   0.082248763163,   0.087883357330],
            [  0.228461538462,   0.065569628690,   0.093053898472,   0.085047230092],
            [  0.236923076923,   0.079339000556,   0.104558137085,   0.082064317933],
            [  0.245384615385,   0.093885885007,   0.116823633505,   0.078926280627],
            [  0.253846153846,   0.109274536189,   0.129925384346,   0.075622958536],
            [  0.262307692308,   0.125584577227,   0.143955098187,   0.072141964988],
            [  0.270769230769,   0.142916452543,   0.159026727602,   0.068467628256],
            [  0.279230769231,   0.161399345379,   0.175284710601,   0.064579888030],
            [  0.287692307692,   0.181203560137,   0.192916540415,   0.060452001835],
            [  0.296153846154,   0.202560519354,   0.212173405717,   0.056047790757],
            [  0.304615384615,   0.225797394948,   0.233405198102,   0.051315301670],
            [  0.313076923077,   0.251401536447,   0.257126013411,   0.046176214187],
            [  0.321538461538,   0.280154144246,   0.284149433408,   0.040502863522],
            [  0.330000000000,   0.313453469844,   0.315915015753,   0.034062599427],
        ], dtype=self._dtype_cpu)

        BCC_0_1_2 = np.array([
            [  0.000000000000,  -0.151035574527,  -0.094238466487,   0.116548117195,   0.058162543126],
            [  0.007692307692,  -0.150748937972,  -0.093991047748,   0.116536055300,   0.058050179929],
            [  0.015384615385,  -0.149890931155,  -0.093250332705,   0.116499184033,   0.057714780531],
            [  0.023076923077,  -0.148467186620,  -0.092020899831,   0.116435475569,   0.057161368451],
            [  0.030769230769,  -0.146486840822,  -0.090310228971,   0.116341644975,   0.056398158764],
            [  0.038461538462,  -0.143962150424,  -0.088128458406,   0.116213287300,   0.055436301084],
            [  0.046153846154,  -0.140908062055,  -0.085488127090,   0.116045053232,   0.054289632553],
            [  0.053846153846,  -0.137341566447,  -0.082403726451,   0.115830863325,   0.052974205609],
            [  0.061538461538,  -0.133281083786,  -0.078891277282,   0.115564135408,   0.051507879434],
            [  0.069230769231,  -0.128745772405,  -0.074967813458,   0.115238019510,   0.049909813622],
            [  0.076923076923,  -0.123754844857,  -0.070650837926,   0.114845623586,   0.048199947188],
            [  0.084615384615,  -0.118326917242,  -0.065957765885,   0.114380217834,   0.046398476161],
            [  0.092307692308,  -0.112479421083,  -0.060905378531,   0.113835407156,   0.044525353079],
            [  0.100000000000,  -0.106228100549,  -0.055509310054,   0.113205264368,   0.042599830375],
            [  0.107692307692,  -0.099586609629,  -0.049783587650,   0.112484420241,   0.040640066478],
            [  0.115384615385,  -0.092566214816,  -0.043740239306,   0.111668109949,   0.038662808526],
            [  0.123076923077,  -0.085175600154,  -0.037388977815,   0.110752178694,   0.036683159638],
            [  0.130769230769,  -0.077420763948,  -0.030736962586,   0.109733051674,   0.034714432460],
            [  0.138461538462,  -0.069304990845,  -0.023788634202,   0.108607675119,   0.032768084907],
            [  0.146153846154,  -0.060828879439,  -0.016545611125,   0.107373435738,   0.030853729382],
            [  0.153846153846,  -0.051990404106,  -0.009006633680,   0.106028065680,   0.028979203506],
            [  0.161538461538,  -0.042784989915,  -0.001167537769,   0.104569539211,   0.027150688801],
            [  0.169230769231,  -0.033205580683,   0.006978760638,   0.102995966048,   0.025372863551],
            [  0.176923076923,  -0.023242681922,   0.015442289649,   0.101305484733,   0.023649077059],
            [  0.184615384615,  -0.012884361955,   0.024236073585,   0.099496157912,   0.021981534237],
            [  0.192307692308,  -0.002116195335,   0.033376232603,   0.097565869874,   0.020371481657],
            [  0.200000000000,   0.009078867579,   0.042882149975,   0.095512225376,   0.018819388491],
            [  0.207692307692,   0.020720722936,   0.052776729250,   0.093332447468,   0.017325117925],
            [  0.215384615385,   0.032832449687,   0.063086767476,   0.091023270845,   0.015888086645],
            [  0.223076923077,   0.045440788936,   0.073843477676,   0.088580826323,   0.014507411824],
            [  0.230769230769,   0.058576806221,   0.085083206194,   0.086000513876,   0.013182048043],
            [  0.238461538462,   0.072276790493,   0.096848426862,   0.083276907029,   0.011910939610],
            [  0.246153846154,   0.086583539377,   0.109188858684,   0.080403039640,   0.010692899190],
            [  0.253846153846,   0.101579482881,   0.122643699160,   0.078416384643,   0.010000000000],
            [  0.261538461538,   0.117581250860,   0.137336963078,   0.077368605645,   0.010000000000],
            [  0.269230769231,   0.134714007292,   0.152746881520,   0.075761208351,   0.010000000000],
            [  0.276923076923,   0.153122449907,   0.169149716443,   0.073699650637,   0.010000000000],
            [  0.284615384615,   0.173036002041,   0.186865320930,   0.071220101937,   0.010000000000],
            [  0.292307692308,   0.194800409761,   0.206310369061,   0.068309392141,   0.010000000000],
            [  0.300000000000,   0.218961581241,   0.228094214478,   0.064905692838,   0.010000000000],
        ], dtype=self._dtype_cpu)

        BCC_0_1_2_3 = np.array([
            [  0.000000000000,  -0.158574315179,  -0.108067567205,   0.111291220964,   0.056111206311,   0.005813370922],
            [  0.006410256410,  -0.158370074447,  -0.107884879537,   0.111285418991,   0.056039552625,   0.005809707006],
            [  0.012820512821,  -0.157758276444,  -0.107337608915,   0.111267709931,   0.055825279485,   0.005798732276],
            [  0.019230769231,  -0.156741669549,  -0.106428116576,   0.111237192048,   0.055470439114,   0.005780497604],
            [  0.025641025641,  -0.155324754766,  -0.105160286093,   0.111192387432,   0.054978413423,   0.005755087222],
            [  0.032051282051,  -0.153513668472,  -0.103539445839,   0.111131278675,   0.054353856425,   0.005722617847],
            [  0.038461538462,  -0.151316023054,  -0.101572261555,   0.111051358527,   0.053602614307,   0.005683237390],
            [  0.044871794872,  -0.148740693955,  -0.099266586506,   0.110949690248,   0.052731604330,   0.005637123001],
            [  0.051282051282,  -0.145797645869,  -0.096631344688,   0.110822979645,   0.051748758993,   0.005584479460],
            [  0.057692307692,  -0.142497612030,  -0.093676286450,   0.110667649810,   0.050662803480,   0.005525535543],
            [  0.064102564103,  -0.138851866374,  -0.090411816275,   0.110479923191,   0.049483149389,   0.005460541190],
            [  0.070512820513,  -0.134871940203,  -0.086848764656,   0.110255903078,   0.048219718139,   0.005389763591],
            [  0.076923076923,  -0.130569349795,  -0.082998158342,   0.109991653199,   0.046882767782,   0.005313482988],
            [  0.083333333333,  -0.125955335015,  -0.078870987884,   0.109683272216,   0.045482717504,   0.005231988194],
            [  0.089743589744,  -0.121040616917,  -0.074477979123,   0.109326960451,   0.044029975065,   0.005145571978],
            [  0.096153846154,  -0.115835181087,  -0.069829375095,   0.108919076659,   0.042534772456,   0.005054526484],
            [  0.102564102564,  -0.110348091760,  -0.064934734245,   0.108456183203,   0.041007014763,   0.004959138851],
            [  0.108974358974,  -0.104587339833,  -0.059802749822,   0.107935078681,   0.039456146547,   0.004859687193],
            [  0.115384615385,  -0.098559725832,  -0.054441093994,   0.107352817701,   0.037891039167,   0.004756437079],
            [  0.121794871795,  -0.092270776911,  -0.048856288671,   0.106706718186,   0.036319901312,   0.004649638626],
            [  0.128205128205,  -0.085724695245,  -0.043053603399,   0.105994357079,   0.034750213813,   0.004539524269],
            [  0.134615384615,  -0.078924333711,  -0.037036979130,   0.105213555804,   0.033188688597,   0.004426307252],
            [  0.141025641026,  -0.071871193758,  -0.030808975272,   0.104362357058,   0.031641250572,   0.004310180827],
            [  0.147435897436,  -0.064565439733,  -0.024370736302,   0.103438994651,   0.030113040354,   0.004191318145],
            [  0.153846153846,  -0.057005923661,  -0.017721973355,   0.102441858071,   0.028608435101,   0.004069872778],
            [  0.160256410256,  -0.049190214522,  -0.010860955655,   0.101369453312,   0.027131084358,   0.003945979796],
            [  0.166666666667,  -0.041114626312,  -0.003784506313,   0.100220361274,   0.025683957668,   0.003819757347],
            [  0.173076923077,  -0.032774239515,   0.003512003087,   0.098993194743,   0.024269400790,   0.003691308637],
            [  0.179487179487,  -0.024162910936,   0.011034664766,   0.097686554645,   0.022889197613,   0.003560724262],
            [  0.185897435897,  -0.015273267158,   0.018791059519,   0.096298985934,   0.021544635212,   0.003428084834],
            [  0.192307692308,  -0.006096676885,   0.026790299748,   0.094828933151,   0.020236569915,   0.003293463845],
            [  0.198717948718,   0.003376802645,   0.035043096761,   0.093274695352,   0.018965492707,   0.003156930768],
            [  0.205128205128,   0.013158510612,   0.043561858964,   0.091634379815,   0.017731592744,   0.003018554365],
            [  0.211538461538,   0.023261305246,   0.052360828478,   0.089905853574,   0.016534818162,   0.002878406234],
            [  0.217948717949,   0.033699732031,   0.061456265323,   0.088086691539,   0.015374933750,   0.002736564626],
            [  0.224358974359,   0.044490247677,   0.070866690596,   0.086174119622,   0.014251575452,   0.002593118585],
            [  0.230769230769,   0.055651513313,   0.080613203775,   0.084164951532,   0.013164302275,   0.002448172491],
            [  0.237179487179,   0.067204774682,   0.090719896777,   0.082055524367,   0.012112649858,   0.002301851121],
            [  0.243589743590,   0.079174346893,   0.101214454622,   0.079841789656,   0.011096261610,   0.002154305061],
            [  0.250000000000,   0.091588339215,   0.112128428186,   0.077518092308,   0.010114411731,   0.002005717982],
        ], dtype=self._dtype_cpu)

        FCC_0_1 = np.array([
            [  0.000000000000,   0.000254284588,   0.000661859281,   0.010078291801],
            [  0.006410256410,   0.000382211010,   0.000783414799,   0.010000000000],
            [  0.012820512821,   0.000777903252,   0.001178945090,   0.010000000049],
            [  0.019230769231,   0.001437743961,   0.001838514850,   0.010000016023],
            [  0.025641025641,   0.002362268468,   0.002762678479,   0.010000000000],
            [  0.032051282051,   0.003554974435,   0.003959331032,   0.010055002579],
            [  0.038461538462,   0.005008688193,   0.005408343376,   0.010003804741],
            [  0.044871794872,   0.006732417979,   0.007131069464,   0.010000050452],
            [  0.051282051282,   0.008938517477,   0.009663712020,   0.013483755378],
            [  0.057692307692,   0.010989187188,   0.011386167754,   0.010000970769],
            [  0.064102564103,   0.013529013346,   0.013929916428,   0.010063384092],
            [  0.070512820513,   0.016341145003,   0.016741612927,   0.010073145255],
            [  0.076923076923,   0.019429585545,   0.019827532865,   0.010058229665],
            [  0.083333333333,   0.022796283812,   0.023188241031,   0.010000758204],
            [  0.089743589744,   0.026450067096,   0.026840426306,   0.010000727032],
            [  0.096153846154,   0.030391542499,   0.030780126452,   0.010000363705],
            [  0.102564102564,   0.034625247053,   0.035012304397,   0.010005239612],
            [  0.108974358974,   0.039155022265,   0.039539602858,   0.010000000000],
            [  0.115384615385,   0.043987052538,   0.044369381491,   0.010000000000],
            [  0.121794871795,   0.049127271138,   0.049507541298,   0.010004917889],
            [  0.128205128205,   0.054585724837,   0.054968114012,   0.010067484691],
            [  0.134615384615,   0.060358532090,   0.060733685760,   0.010009853160],
            [  0.141025641026,   0.066465166561,   0.066836957093,   0.010005990931],
            [  0.147435897436,   0.072912149992,   0.073280901552,   0.010009761015],
            [  0.153846153846,   0.079721364286,   0.080098875329,   0.010177006441],
            [  0.160256410256,   0.086918520006,   0.087328682588,   0.010662151187],
            [  0.166666666667,   0.094440507969,   0.094831022189,   0.010464949297],
            [  0.173076923077,   0.102458732846,   0.102924822965,   0.011499697759],
            [  0.179487179487,   0.110703036825,   0.111078326955,   0.010395852625],
            [  0.185897435897,   0.119438457241,   0.119780431034,   0.010000455554],
            [  0.192307692308,   0.128660961614,   0.128997413370,   0.010002427716],
            [  0.198717948718,   0.138679090416,   0.139236830792,   0.012977203422],
            [  0.205128205128,   0.148584010197,   0.148907705027,   0.010001683831],
            [  0.211538461538,   0.159360207031,   0.159679376607,   0.010043439990],
            [  0.217948717949,   0.170974669450,   0.171428054132,   0.012108070541],
            [  0.224358974359,   0.182765508127,   0.183071927963,   0.010104577429],
            [  0.230769230769,   0.195921517020,   0.196417444090,   0.013039187282],
            [  0.237179487179,   0.209087778103,   0.209378879382,   0.010185422510],
            [  0.243589743590,   0.223543001226,   0.223812604531,   0.010005014892],
            [  0.250000000000,   0.239081751970,   0.239338897840,   0.010005613567],
        ], dtype=self._dtype_cpu)

        FCC_0_1_2 = np.array([
            [  0.000000000000,  -0.127233738563,  -0.065826817872,   0.127697395147,   0.097486643369],
            [  0.006410256410,  -0.127001591619,  -0.065643948522,   0.127659404870,   0.097433986897],
            [  0.012820512821,  -0.126305114456,  -0.065095197675,   0.127545421617,   0.097276025191],
            [  0.019230769231,  -0.125144153138,  -0.064180199904,   0.127355287891,   0.097012693435],
            [  0.025641025641,  -0.123518455904,  -0.062898331814,   0.127088759312,   0.096643896722],
            [  0.032051282051,  -0.121427664887,  -0.061248712976,   0.126745489759,   0.096169499009],
            [  0.038461538462,  -0.118871310715,  -0.059230198254,   0.126325027218,   0.095589319888],
            [  0.044871794872,  -0.115848805623,  -0.056841367930,   0.125826808245,   0.094903130364],
            [  0.051282051282,  -0.112359425152,  -0.054080525254,   0.125250129767,   0.094110628557],
            [  0.057692307692,  -0.108402331341,  -0.050945649869,   0.124594210249,   0.093211496645],
            [  0.064102564103,  -0.103976519166,  -0.047434423160,   0.123858088500,   0.092205311793],
            [  0.070512820513,  -0.099080828987,  -0.043544185295,   0.123040662943,   0.091091584970],
            [  0.076923076923,  -0.093713927641,  -0.039271920211,   0.122140663559,   0.089869740178],
            [  0.083333333333,  -0.087874294250,  -0.034614233362,   0.121156632084,   0.088539102806],
            [  0.089743589744,  -0.081560205544,  -0.029567327977,   0.120086897986,   0.087098886388],
            [  0.096153846154,  -0.074769721179,  -0.024126980242,   0.118929549280,   0.085548177617],
            [  0.102564102564,  -0.067500669669,  -0.018288514029,   0.117682396984,   0.083885919489],
            [  0.108974358974,  -0.059750635813,  -0.012046776070,   0.116342931671,   0.082110892403],
            [  0.115384615385,  -0.051516950796,  -0.005396112807,   0.114908270131,   0.080221693103],
            [  0.121794871795,  -0.042796686582,   0.001669649303,   0.113375089548,   0.078216711342],
            [  0.128205128205,  -0.033586656756,   0.009157217623,   0.111739545817,   0.076094104206],
            [  0.134615384615,  -0.023883426764,   0.017073837961,   0.109997171510,   0.073851768146],
            [  0.141025641026,  -0.013683337468,   0.025427285404,   0.108142747546,   0.071487308912],
            [  0.147435897436,  -0.002982547283,   0.034225835151,   0.106170140529,   0.068998009838],
            [  0.153846153846,   0.008222900108,   0.043478204920,   0.104072094829,   0.066380799326],
            [  0.160256410256,   0.019936973197,   0.053193457617,   0.101839964359,   0.063632218979],
            [  0.166666666667,   0.032163503081,   0.063380849176,   0.099463362967,   0.060748394734],
            [  0.173076923077,   0.044905971066,   0.074049601661,   0.096929703532,   0.057725014708],
            [  0.179487179487,   0.058167213166,   0.085208575729,   0.094223582451,   0.054557319479],
            [  0.185897435897,   0.071949015829,   0.096865809169,   0.091325945405,   0.051240113564],
            [  0.192307692308,   0.086251599570,   0.109027858038,   0.088212831236,   0.047767736636],
            [  0.198717948718,   0.101072822514,   0.121699003882,   0.084854038674,   0.044134368259],
            [  0.205128205128,   0.116407351049,   0.134880023790,   0.081210374574,   0.040333919802],
            [  0.211538461538,   0.132245427104,   0.148566764175,   0.077230296349,   0.036360449882],
            [  0.217948717949,   0.148571442065,   0.162748270933,   0.072844224679,   0.032208517355],
            [  0.224358974359,   0.165362240764,   0.177404553955,   0.067955168525,   0.027873763815],
            [  0.230769230769,   0.182585257682,   0.192504021360,   0.062421720244,   0.023353672369],
            [  0.237179487179,   0.200196644244,   0.208000788895,   0.056023918243,   0.018648571351],
            [  0.243589743590,   0.218139700155,   0.223832212643,   0.048383182459,   0.013762818855],
            [  0.250000000000,   0.236399136350,   0.240549721292,   0.041907026223,   0.010000000000],
        ], dtype=self._dtype_cpu)

        FCC_0_1_2_3 = np.array([
            [  0.000000000000,  -0.138357211062,  -0.081227384109,   0.125151338260,   0.097120295620,   0.009505793285],
            [  0.006410256410,  -0.138116726937,  -0.081030276163,   0.125117724002,   0.097067272606,   0.009498744446],
            [  0.012820512821,  -0.137395247605,  -0.080438878437,   0.125016814413,   0.096908194429,   0.009477612147],
            [  0.019230769231,  -0.136192674226,  -0.079452933296,   0.124848412045,   0.096643031903,   0.009442434240],
            [  0.025641025641,  -0.134508842413,  -0.078072013703,   0.124612185636,   0.096271735025,   0.009393274607],
            [  0.032051282051,  -0.132343516699,  -0.076295514555,   0.124307667926,   0.095794230403,   0.009330222662],
            [  0.038461538462,  -0.129696385238,  -0.074122645423,   0.123934252009,   0.095210417812,   0.009253393343],
            [  0.044871794872,  -0.126567053038,  -0.071552421219,   0.123491186447,   0.094520165741,   0.009162927109],
            [  0.051282051282,  -0.122955024686,  -0.068583658671,   0.122977550206,   0.093723287905,   0.009058989257],
            [  0.057692307692,  -0.118859725018,  -0.065214936434,   0.122392306880,   0.092819596291,   0.008941772251],
            [  0.064102564103,  -0.114280449450,  -0.061444613618,   0.121734214930,   0.091808816632,   0.008811492838],
            [  0.070512820513,  -0.109216374750,  -0.057270792344,   0.121001862259,   0.090690624278,   0.008668393873],
            [  0.076923076923,  -0.103666541112,  -0.052691302763,   0.120193641104,   0.089464624026,   0.008512744103],
            [  0.083333333333,  -0.097629838574,  -0.047703681914,   0.119307730115,   0.088130338616,   0.008344838370],
            [  0.089743589744,  -0.091104993068,  -0.042305151074,   0.118342072564,   0.086687195857,   0.008164997899],
            [  0.096153846154,  -0.084090552540,  -0.036492591984,   0.117294349859,   0.085134514328,   0.007973570622],
            [  0.102564102564,  -0.076584873768,  -0.030262522459,   0.116161949283,   0.083471487605,   0.007770931570],
            [  0.108974358974,  -0.068586110657,  -0.023611072130,   0.114941924595,   0.081697166981,   0.007557483271],
            [  0.115384615385,  -0.060092205141,  -0.016533959394,   0.113630947776,   0.079810442702,   0.007333656137],
            [  0.121794871795,  -0.051100882115,  -0.009026471023,   0.112225249657,   0.077810023805,   0.007099908744],
            [  0.128205128205,  -0.041609650363,  -0.001083446483,   0.110720546532,   0.075694416733,   0.006856727901],
            [  0.134615384615,  -0.031615812042,   0.007300730236,   0.109111948931,   0.073461903113,   0.006604628327],
            [  0.141025641026,  -0.021116484144,   0.016132127487,   0.107393847525,   0.071110517283,   0.006344151662],
            [  0.147435897436,  -0.010108636435,   0.025417246353,   0.105559769394,   0.068638024592,   0.006075864436],
            [  0.153846153846,   0.001410848244,   0.035162977648,   0.103602195597,   0.066041902018,   0.005800354421],
            [  0.160256410256,   0.013445083634,   0.045376526293,   0.101512327600,   0.063319323516,   0.005518224592],
            [  0.166666666667,   0.025997049787,   0.056065290894,   0.099279785356,   0.060467153673,   0.005230083543],
            [  0.173076923077,   0.039069410328,   0.067236682706,   0.096892212911,   0.057481955048,   0.004936530814],
            [  0.179487179487,   0.052664261554,   0.078897863764,   0.094334757049,   0.054360017054,   0.004638134985],
            [  0.185897435897,   0.066782793862,   0.091055378137,   0.091589368135,   0.051097417441,   0.004335401691],
            [  0.192307692308,   0.081424883931,   0.103714632473,   0.088633727501,   0.047690047983,   0.004028721216],
            [  0.198717948718,   0.096588404722,   0.116879269666,   0.085440260059,   0.044134035452,   0.003718322003],
            [  0.205128205128,   0.112268637273,   0.130550222877,   0.081973791014,   0.040425698740,   0.003404157578],
            [  0.211538461538,   0.128457296422,   0.144724636833,   0.078188974462,   0.036562117621,   0.003085798588],
            [  0.217948717949,   0.145141494617,   0.159394503555,   0.074025918588,   0.032541655857,   0.002762272509],
            [  0.224358974359,   0.162302590749,   0.174545065358,   0.069403016306,   0.028364630103,   0.002431859243],
            [  0.230769230769,   0.179914974313,   0.190153186177,   0.064204909271,   0.024034405194,   0.002091868500],
            [  0.237179487179,   0.197945080941,   0.206185717930,   0.058258696211,   0.019558363361,   0.001738372372],
            [  0.243589743590,   0.216350790867,   0.222598389925,   0.051282730750,   0.014949186047,   0.001365972282],
            [  0.250000000000,   0.235086858785,   0.239505835823,   0.043627694350,   0.010585160148,   0.001000000000],
        ], dtype=self._dtype_cpu)
       
        if self._struct.upper() == 'BCC':
            if self._npeaks == 1:
                data = BCC_0_1
            elif self._npeaks == 2:
                data = BCC_0_1_2
            elif self._npeaks == 3:
                data = BCC_0_1_2_3
            else:
                raise ValueError("npeaks must be 1-3 for BCC structure.")
        elif self._struct.upper() == 'FCC':
            if self._npeaks == 1:
                data = FCC_0_1
            elif self._npeaks == 2:
                data = FCC_0_1_2
            elif self._npeaks == 3:
                data = FCC_0_1_2_3
            else:
                raise ValueError("npeaks must be 1-3 for FCC structure.")
        elif self._struct.upper() == 'SC':
            if self._npeaks == 1:
                data = SC_0_1
            elif self._npeaks == 2:
                data = SC_0_1_2
            elif self._npeaks == 3:
                data = SC_0_1_2_3
            else:
                raise ValueError("npeaks must be 1-3 for SC structure.")
        else:
            raise ValueError(f"Unsupported structure {self._struct}.")

        # Interpolate data
        sigma_data = data[:, 0]
        nliq_data  = data[:, 1]
        nsol_data  = data[:, 2]
        ampl_data  = data[:, 3:]

        # Get sigma bounds from data
        sigma_min, sigma_max = sigma_data[0], sigma_data[-1]
        
        # Check bounds and warn/error if outside range
        if self._sigma < sigma_min or self._sigma > sigma_max:
            raise ValueError(f"sigma={self._sigma} outside valid range [{sigma_min:.3f}, {sigma_max:.3f}]")

        # Interpolate liquid and solid densities
        nliq = spi.interp1d(sigma_data, nliq_data, kind=interp_kind, fill_value="extrapolate")(self._sigma)
        nsol = spi.interp1d(sigma_data, nsol_data, kind=interp_kind, fill_value="extrapolate")(self._sigma)
        nLnS = np.array([nliq, nsol])

        # Interpolate amplitudes for each mode
        ampl = np.zeros(self._npeaks)
        for i in range(self._npeaks):
            ampl[i] = spi.interp1d(sigma_data, ampl_data[:, i], kind=interp_kind, fill_value="extrapolate")(self._sigma)

        return ampl, nLnS

# =====================================================================================
