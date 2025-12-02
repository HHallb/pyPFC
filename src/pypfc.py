# Copyright (C) 2025 Håkan Hallberg
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE file for full license text

import numpy as np
import torch
import time
import os
from pypfc_io import setup_io
from typing import Union, List, Optional, Tuple, Dict, Any
class setup_simulation(setup_io):
    """
    This is the primary class for conducting PFC simulations, providing complete
    functionality for time evolution, energy evaluation, and phase field analysis.
    It combines all inherited capabilities from the class hierarchy:
    
    - Grid setup and domain discretization (pypfc_grid)
    - Mathematical operations and device management (pypfc_base)  
    - Density field generation and crystal structures (pypfc_pre)
    - File I/O operations and data export (pypfc_io)
    
    The class uses a configuration-driven approach where all simulation parameters
    are managed through a DEFAULTS dictionary. User configurations merge with defaults,
    to provide parameter handling with fallback values.
        
    Notes
    -----
    All grid divisions (ndiv) must be even numbers for FFT compatibility.
    The class automatically validates configuration parameters and provides
    informative error messages for invalid inputs.
    """

    DEFAULTS = {
        'dtime':                    1.0e-4,
        'struct':                   'FCC',
        'alat':                     1.0,
        'sigma':                    0.0,
        'npeaks':                   2,
        'alpha':                    [1, 1, 1],
        'C20_amplitude':            0.0,
        'C20_alpha':                1.0,
        'pf_gauss_var':             1.0,
        'normalize_pf':             True,
        'update_scheme':            '1st_order',
        'update_scheme_params':     [1.0, 1.0, 1.0, None, None, None],
        'device_type':              'gpu',
        'device_number':            0,
        'dtype_cpu':                np.double,
        'dtype_gpu':                torch.float64,
        'verbose':                  False,
        'evaluate_phase_field':     False,
        'density_interp_order':     2,
        'density_threshold':        0.0,
        'density_merge_distance':   0.1,
        'pf_iso_level':             0.5,
        'torch_threads':            os.cpu_count(),
        'torch_threads_interop':    os.cpu_count(),
    }

    def __init__(self, domain_size: Union[List[float], np.ndarray], ndiv: Optional[Union[List[int], np.ndarray]] = None, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize PFC simulation with domain parameters and configuration.
        
        Sets up the complete simulation environment including grid discretization,
        device configuration, and all numerical parameters needed for PFC evolution.
        
        Parameters
        ----------
        domain_size : array_like of float, shape (3,)
            Physical size of simulation domain [Lx, Ly, Lz] in lattice parameter units.
            This defines the spatial extent of the simulation box.
        ndiv : array_like of int, shape (3,), optional
            Number of grid divisions [nx, ny, nz]. All values must be even numbers
            for FFT compatibility. If not provided, automatically calculated as
            `domain_size / alat * 8` (8 points per lattice spacing).
        config : dict, optional
            Configuration parameters as key-value pairs.
            See the [pyPFC overview](core.md) for a complete list of the configuration parameters.
        **kwargs : dict, optional
            Individual configuration parameters passed as keyword arguments.
            These will override any corresponding values in the config dictionary.
            
        Raises
        ------
        ValueError
            If any element in `ndiv` is not an even number.
            
        Examples
        --------
        >>> # Basic simulation setup
        >>> sim = setup_simulation([10.0, 10.0, 10.0])
        
        >>> # Custom grid and parameters using config dictionary
        >>> config = {'dtime': 5e-5, 'struct': 'BCC', 'verbose': True}
        >>> sim = setup_simulation([8.0, 8.0, 8.0], ndiv=[64, 64, 64], config=config)
        
        >>> # Using individual keyword arguments
        >>> sim = setup_simulation([8.0, 8.0, 8.0], pf_iso_level=0.5, dtime=1e-4, verbose=True)
        
        >>> # Mixing config dictionary and keyword arguments (kwargs take precedence)
        >>> sim = setup_simulation([8.0, 8.0, 8.0], config=config, pf_iso_level=0.3)
        """

        # Merge user parameters with defaults, but only use keys present in DEFAULTS
        # ==========================================================================
        cfg = dict(self.DEFAULTS)
        ignored = set()
        
        # First apply config dictionary if provided
        if config is not None:
            filtered_config = {k: v for k, v in config.items() if k in self.DEFAULTS}
            cfg.update(filtered_config)
            ignored.update(set(config.keys()) - set(self.DEFAULTS.keys()))
        
        # Then apply individual keyword arguments (these take precedence over config)
        if kwargs:
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in self.DEFAULTS}
            cfg.update(filtered_kwargs)
            ignored.update(set(kwargs.keys()) - set(self.DEFAULTS.keys()))
        if ignored:
            print(f"Ignored config keys: {ignored}")

        # Ensure domain_size is a numpy array
        # ===================================
        domain_size = np.array(domain_size, dtype=cfg['dtype_cpu'])

        # Ensure ndiv is a numpy array
        # ============================
        if ndiv is not None:
            ndiv = np.array(ndiv, dtype=int)
        else:
            ndiv = np.array(domain_size) / cfg['alat'] * 8 # Default to 8 points per lattice spacing
            ndiv = ndiv.astype(int)

        # Check that all ndiv values are even
        # ===================================
        if not np.all(ndiv % 2 == 0):
            raise ValueError(f"All values in ndiv must be even, but got ndiv={ndiv}")
            
        # Initiate the inherited class
        # ============================
        super().__init__(domain_size, ndiv, config=cfg)

        # Handle input arguments
        # ======================
        self._dtime                = cfg['dtime']
        self._update_scheme        = cfg['update_scheme']
        self._update_scheme_params = cfg['update_scheme_params']
        self._alat                 = cfg['alat']
        self._alpha                = cfg['alpha']
        self._pf_gauss_var         = cfg['pf_gauss_var']
        self._normalize_pf         = cfg['normalize_pf']
        self._evaluate_phase_field = cfg['evaluate_phase_field']
        self._C20_amplitude        = cfg['C20_amplitude']
        self._C20_alpha            = cfg['C20_alpha']

        # Initiate additional class variables
        # ===================================
        self._using_setup_file = False
        self._setup_file_path  = None
        self._use_H2           = False

        # Allocate torch tensors and ensure that they are contiguous in memory
        # ====================================================================
        if self._verbose: tstart = time.time()
        self._tmp_d    = torch.zeros((self._nx, self._ny, self._nz),      dtype=self._dtype_gpu, device=self._device)
        self._den_d    = torch.zeros((self._nx, self._ny, self._nz),      dtype=self._dtype_gpu, device=self._device)
        self._f_tmp_d  = torch.zeros((self._nx, self._ny, self._nz_half), dtype=self._ctype_gpu, device=self._device)
        self._f_den_d  = torch.zeros((self._nx, self._ny, self._nz_half), dtype=self._ctype_gpu, device=self._device)
        self._f_den2_d = torch.zeros((self._nx, self._ny, self._nz_half), dtype=self._ctype_gpu, device=self._device)
        self._f_den3_d = torch.zeros((self._nx, self._ny, self._nz_half), dtype=self._ctype_gpu, device=self._device)

        self._tmp_d    = self._tmp_d.contiguous()
        self._den_d    = self._den_d.contiguous()
        self._f_tmp_d  = self._f_tmp_d.contiguous()
        self._f_den_d  = self._f_den_d.contiguous()
        self._f_den2_d = self._f_den2_d.contiguous()
        self._f_den3_d = self._f_den3_d.contiguous()

        if self._update_scheme=='2nd_order':
            self._f_denOld_d = torch.zeros((self._nx,self._ny,self._nz_half), dtype=self._ctype_gpu, device=self._device)
            self._f_denOld_d = self._f_denOld_d.contiguous()
        else:
            self._f_denOld_d = None

        if self._verbose:
            tend = time.time()
            print(f'Time to allocate tensors: {tend-tstart:.3f} s')

        # Get two-point pair correlation function
        # =======================================
        if self._verbose: tstart = time.time()
        self._C2_d = self.evaluate_C2_d()
        if self._verbose:
            tend = time.time()
            print(f'Time to construct C2_d: {tend-tstart:.3f} s')

        # Set phase field kernels, if needed
        # ==================================
        if self._evaluate_phase_field:
            self.set_phase_field_kernel()
            self.set_phase_field_smoothing_kernel(pf_gauss_var=self._pf_gauss_var)

        # Define scheme for PFC density field time integration
        # ====================================================
        if self._verbose: tstart = time.time()
        self.update_density = self.get_update_scheme()
        if self._verbose:
            tend = time.time()
            print(f'Time to construct the time integration scheme ({self._update_scheme}): {tend-tstart:.3f} s')

# =====================================================================================

    def set_alat(self, alat: float) -> None:
        """
        Set the lattice parameter.
        
        Parameters
        ----------
        alat : float
            Lattice parameter.
        """
        self._alat = alat

# =====================================================================================

    def get_alat(self) -> float:
        """
        Get the current lattice parameter.
        
        Returns
        -------
        float
            Current lattice parameter value.
        """
        return self._alat

# =====================================================================================

    def set_dtime(self, dtime: float) -> None:
        """
        Set the time step.
        
        Parameters
        ----------
        dtime : float
            Time step size. Must be positive and small enough for stability.
        """
        self._dtime = dtime

# =====================================================================================

    def get_dtime(self) -> float:
        """
        Get the current time step size.
        
        Returns
        -------
        float
            Current time step size for integration.
        """
        return self._dtime

# =====================================================================================

    def set_alpha(self, alpha: Union[List[float], np.ndarray]) -> None:
        """
        Set the alpha parameters, controlling the widths of the
        Gaussian peaks of the pair correlation function.
        
        Parameters
        ----------
        alpha : float
            Pair correlation peak widths.
        """
        self._alpha = alpha

# =====================================================================================

    def get_alpha(self) -> np.ndarray:
        """
        Get the current alpha parameters.
        
        Returns
        -------
        float
            Current alpha parameter values.
        """
        return self._alpha

# =====================================================================================

    def set_C2_d(self, C2_d: torch.Tensor) -> None:
        """
        Set the two-point correlation function in Fourier space.
        
        Parameters
        ----------
        C2_d : torch.Tensor
            Two-point correlation function tensor in Fourier space.
            Must match the grid dimensions and be on the correct device.
        """
        self._C2_d = C2_d

# =====================================================================================

    def get_C2_d(self) -> torch.Tensor:
        """
        Get the current two-point correlation function.
        
        Returns
        -------
        torch.Tensor
            Two-point correlation function in Fourier space.
        """
        return self._C2_d

# =====================================================================================

    def set_H2(self, H0: float, Rot: np.ndarray) -> None:
        """
        Set the directional correlation kernel for extended PFC models.
        
        Configures the H2 kernel used in extended phase field crystal
        models to introduce directional correlations.
        
        Parameters
        ----------
        H0 : float
            Amplitude of the directional correlation kernel.
        Rot : array_like or None
            Rotation matrix for orienting the kernel. If None,
            uses identity orientation.
        """
        self._f_H_d  = self.evaluate_directional_correlation_kernel(H0, Rot)
        self._f_H_d  = self._f_H_d.contiguous()
        self._use_H2 = True
        self.update_density = self.get_update_scheme()  # Recompute the update scheme to include H2

# =====================================================================================

    def set_update_scheme(self, update_scheme: str) -> None:
        """
        Set the time integration scheme for density evolution.
        
        Parameters
        ----------
        update_scheme : str
            Time integration method. Options: '1st_order', '2nd_order', 
            'exponential'.
        """
        self._update_scheme = update_scheme
        self.update_density = self.get_update_scheme()

# =====================================================================================

    def set_update_scheme_params(self, params: Union[List[float], np.ndarray]) -> None:
        """
        Set parameters for the time integration scheme.
        
        Parameters
        ----------
        params : array_like
            Parameters specific to the chosen integration scheme.
            Format depends on the selected scheme type.
        """
        self._update_scheme_params = params
        self.update_density = self.get_update_scheme()  

# =====================================================================================

    def get_update_scheme_params(self) -> np.ndarray:
        """
        Get the current integration scheme parameters.
        
        Returns
        -------
        array_like
            Current parameters for the time integration scheme.
        """
        return self._update_scheme_params

# =====================================================================================

    def get_energy(self) -> np.ndarray:
        """
        Get the PFC energy field and its mean value.
        
        Computes the local energy density field and its spatial average using
        the current density field configuration.
        
        Returns
        -------
        ene : ndarray of float, shape (nx,ny,nz)
            Local energy density field on CPU.
        mean_ene : float
            Spatially averaged energy density.
        """
        ene, mean_ene = self.evaluate_energy()
        return ene, mean_ene

# =====================================================================================

    def get_density(self) -> np.ndarray:
        """
        Get the PFC density field and its mean value.
        
        Returns the current density field and its spatial average, transferring
        data from GPU to CPU if necessary.
        
        Returns
        -------
        den : ndarray of float, shape (nx,ny,nz)
            Density field on CPU.
        mean_den : float
            Spatially averaged density.
        """
        den      = self._den_d.detach().cpu().numpy()
        mean_den = torch.mean(self._den_d).detach().cpu().numpy()
        return den, mean_den

# =====================================================================================

    def set_density(self, density: np.ndarray) -> None:
        """
        Set the PFC density field.
        
        Updates the density field and automatically computes its Fourier transform
        for subsequent calculations.
        
        Parameters
        ----------
        density : ndarray of float, shape (nx,ny,nz)
            New density field configuration.
        """
        self._den_d   = torch.from_numpy(density).to(self._device)
        self._f_den_d = torch.fft.rfftn(self._den_d).to(self._f_den_d.dtype)

# =====================================================================================

    def set_phase_field_kernel(self, H0: float = 1.0, Rot: Optional[Union[np.ndarray, List[np.ndarray]]] = None) -> None:
        """
        Set phase field kernel for directional analysis.
        
        Configures the correlation kernel used for phase field evaluation,
        allowing for directional filtering or isotropic analysis.
        
        Parameters
        ----------
        H0 : float, optional
            Kernel amplitude.
        Rot : ndarray of float, shape (3,3), optional
            Rotation matrix for directional kernel. If `None`, uses isotropic
            two-point correlation function.
        """
        if Rot is None:
            self._f_pf_kernel_d = self._C2_d
            self._f_pf_kernel_d = self._f_pf_kernel_d.contiguous()
        else:
            self._f_pf_kernel_d = self.evaluate_directional_correlation_kernel(H0, Rot)
            self._f_pf_kernel_d = self._f_pf_kernel_d.contiguous()

# =====================================================================================

    def set_phase_field_smoothing_kernel(self, pf_gauss_var: Optional[float] = None) -> None:
        """
        Set phase field smoothing kernel.
        
        Configures Gaussian smoothing parameters for phase field calculations.
        
        Parameters
        ----------
        pf_gauss_var : float, optional
            Gaussian variance for smoothing kernel. If `None`, uses current value.
        """
        self._pf_gauss_var = pf_gauss_var
        denom1 = 2 * self._pf_gauss_var**2
        denom2 = self._pf_gauss_var * torch.sqrt(torch.tensor(2.0, device=self._device, dtype=self._dtype_gpu))
        self._f_pf_smoothing_kernel_d = torch.exp(-self._k2_d / denom1) / denom2
        self._f_pf_smoothing_kernel_d = self._f_pf_smoothing_kernel_d.contiguous()

# =====================================================================================

    def cleanup(self) -> None:
        """
        Clean up allocated tensors and free device memory.
        
        Explicitly deletes PyTorch tensors to free GPU/CPU memory. This is
        particularly important for GPU simulations to prevent memory leaks
        and ensure proper resource management.
        
        Notes
        -----
        This method should be called at the end of simulations, especially
        when running multiple simulations sequentially or when GPU memory
        is limited. The method automatically detects which tensors exist
        based on the update scheme and cleans up accordingly.
        """
        del self._tmp_d, self._C2_d, self._f_den_d, self._f_den2_d, self._f_den3_d
        del self._den_d, self._f_tmp_d, self._k2_d
        del self._ampl_d, self._nlns_d

        if self._update_scheme=='1st_order':
            del self._f_Lterm_d

        if self._update_scheme=='2nd_order':
            del self._f_denOld_d
            del self._f_Lterm0_d
            del self._f_Lterm1_d
            del self._f_Lterm2_d
            del self._f_Lterm3_d

        if self._update_scheme=='exponential':
            del self._f_Lterm0_d
            del self._f_Lterm1_d

        if self._evaluate_phase_field:
            del self._f_pf_kernel_d, self._f_pf_smoothing_kernel_d

        if self._use_H2:
            del self._f_H_d

        torch.cuda.empty_cache()  # Frees up unused GPU memory

        # Write finishing time stamp to the setup file, if it is active
        # =============================================================
        if self._using_setup_file:
            self.append_to_info_file(f' ', output_path=self._setup_file_path)
            self.append_to_info_file(f'======================================================', output_path=self._setup_file_path)
            self.append_to_info_file(f'{self.get_time_stamp()}', output_path=self._setup_file_path)
            self.append_to_info_file(f'======================================================', output_path=self._setup_file_path)

# =====================================================================================

    def get_update_scheme(self) -> str:
        """
        Establish the PFC time integration scheme and return method handle.
        
        Configures the selected time integration scheme and returns a function
        handle to the appropriate update method. This method sets up scheme-specific
        parameters and precomputed terms for efficient time stepping.
        
        Returns
        -------
        update_density : callable
            Function handle to the selected time integration method.
            
        Raises
        ------
        ValueError
            If alpha, beta, gamma parameters are not provided for '2nd_order' scheme.
        ValueError
            If f_denOld_d is not provided for '2nd_order' scheme.
        ValueError
            If the specified update_scheme is not supported.
            
        Notes
        -----
        The method automatically precomputes linear terms in Fourier space
        to optimize performance during time stepping. Different schemes require
        different numbers of precomputed terms and have varying stability
        properties and computational costs.
        """

        # Scheme parameters
        # =================
        g1, _, _, alpha, beta, gamma = self._update_scheme_params
        dt = self._dtime

        if self._use_H2 and self._verbose:
            print("Using an orientation-dependent kernel H2 in the time integration scheme.")

        # Pre-compute contants and define the update function
        # ===================================================
        if self._update_scheme == '1st_order':
            if self._use_H2:
                self._f_Lterm_d = -self._k2_d.mul(g1 - self._C2_d - self._f_H_d).contiguous()
            else:
                self._f_Lterm_d = -self._k2_d.mul(g1 - self._C2_d).contiguous()
            self.update_density = self._update_density_1
        elif self._update_scheme == '2nd_order':
            if self._update_scheme_params[3:].any() is None or len(self._update_scheme_params) != 6:
                raise ValueError("alpha, beta, gamma parameters must be provided for the '2nd_order' update_scheme.")
            if self._f_denOld_d is None:
                raise ValueError("f_denOld_d must be provided for '2nd_order' update_scheme.")
            self._f_Lterm0_d = 4 * gamma
            self._f_Lterm1_d = beta * dt - 2 * gamma
            self._f_Lterm2_d = 2 * (dt ** 2) * alpha ** 2 * self._k2_d.contiguous()
            if self._use_H2:
                self._f_Lterm3_d = (2 * gamma + beta * self._dtime +
                                    2 * (dt ** 2) * (alpha ** 2) *
                                    self._k2_d.mul(g1 - self._C2_d - self._f_H_d).contiguous())
            else:
                self._f_Lterm3_d = (2 * gamma + beta * self._dtime +
                                    2 * (dt ** 2) * (alpha ** 2) *
                                    self._k2_d.mul(g1 - self._C2_d).contiguous())
            self.update_density = self._update_density_2
        elif self._update_scheme == 'exponential':
            if self._use_H2:
                self._f_Lterm0_d = g1 - self._C2_d - self._f_H_d
            else:
                self._f_Lterm0_d = g1 - self._C2_d
            self._f_Lterm0_d = torch.where(self._f_Lterm0_d == 0,
                                        torch.tensor(1e-12, device=self._device, dtype=self._dtype_torch),
                                        self._f_Lterm0_d).contiguous()
            self._f_Lterm1_d = torch.exp(-self._k2_d.mul(self._f_Lterm0_d) * dt).contiguous()
            self.update_density = self._update_density_exp
        else:
            raise ValueError(f"Unknown update_scheme: {self._update_scheme}")

        return self.update_density
    
# =====================================================================================

    def do_step_update(self) -> None:
        """
        Update the PFC density field using the selected time integration scheme.
        
        Performs one time step of the PFC evolution equation using the time integration
        method specified by `_update_scheme`. The method operates in Fourier space for
        computational efficiency and automatically handles the FFT/iFFT transformations.
        
        Returns
        -------
        f_den_d : torch.Tensor
            Updated density field in Fourier space. The real-space density field
            is automatically updated in `self._den_d` via inverse FFT.
            
        Raises
        ------
        ValueError
            If the specified update_scheme is not supported.
            
        Notes
        -----
        This method uses precomputed linear terms for efficiency. The appropriate
        update method is selected based on `self._update_scheme` and called with
        the corresponding precomputed Fourier-space operators.
        """

        # Call the selected update method with precomputed constants
        if self._update_scheme == '1st_order':
            self._f_den_d = self.update_density(self._f_Lterm_d)
        elif self._update_scheme == '2nd_order':
            self._f_den_d = self.update_density(self._f_Lterm0_d, self._f_Lterm1_d, self._f_Lterm2_d, self._f_Lterm3_d)
        elif self._update_scheme == 'exponential':
            self._f_den_d = self.update_density(self._f_Lterm0_d, self._f_Lterm1_d)
        else:
            raise ValueError(f"Unknown update_scheme: {self._update_scheme}")

        # Reverse FFT of the updated density field
        torch.fft.irfftn(self._f_den_d, s=self._den_d.shape, out=self._den_d)

# =====================================================================================

    def _update_density_1(self, f_Lterm_d: torch.Tensor) -> None:
        """
        First-order time integration scheme.
        
        Implements the first-order method for PFC density evolution.
        
        Parameters
        ----------
        f_Lterm_d : torch.Tensor
            Precomputed linear operator in Fourier space.
            
        Returns
        -------
        f_den_new : torch.Tensor
            Updated density field in Fourier space.
            
        Notes
        -----
        This is a private method used internally by `do_step_update()`.
        The scheme has first-order accuracy in time but may require
        smaller time steps for stability.
        """

        # Parameters
        _, g2, g3, *_ = self._update_scheme_params

        # Forward FFT of the nonlinear density terms (in-place)
        torch.fft.rfftn(self._den_d.pow(2), out=self._f_den2_d)
        torch.fft.rfftn(self._den_d.pow(3), out=self._f_den3_d)

        # Update the density field in-place
        self._f_den_d.sub_(self._dtime * self._k2_d * (-self._f_den2_d * g2 / 2 + self._f_den3_d * g3 / 3))
        self._f_den_d.div_(1 - self._dtime * f_Lterm_d)

        return self._f_den_d
    
# =====================================================================================

    def _update_density_2(self, f_Lterm0_d: torch.Tensor, f_Lterm1_d: torch.Tensor, f_Lterm2_d: torch.Tensor, f_Lterm3_d: torch.Tensor) -> None:
        """
        Second-order time integration scheme.
        
        Implements a second-order method for improved accuracy
        and stability compared to first-order methods.
        
        Parameters
        ----------
        f_Lterm0_d : torch.Tensor
            First precomputed linear operator term.
        f_Lterm1_d : torch.Tensor  
            Second precomputed linear operator term.
        f_Lterm2_d : torch.Tensor
            Third precomputed linear operator term.
        f_Lterm3_d : torch.Tensor
            Fourth precomputed linear operator term.
            
        Returns
        -------
        f_den_new : torch.Tensor
            Updated density field in Fourier space.
            
        Notes
        -----
        This is a private method used internally by `do_step_update()`.
        The scheme provides second-order accuracy in time with improved
        stability properties at moderate computational cost.
        """
        # Parameters
        _, g2, g3, *_ = self._update_scheme_params

        # Maintain a copy of the old density field in Fourier space
        self._f_denOld_d.copy_(self._f_den_d)

        # Forward FFT of the nonlinear density terms (in-place)
        torch.fft.rfftn(self._den_d.pow(2), out=self._f_den2_d)
        torch.fft.rfftn(self._den_d.pow(3), out=self._f_den3_d)

        # Compute nonlinear term in-place: self._f_tmp_d = f_Lterm2_d * (self._f_den2_d/2 - self._f_den3_d/3)
        self._f_tmp_d.copy_(self._f_den2_d.div(2/g2).sub(self._f_den3_d.div(3/g3)).mul(f_Lterm2_d))

        # Update the density field in-place
        self._f_den_d.mul_(f_Lterm0_d)
        self._f_den_d.add_(f_Lterm1_d * self._f_denOld_d)
        self._f_den_d.add_(self._f_tmp_d)
        self._f_den_d.div_(f_Lterm3_d)

        return self._f_den_d
    
# =====================================================================================

    def _update_density_exp(self, f_Lterm0_d: torch.Tensor, f_Lterm1_d: torch.Tensor) -> None:
        """
        Exponential time integration scheme.
        
        Implements an exponential time integrator that provides good
        stability properties in phase field modeling.
        
        Parameters
        ----------
        f_Lterm0_d : torch.Tensor
            First precomputed exponential operator term.
        f_Lterm1_d : torch.Tensor
            Second precomputed exponential operator term.
            
        Returns
        -------
        f_den_new : torch.Tensor
            Updated density field in Fourier space.
            
        Notes
        -----
        This is a private method used internally by `do_step_update()`.
        """

        # Parameters
        _, g2, g3, *_ = self._update_scheme_params

        # Forward FFT of the nonlinear density terms
        torch.fft.rfftn(self._den_d.pow(2), out=self._f_den2_d)
        torch.fft.rfftn(self._den_d.pow(3), out=self._f_den3_d)

        # Compute nonlinear term out-of-place
        self._f_tmp_d.copy_((-self._f_den2_d * g2 / 2) + (self._f_den3_d * g3 / 3))

        # Update self._f_den_d in-place:
        self._f_den_d.mul_(f_Lterm1_d)
        self._f_tmp_d.mul_(f_Lterm1_d - 1)
        self._f_tmp_d.div_(f_Lterm0_d)
        self._f_den_d.add_(self._f_tmp_d)

        return self._f_den_d

# =====================================================================================

    def evaluate_energy(self) -> float:
        """
        Evaluate the PFC energy.

        Computes the total free energy of the system using the phase field
        crystal energy functional.

        Returns
        -------
        ene : torch.Tensor
            Energy field with shape [nx, ny, nz].
        eneAv : float
            Average free energy density.
            
        Notes
        -----
        Energy is computed in Fourier space for efficiency and transformed
        back to real space for local energy density visualization.
        """

        if self._verbose: tstart = time.time()

        # Grid
        nx,ny,nz = self._ndiv

        # Evaluate convolution in Fourier space and retrieve the result back to real space
        self._tmp_d = torch.fft.irfftn(self._f_den_d*self._C2_d, s=self._tmp_d.shape)
        
        # Evaluate free energy (on device)
        self._tmp_d = self._den_d.pow(2)/2 - self._den_d.pow(3)/6 + self._den_d.pow(4)/12 - 0.5*self._den_d.mul(self._tmp_d)

        # Evaluate the average free energy
        eneAv = torch.sum(self._tmp_d) / (nx * ny * nz)

        # Copy the resulting energy back to host
        ene = self._tmp_d.detach().cpu().numpy()

        if self._verbose:
            tend = time.time()
            print(f'Time to evaluate energy: {tend-tstart:.3f} s')

        return ene, eneAv.item() # .item() converts eneAv to a Python scalar

# =====================================================================================

    def get_phase_field(self) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Evaluate phase field using wavelet filtering.
        
        Computes the phase field by applying wavelet convolution followed by
        Gaussian smoothing. The phase field identifies crystalline regions
        and their orientations within the PFC density field.
        
        Returns
        -------
        pf : ndarray of float, shape (nx,ny,nz) or list of such arrays
            Phase field(s) on CPU. Returns a single array for isotropic kernels
            or a list of arrays for directional analysis with multiple kernels.
            
        Notes
        -----
        The method automatically handles both single and multiple wavelet kernels
        for comprehensive grain orientation analysis.
        """

        if self._verbose: tstart = time.time()

        def compute_pf(f_wavelet_d: torch.Tensor) -> np.ndarray:
        #def compute_pf(f_wavelet_d, k2_d, varGauss, f_den_d, normalizePF):
            # Perform the first convolution and retrieve the result to real space
            torch.fft.irfftn(self._f_den_d * f_wavelet_d, s=self._tmp_d.shape, out=self._tmp_d)

            # Only keep positive values
            self._tmp_d = torch.where(self._tmp_d < 0.0, torch.tensor(0.0, device=self._device), self._tmp_d)

            # Perform forward FFT
            torch.fft.rfftn(self._tmp_d, s=self._tmp_d.shape, out=self._f_tmp_d)

            # Perform the second convolution and retrieve the result to real space
            torch.fft.irfftn(self._f_tmp_d * self._f_pf_smoothing_kernel_d, s=self._tmp_d.shape, out=self._tmp_d)

            # Normalize the phase field to lie in the range [0, 1]
            if self._normalize_pf:
                pf_min = torch.min(self._tmp_d)
                pf_max = torch.max(self._tmp_d)
                self._tmp_d.sub_(pf_min)
                self._tmp_d.div_(pf_max - pf_min + 1.0e-15)  # Avoid division by zero

            return self._tmp_d.detach().cpu().numpy()

        # Check if f_wavelet_d is a list
        if isinstance(self._f_pf_kernel_d, list):
            # If it is a list, compute pf for each f_wavelet_d
            pf_list = [compute_pf(wavelet) for wavelet in self._f_pf_kernel_d]

            if self._verbose:
                 tend = time.time()
                 print(f'Time to evaluate phase field: {tend-tstart:.3f} s')

            return pf_list
        else:
            # If it is not a list, compute pf for the single f_wavelet_d
            pf = compute_pf(self._f_pf_kernel_d)

            if self._verbose:
                 tend = time.time()
                 print(f'Time to evaluate phase field: {tend-tstart:.3f} s')

            return pf
        
# =====================================================================================
