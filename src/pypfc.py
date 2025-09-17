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
import torch
import time
import os
from pypfc_io import setup_io
class setup_simulation(setup_io):

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
        'density_merge_distance':   None,
        'pf_iso_level':             0.5,
        'torch_threads':            os.cpu_count(),
        'torch_threads_interop':    os.cpu_count(),
    }

    def __init__(self, ndiv, ddiv, config=None):

        # Merge user parameters with defaults, but only use keys present in DEFAULTS
        # ==========================================================================
        cfg = dict(self.DEFAULTS)
        if config is not None:
            # Only update with keys that are in DEFAULTS
            filtered_config = {k: v for k, v in config.items() if k in self.DEFAULTS}
            cfg.update(filtered_config)
        # Warn about any keys in config that are not in DEFAULTS
        ignored = set(config.keys()) - set(self.DEFAULTS.keys())
        if ignored:
            print(f"Ignored config keys: {ignored}")

        # Initiate the inherited class
        # ============================
        subset_cfg = {k: cfg[k] for k in ['struct', 'sigma', 'npeaks', 'device_number',
                                          'device_type', 'dtype_cpu', 'dtype_gpu', 'verbose',
                                          'density_interp_order', 'density_threshold', 'density_merge_distance',
                                          'pf_iso_level', 'torch_threads', 'torch_threads_interop'] if k in cfg}
        super().__init__(ndiv, ddiv, config=subset_cfg)

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
            print(f'Time to construct time integration scheme ({self._update_scheme}): {tend-tstart:.3f} s')

# =====================================================================================

    def set_alat(self, alat):
        self._alat = alat

    def get_alat(self):
        return self._alat

    def set_dtime(self, dtime):
        self._dtime = dtime

    def get_dtime(self):
        return self._dtime

    def set_alpha(self, alpha):
        self._alpha = alpha

    def get_alpha(self):
        return self._alpha

    def set_C2_d(self, C2_d):
        self._C2_d = C2_d

    def get_C2_d(self):
        return self._C2_d

    def set_H2(self, H0, Rot):
        self._f_H_d  = self.evaluate_directional_correlation_kernel(H0, Rot)
        self._f_H_d  = self._f_H_d.contiguous()
        self._use_H2 = True
        self.update_density = self.get_update_scheme()  # Recompute the update scheme to include H2

    def set_update_scheme(self, update_scheme):
        self._update_scheme = update_scheme
        self.update_density = self.get_update_scheme()

    def get_update_scheme(self):
        return self._update_scheme
    
    def set_update_scheme_params(self, params):
        self._update_scheme_params = params
        self.update_density = self.get_update_scheme()  

    def get_update_scheme_params(self):
        return self._update_scheme_params

    def get_energy(self):
        ene, mean_ene = self.evaluate_energy()
        return ene, mean_ene

    def get_density(self):
        den      = self._den_d.detach().cpu().numpy()
        mean_den = torch.mean(self._den_d).detach().cpu().numpy()
        return den, mean_den

    def set_density(self, density):
        self._den_d   = torch.from_numpy(density).to(self._device)
        self._f_den_d = torch.fft.rfftn(self._den_d).to(self._f_den_d.dtype)  # Forward FFT of the density field

    def set_phase_field_kernel(self, H0=1.0, Rot=None):
        if Rot is None:
            self._f_pf_kernel_d = self._C2_d
            self._f_pf_kernel_d = self._f_pf_kernel_d.contiguous()
        else:
            self._f_pf_kernel_d = self.evaluate_directional_correlation_kernel(H0, Rot)
            self._f_pf_kernel_d = self._f_pf_kernel_d.contiguous()

    def set_phase_field_smoothing_kernel(self, pf_gauss_var=None):
        self._pf_gauss_var = pf_gauss_var
        denom1 = 2 * self._pf_gauss_var**2
        denom2 = self._pf_gauss_var * torch.sqrt(torch.tensor(2.0, device=self._device, dtype=self._dtype_gpu))
        self._f_pf_smoothing_kernel_d = torch.exp(-self._k2_d / denom1) / denom2
        self._f_pf_smoothing_kernel_d = self._f_pf_smoothing_kernel_d.contiguous()

# =====================================================================================

    def cleanup(self):
        '''
        PURPOSE
            Clean up variables.

        INPUT

        OUTPUT

        Last revision:
        H. Hallberg 2025-09-17
        '''

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

    def evaluate_C2_d(self):
        """
        PURPOSE
            Establish the two-point correlation function for a particular crystal structure.

        INPUT

        OUTPUT
            C2_d          Two-point pair correlation function [nx, ny, nz/2+1] (on the device)

        Last revision:
        H. Hallberg 2025-09-16
        """

        # Get reciprocal planes
        kpl, npl, denpl = self.evaluate_reciprocal_planes()

        # Convert to PyTorch tensors and move to device
        kpl_d   = torch.tensor(kpl,   dtype=self._k2_d.dtype, device=self._k2_d.device)
        denpl_d = torch.tensor(denpl, dtype=self._k2_d.dtype, device=self._k2_d.device)
        alpha_d = torch.tensor(self._alpha, dtype=self._k2_d.dtype, device=self._k2_d.device)
        npl_d   = torch.tensor(npl,   dtype=self._k2_d.dtype, device=self._k2_d.device)

        # Evaluate the exponential pre-factor (Debye-Waller-like)
        DWF_d = torch.exp(-(self._sigma**2) * (kpl_d**2) / (2 * denpl_d * npl_d))

        # Precompute quantities
        denom_d   = 2 * alpha_d**2
        k2_sqrt_d = torch.sqrt(self._k2_d)

        # # Reshape kpl, DWF, and alpha for broadcasting
        # kpl_d   = kpl_d.view(1, 1, 1, self._npeaks)
        # DWF_d   = DWF_d.view(1, 1, 1, self._npeaks)
        # denom_d = denom_d.view(1, 1, 1, self._npeaks)

        # # Compute the correlation function
        # C2testval_d = DWF_d * torch.exp(-(k2_sqrt_d.unsqueeze(-1) - kpl_d) ** 2 / denom_d)
        # C2_d = torch.max(C2testval_d, dim=-1).values
        # C2_d = C2_d.contiguous()

        # Zero-mode peak
        if self._C20_amplitude != 0.0:
            if self._C20_alpha < 0.0:
                raise ValueError("C20_alpha must be positive when C20_amplitude is non-zero.")
            zero_peak = self._C20_amplitude * torch.exp(-k2_sqrt_d ** 2 / self._C20_alpha)
        else:
            zero_peak = torch.zeros_like(k2_sqrt_d)

        # Use f_tmp_d as workspace (complex type)
        self._f_tmp_d.zero_()
        # Take real part for max operation
        self._f_tmp_d.real.copy_(zero_peak)

        # Compute the correlation function for all peaks
        for ipeak in range(self._npeaks):
            peak_val = DWF_d[ipeak] * torch.exp( -(k2_sqrt_d - kpl_d[ipeak]) ** 2 / denom_d[ipeak] )
            self._f_tmp_d.real = torch.maximum(self._f_tmp_d.real, peak_val)

        # Return the real part as the result
        C2_d = self._f_tmp_d.real.contiguous()

        return C2_d

# =====================================================================================

    def evaluate_reciprocal_planes(self):
        '''
        PURPOSE
            Establish reciprocal vectors/planes for a particular crystal structure.

        INPUT

        OUTPUT
            kPlane        Reciprocal lattice plane spacing (a.k.a. "d-spacing"). For cubic systems, the formulae
                            is:
                                    d = a / sqrt(h^2 + k^2 + l^2)

                            where a is the lattice parameter. The reciprocal spacing is

                                    kPlane = 2pi/d

                            Theorem: For any family of lattice planes separated by distance d, there are 
                                    reciprocal lattice vectors perpendicular to the planes, the shortest
                                    being 2pi/d.

            nPlane        Number of symmetrical planes of each family
            denPlane      Atomic density within a plane (i.e. "planar density")

        Last revision:
        H. Hallberg 2025-08-26
        '''

        kPlane   = np.zeros(self._npeaks, dtype=float)
        denPlane = np.zeros(self._npeaks, dtype=float)
        nPlane   = np.zeros(self._npeaks, dtype=int)

        # Define reciprocal vectors
        match self._struct.upper():
            case 'SC': #= SC in reciprocal space
                # {100}, {110}, {111}
                nvals = 3
                kpl   = (2*np.pi/self._alat) * np.array([1, np.sqrt(2), np.sqrt(3)], dtype=float)
                pl    = np.array([6, 12, 8], dtype=int)
                denpl = (1/self._alat**2) * np.array([1, 1/np.sqrt(2), 1/np.sqrt(3)], dtype=float)
            case 'BCC': # = FCC in reciprocal space
                # {110}, {200}       (...the next would be {211}, {220}, {310}, {222})
                nvals = 2
                kpl   = (2*np.pi/self._alat) * np.array([np.sqrt(2), 2], dtype=float)
                pl    = np.array([12, 6, 24], dtype=int)
                denpl = (1/self._alat**2) * np.array([2/np.sqrt(2), 1], dtype=float)
            case 'FCC': # = BCC in reciprocal space
                # {111}, {200}, {220}        (...the next would be {311}, {222})
                nvals = 3
                kpl   = (2*np.pi/self._alat) * np.array([np.sqrt(3), 2, np.sqrt(8)], dtype=float)
                pl    = np.array([8, 6, 12], dtype=int)
                denpl = (1/self._alat**2) * np.array([4/np.sqrt(3), 2, 4/np.sqrt(2)], dtype=float)
            case 'DC': # Diamond Cubic (3D)
                # {111}, {220}, {311}         (...the next would be {400}, {331}, {422}, {511})
                nvals = 3
                kpl   = (2*np.pi/self._alat) * np.array([np.sqrt(3), np.sqrt(8), np.sqrt(11)], dtype=float)
                pl    = np.array([8, 12, 24], dtype=int)                                                   
                denpl = (1/self._alat**2) * np.array([4/np.sqrt(3), 4/np.sqrt(2), 1.385641467389298], dtype=float)
            case _:
                raise ValueError(f'Unsupported crystal structure: struct={self._struct.upper()}')

        # Retrieve output data
        if nvals>=self._npeaks:
            kPlane   = kpl[0:self._npeaks]
            nPlane   = pl[0:self._npeaks]
            denPlane = denpl[0:self._npeaks]
        else:
            raise ValueError(f'Not enough peaks defined, npeaks={self._npeaks}')

        return kPlane, nPlane, denPlane

# =====================================================================================

    def get_update_scheme(self):

        '''
        PURPOSE
            Establish the PFC time integration scheme.

        INPUT

        OUTPUT
            update_density     Function handle to the selected time integration scheme

        Last revision:
        H. Hallberg 2025-09-16
        '''

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
            self.update_density = self.update_density_1
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
            self.update_density = self.update_density_2
        elif self._update_scheme == 'exponential':
            if self._use_H2:
                self._f_Lterm0_d = g1 - self._C2_d - self._f_H_d
            else:
                self._f_Lterm0_d = g1 - self._C2_d
            self._f_Lterm0_d = torch.where(self._f_Lterm0_d == 0,
                                        torch.tensor(1e-12, device=self._device, dtype=self._dtype_torch),
                                        self._f_Lterm0_d).contiguous()
            self._f_Lterm1_d = torch.exp(-self._k2_d.mul(self._f_Lterm0_d) * dt).contiguous()
            self.update_density = self.update_density_exp
        else:
            raise ValueError(f"Unknown update_scheme: {self._update_scheme}")

        return self.update_density
    
# =====================================================================================

    def do_step_update(self):
        '''
        PURPOSE
            Update the (X)PFC density field using the time integration scheme defined by
            set_update_scheme.

        INPUT

        OUTPUT
            f_den_d     Updated density field in the frequency domain

        Last revision:
        H. Hallberg 2025-09-16
        '''

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

    def update_density_1(self, f_Lterm_d):
        '''
        PURPOSE
            Update the (X)PFC density field.

        INPUT
            dtime       Time increment
            den_d       Previous density field
            f_Lterm_d   Constant linear operator in the updating scheme
            k2_d        Squared wave vectors
            f_den_d     Density field in Fourier space
            f_den2_d    Temporary tensor
            f_den3_d    Temporary tensor

        OUTPUT
            f_den_d     Updated density field in the frequency domain

        Last revision:
        H. Hallberg 2025-09-16
        '''

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

    def update_density_2(self, f_Lterm0_d, f_Lterm1_d, f_Lterm2_d, f_Lterm3_d):
        '''
        PURPOSE
            Update the (X)PFC density field to step n+1.
            Time integration considering the second derivative w.r.t. time is used.

        INPUT
            den_d       Density field in step n
            dtime       Time increment
            f_Lterm0_d  Constant operator in the updating scheme
            f_Lterm1_d  Constant operator in the updating scheme
            f_Lterm2_d  Constant operator in the updating scheme
            f_Lterm3_d  Constant operator in the updating scheme
            f_den_d     Density field in step n in Fourier space
            f_denOld_d  Density field in step n-1 in Fourier space
            f_den2_d    Temporary tensor
            f_den3_d    Temporary tensor

        OUTPUT
            f_den_d     Updated density field in step n+1 in the frequency domain

        Last revision:
        H. Hallberg 2025-09-16
        '''
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

    def update_density_exp(self, f_Lterm0_d, f_Lterm1_d):
        '''
        PURPOSE
            Update the (X)PFC density field using the exponential time integration scheme,
            using only class attributes and in-place operations for memory efficiency.

        INPUT
            None (uses self._den_d, self._f_den_d, etc.)

        OUTPUT
            Updates self._f_den_d in-place

        Last revision:
        H. Hallberg 2025-09-16
        '''

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

    def evaluate_energy(self):
        '''
        PURPOSE
            Evaluate the free energy for the 3D XPFC model.

        INPUT
            den_d           Density field (on the device), [nx x ny x nz]
            f_den_d         Density field in the frequency domain (on the device), [nx x ny x nz/2+1]
            C2_d            Pair correlation function in the frequency domain (on the device), [nx x ny x nz/2+1]
            tmp_d           Temporary array (on the device), [nx x ny x nz]
    
        OUTPUT
            ene             Energy field,  [nx x ny x nz]
            eneAv           Average free energy

        Last revision:
        H. Hallberg 2025-03-01
        '''

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

    def get_phase_field(self):
        """
        PURPOSE
            Evaluate the phase field (orientation field) using a wavelet transform.
            The phase field is calculated as
                pf = (density_field*wavelet)*smoothing_kernel
            where * denotes a convolution.

        INPUT
            f_pfwavelet_d   Wavelet kernel, defined in Fourier space, [nx, ny, nz/2+1] or list of such kernels
            k2_d            Sum of squared wave vectors, [nx, ny, nz]
            varGauss        Variance (sigma) of the Gaussian smoothing kernel
            den_d           Density field in real space (on the device), [nx, ny, nz]
            f_den_d         Density field in Fourier space (on the device), [nx, ny, nz/2+1]
            normalizePF     Normalize the phase field or not

        OUTPUT
            pf              Phase field, [nx, ny, nz] or list of such fields

        Last revision:
        H. Hallberg 2025-08-28
        """

        if self._verbose: tstart = time.time()

        def compute_pf(f_wavelet_d):
        #def compute_pf(f_wavelet_d, k2_d, varGauss, f_den_d, normalizePF):
            # Perform the first convolution and retrieve the result to real space
            #self._tmp_d = torch.fft.irfftn(f_den_d * f_wavelet_d)
            torch.fft.irfftn(self._f_den_d * f_wavelet_d, s=self._tmp_d.shape, out=self._tmp_d)

            # Only keep positive values
            self._tmp_d = torch.where(self._tmp_d < 0.0, torch.tensor(0.0, device=self._device), self._tmp_d)

            # Perform forward FFT
            #self._f_tmp_d = torch.fft.rfftn(self._tmp_d)
            torch.fft.rfftn(self._tmp_d, s=self._tmp_d.shape, out=self._f_tmp_d)

            # Evaluate the Gaussian smoothing kernel
            #denom1 = 2 * varGauss**2
            ##denom2 = varGauss * torch.sqrt(torch.tensor(2.0, device=self._f_tmp_d.device))
            ##f_GaussKern_d = torch.exp(-k2_d / denom1) / denom2
            #denom2 = varGauss * torch.sqrt(torch.tensor(2.0, device=self._device, dtype=self._dtype_gpu))
            #f_GaussKern_d = torch.exp(-self._k2_d / denom1) / denom2

            # Perform the second convolution and retrieve the result to real space
            #self._tmp_d = torch.fft.irfftn(self._f_tmp_d * f_GaussKern_d)
            torch.fft.irfftn(self._f_tmp_d * self._f_pf_smoothing_kernel_d, s=self._tmp_d.shape, out=self._tmp_d)

            # Normalize the phase field to lie in the range [0, 1]
            if self._normalize_pf:
                pf_min = torch.min(self._tmp_d)
                pf_max = torch.max(self._tmp_d)
                #self._tmp_d = (self._tmp_d - pf_min) / (pf_max - pf_min + 1.0e-15)  # Avoid division by zero
                self._tmp_d.sub_(pf_min)
                self._tmp_d.div_(pf_max - pf_min + 1.0e-15)  # Avoid division by zero

            #return self._tmp_d
            return self._tmp_d.detach().cpu().numpy()

        # Check if f_wavelet_d is a list
        if isinstance(self._f_pf_kernel_d, list):
            # If it is a list, compute pf for each f_wavelet_d
            #pf_list = [compute_pf(wavelet, self._k2_d, varGauss, self._f_den_d, normalize_pf) for wavelet in self._f_pfkernel_d]
            pf_list = [compute_pf(wavelet) for wavelet in self._f_pf_kernel_d]
            # Transfer the result to host
            #pf_list = [pf.detach().cpu().numpy() for pf in pf_list]

            if self._verbose:
                 tend = time.time()
                 print(f'Time to evaluate phase field: {tend-tstart:.3f} s')

            return pf_list
        else:
            # If it is not a list, compute pf for the single f_wavelet_d
            #pf = compute_pf(self._f_pfkernel_d, self._k2_d, varGauss, self._f_den_d, normalize_pf)
            pf = compute_pf(self._f_pf_kernel_d)
            # Transfer the result to host
            #pf = pf.detach().cpu().numpy()

            if self._verbose:
                 tend = time.time()
                 print(f'Time to evaluate phase field: {tend-tstart:.3f} s')

            return pf
        
# =====================================================================================

    def evaluate_directional_correlation_kernel(self, H0, Rot):
        '''
        PURPOSE
            Establish the directional correlation kernel for a particular crystal structure.

        INPUT
            kx              Wave vector along the x-axis, [nx]
            ky              Wave vector along the y-axis, [ny]
            kz              Wave vector along the z-axis, [nz]
            latticePar      Lattice parameter
            struct          Crystal structure: SC, BCC, FCC, DC
            H0              Constant modulation of the peak height
            Rot             Lattice rotation matrix, [3, 3]
    
        OUTPUT
            f_H             Directional correlation kernel, [nx, ny, nz/2+1]

        Last revision:
        H. Hallberg 2024-10-21
        '''

        if self._verbose: tstart = time.time()

        # Allocate output array
        f_H = np.zeros((self._nx, self._ny, self._nz_half), dtype=self._dtype_cpu)

        # Define reciprocal lattice vectors (RLV)
        rlv  = self.get_rlv(self._struct, self._alat)  # Shape: [nrlv, 3]
        nrlv = rlv.shape[0]
        
        # Gauss peak width parameters
        gamma = np.ones(nrlv, dtype=np.double)
        denom = 2 * gamma**2

        # Rotate the reciprocal lattice vectors
        rlv_rotated = np.dot(rlv, Rot.T)  # Shape: [nrlv, 3]

        # Create 3D grids for kx, ky, kz
        kx = self.get_k(self._nx, self._dx)
        ky = self.get_k(self._ny, self._dy)
        kz = self.get_k(self._nz, self._dz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz[:self._nz_half], indexing='ij')

        # Loop over reciprocal lattice vectors (small dimension)
        for p in range(nrlv):
            # Compute squared differences for each reciprocal lattice vector
            diff_kx = (KX - rlv_rotated[p, 0])**2
            diff_ky = (KY - rlv_rotated[p, 1])**2
            diff_kz = (KZ - rlv_rotated[p, 2])**2

            # Compute the Gaussian contribution for this lattice vector
            Htestval = H0 * np.exp(-(diff_kx + diff_ky + diff_kz) / denom[p])

            # Update the directional correlation kernel by taking the maximum
            f_H = np.maximum(f_H, Htestval)

        f_H_d = torch.from_numpy(f_H).to(self._device) # Copy to GPU device
        f_H_d = f_H_d.contiguous()                     # Ensure that the tensor is contiguous in memory

        if self._verbose:
            tend = time.time()
            print(f'Time to evaluate directional convolution kernel: {tend-tstart:.3f} s')

        return f_H_d

# =====================================================================================

    def get_rlv(self, struct, alat):
        '''
        PURPOSE
            Get the reciprocal lattice vectors for a particular crystal structure.

        INPUT
            struct      Crystal structure: SC, BCC, FCC, DC
            latticePar  Lattice parameter
    
        OUTPUT
            RLV         Reciprocal lattice vectors, [nRLV x 3]

        Last revision:
        H. Hallberg 2025-08-27
        '''

        # Define reciprocal lattice vectors
        structures = {
                'SC': [
                    [ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1],
                    [-1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]
                ],
                'BCC': [
                    [ 0,  1,  1], [ 0, -1,  1], [ 0,  1, -1], [ 0, -1, -1],
                    [ 1,  0,  1], [-1,  0,  1], [ 1,  0, -1], [-1,  0, -1],
                    [ 1,  1,  0], [-1,  1,  0], [ 1, -1,  0], [-1, -1,  0]
                ],
                'FCC': [
                    [ 1,  1,  1], [-1,  1,  1], [ 1, -1,  1], [ 1,  1, -1],
                    [-1, -1,  1], [ 1, -1, -1], [-1,  1, -1], [-1, -1, -1]
                ],
                'DC': [
                    [ 1,  1,  1], [-1,  1,  1], [ 1, -1,  1], [ 1,  1, -1],
                    [-1, -1,  1], [ 1, -1, -1], [-1,  1, -1], [-1, -1, -1],
                    [ 1,  1,  0], [-1,  1,  0], [ 1, -1,  0], [-1, -1,  0],
                    [ 1,  0,  1], [-1,  0,  1], [ 1,  0, -1], [-1,  0, -1],
                    [ 0,  1,  1], [ 0, -1,  1], [ 0,  1, -1], [ 0, -1, -1]
                ],
            }

        if struct.upper() not in structures:
            raise ValueError(f'Unsupported crystal structure ({struct.upper()}) in get_rlv')
        
        rlv = np.array(structures[struct], dtype=float)
        rlv = rlv * (2*np.pi/alat)

        return rlv

# =====================================================================================