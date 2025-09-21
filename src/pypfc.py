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
        'density_merge_distance':   0.1,
        'pf_iso_level':             0.5,
        'torch_threads':            os.cpu_count(),
        'torch_threads_interop':    os.cpu_count(),
    }

    def __init__(self, domain_size, ndiv=None, config=None):

        # Merge user parameters with defaults, but only use keys present in DEFAULTS
        # ==========================================================================
        cfg = dict(self.DEFAULTS)
        ignored = set()
        if config is not None:
            filtered_config = {k: v for k, v in config.items() if k in self.DEFAULTS}
            cfg.update(filtered_config)
            ignored = set(config.keys()) - set(self.DEFAULTS.keys())
        if ignored:
            print(f"Ignored config keys: {ignored}")

        # Ensure domain_size is a numpy array
        # ===================================
        domain_size = np.array(domain_size, dtype=float)

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
            self.update_density = self.__update_density_1
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
            self.update_density = self.__update_density_2
        elif self._update_scheme == 'exponential':
            if self._use_H2:
                self._f_Lterm0_d = g1 - self._C2_d - self._f_H_d
            else:
                self._f_Lterm0_d = g1 - self._C2_d
            self._f_Lterm0_d = torch.where(self._f_Lterm0_d == 0,
                                        torch.tensor(1e-12, device=self._device, dtype=self._dtype_torch),
                                        self._f_Lterm0_d).contiguous()
            self._f_Lterm1_d = torch.exp(-self._k2_d.mul(self._f_Lterm0_d) * dt).contiguous()
            self.update_density = self.__update_density_exp
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

    def __update_density_1(self, f_Lterm_d):
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

    def __update_density_2(self, f_Lterm0_d, f_Lterm1_d, f_Lterm2_d, f_Lterm3_d):
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

    def __update_density_exp(self, f_Lterm0_d, f_Lterm1_d):
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
            Evaluate the phase field using a wavelet filtering.
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
