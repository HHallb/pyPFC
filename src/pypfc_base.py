'''
pyPFC: A Python Package for Phase Field Crystal Simulations
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
import datetime
import torch
import time
import os
from pypfc_grid import setup_grid

class setup_base(setup_grid):

    DEFAULTS = {
        'dtype_cpu':                np.double,
        'dtype_gpu':                torch.float64,
        'device_type':              'gpu',
        'device_number':            0,
        'torch_threads':            os.cpu_count(),
        'torch_threads_interop':    os.cpu_count(),
        'verbose':                  False,
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

        # Initiate the inherited grid class
        # =================================
        super().__init__(ndiv, ddiv)

        # Set the data types
        self._dtype_cpu               = cfg['dtype_cpu']
        self._dtype_gpu               = cfg['dtype_gpu']
        self._device_number           = cfg['device_number']
        self._device_type             = cfg['device_type']
        self._set_num_threads         = cfg['torch_threads']
        self._set_num_interop_threads = cfg['torch_threads_interop']
        self._verbose                 = cfg['verbose']

        # Set complex GPU array precision based on dtype_gpu
        # ==================================================
        if self._dtype_gpu == torch.float32:
            self._ctype_gpu = torch.cfloat
        elif self._dtype_gpu == torch.float64:
            self._ctype_gpu = torch.cdouble
        else:
            raise ValueError("dtype_gpu must be torch.float32 or torch.float64")

        # Set computing environment (CPU/GPU)
        # ===================================
        nGPU = torch.cuda.device_count()
        if nGPU>0 and self._device_type.upper() == 'GPU':
            self._device = torch.device('cuda')
            torch.cuda.set_device(self._device_number)
            # Additional info when using GPU
            if self._verbose:
                for gpuNr in range(nGPU):
                    print(f'GPU {gpuNr}: {torch.cuda.get_device_name(gpuNr)}')
                    print(f'       Compute capability:    {torch.cuda.get_device_properties(gpuNr).major}.{torch.cuda.get_device_properties(gpuNr).minor}')
                    print(f'       Total memory:          {round(torch.cuda.get_device_properties(gpuNr).total_memory/1024**3,2)} GB')
                    print(f'       Allocated memory:      {round(torch.cuda.memory_allocated(gpuNr)/1024**3,2)} GB') # Returns the maximum GPU memory managed by the caching allocator in bytes for a given device
                    print(f'       Cached memory:         {round(torch.cuda.memory_reserved(gpuNr)/1024**3,2)} GB')  # Returns the current GPU memory usage by tensors in bytes for a given device
                    print(f'       Multi processor count: {torch.cuda.get_device_properties(gpuNr).multi_processor_count}')
                    #print(f'       GPU temperature:       {torch.cuda.temperature(gpuNr)} degC')
                    print(f'')
                print(f'Current GPU: {torch.cuda.current_device()}')
            torch.cuda.empty_cache() # Clear GPU cache
        elif nGPU==0 and self._device_type.upper() == 'GPU':
            raise ValueError(f'No GPU available, but GPU requested: device_number={self._device_number}')
        elif self._device_type.upper() == 'CPU':
            self._device = torch.device('cpu') 
            torch.set_num_threads(self._set_num_threads)
            torch.set_num_interop_threads(self._set_num_interop_threads)
            if self._verbose:
                print(f"Using {self._set_num_threads} CPU threads and {self._set_num_interop_threads} interop threads.")
        if self._verbose:
            print(f'Using device: {self._device}')

        # Get wave vector operator
        # ========================
        if self._verbose: tstart = time.time()
        self._k2_d = self.evaluate_k2_d()
        if self._verbose:
            tend = time.time()
            print(f'Time to evaluate k2_d: {tend-tstart:.3f} s')

# =====================================================================================

    def set_verbose(self, verbose):
        self._verbose = verbose

    def get_verbose(self):
        return self._verbose

    def set_dtype_cpu(self, dtype):
        self._dtype_cpu = dtype

    def get_dtype_cpu(self):
        return self._dtype_cpu

    def set_dtype_gpu(self, dtype):
        self._dtype_gpu = dtype

    def get_dtype_gpu(self):
        return self._dtype_gpu
    
    def set_device_type(self, device_type):
        self._device_type = device_type

    def get_device_type(self):
        return self._device_type

    def set_device_number(self, device_number):
        self._device_number = device_number

    def get_device_number(self):
        return self._device_number

    def set_k2_d(self, k2_d):
        self._k2_d = k2_d

    def get_k2_d(self):
        return self._k2_d

    def get_torch_threads(self):
        return torch.get_num_threads(), torch.get_num_interop_threads()
    
    def set_torch_threads(self, nthreads, nthreads_interop):
        torch.set_num_threads(nthreads)
        torch.set_num_interop_threads(nthreads_interop)
        self._set_num_threads         = nthreads
        self._set_num_interop_threads = nthreads_interop

# =====================================================================================

    def get_time_stamp(self):
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

# =====================================================================================

    def get_k(self, npoints, dspacing):
        '''
        PURPOSE
            Define a 1D wave vector.

        INPUT
            npoints     Number of grid points
            dspacing    Grid spacing

        OUTPUT
            k           Wave vector

        Last revision:
        H. Hallberg 2025-08-26
        '''

        # Check input
        if np.mod(npoints,2) != 0:
            raise ValueError(f'The number of grid points must be an even number, npoints={npoints}')

        delk = 2*np.pi / (npoints*dspacing)
        k    = np.zeros(npoints, dtype=dspacing.dtype)

        k[:npoints//2] = np.arange(0, npoints//2) * delk
        k[npoints//2:] = np.arange(-npoints//2, 0) * delk

        return k
    
    # =====================================================================================

    def evaluate_k2_d(self):
        '''
        PURPOSE
            Evaluate the sum of the squared wave vectors.

        INPUT

        OUTPUT
            k2_d         k2=kx**2+ky**2 +kz**2,  [nx, ny, nz] (on the device)

        Last revision:
        H. Hallberg 2025-08-26
        '''

        kx = self.get_k(self._nx, self._dx)
        ky = self.get_k(self._ny, self._dy)
        kz = self.get_k(self._nz, self._dz)

        kx2 = kx[:, np.newaxis, np.newaxis] ** 2
        ky2 = ky[np.newaxis, :, np.newaxis] ** 2
        kz2 = kz[np.newaxis, np.newaxis, :] ** 2
        k2  = kx2 + ky2 + kz2

        k2_d = torch.from_numpy(k2[:,:,:self._nz_half]).to(self._device) # Copy to device
        k2_d = k2_d.to(dtype=self._dtype_gpu) # Ensure correct dtype
        k2_d = k2_d.contiguous()

        return k2_d

    # =====================================================================================

    def fit_circle_2d(self, points):
        """
        PURPOSE
            Fits a circle to a set of 2D points using least squares.

        INPUT
            points  Array-like, shape (N, 2)

        OUTPUT
            cx, cy  Center point coordinates
            r       Radius

        Last revision:
        H. Hallberg 2025-08-18
        """

        points = np.asarray(points)
        x = points[:, 0]
        y = points[:, 1]
        
        # Assemble the system: (x-cx)^2 + (y-cy)^2 = r^2
        A = np.c_[2*x, 2*y, np.ones_like(x)]
        b = x**2 + y**2
        
        # Solve for cx, cy, r^2
        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, c = sol
        r = np.sqrt(c + cx**2 + cy**2)
        
        return cx, cy, r

# =====================================================================================
