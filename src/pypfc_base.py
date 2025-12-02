# Copyright (C) 2025 Håkan Hallberg
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE file for full license text

import numpy as np
import datetime
import torch
import time
from scipy.spatial import cKDTree
from scipy.ndimage import zoom
from skimage import measure
from scipy import ndimage as ndi
from typing import Union, Tuple, Optional, Dict, Any, List
from pypfc_grid import setup_grid

class setup_base(setup_grid):

    def __init__(self, domain_size: np.ndarray, ndiv: np.ndarray, config: Dict[str, Any]) -> None:
        """
        Initialize the base PFC setup with domain parameters and device configuration.
        
        Parameters
        ----------
        domain_size : ndarray of float, shape (3,)
            Physical size of the simulation domain [Lx, Ly, Lz] in lattice parameter units.
        ndiv : ndarray of int, shape (3,)
            Number of grid divisions [nx, ny, nz]. Must be even numbers for FFT compatibility.
        config : dict
            Configuration parameters as key-value pairs.
            See the [pyPFC overview](core.md) for a complete list of the configuration parameters.
            
        Raises
        ------
        ValueError
            If dtype_gpu is not torch.float32 or torch.float64.
        ValueError
            If GPU is requested but no GPU is available.
        """

        # Initiate the inherited grid class
        # =================================
        super().__init__(domain_size, ndiv)

        # Set the data types
        self._struct                  = config['struct']
        self._alat                    = config['alat']
        self._sigma                   = config['sigma']
        self._npeaks                  = config['npeaks']
        self._alpha                   = np.array(config['alpha'], dtype=config['dtype_cpu'])
        self._dtype_cpu               = config['dtype_cpu']
        self._dtype_gpu               = config['dtype_gpu']
        self._device_number           = config['device_number']
        self._device_type             = config['device_type']
        self._set_num_threads         = config['torch_threads']
        self._set_num_interop_threads = config['torch_threads_interop']
        self._verbose                 = config['verbose']
        self._density_interp_order    = config['density_interp_order']
        self._density_threshold       = config['density_threshold']
        self._density_merge_distance  = config['density_merge_distance']
        self._pf_iso_level            = config['pf_iso_level']

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
                    print(f'       Allocated memory:      {round(torch.cuda.memory_allocated(gpuNr)/1024**3,2)} GB')
                    print(f'       Cached memory:         {round(torch.cuda.memory_reserved(gpuNr)/1024**3,2)} GB')
                    print(f'       Multi processor count: {torch.cuda.get_device_properties(gpuNr).multi_processor_count}')
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

    def set_verbose(self, verbose: bool) -> None:
        """
        Set verbose output mode for debugging and monitoring.
        
        Parameters
        ----------
        verbose : bool
            If True, enables detailed timing and progress output.
        """
        self._verbose = verbose

# =====================================================================================

    def get_verbose(self) -> bool:
        """
        Get the current verbose output setting.
        
        Returns
        -------
        bool
            Current verbose mode setting.
        """
        return self._verbose

# =====================================================================================

    def set_dtype_cpu(self, dtype: type) -> None:
        """
        Set the CPU data type for numpy arrays.
        
        Parameters
        ----------
        dtype : numpy.dtype
            NumPy data type for CPU computations (e.g., np.float32, np.float64).
        """
        self._dtype_cpu = dtype

# =====================================================================================

    def get_dtype_cpu(self) -> type:
        """
        Get the current CPU data type.
        
        Returns
        -------
        numpy.dtype
            Current NumPy data type used for CPU arrays.
        """
        return self._dtype_cpu

# =====================================================================================

    def set_dtype_gpu(self, dtype: torch.dtype) -> None:
        """
        Set the GPU data type for PyTorch tensors.
        
        Parameters
        ----------
        dtype : torch.dtype
            PyTorch data type for GPU computations (e.g., torch.float32, torch.float64).
        """
        self._dtype_gpu = dtype

# =====================================================================================

    def get_dtype_gpu(self) -> torch.dtype:
        """
        Get the current GPU data type.
        
        Returns
        -------
        torch.dtype
            Current PyTorch data type used for GPU tensors.
        """
        return self._dtype_gpu

# =====================================================================================
#     
    def set_device_type(self, device_type: str) -> None:
        """
        Set the computation device type.
        
        Parameters
        ----------
        device_type : str
            Device type for computations. Options: 'CPU', 'GPU'.
        """
        self._device_type = device_type

# =====================================================================================

    def get_device_type(self) -> str:
        """
        Get the current computation device type.
        
        Returns
        -------
        str
            Current device type ('CPU' or 'GPU').
        """
        return self._device_type

# =====================================================================================

    def set_device_number(self, device_number: int) -> None:
        """
        Set the GPU device number for multi-GPU systems.
        
        Parameters
        ----------
        device_number : int
            GPU device index (0, 1, 2, ...) for CUDA computations.
        """
        self._device_number = device_number

# =====================================================================================

    def get_device_number(self) -> int:
        """
        Get the current GPU device number.
        
        Returns
        -------
        int
            Current GPU device index.
        """
        return self._device_number

# =====================================================================================

    def set_k2_d(self, k2_d: torch.Tensor) -> None:
        """
        Set the wave vector magnitude squared tensor.
        
        Parameters
        ----------
        k2_d : torch.Tensor
            Wave vector magnitude squared (k²) tensor in Fourier space.
            Used for FFT-based operations and differential operators.
        """
        self._k2_d = k2_d

# =====================================================================================

    def get_k2_d(self) -> torch.Tensor:
        """
        Get the wave vector magnitude squared tensor.
        
        Returns
        -------
        torch.Tensor
            Wave vector magnitude squared (k²) tensor in Fourier space.
        """
        return self._k2_d

# =====================================================================================

    def get_torch_threads(self) -> Tuple[int, int]:
        """
        Get the current PyTorch thread configuration.
        
        Returns
        -------
        tuple of int
            (num_threads, num_interop_threads) for PyTorch operations.
        """
        return torch.get_num_threads(), torch.get_num_interop_threads()

# =====================================================================================
#     
    def set_torch_threads(self, nthreads: int, nthreads_interop: int) -> None:
        """
        Set PyTorch thread configuration for CPU operations.
        
        Parameters
        ----------
        nthreads : int
            Number of threads for intra-op parallelism.
        nthreads_interop : int
            Number of threads for inter-op parallelism.
        """
        torch.set_num_threads(nthreads)
        torch.set_num_interop_threads(nthreads_interop)
        self._set_num_threads         = nthreads
        self._set_num_interop_threads = nthreads_interop

# =====================================================================================

    def set_alpha(self, alpha: Union[List[float], np.ndarray]) -> None:
        """
        Set the Gaussian peak widths for the two-point correlation function.
        
        Parameters
        ----------
        alpha : array_like of float
            Gaussian peak widths (α_i) for each peak in the correlation function.
        """
        self._alpha = alpha

# =====================================================================================

    def get_alpha(self) -> np.ndarray:
        """
        Get the Gaussian peak widths for the two-point correlation function.
        
        Returns
        -------
        alpha : ndarray of float
            Gaussian peak widths (α_i) for each peak in the correlation function.
        """
        return self._alpha

# =====================================================================================

    def get_time_stamp(self) -> str:
        """
        Get current timestamp string.
        
        Returns
        -------
        timestamp : str
            Current date and time in format: YYYY-MM-DD HH:MM.
        """
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

# =====================================================================================

    def get_k(self, npoints: int, dspacing: float) -> np.ndarray:
        """
        Define a 1D wave vector for Fourier space operations.

        Parameters
        ----------
        npoints : int
            Number of grid points. Must be even.
        dspacing : float
            Grid spacing in real space.

        Returns
        -------
        k : ndarray of float
            1D wave vector array with proper frequency ordering for FFTs.
            
        Raises
        ------
        ValueError
            If npoints is not an even number.
        """

        # Check input
        if np.mod(npoints,2) != 0:
            raise ValueError(f"The number of grid points must be an even number, got npoints={npoints}")

        delk = 2*np.pi / (npoints*dspacing)
        k    = np.zeros(npoints, dtype=self._dtype_cpu)

        k[:npoints//2] = np.arange(0, npoints//2) * delk
        k[npoints//2:] = np.arange(-npoints//2, 0) * delk

        return k
    
    # =====================================================================================

    def evaluate_k2_d(self) -> torch.Tensor:
        """
        Evaluate the sum of squared wave vectors for FFT operations.
        
        Computes $k^2 = k_x^2 + k_y^2 + k_z^2$ on the computational device
        using PyTorch FFT frequency grids. This is fundamental for Fourier-space
        operations in PFC simulations.
        
        Returns
        -------
        k2_d : torch.Tensor, shape (nx, ny, nz_half)
            Sum of squared wave vectors on the device. The z-dimension is 
            reduced due to real FFT symmetry (nz_half = nz//2 + 1).
        """

        kx    = 2 * torch.pi * torch.fft.fftfreq(self._nx, d=self._dx, device=self._device, dtype=self._dtype_gpu)
        ky    = 2 * torch.pi * torch.fft.fftfreq(self._ny, d=self._dy, device=self._device, dtype=self._dtype_gpu)
        kz    = 2 * torch.pi * torch.fft.fftfreq(self._nz, d=self._dz, device=self._device, dtype=self._dtype_gpu)
        k2_d  = torch.zeros((self._nx, self._ny, self._nz_half), dtype=self._dtype_gpu, device=self._device)
        k2_d += kx[:, None, None] ** 2
        k2_d += ky[None, :, None] ** 2
        k2_d += kz[None, None, :self._nz_half] ** 2
        
        return k2_d.contiguous()
    
    # =====================================================================================

    def get_integrated_field_in_volume(self, field: np.ndarray, limits: Union[List[float], np.ndarray]) -> float:
        """
        Integrate a field variable within a defined volume.
        
        Performs numerical integration of a field variable over a specified
        3D volume on a Cartesian grid.

        Parameters
        ----------
        field : ndarray of float, shape (nx, ny, nz)
            Field to be integrated over the specified volume.
        limits : array_like of float, length 6
            Spatial integration limits: [xmin, xmax, ymin, ymax, zmin, zmax].

        Returns
        -------
        result : float
            Result of the volume integration.
        """

        # Grid
        nx,ny,nz = self._ndiv
        dx,dy,dz = self._ddiv

        # Integration limits
        xmin,xmax,ymin,ymax,zmin,zmax = limits

        # Create a grid of coordinates
        x = np.linspace(0, (nx-1) * dx, nx)
        y = np.linspace(0, (ny-1) * dy, ny)
        z = np.linspace(0, (nz-1) * dz, nz)
        
        # Create a meshgrid of coordinates
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

        # Create a boolean mask for the integration limits
        mask = ((X >= xmin) & (X <= xmax) &
                (Y >= ymin) & (Y <= ymax) &
                (Z >= zmin) & (Z <= zmax))

        # Perform integration using the mask
        result = np.sum(field[mask]) * dx * dy * dz

        return result
      
# =====================================================================================

    def get_field_average_along_axis(self, field: np.ndarray, axis: int) -> np.ndarray:
        """
        Evaluate the mean value of a field variable along a specified axis.
        
        Computes the spatial average of a 3D field along one axis, 
        reducing the dimensionality by averaging over the other two axes.

        Parameters
        ----------
        field : ndarray of float, shape (nx, ny, nz)
            3D field variable to be averaged.
        axis : str
            Axis to average along: x, y or z (case insensitive).

        Returns
        -------
        result : ndarray of float
            1D array containing mean values along the specified axis.
            Shape depends on the axis: (nx,), (ny,), or (nz,).
            
        Raises
        ------
        ValueError
            If axis is not x, y or z.
        """

        # Evaluate the mean field value along the specified axis
        # ======================================================
        if axis.upper() == 'X':
            result = np.mean(field, axis=(1,2))
        elif axis.upper() == 'Y':
            result = np.mean(field, axis=(0,2))
        elif axis.upper() == 'Z':
            result = np.mean(field, axis=(0,1))
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        return result
      
# =====================================================================================

    def get_integrated_field_along_axis(self, field: np.ndarray, axis: int) -> np.ndarray:
        """
        Integrate a field variable along a specified axis.
        
        Performs numerical integration of a 3D field variable along one axis,
        integrating over the two orthogonal directions.

        Parameters
        ----------
        field : ndarray of float, shape (nx, ny, nz)
            3D field variable to be integrated.
        axis : str
            Axis to integrate along: x, y or z (case insensitive).

        Returns
        -------
        result : ndarray of float
            1D array containing integrated values along the specified axis.
            Shape depends on the axis: (nx,), (ny,), or (nz,).
            
        Raises
        ------
        ValueError
            If axis is not x, y or z.
        """

        # Grid
        # ====
        dx,dy,dz = self._ddiv

        # Integrate along the specified axis
        # ==================================
        if axis.upper() == 'X':
            # Integrate over y and z for each x
            result = np.sum(field, axis=(1,2)) * dy * dz
        elif axis.upper() == 'Y':
            # Integrate over x and z for each y
            result = np.sum(field, axis=(0,2)) * dx * dz
        elif axis.upper() == 'Z':
            # Integrate over x and y for each z
            result = np.sum(field, axis=(0,1)) * dx * dy
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        return result
      
# =====================================================================================

    def interpolate_atoms(self, intrp_pos: np.ndarray, pos: np.ndarray, values: np.ndarray, num_nnb: int = 8, power: int = 2) -> np.ndarray:
        """
        Interpolate values at given positions using inverse distance weighting.
        
        Performs 3D interpolation in a periodic domain using inverse distance
        weighting: interpolated_value = Σ(wi × vi) / Σ(wi), where wi = 1 / (di^power).
        
        Parameters
        ----------
        intrp_pos : ndarray of float, shape (n_intrp, 3)
            3D coordinates of positions where values should be interpolated.
        pos : ndarray of float, shape (n_particles, 3)
            3D coordinates of particles with known values.
        values : ndarray of float, shape (n_particles,)
            Values at the particle positions to be interpolated.
        num_nnb : int, optional
            Number of nearest neighbors to use for interpolation.
        power : float, optional
            Power for inverse distance weighting.
            
        Returns
        -------
        interp_val : ndarray of float, shape (n_intrp,)
            Interpolated values at the specified positions.
        """

        n_interp   = intrp_pos.shape[0]
        interp_val = np.zeros(n_interp, dtype=self._dtype_cpu)

        # Generate periodic images of the source positions
        images = np.vstack([pos + np.array([dx, dy, dz]) * self._domain_size
                            for dx in (-1, 0, 1)
                            for dy in (-1, 0, 1)
                            for dz in (-1, 0, 1)])
        
        # Replicate values for all periodic images
        values_periodic = np.tile(values, 27)  # 3^3 = 27 periodic images
        
        # Create KDTree for efficient neighbor search
        tree = cKDTree(images)
        
        # Parameters for inverse distance weighting
        k_neighbors = min(num_nnb, len(pos))  # Number of nearest neighbors to use
        epsilon     = 1e-12  # Small value to avoid division by zero
        
        # Vectorized neighbor search for all interpolation points at once
        distances, indices = tree.query(intrp_pos, k=k_neighbors)
        
        # Handle exact matches (distance < epsilon)
        exact_matches = distances[:, 0] < epsilon
        
        # Initialize output array
        interp_val = np.zeros(n_interp, dtype=self._dtype_cpu)
        
        # For exact matches, use the nearest neighbor value directly
        if np.any(exact_matches):
            interp_val[exact_matches] = values_periodic[indices[exact_matches, 0]]
        
        # For non-exact matches, use inverse distance weighting
        non_exact = ~exact_matches
        if np.any(non_exact):
            # Get distances and indices for non-exact matches
            dist_subset = distances[non_exact]
            idx_subset = indices[non_exact]
            
            # Compute weights: 1 / distance^power
            weights = 1.0 / (dist_subset ** power)
            
            # Get values for all neighbors
            neighbor_values = values_periodic[idx_subset]
            
            # Compute weighted sum and total weights
            weighted_sum = np.sum(weights * neighbor_values, axis=1)
            total_weight = np.sum(weights, axis=1)
            
            # Store interpolated values
            interp_val[non_exact] = weighted_sum / total_weight

        return interp_val

# =====================================================================================

    def interpolate_density_maxima(self, den: Union[np.ndarray, torch.Tensor], ene: Optional[Union[np.ndarray, torch.Tensor]] = None, pf: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find density field maxima and interpolate atomic positions and properties.
        
        Identifies local maxima in the density field as atomic positions and performs
        high-order interpolation to obtain sub-grid precision coordinates. Also 
        interpolates associated field values (density, energy, phase fields) at
        the atomic positions.
        
        Parameters
        ----------
        den : ndarray of float, shape (nx,ny,nz)
            Density field from PFC simulation.
        ene : ndarray of float, shape (nx,ny,nz), optional
            Energy field for interpolation at atomic positions.
        pf : list of ndarray, optional
            List of phase fields for interpolation at atomic positions.
            Each array should have shape (nx,ny,nz).
            
        Returns
        -------
        atom_coord : ndarray of float, shape (n_maxima,3)
            Interpolated coordinates of density maxima (atomic positions).
        atom_data : ndarray of float, shape (n_maxima, 2+n_phase_fields)
            Interpolated field values at atomic positions.
            Columns: [density, energy, pf1, pf2, ..., pfN]
            
        Notes
        -----
        The method uses scipy.ndimage for high-order interpolation and applies
        density thresholding and merging of nearby maxima to remove spurious peaks.
        The interpolation order is controlled by the `_density_interp_order` attribute.
        """

        if self._verbose: tstart = time.time()

        # Grid spacing
        dx, dy, dz = self._ddiv
        
        # Get density threshold early to avoid recomputation
        max_den = np.max(den)
        density_threshold = self._density_threshold * max_den
        # Optimized local maxima detection using maximum_filter (this is actually quite efficient)
        size = 1 + 2 * self._density_interp_order
        footprint = np.ones((size, size, size), dtype=bool)
        footprint[self._density_interp_order, self._density_interp_order, self._density_interp_order] = False
        
        # Find local maxima - maximum_filter is optimized in scipy
        filtered = ndi.maximum_filter(den, footprint=footprint, mode='wrap')
        
        # Combine maxima detection and threshold filtering in one operation
        valid_maxima_mask = (den > filtered) & (den >= density_threshold)
        
        # Early exit if no maxima found
        if not np.any(valid_maxima_mask):
            atom_coord = np.array([]).reshape(0, 3)
            atom_data = np.array([]).reshape(0, 1)
            return atom_coord, atom_data
            
        # Extract coordinates efficiently - avoid transpose operations
        maxima_indices = np.where(valid_maxima_mask)
        n_maxima = len(maxima_indices[0])
        
        # Pre-allocate coordinate array and fill directly
        coords = np.empty((n_maxima, 3), dtype=self._dtype_cpu)
        coords[:, 0] = maxima_indices[0] * dx
        coords[:, 1] = maxima_indices[1] * dy 
        coords[:, 2] = maxima_indices[2] * dz
        
        # Extract field values directly using the indices
        denpos = den[maxima_indices]
        enepos = ene[maxima_indices] if ene is not None else None

        # Simplified clustering - most efficient for typical PFC use cases
        if self._density_merge_distance > 0.0 and n_maxima > 1:
            # Use KDTree for all cases - it's consistently fast and memory efficient
            tree = cKDTree(coords)
            visited = np.zeros(n_maxima, dtype=bool)
            
            cluster_coords = []
            cluster_denpos = []
            cluster_enepos = [] if ene is not None else None
            
            for i in range(n_maxima):
                if visited[i]:
                    continue
                    
                # Find all points within merge distance using KDTree
                neighbors = tree.query_ball_point(coords[i], r=self._density_merge_distance)
                
                # Mark as visited
                visited[neighbors] = True
                
                # Average the cluster - use numpy indexing directly
                if len(neighbors) == 1:
                    # Single point - no averaging needed
                    cluster_coords.append(coords[i])
                    cluster_denpos.append(denpos[i])
                    if ene is not None:
                        cluster_enepos.append(enepos[i])
                else:
                    # Multiple points - compute averages
                    cluster_coords.append(np.mean(coords[neighbors], axis=0))
                    cluster_denpos.append(np.mean(denpos[neighbors]))
                    if ene is not None:
                        cluster_enepos.append(np.mean(enepos[neighbors]))
            
            atom_coord = np.array(cluster_coords)
            denpos = np.array(cluster_denpos)
            if ene is not None:
                enepos = np.array(cluster_enepos)
        else:
            atom_coord = coords

        # Handle phase field(s) efficiently
        if pf is not None:
            if isinstance(pf, np.ndarray) and pf.ndim == 3:
                pf_list = [pf]
            else:
                pf_list = list(pf)
            
            n_pf = len(pf_list)
            n_atoms = len(atom_coord)
            
            # Extract phase field values efficiently
            pfpos = np.empty((n_atoms, n_pf), dtype=self._dtype_cpu)
            for pf_idx, phase_field in enumerate(pf_list):
                if self._density_merge_distance > 0.0 and n_maxima != n_atoms:
                    # Merging occurred - use first value (approximate)
                    pf_values = phase_field[maxima_indices]
                    pfpos[:, pf_idx] = pf_values[:n_atoms]
                else:
                    # No merging - direct extraction
                    pfpos[:, pf_idx] = phase_field[maxima_indices]
            
            # Assemble final data array efficiently
            if ene is not None:
                atom_data = np.column_stack((denpos, enepos, pfpos))
            else:
                atom_data = np.column_stack((denpos, pfpos))
        else:
            # No phase fields - simpler assembly
            if ene is not None:
                atom_data = np.column_stack((denpos, enepos))
            else:
                atom_data = denpos[:, np.newaxis]

        if self._verbose:
            tend = time.time()
            print(f'Time to interpolate {atom_coord.shape[0]} density maxima: {tend-tstart:.3f} s')

        return atom_coord, atom_data
    
# =====================================================================================

    # def interpolate_density_maxima_BACKUP25050924(self, den, ene=None, pf=None):
    #         '''
    #         PURPOSE
    #             Find the coordinates of the maxima in the density field (='atom' positions)
    #             The domain is assumed to be defined such that all maxima
    #             have coordinates (x,y,z) >= (0,0,0).
    #             The density and, optionally, the energy and the phase field value(s)
    #             at the individual maxima are interpolated too.

    #         INPUT
    #             den                     Density field, [nx, ny, nz]
    #             ene                     Energy field, [nx, ny, nz]
    #             pf                      Optional list of phase fields, [nx, ny, nz]

    #         OUTPUT
    #             atom_coord              Coordinates of the density maxima, [nmaxima x 3]
    #             atom_data               Interpolated field values at the density maxima,
    #                                     [nmaxima x 2+nPhaseFields].
    #                                     The columns hold point data in the order:
    #                                     [den ene pf1 pf2 ... pfN]

    #         Last revision:
    #         H. Hallberg 2025-09-20
    #         '''

    #         if self._verbose: tstart = time.time()

    #         # Grid
    #         dx,dy,dz = self._ddiv

    #         size = 1 + 2 * self._density_interp_order
    #         footprint = np.ones((size, size, size))
    #         footprint[self._density_interp_order, self._density_interp_order, self._density_interp_order] = 0

    #         filtered = ndi.maximum_filter(den, footprint=footprint, mode='wrap')

    #         mask_local_maxima = den > filtered
    #         coords = np.asarray(np.where(mask_local_maxima),dtype=self._dtype_cpu).T

    #         # ndi.maximum_filter works in voxel coordinates, convert to physical coordinates
    #         coords[:,0] *= dx
    #         coords[:,1] *= dy
    #         coords[:,2] *= dz

    #         # Filter maxima based on density threshold
    #         max_den = np.max(den)
    #         valid_maxima = den[mask_local_maxima] >= (self._density_threshold * max_den)
    #         coords = coords[valid_maxima]

    #         denpos = den[mask_local_maxima][valid_maxima]
    #         if ene is not None:
    #             enepos = ene[mask_local_maxima][valid_maxima]

    #         # Merge maxima within the merge_distance
    #         if self._density_merge_distance > 0.0 and len(coords) > 0:
    #             tree = cKDTree(coords)
    #             clusters = tree.query_ball_tree(tree, r=self._density_merge_distance)
    #             unique_clusters = []
    #             seen = set()
    #             for cluster in clusters:
    #                 cluster = tuple(sorted(cluster))
    #                 if cluster not in seen:
    #                     seen.add(cluster)
    #                     unique_clusters.append(cluster)

    #             merged_coords = []
    #             merged_denpos = []
    #             merged_enepos = [] if ene is not None else None
    #             for cluster in unique_clusters:
    #                 cluster_coords = coords[list(cluster)]
    #                 cluster_denpos = denpos[list(cluster)]
    #                 merged_coords.append(np.mean(cluster_coords, axis=0))
    #                 merged_denpos.append(np.mean(cluster_denpos))
    #                 if ene is not None:
    #                     cluster_enepos = enepos[list(cluster)]
    #                     merged_enepos.append(np.mean(cluster_enepos))
    #             atom_coord = np.array(merged_coords)
    #             denpos = np.array(merged_denpos)
    #             if ene is not None:
    #                 enepos = np.array(merged_enepos)
    #         else:
    #             atom_coord = coords
    #             # denpos and enepos are already set above
    #             # Only set enepos if ene is not None
    #             if ene is not None:
    #                 enepos = enepos

    #         # Handle phase field(s), either as a list of fields or as a single field
    #         if pf is not None:
    #             # If pf is a single array, wrap it in a list
    #             if isinstance(pf, np.ndarray) and pf.ndim == 3:
    #                 pf_list = [pf]
    #             else:
    #                 pf_list = list(pf)
    #             nPf = len(pf_list)
    #             pfpos = np.zeros((coords.shape[0], nPf), dtype=self._dtype_cpu)
    #             for pfNr, phaseField in enumerate(pf_list):
    #                 pfpos[:, pfNr] = phaseField[mask_local_maxima][valid_maxima][:coords.shape[0]]
    #             if ene is not None:
    #                 atom_data = np.hstack((denpos[:, None], enepos[:, None], pfpos))
    #             else:
    #                 atom_data = np.hstack((denpos[:, None], pfpos))
    #         else:
    #             if ene is not None:
    #                 atom_data = np.hstack((denpos[:, None], enepos[:, None]))
    #             else:
    #                 atom_data = denpos[:, None]

    #         if self._verbose:
    #             tend = time.time()
    #             print(f'Time to interpolate {atom_coord.shape[0]} density maxima: {tend-tstart:.3f} s')

    #         return atom_coord, atom_data
    
# =====================================================================================

    def get_phase_field_contour(self, pf: Union[np.ndarray, torch.Tensor], pf_zoom: float = 1.0, evaluate_volume: bool = True) -> Union[Tuple[np.ndarray, float], np.ndarray]:
        """
        Find the iso-contour surface of a 3D phase field using marching cubes.
        
        Extracts iso-surfaces from 3D phase field data using the marching cubes
        algorithm, with optional volume calculation for enclosed regions.
        
        Parameters
        ----------
        pf : ndarray of float, shape (nx, ny, nz)
            3D phase field data for iso-surface extraction.
        pf_zoom : float, optional
            Zoom factor for spatial coarsening/refinement.
        evaluate_volume : bool, optional
            If True, calculates the volume enclosed by the iso-surface.
            
        Returns
        -------
        verts : ndarray of float, shape (n_vertices, 3)
            Vertices of the iso-surface triangulation.
        faces : ndarray of int, shape (n_faces, 3)
            Surface triangulation topology (vertex indices).
        volume : float, optional
            Volume enclosed by the iso-surface (only if evaluate_volume=True).
        """

        verts, faces, *_ = measure.marching_cubes(zoom(pf,pf_zoom), self._pf_iso_level, spacing=self._ddiv)
        verts            = verts / pf_zoom

        if evaluate_volume:
            v0 = verts[faces[:, 0]]
            v1 = verts[faces[:, 1]]
            v2 = verts[faces[:, 2]]
            cross_product  = np.cross(v1-v0, v2-v0)
            signed_volumes = np.einsum('ij,ij->i', v0, cross_product)
            volume         = np.abs(np.sum(signed_volumes) / 6.0)
            return verts, faces, volume
        else:
            return verts, faces

# =====================================================================================

    def get_rlv(self, struct: str, alat: float) -> np.ndarray:
        """
        Get the reciprocal lattice vectors for a crystal structure.
        
        Computes reciprocal lattice vectors for common crystal structures
        used in phase field crystal modeling.
        
        Parameters
        ----------
        struct : str
            Crystal structure type. Options: 'SC', 'BCC', 'FCC', 'DC'.
        alat : float
            Lattice parameter.
            
        Returns
        -------
        rlv : ndarray of float, shape (nrlv, 3)
            Reciprocal lattice vectors for the specified crystal structure.
            
        Raises
        ------
        ValueError
            If the crystal structure is not supported.
        """

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
        
        rlv = np.array(structures[struct], dtype=self._dtype_cpu)
        rlv = rlv * (2*np.pi/alat)

        return rlv

# =====================================================================================

    def evaluate_reciprocal_planes(self) -> torch.Tensor:
        """
        Establish reciprocal vectors/planes for a crystal structure.
        
        Computes reciprocal lattice plane spacing (d-spacing) and wave vectors
        for crystallographic planes. For cubic systems: d = a / sqrt(h² + k² + l²)
        where a is the lattice parameter, and reciprocal spacing is k = 2π/d.
        
        Returns
        -------
        k_plane : ndarray of float
            Reciprocal lattice plane spacings (wave vector magnitudes).
        n_plane : ndarray of int
            Number of symmetrical planes in each family.
        den_plane : ndarray of float
            Atomic density within each plane family.
            
        Raises
        ------
        ValueError
            If the crystal structure is not supported.
        ValueError
            If there are not enough peaks defined for the requested number of peaks.
            
        Notes
        -----
        For any family of lattice planes separated by distance d, there are
        reciprocal lattice points at intervals of 2π/d in reciprocal space.
        """

        k_plane   = np.zeros(self._npeaks, dtype=self._dtype_cpu)
        den_plane = np.zeros(self._npeaks, dtype=self._dtype_cpu)
        n_plane   = np.zeros(self._npeaks, dtype=int)

        # Define reciprocal vectors
        match self._struct.upper():
            case 'SC': #= SC in reciprocal space
                # {100}, {110}, {111}
                nvals = 3
                kpl   = (2*np.pi/self._alat) * np.array([1, np.sqrt(2), np.sqrt(3)], dtype=self._dtype_cpu)
                pl    = np.array([6, 12, 8], dtype=int)
                denpl = (1/self._alat**2) * np.array([1, 1/np.sqrt(2), 1/np.sqrt(3)], dtype=self._dtype_cpu)
            case 'BCC': # = FCC in reciprocal space
                # {110}, {200}       (...the next would be {211}, {220}, {310}, {222})
                nvals = 2
                kpl   = (2*np.pi/self._alat) * np.array([np.sqrt(2), 2], dtype=self._dtype_cpu)
                pl    = np.array([12, 6, 24], dtype=int)
                denpl = (1/self._alat**2) * np.array([2/np.sqrt(2), 1], dtype=self._dtype_cpu)
            case 'FCC': # = BCC in reciprocal space
                # {111}, {200}, {220}        (...the next would be {311}, {222})
                nvals = 3
                kpl   = (2*np.pi/self._alat) * np.array([np.sqrt(3), 2, np.sqrt(8)], dtype=self._dtype_cpu)
                pl    = np.array([8, 6, 12], dtype=int)
                denpl = (1/self._alat**2) * np.array([4/np.sqrt(3), 2, 4/np.sqrt(2)], dtype=self._dtype_cpu)
            case 'DC': # Diamond Cubic (3D)
                # {111}, {220}, {311}         (...the next would be {400}, {331}, {422}, {511})
                nvals = 3
                kpl   = (2*np.pi/self._alat) * np.array([np.sqrt(3), np.sqrt(8), np.sqrt(11)], dtype=self._dtype_cpu)
                pl    = np.array([8, 12, 24], dtype=int)                                                   
                denpl = (1/self._alat**2) * np.array([4/np.sqrt(3), 4/np.sqrt(2), 1.385641467389298], dtype=self._dtype_cpu)
            case _:
                raise ValueError(f'Unsupported crystal structure: struct={self._struct.upper()}')

        # Retrieve output data
        if nvals>=self._npeaks:
            k_plane   = kpl[0:self._npeaks]
            n_plane   = pl[0:self._npeaks]
            den_plane = denpl[0:self._npeaks]
        else:
            raise ValueError(f'Not enough peaks defined, npeaks={self._npeaks}')

        return k_plane, n_plane, den_plane

# =====================================================================================

    def evaluate_C2_d(self) -> torch.Tensor:
        """
        Establish the two-point correlation function for a crystal structure.
        
        Computes the two-point pair correlation function in Fourier space
        for the specified crystal structure using Gaussian peaks at 
        reciprocal lattice positions.
        
        Returns
        -------
        C2_d : torch.Tensor, shape (nx, ny, nz//2+1)
            Two-point pair correlation function on the computational device.
            
        Raises
        ------
        ValueError
            If C20_alpha is negative when C20_amplitude is non-zero.
        """

        # Get reciprocal planes
        kpl, npl, denpl = self.evaluate_reciprocal_planes()

        # Convert to PyTorch tensors and move to device
        kpl_d   = torch.tensor(kpl,   dtype=self._dtype_gpu, device=self._k2_d.device)
        denpl_d = torch.tensor(denpl, dtype=self._dtype_gpu, device=self._k2_d.device)
        alpha_d = torch.tensor(self._alpha, dtype=self._dtype_gpu, device=self._k2_d.device)
        npl_d   = torch.tensor(npl,   dtype=self._dtype_gpu, device=self._k2_d.device)

        # Evaluate the exponential pre-factor (Debye-Waller-like)
        DWF_d = torch.exp(-(self._sigma**2) * (kpl_d**2) / (2 * denpl_d * npl_d))

        # Precompute quantities
        denom_d   = 2 * alpha_d**2
        k2_sqrt_d = torch.sqrt(self._k2_d)

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
        if self._C20_amplitude < 0.0:
            # Envelope as the largest absolute value at each grid point. This is needed if the zero-mode
            # peak has a negative amplitude, but consumes slightly more memory
            for ipeak in range(self._npeaks):
                peak_val = DWF_d[ipeak] * torch.exp( -(k2_sqrt_d - kpl_d[ipeak]) ** 2 / denom_d[ipeak] )
                mask = peak_val.abs() > self._f_tmp_d.real.abs()
                self._f_tmp_d.real[mask] = peak_val[mask]
        else:
            for ipeak in range(self._npeaks):
                peak_val = DWF_d[ipeak] * torch.exp( -(k2_sqrt_d - kpl_d[ipeak]) ** 2 / denom_d[ipeak] )
                self._f_tmp_d.real = torch.maximum(self._f_tmp_d.real, peak_val)

        # Return the real part as the result
        C2_d = self._f_tmp_d.real.contiguous()

        return C2_d

# =====================================================================================

    def evaluate_directional_correlation_kernel(self, H0: np.ndarray, Rot: np.ndarray) -> torch.Tensor:
        """
        Establish directional correlation kernel for a crystal structure.
        
        Computes directional correlation kernels used in extended PFC models
        to introduce orientational dependence.
        
        Parameters
        ----------
        H0 : float
            Constant modulation of the peak height.
        Rot : ndarray of float, shape (3, 3) or None
            Lattice rotation matrix. If None, uses identity matrix.
            
        Returns
        -------
        H_d : torch.Tensor, shape (nx, ny, nz//2+1)
            Directional correlation kernel on the computational device.
        """

        if self._verbose: tstart = time.time()

        # Allocate output array
        f_H = np.zeros((self._nx, self._ny, self._nz_half), dtype=self._dtype_cpu)

        # Define reciprocal lattice vectors (RLV)
        rlv  = self.get_rlv(self._struct, self._alat)  # Shape: [nrlv, 3]
        nrlv = rlv.shape[0]
        
        # Gauss peak width parameters
        gamma = np.ones(nrlv, dtype=self._dtype_cpu)
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

    def get_xtal_nearest_neighbors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get nearest neighbor information for crystal structures.
        
        Computes nearest neighbor distances and coordination numbers
        for common crystal structures used in phase field crystal modeling.
        
        Returns
        -------
        nnb : ndarray of int
            Number of nearest and next-nearest neighbors.
        nnb_dist : ndarray of float
            Distances to the nearest and next-nearest neighbors.
            
        Raises
        ------
        ValueError
            If the crystal structure is not supported.
        """

        # Nearest and next nearest neighbor positions
        if self._struct.upper() == 'SC':
            # SC
            nnb_dist = self._alat * np.array([1.0, 1.0], dtype=self._dtype_cpu)
            nnb      = np.array([6, 12], dtype=int)
        elif self._struct.upper() == 'BCC':
            # BCC
            nnb_dist = self._alat * np.array([np.sqrt(3)/2, 1.0], dtype=self._dtype_cpu)
            nnb      = np.array([8, 6], dtype=int)
        elif self._struct.upper() == 'FCC':
            # FCC
            nnb_dist = self._alat * np.array([1/np.sqrt(2), 1.0, np.sqrt(3/2), np.sqrt(2), np.sqrt(5/2), np.sqrt(3), np.sqrt(7/2), 2.0], dtype=self._dtype_cpu)
            nnb      = np.array([12, 6, 24, 12, 24, 8, 48, 6], dtype=int)
        elif self._struct.upper() == 'DC':
            # DC
            nnb_dist = self._alat * np.array([np.sqrt(3)/4, 1/np.sqrt(2)], dtype=self._dtype_cpu)
            nnb      = np.array([4, 12], dtype=int)
        else:
            raise ValueError(f'Unsupported crystal structure: {self._struct}')

        return nnb, nnb_dist

# =====================================================================================

    def get_csp(self, pos: np.ndarray, normalize_csp: bool = False) -> np.ndarray:
        """
        Calculate the centro-symmetry parameter (CSP) for atoms.
        
        Computes CSP values for atoms in a 3D periodic domain to identify
        crystal defects and disorder. CSP quantifies deviation from
        centro-symmetric local environments.
        
        Parameters
        ----------
        pos : ndarray of float, shape (natoms, 3)
            3D coordinates of atoms.
        normalize_csp : bool, optional
            If True, normalizes CSP values to range [0,1].
            
        Returns
        -------
        csp : ndarray of float, shape (natoms,)
            Centro-symmetry parameter for each atom.
            
        References
        ----------
        C.L. Kelchner, S.J. Plimpton and J.C. Hamilton, Dislocation nucleation and defect
        structure during surface indentation, Phys. Rev. B, 58(17):11085-11088, 1998.
        https://doi.org/10.1103/PhysRevB.58.11085
        """

        if self._verbose:
            tstart = time.time()

        # Determine the number of nearest neighbors based on crystal structure
        nnb, _      = self.get_xtal_nearest_neighbors()
        n_neighbors = nnb[0]

        # Ensure n_neighbors is even for CSP calculation
        if n_neighbors % 2 != 0:
            n_neighbors += 1
            if self._verbose:
                print(f"Warning: Adjusted num_neighbors to {n_neighbors} (must be even for CSP)")

        # Generate periodic images more efficiently - only if needed for boundary atoms
        # For most atoms, neighbors are likely within the main domain
        offsets = np.array([[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)])
        periodic_images = np.vstack([pos + offset * self._domain_size for offset in offsets])

        # Create KDTree for efficient neighbor search
        tree = cKDTree(periodic_images)

        # Pre-compute triangular indices for efficiency (avoid recomputation in loop)
        triu_indices = np.triu_indices(n_neighbors, k=1)
        n_pairs = n_neighbors // 2
        
        # Vectorized neighbor finding for all atoms at once
        distances, indices = tree.query(pos, k=n_neighbors + 1)  # +1 to exclude self
        neighbor_indices = indices[:, 1:]  # Exclude the atom itself
        
        # Get all neighbor positions for all atoms at once
        all_neighbor_positions = periodic_images[neighbor_indices]  # Shape: (n_atoms, n_neighbors, 3)
        
        # Calculate relative positions for all atoms at once
        pos_expanded = pos[:, np.newaxis, :]  # Shape: (n_atoms, 1, 3)
        neighbors_rel = all_neighbor_positions - pos_expanded  # Shape: (n_atoms, n_neighbors, 3)
        
        # Vectorized CSP calculation for all atoms at once
        # Create pairwise sums for all atoms simultaneously
        neighbors_i   = neighbors_rel[:, :, np.newaxis, :]  # Shape: (n_atoms, n_neighbors, 1, 3)
        neighbors_j   = neighbors_rel[:, np.newaxis, :, :]  # Shape: (n_atoms, 1, n_neighbors, 3)
        pairwise_sums = neighbors_i + neighbors_j           # Shape: (n_atoms, n_neighbors, n_neighbors, 3)
        
        # Compute squared magnitudes for all pairs, all atoms
        pairwise_contributions = np.sum(pairwise_sums**2, axis=3)  # Shape: (n_atoms, n_neighbors, n_neighbors)
        
        # Extract upper triangular parts for all atoms at once
        pair_contributions_all = pairwise_contributions[:, triu_indices[0], triu_indices[1]]  # Shape: (n_atoms, n_unique_pairs)
        
        # Get the N/2 smallest contributions for each atom
        smallest_contributions = np.partition(pair_contributions_all, n_pairs - 1, axis=1)[:, :n_pairs]
        csp = np.sum(smallest_contributions, axis=1)

        # Normalize CSP values to the range [0, 1] if requested
        if normalize_csp:
            csp_min = np.min(csp)
            csp_max = np.max(csp)
            if csp_max > csp_min:
                csp = (csp - csp_min) / (csp_max - csp_min)
            else:
                csp = np.zeros_like(csp)

        if self._verbose:
            tend = time.time()
            print(f"Time to evaluate CSP for {pos.shape[0]} atoms: {tend-tstart:.3f} s")
            print(f"   Using {n_neighbors} neighbors for {self._struct} structure")
            print(f"   CSP range: [{np.min(csp):.6f}, {np.max(csp):.6f}]")
            print(f"   Mean CSP:   {np.mean(csp):.6f}")

        return csp

# =====================================================================================
