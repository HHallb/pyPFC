# Copyright (C) 2026 Håkan Hallberg
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE file for full license text

import numpy as np
import datetime
import torch
import time
from fractions import Fraction
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
            if torch.get_num_interop_threads() != self._set_num_interop_threads:
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

    def interpolate_density_maxima(self, den: Union[np.ndarray, torch.Tensor], ene: Optional[Union[np.ndarray, torch.Tensor]] = None, fields: Optional[Union[np.ndarray, torch.Tensor]] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Find density field maxima and interpolate atomic positions and properties.
        
        Identifies local maxima in the density field as atomic positions and performs
        high-order interpolation to obtain sub-grid precision coordinates. Also 
        interpolates associated field values at the atomic positions.
        
        Parameters
        ----------
        den : ndarray of float, shape (nx,ny,nz)
            Density field from PFC simulation.
        ene : ndarray of float, shape (nx,ny,nz), optional
            Energy field for interpolation at atomic positions.
        fields : list of ndarray, optional
            List of additional fields for interpolation at atomic positions.
            Each array should have shape (nx,ny,nz).
            
        Returns
        -------
        atom_coord : ndarray of float, shape (n_maxima,3)
            Interpolated coordinates of density maxima (atomic positions).
        atom_data : ndarray of float, shape (n_maxima, 2+n_phase_fields)
            Interpolated field values at atomic positions.
            Columns: [density, energy, field1, field2, ..., fieldN]
            
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
        if fields is not None:
            if isinstance(fields, np.ndarray) and fields.ndim == 3:
                field_list = [fields]
            else:
                field_list = list(fields)
            
            n_fields = len(field_list)
            n_atoms = len(atom_coord)
            
            # Extract field values efficiently
            field_pos = np.empty((n_atoms, n_fields), dtype=self._dtype_cpu)
            for field_idx, phase_field in enumerate(field_list):
                if self._density_merge_distance > 0.0 and n_maxima != n_atoms:
                    # Merging occurred - use first value (approximate)
                    field_values = phase_field[maxima_indices]
                    field_pos[:, field_idx] = field_values[:n_atoms]
                else:
                    # No merging - direct extraction
                    field_pos[:, field_idx] = phase_field[maxima_indices]
            
            # Assemble final data array efficiently
            if ene is not None:
                atom_data = np.column_stack((denpos, enepos, field_pos))
            else:
                atom_data = np.column_stack((denpos, field_pos))
        else:
            # No additional fields - simpler assembly
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

    def interpolate_gb_from_phase_field(self, pf: Union[np.ndarray, torch.Tensor], search_direction: str = 'x') -> np.ndarray:
        """
        Interpolate grain-boundary points from a phase-field iso-contour.
        
        The GB position is defined as the first crossing of the iso-level
        ``pf_iso_level`` along the selected search direction. The search
        is periodic along the sweep axis, so the segment connecting the last and
        first grid point is also considered.
        
        Parameters
        ----------
        pf : ndarray of float, shape (nx, ny, nz)
            3D phase field data for GB interpolation.
        search_direction : str, optional
            Direction along which to search for the GB. Options: 'x', '-x', 'y', '-y', 'z', '-z'.
            
        Returns
        -------
        gb_point_coords : ndarray of float, shape (n_points, 3)
            Coordinates of grain-boundary points. For each line in the plane
            perpendicular to the search axis, one point is returned. If no
            crossing is found for a line, the coordinate in the search axis is
            NaN.
            
        Raises
        ------
        ValueError
            If search direction or phase-field shape is invalid.
        """
        
        if torch.is_tensor(pf):
            pf_np = pf.detach().cpu().numpy()
        else:
            pf_np = np.asarray(pf)

        expected_shape = (self._nx, self._ny, self._nz)
        if pf_np.shape != expected_shape:
            raise ValueError(f"Expected pf shape {expected_shape}, got {pf_np.shape}")

        direction = search_direction.strip().lower()
        if direction not in ('x', '-x', 'y', '-y', 'z', '-z'):
            raise ValueError(f"Unsupported search direction: {search_direction}")

        axis = {'x': 0, '-x': 0, 'y': 1, '-y': 1, 'z': 2, '-z': 2}[direction]
        forward = not direction.startswith('-')

        n_axis = expected_shape[axis]
        other_axes = [ax for ax in (0, 1, 2) if ax != axis]
        n0 = expected_shape[other_axes[0]]
        n1 = expected_shape[other_axes[1]]

        gb_point_coords = np.full((n0 * n1, 3), np.nan, dtype=self._dtype_cpu)
        tol = 1e-14

        for i0 in range(n0):
            for i1 in range(n1):
                point_nr = i0 * n1 + i1

                line_selector = [slice(None), slice(None), slice(None)]
                line_selector[other_axes[0]] = i0
                line_selector[other_axes[1]] = i1
                line = pf_np[tuple(line_selector)]

                coord = np.array([
                    np.nan,
                    np.nan,
                    np.nan
                ], dtype=self._dtype_cpu)
                coord[other_axes[0]] = i0 * self._ddiv[other_axes[0]]
                coord[other_axes[1]] = i1 * self._ddiv[other_axes[1]]

                k_values = range(n_axis) if forward else range(n_axis - 1, -1, -1)
                hit_found = False

                for k_a in k_values:
                    if forward:
                        k_b = (k_a + 1) % n_axis
                    else:
                        k_b = (k_a - 1) % n_axis

                    pf_a = line[k_a]
                    pf_b = line[k_b]
                    da = pf_a - self._pf_iso_level
                    db = pf_b - self._pf_iso_level

                    if abs(da) < tol:
                        t = 0.0
                    elif abs(db) < tol:
                        t = 1.0
                    elif da * db < 0.0:
                        t = (self._pf_iso_level - pf_a) / (pf_b - pf_a)
                    else:
                        continue

                    if forward:
                        k_iso = (k_a + t) % n_axis
                    else:
                        k_iso = (k_a - t) % n_axis

                    coord[axis] = k_iso * self._ddiv[axis]
                    hit_found = True
                    break

                if hit_found:
                    gb_point_coords[point_nr] = coord
                else:
                    # Keep transverse coordinates even if no crossing was found.
                    gb_point_coords[point_nr, other_axes[0]] = coord[other_axes[0]]
                    gb_point_coords[point_nr, other_axes[1]] = coord[other_axes[1]]

        return gb_point_coords

# =====================================================================================

    def minimum_periodic_domain_single_crystal(self, orientation: np.ndarray, alat: float = 1.0, struct: str = 'FCC', target_ddiv: float = 0.125) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the minimum periodic 3D domain for a single oriented crystal.

        The function estimates the smallest axis-aligned domain that is periodic
        for a crystal whose orientation is given by a rotation matrix.
        The returned grid divisions are chosen so the grid spacing in each direction
        does not exceed ``target_ddiv`` and the divisions remain even.

        Parameters
        ----------
        orientation : ndarray of float, shape (3, 3)
            Crystal orientation matrix that maps the crystal frame to the lab frame.
            This should be a proper rotation matrix (orthonormal with determinant +1).
        alat : float, optional
            Lattice parameter. Default is 1.0.
        struct : str, optional
            Crystal structure. Options: 'FCC', 'BCC', 'SC'. Default is 'FCC'.
        target_ddiv : float, optional
            Maximum allowed grid spacing. Default is 0.125.

        Returns
        -------
        min_domain_size : ndarray of float, shape (3,)
            Minimum periodic domain dimensions.
        min_ndiv : ndarray of int, shape (3,)
            Minimum number of divisions for the periodic domain.
        ddiv : ndarray of float, shape (3,)
            Actual grid spacing in each direction, which does not exceed ``target_ddiv``.

        Raises
        ------
        ValueError
            If orientation is invalid, the structure is unsupported, or a periodic
            domain cannot be determined.
        """

        orientation = np.asarray(orientation, dtype=self._dtype_cpu)
        if orientation.shape != (3, 3):
            raise ValueError(f"orientation must have shape (3, 3), got {orientation.shape}")
        if alat <= 0.0:
            raise ValueError(f"alat must be positive, got alat={alat}")
        if target_ddiv <= 0.0:
            raise ValueError(f"target_ddiv must be positive, got target_ddiv={target_ddiv}")

        # Guard against malformed orientation matrices to avoid silent geometry errors.
        if not np.allclose(orientation.T @ orientation, np.eye(3), atol=1e-10, rtol=0.0):
            raise ValueError("orientation must be orthonormal (R.T @ R = I).")
        if not np.isclose(np.linalg.det(orientation), 1.0, atol=1e-10, rtol=0.0):
            raise ValueError("orientation must have determinant +1.")

        struct_u = struct.upper()
        if struct_u == 'SC':
            primitive_vectors = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=self._dtype_cpu)
        elif struct_u == 'BCC':
            primitive_vectors = np.array([
                [0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ], dtype=self._dtype_cpu)
        elif struct_u == 'FCC':
            primitive_vectors = np.array([
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ], dtype=self._dtype_cpu)
        else:
            raise ValueError(f"Unsupported crystal structure: {struct}")

        basis = (orientation @ (primitive_vectors * alat).T).T

        def _lcm(a: int, b: int) -> int:
            return abs(a * b) // np.gcd(a, b)

        def _reduce_integer_vector(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=int)
            nonzero = values[values != 0]
            if nonzero.size == 0:
                return values
            divisor = int(np.abs(nonzero[0]))
            for value in nonzero[1:]:
                divisor = int(np.gcd(divisor, int(np.abs(value))))
            if divisor > 1:
                values = values // divisor
            return values

        def _period_along_axis(axis_index: int) -> float:
            # If t = n @ basis is parallel to a lab axis, then n = L * w,
            # where w is the corresponding row of inv(basis). We search for the
            # smallest L that makes all components of n integers.
            inv_basis = np.linalg.inv(basis)
            w = inv_basis[axis_index, :]

            tol = 1e-12
            nonzero = np.where(np.abs(w) > tol)[0]
            if nonzero.size == 0:
                raise ValueError(f"Could not determine a periodic translation along axis {axis_index}.")

            ref_idx = nonzero[np.argmax(np.abs(w[nonzero]))]
            ref = w[ref_idx]
            if np.abs(ref) < tol:
                raise ValueError(f"Could not determine a periodic translation along axis {axis_index}.")

            max_denominator = 4096
            for _ in range(4):
                ratios = w / ref
                rational_parts = []
                for ratio in ratios:
                    if np.abs(ratio) < tol:
                        rational_parts.append(Fraction(0, 1))
                    else:
                        rational_parts.append(Fraction(float(ratio)).limit_denominator(max_denominator))

                common_den = 1
                for frac in rational_parts:
                    common_den = _lcm(common_den, frac.denominator)

                integer_direction = np.array([
                    frac.numerator * (common_den // frac.denominator)
                    for frac in rational_parts
                ], dtype=int)
                integer_direction = _reduce_integer_vector(integer_direction)

                scale = integer_direction[ref_idx] / ref
                candidate = scale * w
                if np.allclose(candidate, np.rint(candidate), atol=1e-8, rtol=0.0):
                    return float(np.abs(scale))

                max_denominator *= 4

            raise ValueError(
                f"Could not find a commensurate periodic length along axis {axis_index} for struct={struct_u} and the provided orientation matrix."
            )

        min_domain_size = np.empty(3, dtype=self._dtype_cpu)
        min_ndiv = np.empty(3, dtype=int)

        for axis_index in range(3):
            min_domain_size[axis_index] = _period_along_axis(axis_index)
            ndiv_axis = int(np.ceil(min_domain_size[axis_index] / target_ddiv))
            if ndiv_axis < 2:
                ndiv_axis = 2
            if ndiv_axis % 2 != 0:
                ndiv_axis += 1
            min_ndiv[axis_index] = ndiv_axis

        return min_domain_size, min_ndiv, min_domain_size / min_ndiv

# =====================================================================================

    def minimum_periodic_domain_bicrystal(self, axis: np.ndarray, angle: float, alat: float = 1.0, struct: str = 'FCC', target_ddiv: float = 0.125, gb_type: str = 'tilt') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find the minimum periodic 3D domain for a given crystal structure and
        grain boundary configuration.

        The function estimates the smallest axis-aligned domain that is periodic
        after rotating a crystal by ``angle`` around the GB axis ``[h k l]``.
        The returned grid divisions are chosen so the grid spacing in each direction
        does not exceed ``target_ddiv`` and the divisions remain even.

        Parameters
        ----------
        axis : ndarray of float, shape (3,)
            Grain-boundary axis given by Miller indices ``[h k l]``.
            For ``gb_type='tilt'``, this is the tilt axis.
            For ``gb_type='twist'``, this is the twist axis (parallel to GB normal).
        angle : float
            Rotation angle in radians.
        alat : float, optional
            Lattice parameter. Default is 1.0.
        struct : str, optional
            Crystal structure. Options: 'FCC', 'BCC', 'SC'. Default is 'FCC'.
        target_ddiv : float, optional
            Maximum allowed grid spacing. Default is 0.125.
        gb_type : str, optional
            Type of grain boundary. Options: 'tilt', 'twist'. Default is 'tilt'.

        Returns
        -------
        min_domain_size : ndarray of float, shape (3,)
            Minimum periodic domain dimensions.
        min_ndiv : ndarray of int, shape (3,)
            Minimum number of divisions for the periodic domain.
        ddiv : ndarray of float, shape (3,)
            Actual grid spacing in each direction, which does not exceed ``target_ddiv``.
        g1 : ndarray of float, shape (3, 3)
            Rotation matrix for crystal 1, rotated by ``+angle`` around ``[h k l]``.
        g2 : ndarray of float, shape (3, 3)
            Rotation matrix for crystal 2, rotated by ``-angle`` around ``[h k l]``.

        Raises
        ------
        ValueError
            If the axis is invalid, the structure is unsupported, or a periodic
            domain cannot be determined.
        """

        axis = np.asarray(axis, dtype=self._dtype_cpu).reshape(3)
        axis_norm = np.linalg.norm(axis)
        if axis_norm == 0.0:
            raise ValueError("Rotation axis must be non-zero.")
        if alat <= 0.0:
            raise ValueError(f"alat must be positive, got alat={alat}")
        if target_ddiv <= 0.0:
            raise ValueError(f"target_ddiv must be positive, got target_ddiv={target_ddiv}")

        gb_type_u = gb_type.lower()
        if gb_type_u not in ('tilt', 'twist'):
            raise ValueError(f"Unsupported gb_type: {gb_type}. Use 'tilt' or 'twist'.")

        struct_u = struct.upper()
        if struct_u == 'SC':
            primitive_vectors = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=self._dtype_cpu)
        elif struct_u == 'BCC':
            primitive_vectors = np.array([
                [0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ], dtype=self._dtype_cpu)
        elif struct_u == 'FCC':
            primitive_vectors = np.array([
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ], dtype=self._dtype_cpu)
        else:
            raise ValueError(f"Unsupported crystal structure: {struct}")

        # Rotate the crystal basis using Rodrigues' formula.
        # For both symmetric tilt and symmetric twist boundaries, the two crystals
        # are represented by equal and opposite rotations around ``axis``.
        axis_unit = axis / axis_norm
        ux, uy, uz = axis_unit
        c = np.cos(angle)
        s = np.sin(angle)
        one_c = 1.0 - c
        rot = np.array([
            [c + ux * ux * one_c,      ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
            [uy * ux * one_c + uz * s,  c + uy * uy * one_c,      uy * uz * one_c - ux * s],
            [uz * ux * one_c - uy * s,  uz * uy * one_c + ux * s, c + uz * uz * one_c     ],
        ], dtype=self._dtype_cpu)

        basis = (rot @ (primitive_vectors * alat).T).T

        # Rotation matrices for the two crystals (±angle around the GB axis [h k l]).
        g1 = rot.copy()
        g2 = np.array([
            [c + ux * ux * one_c,      ux * uy * one_c + uz * s, ux * uz * one_c - uy * s],
            [uy * ux * one_c - uz * s,  c + uy * uy * one_c,      uy * uz * one_c + ux * s],
            [uz * ux * one_c + uy * s,  uz * uy * one_c - ux * s, c + uz * uz * one_c     ],
        ], dtype=self._dtype_cpu)

        def _lcm(a: int, b: int) -> int:
            return abs(a * b) // np.gcd(a, b)

        def _reduce_integer_vector(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=int)
            nonzero = values[values != 0]
            if nonzero.size == 0:
                return values
            divisor = int(np.abs(nonzero[0]))
            for value in nonzero[1:]:
                divisor = int(np.gcd(divisor, int(np.abs(value))))
            if divisor > 1:
                values = values // divisor
            return values

        def _period_along_axis(axis_index: int) -> float:
            # If t = n @ basis is parallel to a lab axis, then n = L * w,
            # where w is the corresponding row of inv(basis). We search for the
            # smallest L that makes all components of n integers.
            inv_basis = np.linalg.inv(basis)
            w = inv_basis[axis_index, :]

            tol = 1e-12
            nonzero = np.where(np.abs(w) > tol)[0]
            if nonzero.size == 0:
                raise ValueError(f"Could not determine a periodic translation along axis {axis_index}.")

            ref_idx = nonzero[np.argmax(np.abs(w[nonzero]))]
            ref = w[ref_idx]
            if np.abs(ref) < tol:
                raise ValueError(f"Could not determine a periodic translation along axis {axis_index}.")

            max_denominator = 4096
            for _ in range(4):
                ratios = w / ref
                rational_parts = []
                for ratio in ratios:
                    if np.abs(ratio) < tol:
                        rational_parts.append(Fraction(0, 1))
                    else:
                        rational_parts.append(Fraction(float(ratio)).limit_denominator(max_denominator))

                common_den = 1
                for frac in rational_parts:
                    common_den = _lcm(common_den, frac.denominator)

                integer_direction = np.array([
                    frac.numerator * (common_den // frac.denominator)
                    for frac in rational_parts
                ], dtype=int)
                integer_direction = _reduce_integer_vector(integer_direction)

                scale = integer_direction[ref_idx] / ref
                candidate = scale * w
                if np.allclose(candidate, np.rint(candidate), atol=1e-8, rtol=0.0):
                    return float(np.abs(scale))

                max_denominator *= 4

            raise ValueError(
                f"Could not find a commensurate periodic length along axis {axis_index} for struct={struct_u} and angle={np.rad2deg(angle):.3f} degrees."
            )

        min_domain_size = np.empty(3, dtype=self._dtype_cpu)
        min_ndiv = np.empty(3, dtype=int)

        for axis_index in range(3):
            min_domain_size[axis_index] = _period_along_axis(axis_index)
            ndiv_axis = int(np.ceil(min_domain_size[axis_index] / target_ddiv))
            if ndiv_axis < 2:
                ndiv_axis = 2
            if ndiv_axis % 2 != 0:
                ndiv_axis += 1
            min_ndiv[axis_index] = ndiv_axis

        return min_domain_size, min_ndiv, min_domain_size / min_ndiv, g1, g2

# =====================================================================================

    def get_csl_config(self, axis: Union[List[int], np.ndarray], max_csl: int, struct: str = 'FCC', gb_type: str = 'tilt') -> List[Dict[str, Any]]:
        """
        Generate candidate CSL configurations for symmetric tilt or twist
        grain boundaries.

        The returned list contains candidate misorientation angles and associated
        GB normals ``[u v w]`` for a specified GB axis ``[h k l]``. The list can be used to pick
        a commensurate angle and then call ``minimum_periodic_domain`` with
        ``angle = theta/2``.

        Parameters
        ----------
        axis : array_like of int, shape (3,)
            Boundary-defining axis ``[h k l]``.
            For ``gb_type='tilt'``, this is the tilt axis.
            For ``gb_type='twist'``, this is the twist axis (parallel to GB normal).
        max_csl : int
            Maximum CSL number (Sigma) to include.
        struct : str, optional
            Crystal structure. Options: 'FCC', 'BCC', 'SC'.
        gb_type : str, optional
            Type of grain boundary. Options: 'tilt', 'twist'. Default is 'tilt'.

        Returns
        -------
        csl_list : list of dict
            Each item contains:
            - 'theta' (radians)
            - 'theta_deg' (degrees)
            - 'gb_normal' (list [u, v, w])
            - 'csl' (Sigma)
            - 'pair' (integer generator pair (m, n))
            - 'axis' (list [h, k, l])
            - 'gb_type' ('tilt' or 'twist')
        """

        if max_csl < 1:
            raise ValueError(f"max_csl must be >= 1, got max_csl={max_csl}")

        struct_u = struct.upper()
        if struct_u not in ('FCC', 'BCC', 'SC'):
            raise ValueError(f"Unsupported crystal structure: {struct}")

        gb_type_u = gb_type.lower()
        if gb_type_u not in ('tilt', 'twist'):
            raise ValueError(f"Unsupported gb_type: {gb_type}. Use 'tilt' or 'twist'.")

        axis = np.asarray(axis, dtype=int).reshape(3)
        if np.all(axis == 0):
            raise ValueError("axis must be non-zero.")

        def _reduce_int_vector(vec: np.ndarray) -> np.ndarray:
            vec = np.asarray(vec, dtype=int)
            nonzero = vec[vec != 0]
            if nonzero.size == 0:
                return vec
            g = int(np.abs(nonzero[0]))
            for val in nonzero[1:]:
                g = int(np.gcd(g, int(np.abs(val))))
            if g > 1:
                vec = vec // g
            for val in vec:
                if val != 0:
                    if val < 0:
                        vec = -vec
                    break
            return vec

        axis = _reduce_int_vector(axis)

        # Build an integer basis in the plane perpendicular to the tilt axis.
        trial_basis = [
            np.array([1, 0, 0], dtype=int),
            np.array([0, 1, 0], dtype=int),
            np.array([0, 0, 1], dtype=int),
        ]

        p = None
        for e in trial_basis:
            cand = np.cross(axis, e)
            if np.any(cand != 0):
                p = _reduce_int_vector(cand)
                break
        if p is None:
            raise ValueError(f"Could not construct a perpendicular basis for axis={axis.tolist()}")

        q = _reduce_int_vector(np.cross(axis, p))
        if np.all(q == 0):
            raise ValueError(f"Could not construct a second perpendicular basis vector for axis={axis.tolist()}")

        max_m = int(np.ceil(np.sqrt(2 * max_csl))) + 1
        csl_list = []
        seen = set()

        for m in range(1, max_m + 1):
            for n in range(1, max_m + 1):
                if np.gcd(m, n) != 1:
                    continue

                sigma_raw = m * m + n * n
                if struct_u in ('FCC', 'BCC') and (m % 2 == 1 and n % 2 == 1):
                    sigma = sigma_raw // 2
                else:
                    sigma = sigma_raw

                if sigma > max_csl:
                    continue

                theta = 2.0 * np.arctan2(n, m)
                if gb_type_u == 'tilt':
                    normal = _reduce_int_vector(m * p + n * q)
                else:
                    # For symmetric twist boundaries, the GB normal is parallel
                    # to the twist axis.
                    normal = axis.copy()

                key = (int(sigma), tuple(normal.tolist()), round(float(theta), 12))
                if key in seen:
                    continue
                seen.add(key)

                csl_list.append({
                    'theta': float(theta),
                    'gb_normal': normal.tolist(),
                    'csl': int(sigma),
                    'pair': (int(m), int(n)),
                    'axis': axis.tolist(),
                    'gb_type': gb_type_u,
                })

        csl_list.sort(key=lambda item: (item['theta'], item['csl']))

        return csl_list

# =====================================================================================

    def expand_minimum_domain(self, min_domain_size: np.ndarray, min_ndiv: np.ndarray, target_domain_size: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Repeat a minimum domain size, required for 3D periodicity, to match a
        target domain size.

        The returned grid parameters are adjusted to provide the closest possible
        match to the target domain size, without being smaller.

        Parameters
        ----------
        min_domain_size : ndarray of float, shape (3,)
            Minimum periodic domain size in each direction (e.g. output from
            ``minimum_periodic_domain``).
        min_ndiv : ndarray of int, shape (3,)
            Minimum number of grid divisions for ``min_domain_size`` (e.g. output
            from ``minimum_periodic_domain``).
        target_domain_size : ndarray of float, shape (3,)
            Desired domain size in each direction.

        Returns
        -------
        domain_size_exp : ndarray of float, shape (3,)
            Adjusted domain size in each direction.
        ndiv_exp : ndarray of int, shape (3,)
            Adjusted number of grid divisions in each direction.

        Raises
        ------
        ValueError
            If input shapes are invalid, if any target domain component is
            non-positive, or if any minimum domain/grid component is non-positive.
        """

        min_domain_size    = np.asarray(min_domain_size, dtype=self._dtype_cpu).reshape(3)
        min_ndiv           = np.asarray(min_ndiv, dtype=int).reshape(3)
        target_domain_size = np.asarray(target_domain_size, dtype=self._dtype_cpu).reshape(3)

        if np.any(min_domain_size <= 0.0):
            raise ValueError("min_domain_size must be positive in all directions.")
        if np.any(min_ndiv <= 0):
            raise ValueError("min_ndiv must be positive in all directions.")
        if np.any(target_domain_size <= 0.0):
            raise ValueError("target_domain_size must be positive in all directions.")

        # Repeat each minimum periodic cell enough times to avoid undershooting.
        nrep = np.ceil(target_domain_size / min_domain_size).astype(int)
        nrep = np.maximum(nrep, 1)

        domain_size_exp = min_domain_size * nrep
        ndiv_exp        = min_ndiv * nrep

        return domain_size_exp, ndiv_exp, nrep

# =====================================================================================

    def get_atom_bond_data(self, atom_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate min/max neighbor bond angles and bond lengths for each atom
        in a set of atoms.

        The neighbor search is performed with periodic boundary conditions in all
        three directions using the simulation domain size.

        Parameters
        ----------
        atom_coord : ndarray of float, shape (n_atoms, 3)
            Coordinates of the atoms.

        Returns
        -------
        bond_angles : ndarray of float, shape (2, n_atoms)
            Min/Max neighbor bond angles for each atom.
        bond_lengths : ndarray of float, shape (2, n_atoms)
            Min/Max neighbor bond lengths for each atom.
        """

        atom_coord = np.asarray(atom_coord, dtype=self._dtype_cpu)
        if atom_coord.ndim != 2 or atom_coord.shape[1] != 3:
            raise ValueError(f"Expected atom_coord shape (n_atoms, 3), got {atom_coord.shape}")

        n_atoms = atom_coord.shape[0]
        bond_angles = np.zeros((2, n_atoms), dtype=self._dtype_cpu)
        bond_lengths = np.zeros((2, n_atoms), dtype=self._dtype_cpu)
        if n_atoms < 2:
            return bond_angles, bond_lengths

        # Determine the number of nearest neighbors based on crystal structure.
        nnb, _ = self.get_xtal_nearest_neighbors()
        n_neighbors = int(nnb[0])
        n_neighbors = min(n_neighbors, n_atoms - 1)
        if n_neighbors < 1:
            return bond_angles, bond_lengths

        # cKDTree periodic mode requires points wrapped to [0, boxsize).
        boxsize = np.asarray(self._domain_size, dtype=self._dtype_cpu)
        if np.any(boxsize <= 0.0):
            raise ValueError(f"Invalid periodic domain size: {boxsize}")
        coords = np.mod(atom_coord, boxsize)

        tree = cKDTree(coords, boxsize=boxsize)
        _, indices = tree.query(coords, k=n_neighbors + 1)

        # Neighbor vectors with minimum-image convention.
        neighbor_idx = indices[:, 1:]
        center = coords[:, None, :]
        neighbor = coords[neighbor_idx]
        vec = neighbor - center
        vec -= boxsize[None, None, :] * np.round(vec / boxsize[None, None, :])

        lengths = np.linalg.norm(vec, axis=2)
        bond_lengths[0, :] = np.min(lengths, axis=1)
        bond_lengths[1, :] = np.max(lengths, axis=1)

        if n_neighbors < 2:
            return bond_angles, bond_lengths

        # Vectorized pair-angle evaluation among all neighbor pairs.
        eps = np.finfo(self._dtype_cpu).eps
        unit = np.zeros_like(vec)
        nonzero = lengths > eps
        unit[nonzero] = vec[nonzero] / lengths[nonzero, None]

        cos_all = np.einsum('ijk,ilk->ijl', unit, unit)
        tri = np.triu_indices(n_neighbors, k=1)
        cos_pairs = np.clip(cos_all[:, tri[0], tri[1]], -1.0, 1.0)

        valid_pairs = nonzero[:, tri[0]] & nonzero[:, tri[1]]
        ang_pairs = np.arccos(cos_pairs)
        ang_pairs[~valid_pairs] = np.nan

        has_valid = np.any(valid_pairs, axis=1)
        if np.any(has_valid):
            bond_angles[0, has_valid] = np.nanmin(ang_pairs[has_valid], axis=1)
            bond_angles[1, has_valid] = np.nanmax(ang_pairs[has_valid], axis=1)

        return bond_angles, bond_lengths

# =====================================================================================


    def rotation_map_vector_to_x(self, n, eps=1.0e-14) -> np.ndarray:
        """
        Construct a rotation matrix A such that

            A @ n = e_x,

        where e_x = [1, 0, 0].

        Parameters
        ----------
        n : ndarray of float, shape (3,)
            Vector to be rotated to align with the x-axis.
        eps : float, optional
            Tolerance for numerical comparisons (default is 1.0e-14).

        Returns
        -------
        A : ndarray of float, shape (3, 3)
            Rotation matrix such that A @ n = e_x.
        """
        n = np.asarray(n, dtype=float)
        n = n / np.linalg.norm(n)

        ex = np.array([1.0, 0.0, 0.0])

        c = np.dot(n, ex)

        if c > 1.0 - eps:
            return np.eye(3)

        if c < -1.0 + eps:
            return np.array([
                [-1.0,  0.0,  0.0],
                [ 0.0,  1.0,  0.0],
                [ 0.0,  0.0, -1.0],
            ])

        v = np.cross(n, ex)
        s = np.linalg.norm(v)

        vx = np.array([
            [0.0,   -v[2],  v[1]],
            [v[2],   0.0, -v[0]],
            [-v[1], v[0],  0.0],
        ])

        A = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))

        return A

# =====================================================================================