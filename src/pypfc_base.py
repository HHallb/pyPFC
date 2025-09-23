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
import datetime
import torch
import time
from scipy.spatial import cKDTree
from scipy.ndimage import zoom
from skimage import measure
from scipy import ndimage as ndi
from pypfc_grid import setup_grid

class setup_base(setup_grid):

    def __init__(self, domain_size, ndiv, config):

        # Initiate the inherited grid class
        # =================================
        super().__init__(domain_size, ndiv)

        # Set the data types
        self._struct                  = config['struct']
        self._alat                    = config['alat']
        self._sigma                   = config['sigma']
        self._npeaks                  = config['npeaks']
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
        k    = np.zeros(npoints, dtype=self._dtype_cpu)

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

        k2_d = torch.from_numpy(k2[:,:,:self._nz_half]).to(self._device)
        k2_d = k2_d.to(dtype=self._dtype_gpu)
        k2_d = k2_d.contiguous()

        return k2_d

    # =====================================================================================

    def get_integrated_field_in_volume(self, field, limits):
        '''
        PURPOSE
            Integrate a field variable within a defined volume, defined on a fixed Cartesian 3D grid.

        INPUT
            field       Field to be integrated, [nx x ny x nz]
            limits      Spatial integration limits, [6]:
                            limits = [xmin xmax ymin ymax zmin zmax]

        OUTPUT
            result      Result of the integration

        Last revision:
        H. Hallberg 2024-09-16
        '''

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

    def get_field_average_along_axis(self, field, axis):
        '''
        PURPOSE
            Evaluate the mean value of a field variable along a certain axis,
            defined on a fixed Cartesian 3D grid.

        INPUT
            field       Field to be integrated, [nx x ny x nz]
            axis        Axis to integrate along: 'x', 'y' or 'z'

        OUTPUT
            result      Result of the integration

        Last revision:
        H. Hallberg 2024-09-17
        '''

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

    def get_integrated_field_along_axis(self, field, axis):
        '''
        PURPOSE
            Integrate a field variable along a certain axis, defined on a fixed Cartesian 3D grid.

        INPUT
            field       Field to be integrated, [nx x ny x nz]
            axis        Axis to integrate along: 'x', 'y' or 'z'

        OUTPUT
            result      Result of the integration

        Last revision:
        H. Hallberg 2024-09-16
        '''

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

    def interpolate_atoms(self, intrpPos, pos, values, num_nnb=8, power=2):
        """
        PURPOSE
            Interpolate values at given positions in a 3D periodic domain using inverse distance weighting.
            interpolated_value = Σ(wi x vi) / Σ(wi)
            where wi = 1 / (di^power), di is the distance to the i-th nearest neighbor, and
            vi is the value at that neighbor.

        INPUT
            intrpPos        Array of shape [n_intrp, 3] containing the
                            3D coordinates of the particles to be interpolated
            pos             Array of shape [n_particles, 3] containing the 3D coordinates of
                            the particles among which to interpolate
            values          Array of shape [n_particles] containing the values to be interpolated
            num_nnb         Number of nearest neighbors to use for interpolation
            power           Power for inverse distance weighting (default is 2)

        OUTPUT
            interpVal       Interpolated values at given positions in
                            intrpPos [n_interp]

        Last revision:
            H. Hallberg 2025-08-03
        """

        n_interp = intrpPos.shape[0]
        interpVal = np.zeros(n_interp, dtype=self._dtype_cpu)

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
        distances, indices = tree.query(intrpPos, k=k_neighbors)
        
        # Handle exact matches (distance < epsilon)
        exact_matches = distances[:, 0] < epsilon
        
        # Initialize output array
        interpVal = np.zeros(n_interp, dtype=self._dtype_cpu)
        
        # For exact matches, use the nearest neighbor value directly
        if np.any(exact_matches):
            interpVal[exact_matches] = values_periodic[indices[exact_matches, 0]]
        
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
            interpVal[non_exact] = weighted_sum / total_weight

        return interpVal

# =====================================================================================

    def interpolate_density_maxima(self, den, ene=None, pf=None):
        '''
        PURPOSE
            Find the coordinates of the maxima in the density field (='atom' positions)
            The domain is assumed to be defined such that all maxima
            have coordinates (x,y,z) >= (0,0,0).
            The density and, optionally, the energy and the phase field value(s)
            at the individual maxima are interpolated too.

        INPUT
            den                     Density field, [nx, ny, nz]
            ene                     Energy field, [nx, ny, nz]
            pf                      Optional list of phase fields, [nx, ny, nz]

        OUTPUT
            atom_coord              Coordinates of the density maxima, [nmaxima x 3]
            atom_data               Interpolated field values at the density maxima,
                                    [nmaxima x 2+nPhaseFields].
                                    The columns hold point data in the order:
                                    [den ene pf1 pf2 ... pfN]

        Last revision:
        H. Hallberg 2025-09-20
        '''

        if self._verbose: tstart = time.time()

        # Grid
        dx,dy,dz = self._ddiv

        size = 1 + 2 * self._density_interp_order
        footprint = np.ones((size, size, size))
        footprint[self._density_interp_order, self._density_interp_order, self._density_interp_order] = 0

        filtered = ndi.maximum_filter(den, footprint=footprint, mode='wrap')

        mask_local_maxima = den > filtered
        coords = np.asarray(np.where(mask_local_maxima),dtype=self._dtype_cpu).T

        # ndi.maximum_filter works in voxel coordinates, convert to physical coordinates
        coords[:,0] *= dx
        coords[:,1] *= dy
        coords[:,2] *= dz

        # Filter maxima based on density threshold
        max_den = np.max(den)
        valid_maxima = den[mask_local_maxima] >= (self._density_threshold * max_den)
        coords = coords[valid_maxima]

        denpos = den[mask_local_maxima][valid_maxima]
        if ene is not None:
            enepos = ene[mask_local_maxima][valid_maxima]

        # Merge maxima within the merge_distance
        if self._density_merge_distance > 0.0 and len(coords) > 0:
            tree = cKDTree(coords)
            clusters = tree.query_ball_tree(tree, r=self._density_merge_distance)
            unique_clusters = []
            seen = set()
            for cluster in clusters:
                cluster = tuple(sorted(cluster))
                if cluster not in seen:
                    seen.add(cluster)
                    unique_clusters.append(cluster)

            merged_coords = []
            merged_denpos = []
            merged_enepos = [] if ene is not None else None
            for cluster in unique_clusters:
                cluster_coords = coords[list(cluster)]
                cluster_denpos = denpos[list(cluster)]
                merged_coords.append(np.mean(cluster_coords, axis=0))
                merged_denpos.append(np.mean(cluster_denpos))
                if ene is not None:
                    cluster_enepos = enepos[list(cluster)]
                    merged_enepos.append(np.mean(cluster_enepos))
            atom_coord = np.array(merged_coords)
            denpos = np.array(merged_denpos)
            if ene is not None:
                enepos = np.array(merged_enepos)
        else:
            atom_coord = coords
            # denpos and enepos are already set above
            # Only set enepos if ene is not None
            if ene is not None:
                enepos = enepos

        # Handle phase field(s), either as a list of fields or as a single field
        if pf is not None:
            # If pf is a single array, wrap it in a list
            if isinstance(pf, np.ndarray) and pf.ndim == 3:
                pf_list = [pf]
            else:
                pf_list = list(pf)
            nPf = len(pf_list)
            pfpos = np.zeros((coords.shape[0], nPf), dtype=self._dtype_cpu)
            for pfNr, phaseField in enumerate(pf_list):
                pfpos[:, pfNr] = phaseField[mask_local_maxima][valid_maxima][:coords.shape[0]]
            if ene is not None:
                atom_data = np.hstack((denpos[:, None], enepos[:, None], pfpos))
            else:
                atom_data = np.hstack((denpos[:, None], pfpos))
        else:
            if ene is not None:
                atom_data = np.hstack((denpos[:, None], enepos[:, None]))
            else:
                atom_data = denpos[:, None]

        if self._verbose:
            tend = time.time()
            print(f'Time to interpolate density maxima: {tend-tstart:.3f} s')

        return atom_coord, atom_data
    
# =====================================================================================

    def get_phase_field_contour(self, pf, pf_zoom=1.0, evaluate_volume=True):
        """
        PURPOSE
            Find the iso-contour surface of a 3D phase field using marching cubes
        
        INPUT
            pf                  Phase field, [nx, ny, nz]
            pf_zoom             Zoom factor for coarsening/refinement
            evaluate_volume     If True, also evaluate the volume enclosed by the iso-surface

        OUTPUT
            verts               Vertices of the iso-surface triangulation
            faces               Surface triangulation topology
            volume              (optional) Volume enclosed by the iso-surface

        Last revision:
            H. Hallberg 2025-09-06
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
        
        rlv = np.array(structures[struct], dtype=self._dtype_cpu)
        rlv = rlv * (2*np.pi/alat)

        return rlv

# =====================================================================================

    def evaluate_reciprocal_planes(self):
        '''
        PURPOSE
            Establish the reciprocal vectors/planes for a particular crystal structure.

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

        kPlane   = np.zeros(self._npeaks, dtype=self._dtype_cpu)
        denPlane = np.zeros(self._npeaks, dtype=self._dtype_cpu)
        nPlane   = np.zeros(self._npeaks, dtype=int)

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
            kPlane   = kpl[0:self._npeaks]
            nPlane   = pl[0:self._npeaks]
            denPlane = denpl[0:self._npeaks]
        else:
            raise ValueError(f'Not enough peaks defined, npeaks={self._npeaks}')

        return kPlane, nPlane, denPlane

# =====================================================================================

    def evaluate_C2_d(self):
        """
        PURPOSE
            Establish the two-point correlation function for a particular crystal structure.

        INPUT

        OUTPUT
            C2_d          Two-point pair correlation function [nx, ny, nz/2+1] (on the device)

        Last revision:
        H. Hallberg 2025-09-22
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