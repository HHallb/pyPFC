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

import os
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
import time
import torch
from pypfc_base import setup_base
from scipy.ndimage import zoom
from skimage import measure
class setup_aux(setup_base):

    DEFAULTS = {
        'struct':                   'FCC',
        'sigma':                    0.0,
        'npeaks':                   2,
        'dtype_cpu':                np.double,
        'dtype_gpu':                torch.float64,
        'device_type':              'gpu',
        'device_number':            0,
        'verbose':                  False,
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
        subset_cfg = {k: cfg[k] for k in ['torch_threads', 'torch_threads_interop', 'device_number', 'device_type', 'dtype_cpu', 'dtype_gpu', 'verbose'] if k in cfg}
        super().__init__(ndiv, ddiv, config=subset_cfg)

        # Handle input arguments
        # ======================
        nx,ny,nz = self.get_ndiv()

        self._den                    = np.zeros((nx, ny, nz), dtype=cfg['dtype_cpu'])
        self._ene                    = np.zeros((nx, ny, nz), dtype=cfg['dtype_cpu'])
        self._struct                 = cfg['struct']
        self._sigma                  = cfg['sigma']
        self._npeaks                 = cfg['npeaks']
        self._density_interp_order   = cfg['density_interp_order']
        self._density_threshold      = cfg['density_threshold']
        self._density_merge_distance = cfg['density_merge_distance']
        self._pf_iso_level           = cfg['pf_iso_level']

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
        self._struct = struct

    def get_struct(self):
        return self._struct

    def set_density(self, den):
        self._den = den

    def get_density(self):
        return self._den

    def set_energy(self, ene):
        self._ene = ene

    def get_energy(self):
        return self._ene

    def get_ampl(self):
        return self._ampl

    def get_nlns(self):
        return self._nlns

    def set_sigma(self, sigma):
        self._sigma = sigma

    def get_sigma(self):
        return self._sigma

    def set_npeaks(self, npeaks):
        self._npeaks = npeaks

    def get_npeaks(self):
        return self._npeaks

# =====================================================================================

    def do_single_crystal(self, xtalRot, params=None, model=0):
        '''
        PURPOSE
            Define a centered crystal in a periodic 3D domain.
    
        INPUT
            xtalRot       Crystal orientation (rotation matrix): [3 x 3]
            params        List containing parameters for the single crystal model
            model         Density field layout:
                            0 = Spherical crystal
                            1 = A crystal extending throughout y and z, while only covering an interval in x
    
        OUTPUT
            density       Density field, real rank-3 array of size [nx x ny x nz]

        Last revision:
        H. Hallberg 2025-09-11
        '''

        # Grid
        nx,ny,nz = self._ndiv
        dx,dy,dz = self._ddiv
        Lx,Ly,Lz = self._ndiv*self._ddiv

        # Allocate output array
        density = np.full((nx, ny, nz), self._nlns[1], dtype=self._dtype_cpu)

        # Crystal orientations (passive rotations)
        Rot = xtalRot[:,:].T

        xc = np.linspace(0, (nx-1)*dx, nx)
        yc = np.linspace(0, (ny-1)*dy, ny)
        zc = np.linspace(0, (nz-1)*dz, nz)

        # Generate crystal
        Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')
        
        if model==0:
            xtalRadius = params[0]
            condition  = (np.sqrt((Xc-Lx/2)**2 + (Yc-Ly/2)**2 + (Zc-Lz/2)**2) <= xtalRadius)

        elif model==1:
            start_x   = params[0]
            end_x     = params[1]
            condition = (Xc >= start_x) & (Xc <= end_x)

        else:
            raise ValueError(f'Unsupported seed layout: model={model}')

        crd = np.array([Xc[condition], Yc[condition], Zc[condition]])
        density[condition] = self.generate_density_field(crd, Rot)

        return density

# =====================================================================================

    def do_bicrystal(self, xtalRot, params=None, liq_width=0.0, model=0):
        '''
        PURPOSE
            Define a centered crystal, embedded inside a matrix crystal, in
            a periodic 3D domain.
    
        INPUT
            xtalRot       Crystal orientations (rotation matrices): [3 x 3 x 2]
            params        List containing parameters for the bicrystal model
            liq_width     Width of the liquid band along the GB
            model         Density field layout:
                            0 = Cylindrical crystal, extending through z
                            1 = Spherical crystal
                            2 = Bicrystal with two planar grain boundaries, normal to x
    
        OUTPUT
            density       Density field, real rank-3 array of size [nx x ny x nz]

        Last revision:
        H. Hallberg 2025-09-17
        '''

        # Grid
        nx,ny,nz = self._ndiv
        dx,dy,dz = self._ddiv
        Lx,Ly,Lz = self._ndiv*self._ddiv

        # Allocate output array
        density = np.full((nx, ny, nz), self._nlns[1], dtype=self._dtype_cpu)

        # Crystal orientations (passive rotations)
        Rot0 = xtalRot[:,:,0].T
        Rot1 = xtalRot[:,:,1].T

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
            raise ValueError(f'Unsupported seed layout: model={model}')

        crd = np.array([Xc[condition0], Yc[condition0], Zc[condition0]])
        density[condition0] = self.generate_density_field(crd, Rot0)
        crd = np.array([Xc[condition1], Yc[condition1], Zc[condition1]])
        density[condition1] = self.generate_density_field(crd, Rot1)

        return density

# =====================================================================================

    def generate_density_field(self, crd, g):
        '''
        PURPOSE
            Define a 3D density field for (X)PFC modeling.

        INPUT
            crd           Point coordinates: [x,y,z]
            struct        Crystal structure: SC, BCC, FCC, DC
            ampl          Density field amplitudes: [nampl]
            n0            Reference density
            g             Rotation matrix
    
        OUTPUT
            density       Density field

        Last revision:
        H. Hallberg 2025-08-27
        '''

        q    = 2*np.pi
        nAmp = len(self._ampl) # Number of density field modes/amplitudes
        n0   = self._nlns[1]  # Reference density (liquid)

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

    def get_integrated_field_in_volume(self, field, limits):
        '''
        PURPOSE
            Integrate a field variable within a certain volume, defined on a fixed Cartesian 3D grid.

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
        interpVal = np.zeros(n_interp, dtype=values.dtype)

        # Generate periodic images of the source positions
        images = np.vstack([pos + np.array([dx, dy, dz]) * self._dSize
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
        interpVal = np.zeros(n_interp, dtype=values.dtype)
        
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

    def interpolate_density_maxima(self, den, ene, pf=None):
        '''
        PURPOSE
            Find the coordinates of the maxima in the density field (='atom' positions)
            The domain is assumed to be defined such that all maxima
            have coordinates (x,y,z) >= (0,0,0).
            The density, energy and, optionally, the phase field value(s)
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
        H. Hallberg 2025-09-11
        '''

        if self._verbose: tstart = time.time()

        # Grid
        dx,dy,dz = self._ddiv

        size = 1 + 2 * self._density_interp_order
        footprint = np.ones((size, size, size))
        footprint[self._density_interp_order, self._density_interp_order, self._density_interp_order] = 0

        filtered = ndi.maximum_filter(den, footprint=footprint, mode='wrap')
        #filtered = ndi.maximum_filter(den, footprint=footprint, mode='constant')

        mask_local_maxima = den > filtered
        coords = np.asarray(np.where(mask_local_maxima),dtype=float).T

        # ndi.maximum_filter works in voxel coordinates, convert to physical coordinates
        coords[:,0] *= dx
        coords[:,1] *= dy
        coords[:,2] *= dz

        # Filter maxima based on density threshold
        max_den = np.max(den)
        valid_maxima = den[mask_local_maxima] >= (self._density_threshold * max_den)
        coords = coords[valid_maxima]

        denpos = den[mask_local_maxima][valid_maxima]
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
            merged_enepos = []
            for cluster in unique_clusters:
                cluster_coords = coords[list(cluster)]
                cluster_denpos = denpos[list(cluster)]
                cluster_enepos = enepos[list(cluster)]
                merged_coords.append(np.mean(cluster_coords, axis=0))
                merged_denpos.append(np.mean(cluster_denpos))
                merged_enepos.append(np.mean(cluster_enepos))

            atom_coord = np.array(merged_coords)
            denpos = np.array(merged_denpos)
            enepos = np.array(merged_enepos)

        # Handle phase field(s), either as a list of fields or as a single field
        if pf is not None:
            # If pf is a single array, wrap it in a list
            if isinstance(pf, np.ndarray) and pf.ndim == 3:
                pf_list = [pf]
            else:
                pf_list = list(pf)
            nPf = len(pf_list)
            pfpos = np.zeros((coords.shape[0], nPf), dtype=float)
            for pfNr, phaseField in enumerate(pf_list):
                pfpos[:, pfNr] = phaseField[mask_local_maxima][valid_maxima][:coords.shape[0]]
            atom_data = np.hstack((denpos[:, None], enepos[:, None], pfpos))
        else:
            atom_data = np.hstack((denpos[:, None], enepos[:, None]))

        if self._verbose:
            tend = time.time()
            print(f'Time to interpolate density maxima: {tend-tstart:.3f} s')

        return atom_coord, atom_data

# =====================================================================================

    def evaluate_ampl_dens(self):
        '''
        PURPOSE
            Get amplitudes and densities for different density field expansions. For use in XPFC simulations.

        INPUT
            struct      Crystal structure: BCC, FCC
            npeaks      Number of peaks to use in the two-point correlation function
            sigma       Effective temperature in the Debye-Waller factor
            device      Device to allocate the tensors on (CPU or GPU)

        OUTPUT
            ampl        Density field amplitudes, real rank-1 array of size, [npeaks]
            nLnS        Densities in the liquid (nL) and solid (nS) phase, [2]


        Last revision:
        H. Hallberg 2025-08-26
        '''

        if self._struct.upper()=='BCC':
            if self._sigma==0:
                if self._npeaks==2:
                    # Including [110], [200]
                    ampl = np.array([ 0.116548193580713,  0.058162568591367], dtype=float)
                    nLnS = np.array([-0.151035610711215, -0.094238426687741], dtype=float)
                elif self._npeaks==3:
                    # Including [110], [200], [211]
                    ampl = np.array([ 0.111291217521458,  0.056111205274590, 0.005813371421170], dtype=float)
                    nLnS = np.array([-0.158574317081128, -0.108067574994277], dtype=float)
                else:
                    raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
            elif self._sigma==0.1:
                if self._npeaks==2:
                    # Including [110], [200]
                    ampl = np.array([ 0.113205280767407,  0.042599977405133], dtype=float)
                    nLnS = np.array([-0.106228213129645, -0.055509415103115], dtype=float)
                else:
                    raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
            else:
                raise ValueError(f'Unsupported value of sigma={self._sigma}')
        elif self._struct.upper()=='FCC':
            if self._sigma==0:
                if self._npeaks==2:
                    # Including [111], [200]
                    ampl = np.array([ 0.127697395147358,  0.097486643368977], dtype=float)
                    nLnS = np.array([-0.127233738562750, -0.065826817872435], dtype=float)
                elif self._npeaks==3:
                    # Including [111], [200], [220]
                    ampl = np.array([ 0.125151338544038,  0.097120295466816, 0.009505792832995], dtype=float)
                    nLnS = np.array([-0.138357209505865, -0.081227380909546], dtype=float)
                else:
                    raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
            else:
                raise ValueError(f'Unsupported value of sigma={self._sigma}')
        else:
            raise ValueError(f'Amplitudes and densities are not set. Unsupported value of struct={self._struct}')

        return ampl, nLnS

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