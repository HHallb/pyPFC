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
from ovito.data import DataCollection, Particles, SimulationCell
from ovito.pipeline import Pipeline, StaticSource
from ovito.modifiers import PolyhedralTemplateMatchingModifier
from ovito.modifiers import DislocationAnalysisModifier, ReplicateModifier
from pypfc_grid import setup_grid
from scipy.spatial.transform import Rotation as scipyRot

class setup(setup_grid):

    def __init__(self, ndiv, ddiv, dtype_cpu=np.double, struct='FCC', pos=None, verbose=False):


        # Initiate the inherited grid class
        # =================================
        super().__init__(ndiv, ddiv)

        # Handle input arguments
        # ======================
        self._ndiv      = ndiv
        self._ddiv      = ddiv
        self._dSize     = self._ndiv * self._ddiv
        self._struct    = struct
        self._pos       = pos
        self._dtype_cpu = dtype_cpu
        self._verbose   = verbose

# =====================================================================================

    def set_coord(self, coord):
        self._pos = coord

    def get_coord(self):
        return self._pos

    def set_verbose(self, verbose):
        self._verbose = verbose

    def set_struct(self, struct):
        self._struct = struct

    def get_struct(self):
        return self._struct

# =====================================================================================

    def do_ovito_ptm(self, refRot=None, outputEulerAng=False, outputStrain=False):
        """
        PURPOSE
            Evaluate crystal structure, orientation and elastic Green-Lagrange strain from a set of particles
            in a 3D periodic domain using OVITO's Polyhedral Template Matching (PTM) implementation.

            References:
                P.M. Larsen et al. (2016), Robust Structural Identification via Polyhedral
                Template Matching, Modelling Simul. Mater. Sci. Eng. 24, 055007.
                https://doi.10.1088/0965-0393/24/5/055007

                A. Stukowski (2010), Visualization and analysis of atomistic simulation data
                with OVITO - the Open Visualization Tool, Modelling Simul. Mater. Sci. Eng. 18, 015012.
                https://10.1088/0965-0393/18/1/015012

        INPUT
            refRot          Optional 3x3 rotation matrix. If provided, rotations are evaluated with respect to this rotated frame
            outputEulerAng  If True, output Euler angles, else quaternions
            outputStrain    If True, output elastic Green-Lagrange strain tensor components, else only crystal structure and orientation

        OUTPUT
            structure_id    OVITO structure type (integer) for each particle [nparticles]:
                                0 = Other/Unknown
                                1 = FCC (face-centered cubic)
                                2 = HCP (hexagonal close-packed)
                                3 = BCC (body-centered cubic)
                                4 = ICO (icosahedral coordination)
                                5 = SC (simple cubic)
                                6 = Cubic diamond
                                7 = Hexagonal diamond
                                8 = Graphene
            rot             Crystal rotation (quaternion or Euler angles) evaluated at each particle [nparticles, 4] or [nparticles, 3]
            strain          Elastic Green-Lagrange strain tensor components (Voigt notation) evaluated at each particle [nparticles, 6]:
                                [exx, eyy, ezz, eyz, exz, exy]

        Last revision:
            H. Hallberg 2025-08-29
        """

        # Check if particle positions are set
        # ===================================
        if self._pos is None:
            raise ValueError("Particle positions (self._pos) must be set before calling this method.")

        # Prepare OVITO DataCollection
        # ============================
        data      = DataCollection()
        particles = Particles()
        particles.create_property('Position', data=self._pos)
        data.objects.append(particles)

        # Define 3D periodic boundary conditions
        # ======================================
        cell      = SimulationCell(pbc=(True, True, True))
        cell[:,0] = [self._dSize[0], 0, 0]
        cell[:,1] = [0, self._dSize[1], 0]
        cell[:,2] = [0, 0, self._dSize[2]]
        data.objects.append(cell)

        # Create a pipeline and add the PTM modifier
        # ==========================================
        pipeline               = Pipeline()
        pipeline.source        = StaticSource(data=data)
        ptm                    = PolyhedralTemplateMatchingModifier()
        ptm.output_orientation = True
        if outputStrain:
            ptm.output_deformation_gradient = True
        pipeline.modifiers.append(ptm)
        data_out = pipeline.compute()

        # Optionally evaluate the elastic Green_Lagrange tensor components
        # ================================================================
        if outputStrain:
            F = np.array(data_out.particles['Elastic Deformation Gradient'])  # shape (N, 9)
            F = F.reshape(-1, 3, 3)  # shape (N, 3, 3)
            I = np.eye(3)
            # Compute Green-Lagrange strain tensor for each atom
            strain = np.empty((F.shape[0], 6))
            for i in range(F.shape[0]):
                E = 0.5 * (F[i].T @ F[i] - I)
                # Voigt notation: [exx, eyy, ezz, eyz, exz, exy]
                strain[i, 0] = E[0, 0]
                strain[i, 1] = E[1, 1]
                strain[i, 2] = E[2, 2]
                strain[i, 3] = E[1, 2]
                strain[i, 4] = E[0, 2]
                strain[i, 5] = E[0, 1]
        else:
            strain = None

        # Retrieve structure identification results
        # =========================================
        if 'Structure Type' in data_out.particles:
            stypes         = data_out.particles['Structure Type']
            unique, counts = np.unique(stypes, return_counts=True)
            ptm_structure_names = {
                0: 'Other/Unknown',
                1: 'FCC (face-centered cubic)',
                2: 'HCP (hexagonal close-packed)',
                3: 'BCC (body-centered cubic)',
                4: 'ICO (icosahedral coordination)',
                5: 'SC (simple cubic)',
                6: 'Cubic diamond',
                7: 'Hexagonal diamond',
                8: 'Graphene'
            }
            if self._verbose:
                print('doOvitoPTM:')
                for u, c in zip(unique, counts):
                    name = ptm_structure_names.get(u, f'Unknown type ({u})')
                    print(f'  {u}: {c} atoms of type {name}')
            structure_id= np.array(stypes)
        else:
            structure_id = None

        # Retrieve orientations
        # =====================
        quats_ovito = data_out.particles['Orientation']  # shape (N, 4)
        # OVITO: (w, x, y, z) ==> SciPy: (x, y, z, w)
        quats = np.empty_like(quats_ovito)
        quats[:, 0] = quats_ovito[:, 1]
        quats[:, 1] = quats_ovito[:, 2]
        quats[:, 2] = quats_ovito[:, 3]
        quats[:, 3] = quats_ovito[:, 0]
        # Replace invalid quaternions (all zeros) with identity quaternion
        invalid        = np.all(quats == 0, axis=1)
        quats[invalid] = np.array([0, 0, 0, 1])

        # Optionally rotate to reference frame
        # ====================================
        if refRot is not None:
            q_ref     = scipyRot.from_matrix(refRot).as_quat()
            q_ref_inv = scipyRot.from_quat(q_ref).inv()
            q_out     = []
            for q in quats:
                q_rel = (q_ref_inv * scipyRot.from_quat(q)).as_quat()
                q_out.append(q_rel)
            quats = np.array(q_out)

        # Map to the cubic fundamental zone (minimum misorientation)
        # ==========================================================
        cubic_group   = scipyRot.create_group('O')
        sym_quats     = cubic_group.as_quat()  # shape (24, 4)
        N             = quats.shape[0]
        # Tile and repeat to get all combinations (N*24, 4)
        quats_tile    = np.tile(quats, (24, 1))            # (24*N, 4)
        sym_quats_rep = np.repeat(sym_quats, N, axis=0) # (24*N, 4)
        # Compose: sym * quat (elementwise)
        q_syms        = scipyRot.from_quat(sym_quats_rep) * scipyRot.from_quat(quats_tile)  # (24*N,)
        q_syms_quat   = q_syms.as_quat().reshape(24, N, 4).transpose(1, 0, 2)  # (N, 24, 4)
        # Compute rotation angles for all symmetry-equivalent quaternions
        angles        = 2 * np.arccos(np.clip(np.abs(q_syms_quat[..., 3]), -1.0, 1.0))  # shape (N, 24)
        min_indices   = np.argmin(angles, axis=1)  # shape (N,)
        quats         = q_syms_quat[np.arange(N), min_indices]
            
        # Return output
        # =============
        if outputEulerAng:
            # Convert to Euler angles (xyz convention)
            rot = scipyRot.from_quat(quats).as_euler('xyz', degrees=False)
            rot = rot[:, ::-1] # Reverse the order of the columns in rot as the 'xyz' ordering does not seem to be respected
            if outputStrain:
                return structure_id, rot, strain
            else:
                return structure_id, rot
        else:
            if outputStrain:
                return structure_id, quats, strain
            else:
                return structure_id, quats

# =====================================================================================

    def do_ovito_dxa(self, rep=[1,1,1], tol=1.e-8):
        """
        PURPOSE
            Perform dislocation analysis (DXA) in a 3D periodic domain using OVITO's DXA implementation.

            References:
                A. Stukowski, V.V. Bulatov and A. Arsenlis (2012), Automated identification
                and indexing of dislocations in crystal interfaces,
                Modelling Simul. Mater. Sci. Eng. 20, 085007.
                https://doi.org/10.1088/0965-0393/20/8/085007

        INPUT
            rep             Domain replication along x, y and z: [nReplX, nReplY, nReplZ]
            tol             Tolerance for Burgers vector identification

        OUTPUT
            disl_type_ids   Array of dislocation type IDs, [nDisl]:
                                0: 1/1<100>  Composite
                                1: 1/2<110>  Perfect edge
                                2: 1/6<112>  Shockley screw
                                3: 1/6<110>  Stair-rod
                                4: 1/3<100>  Hirth
                                5: 1/3<111>  Frank
                                6: other     Unassigned
            disl_coord      Coordinates of the first point of each dislocation line, [nDisl, 3]
            disl_line_len   Array of dislocation line lengths, [nDisl]
            disl_line_dir   Array of dislocation line direction unit vectors, [nDisl, 3]
            disl_burg_vec   Array of Burgers vectors, [nDisl, 3]
            disl_segm_pts   List of dislocation segment points

        Last revision:
            H. Hallberg 2025-08-29
        """

        # Check if particle positions are set
        # ===================================
        if self._pos is None:
            raise ValueError("Particle positions (self._pos) must be set before calling this method.")

        # Auxiliary function to identify Burgers vector type
        # ==================================================
        def burgers_vector_type(bv, tol=tol):
            # bv_type_id    bv_type     Description
            # 0             '1/1<100>'  Composite
            # 1             '1/2<110>'  Perfect edge
            # 2             '1/6<112>'  Shockley Screw
            # 3             '1/6<110>'  Stair-rod
            # 4             '1/3<100>'  Hirth
            # 5             '1/3<111>'  Frank
            # 6             'other'     Unassigned
            abs_bv = np.abs(bv)
            # <100> family: any single component ±1, others zero
            if np.isclose(np.sum(abs_bv == 1.0), 1) and np.isclose(np.sum(abs_bv == 0.0), 2):
                return 0, '1/1<100>'
            # <110> family: two components ±0.5, one zero
            elif np.isclose(np.sum(abs_bv == 0.5), 2) and np.isclose(np.sum(abs_bv == 0.0), 1):
                return 1, '1/2<110>'
            # <112> family: two components ±1/6, one ±1/3
            elif np.isclose(np.sum(np.isclose(abs_bv, 1/6, atol=tol)), 2) and np.isclose(np.sum(np.isclose(abs_bv, 1/3, atol=tol)), 1):
                return 2, '1/6<112>'
            # <110> stair-rod: two components ±1/6, one zero
            elif np.isclose(np.sum(np.isclose(abs_bv, 1/6, atol=tol)), 2) and np.isclose(np.sum(abs_bv == 0.0), 1):
                return 3, '1/6<110>'
            # <100> Hirth: one component ±1/3, others zero
            elif np.isclose(np.sum(np.isclose(abs_bv, 1/3, atol=tol)), 1) and np.isclose(np.sum(abs_bv == 0.0), 2):
                return 4, '1/3<100>'
            # <111> Frank: all components ±1/3
            elif np.isclose(np.sum(np.isclose(abs_bv, 1/3, atol=tol)), 3):
                return 5, '1/3<111>'
            else:
                return 6, 'other'
            
        # Prepare OVITO DataCollection
        # ============================
        data      = DataCollection()
        particles = Particles()
        particles.create_property('Position', data=self._pos)
        data.objects.append(particles)

        # Define 3D periodic boundary conditions
        # ======================================
        cell      = SimulationCell(pbc=(True, True, True))
        cell[:,0] = [self._dSize[0], 0, 0]
        cell[:,1] = [0, self._dSize[1], 0]
        cell[:,2] = [0, 0, self._dSize[2]]
        data.objects.append(cell)

        # Create a pipeline
        # =================
        pipeline        = Pipeline()
        pipeline.source = StaticSource(data=data)

        # Add replicate modifier
        # ======================
        modifier = ReplicateModifier()
        modifier.num_x = rep[0]
        modifier.num_y = rep[1]
        modifier.num_z = rep[2]
        pipeline.modifiers.append(modifier)

        # Add DXA modifier
        # ===============
        modifier = DislocationAnalysisModifier()
        #modifier.line_coarsening_enabled = False # Default = True
        #modifier.line_smoothing_enabled  = False # Default = True
        #modifier.circuit_stretchability  = 9     # Default = 9
        #modifier.trial_circuit_length    = 14    # Default = 14
        if self._struct.lower() == 'fcc':
            modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.FCC
        elif self._struct.lower() == 'hcp':
            modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.HCP
        elif self._struct.lower() == 'bcc':
            modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.BCC
        elif self._struct.lower() == 'dc':
            modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.CubicDiamond
        elif self._struct.lower() == 'hcp':
            modifier.input_crystal_structure = DislocationAnalysisModifier.Lattice.HexagonalDiamond
        pipeline.modifiers.append(modifier)
        data = pipeline.compute()

        if self._verbose:
            print('doOvitoDXA:')
            print(f'   Found {len(data.dislocations.lines)} dislocations')

        # Extract dislocation data
        # ========================
        disl_line_len = [] # Dislocation line lengths
        disl_coord    = [] # Dislocation line origin coordinates
        disl_line_dir = [] # Unit vector along the dislocation line
        disl_burg_vec = [] # Dislocation Burgers vectors
        disl_type_ids = [] # Dislocation type IDs
        disl_segm_pts = [] # Dislocation line segement points
        for line in data.dislocations.lines:
            bv = np.where(np.abs(line.true_burgers_vector) < tol, 0.0, line.true_burgers_vector)  # Zero near-zero values for Burgers vector identification
            bv_type_id, bv_type = burgers_vector_type(bv)
            disl_line_len.append(line.length)
            disl_coord.append(line.points[0])  # First point of each dislocation line
            # Evaluate line direction as unit vector from first to last point
            p0 = line.points[0]
            p1 = line.points[-1]
            dir_vec = p1 - p0
            norm = np.linalg.norm(dir_vec)
            if norm > 0:
                dir_unit = dir_vec / norm
            else:
                dir_unit = np.zeros_like(dir_vec)
            disl_line_dir.append(dir_unit)
            disl_burg_vec.append(line.true_burgers_vector)
            disl_type_ids.append(bv_type_id)
            disl_segm_pts.append(line.points)
            if self._verbose:
                print(f'   Dislocation {line.id}: length={line.length:.3f}, Burgers vector={line.true_burgers_vector}, type={bv_type}')
        # Convert lists to arrays for output
        disl_line_len = np.array(disl_line_len, dtype=self._dtype_cpu)
        disl_coord    = np.array(disl_coord, dtype=self._dtype_cpu)
        disl_line_dir = np.array(disl_line_dir, dtype=self._dtype_cpu)
        disl_burg_vec = np.array(disl_burg_vec, dtype=self._dtype_cpu)
        disl_type_ids = np.array(disl_type_ids, dtype=int)

        return disl_type_ids, disl_coord, disl_line_len, disl_line_dir, disl_burg_vec, disl_segm_pts
    
# =====================================================================================