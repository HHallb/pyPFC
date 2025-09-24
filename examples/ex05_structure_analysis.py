"""
This is an example to illustrate the use of structure analysis by the centro-symmetry parameter (CSP) method in the pyPFC package.
It demonstrates how to set up a simulation, generate an initial density field, evolve the density field over time,
and perform structure analysis to identify defects in the crystal structure.
The results are saved to VTK files for visualization.

Before running this script, ensure that you have the pyPFC package and its dependencies installed.
"""

import pypfc
import numpy as np
import torch
from scipy.spatial.transform import Rotation # Added import to define crystal rotations
import pypfc_ovito

# Simulation-specific parameters
# ==============================
nstep       = 40000                     # Number of simulation steps
nout        = 1000                      # Evaluate step data in every nout:h step
nfill       = 7                         # Number of figures to use in filenames (pre-pad with zeroes if needed)
output_path = './examples/ex05_output/' # Output path

# Define the computational grid
# =============================
domain_size = np.array([64, 64, 1])      # Domain size along the x, y and z axes
ndiv        = 16 * np.array([64, 64, 1]) # Number of grid divisions along the x, y and z axes
ddiv        = domain_size / ndiv          # Grid spacing along the x, y and z axes
print(f'ndiv:        {ndiv}')
print(f'ddiv:        {ddiv} [a]')
print(f'domain_size: {domain_size} [a]')
print(f'nPoints:     {np.prod(ndiv):,}')

# Create a simulation object
# ==========================
sim = pypfc.setup_simulation(domain_size, ndiv)

# Create an Ovito object
# ======================
ovi = pypfc_ovito.setup(ndiv, ddiv)

# Generate the initial density field
# ==================================
n_xtal         = 6
xtalRot        = np.zeros((3,3,n_xtal), dtype=float)                   # Rotation matrices of the two crystals
xtalRot[:,:,0] = np.eye(3,dtype=float)                                 # Rotation of crystal #1
xtalRot[:,:,1] = Rotation.from_euler('z', np.deg2rad(27)).as_matrix()  # Rotation of crystal #2
xtalRot[:,:,2] = Rotation.from_euler('z', np.deg2rad(-38)).as_matrix() # Rotation of crystal #3
xtalRot[:,:,3] = Rotation.from_euler('z', np.deg2rad(-49)).as_matrix() # Rotation of crystal #4
xtalRot[:,:,4] = Rotation.from_euler('z', np.deg2rad(127)).as_matrix() # Rotation of crystal #5
xtalRot[:,:,5] = Rotation.from_euler('z', np.deg2rad(-80)).as_matrix() # Rotation of crystal #6
liq_width      = 2.0                                                   # Width of the liquid layers between the crystals
den            = sim.do_polycrystal(xtalRot, liq_width=liq_width)      # Generate a polycrystal
sim.set_density(den)                                                   # Sets the new density field in the pyPFC simulation object

# Interpolate density field maxima
# ================================
atom_coord, atom_data = sim.interpolate_density_maxima(den) # Interpolate density maxima positions

# Turn on verbose output
# ======================
sim.set_verbose(True)
ovi.set_verbose(True)

# Evaluate CSP by pyPFC
# =====================
csp = sim.get_csp(atom_coord)

# Evaluate CSP by Ovito
# =====================
ovi.set_coord(atom_coord)      # Send current atom coordinates to Ovito
csp_ovito = ovi.do_ovito_csp() # Evaluate CSP by Ovito

# Save data to file
# =================
plotnr   = str(0).zfill(nfill)
filename = output_path + 'pfc_data_' + plotnr
sim.write_vtk_points(filename, atom_coord, [atom_data[:,0], csp, csp_ovito], ['den', 'csp', 'csp_ovito']) # Save atom data to a VTK file

# Evolve density field
# ====================
for step in range(nstep):

    # Update density
    sim.do_step_update()

    # Evaluate and save data at specified intervals
    if np.mod(step+1,nout)==0 or step+1==nstep:
        
        # Print current step to trace the simulation progress
        # ===================================================
        print('Step: ', step+1)

        # Evaluate data in the current step
        # =================================
        den, _                = sim.get_density()                   # Retrieve the density field
        atom_coord, atom_data = sim.interpolate_density_maxima(den) # Interpolate density maxima positions

        # Evaluate CSP
        # ============
        csp = sim.get_csp(atom_coord)

        # Evaluate CSP by Ovito
        # =====================
        ovi.set_coord(atom_coord)      # Send current atom coordinates to Ovito
        csp_ovito = ovi.do_ovito_csp() # Evaluate CSP by Ovito

        # Save atom data to a VTK file
        # ============================
        plotnr   = str(step+1).zfill(nfill)
        filename = output_path + 'pfc_data_' + plotnr
        sim.write_vtk_points(filename, atom_coord, [atom_data[:,0], csp, csp_ovito], ['den', 'csp', 'csp_ovito']) # Save atom data to a VTK file

# Do cleanup
# ==========
sim.cleanup()