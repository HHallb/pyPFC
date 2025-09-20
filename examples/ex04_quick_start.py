"""
This is a quick start example for using the pyPFC package to perform a simple phase field crystal (PFC) simulation.
It demonstrates how to set up a simulation, generate an initial density field, evolve the density field over time,
and save the results to VTK files for visualization.

The simulation traces the growth of a spherical crystal, centered in a 3D periodic domain.

Before running this script, ensure that you have the pyPFC package and its dependencies installed.
"""

import pypfc
import numpy as np

# Simulation-specific parameters
# ==============================
nstep       = 4000                      # Number of simulation steps
nout        = 1000                      # Evaluate and save data in every nout:h step
output_path = './examples/ex04_output/' # Output path

# Define the computational grid
# =============================
domain_size = np.array([20, 20, 20])    # Domain size along the x, y and z axes, assuming a cubic unit cell with a unit lattice parameter
ndiv        = 10*np.array([20, 20, 20]) # Number of grid divisions along the x, y and z axes

# Create a simulation object
# ==========================
sim = pypfc.setup_simulation(domain_size, ndiv)

# Generate the initial density field
# ==================================
den = sim.do_single_crystal(params=[domain_size[0]*0.25]) # Generates a spherical crystal with a specified radius (=domain_size[0]*0.1)in the center of the domain
sim.set_density(den)                                      # Sets the new density field in the pyPFC simulation object

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
        atom_coord, atom_data = sim.interpolate_density_maxima(den) # Interpolate density maxima

        # Save atom data to a VTK file
        # ============================
        filename = output_path + 'pfc_data_' + str(step+1)
        sim.write_vtk_points(filename, atom_coord, [atom_data[:,0]], ['den'])

# Do cleanup
# ==========
sim.cleanup()