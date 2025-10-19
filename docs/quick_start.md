![pyPFC logo](images/pyPFC_logo_transparent.png)

# Quick Start Guide

This guide will get you running your first pyPFC simulation in minutes.

## Your First Simulation

This is a first quick start example for using the pyPFC package to perform a simple PFC simulation. It demonstrates how to set up a simulation, generate an initial density field, evolve the density field over time and save the results to VTK files for visualization. The simulation traces the growth of a spherical crystal, centered in a 3D periodic domain.

Before running this script, ensure that you have the pyPFC package and its dependencies installed.


```python
import pypfc
import numpy as np

# Set simulation-specific parameters
nstep       = 4000                      # Number of simulation steps
nout        = 1000                      # Evaluate and save data in every nout:h step
output_path = './examples/ex04_output/' # Output path

# Define the simulation box by setting the domain size 
# along the x, y and z axes, measured in lattice parameters
domain_size = [20, 20, 20]

# Create a simulation object
sim = pypfc.setup_simulation(domain_size)

# Generate a spherical crystal with a specified radius (taken as a fraction of
# the domain_size along x) at the center of the domain
den = sim.do_single_crystal(params=[domain_size[0]*0.25])

# Set the new density field in the pyPFC simulation object
sim.set_density(den)

# Evolve the PFC density field
for step in range(nstep):

    # Update density
    sim.do_step_update()

    # Evaluate and save data at specified intervals
    if np.mod(step+1,nout)==0 or step+1==nstep:
        
        # Print the current step number to trace the simulation progress
        print('Step: ', step+1)

        # Evaluate data in the current step
        den, _                = sim.get_density()                   # Retrieve the density field
        atom_coord, atom_data = sim.interpolate_density_maxima(den) # Interpolate density maxima

        # Save atom data to a VTK file
        filename = output_path + 'pfc_data_' + str(step+1)
        sim.write_vtk_points(filename, atom_coord, [atom_data[:,0]], ['den'])

# Final cleanup
sim.cleanup()
```

## Add a plot of the PFC energy

```python
import matplotlib as plt

# Run simulation for 100 steps
nstep = 100
nout  = 10
energies = []
steps = []

for step in range(nstep):
    # Update the system
    sim.do_step_update()
    
    # Monitor energy every nout:h step
    if np.mod(step+1,nout)==0 or step+1==nstep:
        energy = sim.get_energy()
        energies.append(energy)
        steps.append(step)
        print(f"Step {step:3d}: Energy = {energy:.6f}")

# Plot energy evolution
plt.figure(figsize=(8, 5))
plt.plot(steps, energies, 'b-o', linewidth=2, markersize=4)
plt.xlabel('Simulation Step')
plt.ylabel('Free Energy')
plt.title('Energy Evolution')
plt.grid(True, alpha=0.3)
plt.show()
```

## Next Step

Explore the [comprehensive examples](examples.md) to see more of pyPFC's capabilities!