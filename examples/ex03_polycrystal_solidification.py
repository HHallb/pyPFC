# Append path to libraries
import sys
sys.path.append('/home/hlhh/Insync/OneDriveLTH/python/pyPFC/src/')

import pypfc
import numpy as np
import time
import torch

from scipy.spatial.transform import Rotation # Added import to define crystal rotations

# Set pyPFC parameters, these are handled as a dictionary
# The parameters are initialized to default values
# internally, but can be changed as needed
# =======================================================
params = {
    'dtime':                  1.0e-3,                               # Non-dimensional time increment
    'struct':                 'FCC',                                # Crystal structure
    'alat':                   1.0,                                  # Lattice parameter (non-dimensional)
    'sigma':                  0.0,                                  # Temperature parameter (non-dimensional)
    'npeaks':                 2,                                    # Number of Gaussian peaks, excluding the zero-mode peak, to use in the pair correlation function C2
    'alpha':                  [1, 1],                               # C2 Gaussian peak widths, excluding the zero-mode peak
    'C20_amplitude':          0.0,                                  # Amplitude of the zero-mode Gaussian peak in C2
    'C20_alpha':              1.0,                                  # Width of the zero-mode Gaussian peak in C2
    'pf_gauss_var':           1.0,                                  # Variance (sigma) of the Gaussian smoothing kernel used in phase field evaluations
    'normalize_pf':           True,                                 # Normalize the phase fields to [0,1], or not
    'update_scheme':          '1st_order',                          # Time integration scheme
    'update_scheme_params':   [1.0, 1.0, 1.0, None, None, None],    # Parameters in the time integration scheme: g1, g2, g3, alpha, beta, gamma
    'device_type':            'gpu',                                # PyTorch device (CPU or GPU)
    'device_number':          0,                                    # GPU device number (if multiple GPUs are available, defaults to 0)
    'dtype_cpu':              np.double,                            # Set precision of numpy arrays
    'dtype_gpu':              torch.float64,                        # Set precision of PyTorch tensors
    'verbose':                True,                                 # Verbose output (or not)
    'evaluate_phase_field':   True,                                 # Evaluate phase field (or not)
    'density_threshold':      0.5,                                  # Threshold for density maxima detection
    'density_merge_distance': 0.1,                                  # Distance for merging density maxima
    'density_interp_order':   2,                                    # Interpolation order for density maxima localization
    'pf_iso_level':           0.5,                                  # Iso-level for phase field contouring
    'torch_threads':          8,                                    # Number of CPU threads to use if device_type is CPU
    'torch_threads_interop':  8,                                    # Number of interop threads to use if device_type is CPU
}

# Simulation-specific parameters
# ==============================
nstep            = 1000                               # Number of simulation steps
nout             = 1000                               # Evaluate step data in every nout:h step
n_save_step_data = 1000                               # Save step data in every n_save_step_data:th step
nfill            = 7                                  # Number of figures to use in filenames (pre-pad with zeroes if needed)
output_path      = '/home/hlhh/Insync/OneDriveLTH/python/pyPFC/examples/ex03_output/' # Output path
output_file      = 'pypfc_setup.txt'                                                  # Output file name

# Define the computational grid
# =============================
dSize = params['alat'] * np.array([128, 64, 2], dtype=float)   # Domain size along the x, y and z axes
ndiv  = 8 * np.array([128, 64, 2], dtype=int)                  # Number of grid divisions along the x, y and z axes       
ddiv  = dSize / ndiv                                           # Grid spacing along the x, y and z axes
print(f'ndiv:    {ndiv}')
print(f'ddiv:    {ddiv} [a]')
print(f'dSize:   {dSize} [a]')
print(f'nPoints: {np.prod(ndiv):,}')

# Create a simulation object
# ==========================
pypfc = pypfc.setup_simulation(ndiv, ddiv, config=params)

# Save setup information to file
# ==============================
pypfc.write_info_file(output_path+output_file)  # Write setup information to a text file

# Generate the initial density field
# ==================================
n_xtal         = 5
xtalRot        = np.zeros((3,3,n_xtal), dtype=float)                    # Rotation matrices of the two crystals
xtalRot[:,:,0] = np.eye(3,dtype=float)                                  # Rotation of crystal #1
xtalRot[:,:,1] = Rotation.from_euler('z', np.deg2rad(15)).as_matrix()   # Rotation of crystal #2
xtalRot[:,:,2] = Rotation.from_euler('z', np.deg2rad(-38)).as_matrix()  # Rotation of crystal #3
xtalRot[:,:,3] = Rotation.from_euler('z', np.deg2rad(-49)).as_matrix()  # Rotation of crystal #4
xtalRot[:,:,4] = Rotation.from_euler('z', np.deg2rad(127)).as_matrix()  # Rotation of crystal #5
liq_width      = 2*params['alat']                                       # Width of the liquid layers between the crystals
den            = pypfc.do_polycrystal(xtalRot, liq_width=liq_width)     # Generate a polycrystal
pypfc.set_density(den)                                                  # Sets the new density field in the pyPFC simulation object

# Evaluate energy
# ===============
ene, mean_ene = pypfc.get_energy()  # Evaluate the PFC free energy and its mean value

# Evaluate phase field
# ====================
pf = pypfc.get_phase_field() # Evaluate the phase field

# Interpolate density field maxima
# ================================
atom_coord, atom_data = pypfc.interpolate_density_maxima(den, ene, pf)  # Interpolate density maxima positions and associated data (density, energy, phase field)
natoms = atom_coord.shape[0]                                            # Retrieve the number of atoms (= density peaks)

# Save data to VTK files
# ======================
plotnr   = str(0).zfill(nfill)
filename = output_path + 'pfc_data_' + plotnr
pypfc.write_vtk_structured_grid(filename, [den], ['den'])                                                               # Save the continuous density field to structured grid VTK file
pypfc.write_vtk_points(filename, atom_coord, [atom_data[:,0], atom_data[:,1], atom_data[:,2]], ['den', 'ene', 'pf'])    # Save the discrete density maxima (atoms) to VTK point file

# Prepare storage of state data and save the intial state
# =======================================================
total_time         = 0.0                                    # Initialize total simulation time
state_output_idx   = 0                                      # Initialize state output index
state_output       = np.zeros((nstep+1, 5), dtype=float)    # Allocate array for state data: time, natoms, mean_ene, mean_den, volume, cpu_time
_ , mean_den       = pypfc.get_density()                    # Evaluate the mean density
state_output[0,:]  = [0.0, natoms, mean_ene, mean_den, 0.0] # Save initial state data
state_output_idx  += 1                                      # Step up state output index

# Evolve density field
# ====================
tstart = time.time()        # Start timer
pypfc.set_verbose(False)    # Turn off verbose output during the time step loop

for step in range(nstep):

    # Update density
    pypfc.do_step_update()

    # Step up timer
    total_time += params['dtime']   

    # Evaluate and save data at specified intervals
    if np.mod(step+1,nout)==0 or step+1==nstep:

        # Evaluate data in the current step
        # =================================
        den, mean_den         = pypfc.get_density()                            # Evaluate the density field and its mean value
        ene, mean_ene         = pypfc.get_energy()                             # Evaluate the PFC free energy and its mean value
        pf                    = pypfc.get_phase_field()                        # Evaluate the phase field
        atom_coord, atom_data = pypfc.interpolate_density_maxima(den, ene, pf) # Interpolate density maxima positions and associated data (density, energy, phase field)
        natoms                = atom_coord.shape[0]                            # Retrieve the number of atoms (= number of interpolated density peaks)

        # Save state data
        # ===============
        state_output[state_output_idx,:] = [total_time, natoms, mean_ene, mean_den, time.time()-tstart] # Save state data
        state_output_idx += 1                                                                           # Step up state output index

        # Print state to identify the simulation progress
        # ===============================================
        state_string = f"Step {step+1:>10,}: natoms = {natoms:>10,}, den = {mean_den:.5e}, ene = {mean_ene:.5e}, sim_time = {time.time()-tstart:.5e} s"
        print(state_string) # Print the current state to the console

        # Save state data to file
        # =======================
        pypfc.append_to_info_file(state_string, output_path+output_file)

        # Save step data
        # ===============
        if np.mod(step+1,n_save_step_data)==0:

            # Integrate fields along x
            # ========================
            den_av = pypfc.get_field_average_along_axis(den, 'x')
            pf_av  = pypfc.get_field_average_along_axis(pf, 'x')
        
            # Save data to a binary pickle file
            # =================================
            filename = output_path + 'step_' + str(step+1).zfill(nfill)
            pypfc.save_pickle(filename, [ step, total_time, ndiv, ddiv, dSize, den, state_output[:state_output_idx+1,:], den_av, pf_av])

            # Save data to VTK files
            # ======================
            plotnr   = str(step+1).zfill(nfill)
            filename = output_path + 'pfc_data_' + plotnr     
            pypfc.write_vtk_structured_grid(filename, [den], ['den'])
            pypfc.write_vtk_points(filename, atom_coord, [atom_data[:,0], atom_data[:,1], atom_data[:,2]], ['den', 'ene', 'pf'])

tend = time.time()
print(f'Time spent in time step loop: {tend-tstart:.3f} s')

# Do cleanup
# ==========
pypfc.cleanup()

