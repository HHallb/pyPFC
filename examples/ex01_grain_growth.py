# Append path to libraries
import sys
sys.path.append('/home/hlhh/Insync/OneDriveLTH/python/pyPFC/src/')

import pypfc
import numpy as np
import time
import torch

# Set pyPFC parameters
# ====================
params = {
    'dtime':                  1.0e-3,                 # Time increment
    'struct':                 'FCC',                  # Set crystal structure
    'alat':                   1.0,                    # Lattice parameter
    'sigma':                  0.0,                    # Temperature parameter
    'npeaks':                 2,                      # Number of peaks in C2
    'alpha':                  [1, 1],                 # C2 Gaussian peak widths
    'pf_gauss_var':           0.1,                    # Variance (sigma) of the Gaussian smoothing kernel used in phase field evaluations
    'normalize_pf':           True,                   # Normalize the phase fields to [0,1], or not
    'update_scheme':          '1st_order',            # Time integration scheme
    'update_scheme_params':   None,                   # Parameters in the 2nd order time integration scheme: alpha, beta, gamma
    'device_type':            'cpu',#'gpu',                  # PyTorch device (CPU/GPU)
    'device_number':          0,                      # GPU device number (if multiple GPUs are available)
    'dtype_cpu':              np.double,              # Set precision of numpy arrays
    'dtype_gpu':              torch.float64,          # Set precision of PyTorch tensors
    'verbose':                True,                   # Verbose output (or not)
    'evaluate_phase_field':   True,                   # Evaluate phase field (or not)
    'density_threshold':      0.0,                    # Threshold for density maxima detection
    'density_merge_distance': 0.1,                    # Distance for merging density maxima
    'density_interp_order':   1,                      # Interpolation order for density maxima localization
    'pf_iso_level':           0.5,                    # Iso-level for phase field contouring
    'torch_threads':          8,                      # Number of CPU threads to use if device_type is 'cpu'
    'torch_threads_interop':  8,                      # Number of interop threads to use if device_type is 'cpu'
}

# Simulation-specific parameters
# ==============================
nstep            = 6000                               # Number of simulation steps
nout             = 500                                # Evaluate step data in every nout:h step
n_save_step_data = 1000                               # Save step data in every n_save_step_data:th step
nfill            = 7                                  # Number of figures to use in filenames (pre-pad with zeroes if needed)
output_path      = '/home/hlhh/Insync/OneDriveLTH/python/pyPFC/examples/ex01_output/' # Output path
output_file      = 'pypfc_setup.txt'                                                  # Output file name

# Define the computational grid
# =============================
dSize          = params['alat'] * np.array([64, 64, 64], dtype=float)
ndiv           = 8 * np.array([64, 64, 64], dtype=int)
#dSize          = params['alat'] * np.array([128, 128, 128], dtype=float)
#ndiv           = 8 * np.array([128, 128, 128], dtype=int)
ddiv           = dSize / ndiv

# Evaluate and display the grid parameters
# ========================================
#nx,ny,nz = ndiv
#dx,dy,dz = ddiv
#dSize    = ndiv * ddiv
print(f'ndiv:    {ndiv}')
print(f'ddiv:    {ddiv} [a]')
print(f'dSize:   {dSize} [a]')
print(f'nPoints: {np.prod(ndiv):,}')

# Create a simulation object
# ==========================
pypfc = pypfc.setup_simulation(ndiv, ddiv, config=params)

# Save setup information to file
# ==============================
pypfc.write_info_file(output_path+output_file)

# Create a preprocessor object and generate the initial density field:
# A centered spherical nucleus in an otherwise liquid domain
# ====================================================================
xtalRot    = np.eye(3, dtype=float)
xtalRadius = 0.1 * min(dSize)  # Radius of the spherical nucleus
den        = pypfc.do_single_crystal(xtalRot, [xtalRadius])
pypfc.set_density(den) # Sets the new density field in the pypfc simulation object

# Evaluate energy
# ===============
ene, mean_ene = pypfc.get_energy()

# Evaluate phase field
# ====================
pf = pypfc.get_phase_field()
pf_verts, pf_faces, volume = pypfc.get_phase_field_contour(pf)

# Interpolate density field maxima
# ================================
atom_coord, atom_data = pypfc.interpolate_density_maxima(den, ene, pf)
natoms = atom_coord.shape[0] # Retrieve the number of atoms (= density peaks)

# Save data to VTK files
# ======================
plotnr   = str(0).zfill(nfill)
filename = output_path + 'pfc_data_' + plotnr
pypfc.write_vtk_structured_grid(filename, [den], ['den'])
pypfc.write_vtk_points(filename, atom_coord, [atom_data[:,0], atom_data[:,1], atom_data[:,2]], ['den', 'ene', 'pf'])

# Variable initialization
# =======================
start_step       = 0
step             = 0
total_time       = 0.0
state_output_idx = 0

# Prepare storage of state data and save the intial state
# =======================================================
state_output       = np.zeros((nstep+1, 6), dtype=float)
den, mean_den      = pypfc.get_density()
ene, mean_ene      = pypfc.get_energy()
state_output[0,:]  = [0.0, natoms, mean_ene, mean_den, volume, 0.0]
state_output_idx  += 1

# Evolve density field
# ====================
tstart = time.time() # Start timer

# Turn off verbose output during the time step loop
pypfc.set_verbose(False)

for step in range(start_step,nstep):

    # Update density
    pypfc.do_step_update()

    # Step up timer
    total_time += params['dtime']

    if np.mod(step+1,nout)==0 or step+1==nstep:

        # Evaluate data in the current step
        # =================================
        den, mean_den              = pypfc.get_density()
        ene, mean_ene              = pypfc.get_energy()
        pf                         = pypfc.get_phase_field()
        pf_verts, pf_faces, volume = pypfc.get_phase_field_contour(pf)
        atom_coord, atom_data      = pypfc.interpolate_density_maxima(den, ene, pf)
        natoms = atom_coord.shape[0]

        # Save state data
        # ===============
        state_output[state_output_idx,:] = [total_time, natoms, mean_ene, mean_den, volume, time.time()-tstart]
        state_output_idx += 1

        # Print state to identify the simulation progress
        # ===============================================
        state_string = f"Step {step+1:>10,}: natoms = {natoms:>10,}, den = {mean_den:.5e}, ene = {mean_ene:.5e}, vol = {volume:.5e}, sim_time = {time.time()-tstart:.5e} s"
        print(state_string)

        # Save state data to file
        # =======================
        pypfc.append_to_info_file(state_string, output_path+output_file)

        # Save step data
        # ===============
        if np.mod(step+1,n_save_step_data)==0:
            filename = output_path + 'step_' + str(step+1).zfill(nfill)
            pypfc.save_pickle(filename, [ step, total_time, ndiv, ddiv, dSize, den, state_output[:state_output_idx+1,:]])

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