# Append path to libraries
import sys
sys.path.append('/home/hlhh/Insync/OneDriveLTH/python/pyPFC/src/')

import pypfc
import numpy as np
import time
import torch

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
    'pf_gauss_var':           0.5,                                  # Variance (sigma) of the Gaussian smoothing kernel used in phase field evaluations
    'normalize_pf':           True,                                 # Normalize the phase fields to [0,1], or not
    'update_scheme':          '1st_order',                          # Time integration scheme
    'update_scheme_params':   [1.0, 1.0, 1.0, None, None, None],    # Parameters in the time integration scheme: g1, g2, g3, alpha, beta, gamma
    'device_type':            'gpu',                                # PyTorch device (CPU or GPU)
    'device_number':          0,                                    # GPU device number (if multiple GPUs are available, defaults to 0)
    'dtype_cpu':              np.double,                            # Set precision of numpy arrays
    'dtype_gpu':              torch.float64,                        # Set precision of PyTorch tensors
    'verbose':                True,                                 # Verbose output (or not)
    'evaluate_phase_field':   True,                                 # Evaluate phase field (or not)
    'density_threshold':      0.5,                                  # Threshold for density maxima detection (fraction of the maximum density value)
    'density_merge_distance': 0.1,                                  # Distance for merging density maxima
    'density_interp_order':   2,                                    # Interpolation order for density maxima localization
    'pf_iso_level':           0.5,                                  # Iso-level for phase field contouring
    'torch_threads':          8,                                    # Number of CPU threads to use if device_type is CPU
    'torch_threads_interop':  8,                                    # Number of interop threads to use if device_type is CPU
}

# Simulation-specific parameters
# ==============================
nstep            = 500                               # Number of simulation steps
nout             = 500                                # Evaluate step data in every nout:h step
n_save_step_data = 500                                # Save step data in every n_save_step_data:th step
nfill            = 7                                  # Number of figures to use in filenames (pre-pad with zeroes if needed)
output_path      = '/home/hlhh/Insync/OneDriveLTH/python/pyPFC/examples/ex00_output/' # Output path
output_file      = 'pypfc_setup.txt'                                                  # Output file name

# Define the computational grid
# =============================
dSize          = params['alat'] * np.array([32, 1, 1], dtype=float)   # Domain size along the x, y and z axes
ndiv           = 16 * np.array([32, 1, 1], dtype=int)                 # Number of grid divisions along the x, y and z axes       
ddiv           = dSize / ndiv                                         # Grid spacing along the x, y and z axes
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
xtalRot = np.eye(3, dtype=float)                                        # No rotation of the seed crystal
start_x = dSize[0]*0.45                                                 # Crystal starts at this x-coordinate
end_x   = dSize[0]*0.55                                                 # Crystal ends at this x-coordinate
den     = pypfc.do_single_crystal(xtalRot, [start_x, end_x], model=1)   # Generates a single crystal within an interval along x, and extending throughout y and z
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
state_output       = np.zeros((nstep+1, 5), dtype=float)    # Allocate array for state data: time, natoms, mean_ene, mean_den, cpu_time
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





# Plot results
# ============
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import subprocess

unglyph_str   = 'python3 /home/hlhh/Insync/OneDriveLTH/python/pylibs/unglyph.py '
epsPath       = '/home/hlhh/Insync/OneDriveLTH/ideas/phase_field_crystal/pyPFC/manuscript/figures/'
saveEps       = True                 # Whether to save EPS files of figures
axesFontSize  = 12                   # Font size on plot axes
dpi           = 150#300              # DPI for saved figures
linewidth     = 2.0                  # Line width for plots

xvals = np.linspace(0, dSize[0], ndiv[0])  # x-coordinates, length nx

fig = plt.figure(figsize=(12, 4), dpi=dpi)

# First subplot: contour plot
ax0 = plt.subplot2grid((2, 1), (0, 0))
vmin, vmax = -0.25, 1.25
levels = np.linspace(vmin, vmax, 21)  # 21 levels between -0.25 and 1.25
norm = Normalize(vmin=vmin, vmax=vmax)
contour = ax0.contourf(den[:,:,0].T, levels=levels, cmap='coolwarm', extent=(0, dSize[0], 0, dSize[1]), norm=norm, extend='both')
ax0.scatter(atom_coord[:,0], atom_coord[:,1], s=30, c='white', edgecolors='black')
ax0.set_xlim(0, dSize[0])
ax0.set_ylim(0, dSize[1])
ax0.axis('equal')
ax0.axis('off')
ax0.tick_params(labelbottom=False)  # Hide x labels

# Add colorbar centered below the top subplot
#contour.set_clim(-0.25, 1.25)
cbar = fig.colorbar(contour, ax=ax0, orientation='horizontal', pad=-0.5, aspect=40, shrink=0.4)
cbar.set_label('n', fontsize=axesFontSize)
cbar.set_ticks(np.arange(-0.25, 1.26, 0.25))
cbar.set_ticklabels([f'{tick:.2f}' for tick in np.arange(-0.25, 1.26, 0.25)])

# Second subplot: den_av and pf_av with twin y-axes
ax1 = plt.subplot2grid((2, 1), (1, 0), sharex=ax0)

# Plot den_av on left y-axis
line1, = ax1.plot(xvals, den_av, 'k-', linewidth=linewidth, label='den')
ax1.set_ylabel('yden', color='k')
ax1.set_xlim(0, dSize[0])
ax1.set_ylim(-0.3, 0.15)
ax1.set_yticks(np.arange(-0.3, 0.16, 0.15))
ax1.tick_params(axis='y', labelcolor='k', labelsize=axesFontSize)
ax1.tick_params(axis='x', labelsize=axesFontSize)
#ax1.set_xlabel('x')
ax1.set_xticks([0, dSize[0]])
ax1.set_xticklabels(['0', 'Lx'])

# Create a second y-axis for pf_av
ax2 = ax1.twinx()
line2, = ax2.plot(xvals, pf_av, 'b-', linewidth=linewidth, label='pf')
ax2.set_ylabel('ypf', color='b')
ax2.set_ylim(0, 1)
ax2.set_yticks(np.arange(0, 1.1, 0.25))
ax2.tick_params(axis='y', labelcolor='b', labelsize=axesFontSize)

# Find x positions where pf_av crosses 0.5
crossings = np.where(np.diff(np.sign(pf_av - 0.5)))[0]
x_cross = []
for idx in crossings:
    x0, x1 = xvals[idx], xvals[idx+1]
    y0, y1 = pf_av[idx], pf_av[idx+1]
    x_cross.append(x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0))

# Add vertical lines at crossings
for xc in x_cross:
    ax2.axvline(x=xc, color='red', linestyle='--', linewidth=linewidth)

# Optionally add legends
lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper left', frameon=False, fontsize=axesFontSize)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2, top=1.25)

# Optionally save plot to file
# ============================
figName = 'field_variation'
if True:
    epsFile_local = figName +'.eps'
    epsFile = epsPath + figName +'.eps'
    plt.savefig(epsFile_local, bbox_inches='tight', pad_inches=0, dpi=fig.dpi)
    subprocess.call(unglyph_str + epsFile_local + ' ' + epsFile_local, shell=True)
    subprocess.call('mv ' + epsFile_local + ' ' + epsPath, shell=True)

plt.show()
