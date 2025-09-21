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
from pypfc_pre import setup_pre
import pickle
import gzip
import vtk
import re
import os
import time
import torch
import os
from vtk.util.numpy_support import numpy_to_vtk

class setup_io(setup_pre):

    def __init__(self, domain_size, ndiv, config):

        # Initiate the inherited class
        # ============================
        super().__init__(domain_size, ndiv, config=config)

        # Set the data types
        self._dtype_cpu     = config['dtype_cpu']
        self._dtype_gpu     = config['dtype_gpu']
        self._device_number = config['device_number']
        self._device_type   = config['device_type']
        self._verbose       = config['verbose']

# =====================================================================================

    def write_extended_xyz(self, filename, coord, atom_data, atom_data_labels, simulation_time=0.0, gz=True):
        """
        PURPOSE
            Save PFC data in extended XYZ format.

        INPUT
            filename            Name of the output XYZ file
            coord               Array of shape (natoms, 3) containing atom coordinates
            atom_data           List of arrays, each of shape [natoms], containing per-atom data
            atom_data_labels    List of strings, labels for each atom data array
            simulation_time     Simulation time
            gz                  If True, save the file as a gzip-compressed file

        OUTPUT
            Data is written to the file 'filename.xyz' or 'filename.xyz.gz' (if gzip=True)

        Last revision:
        H. Hallberg 2025-09-02
        """

        if self._verbose:
            tstart = time.time()

        # Checks
        # ======
        natoms = coord.shape[0]
        if not isinstance(atom_data, list) or not isinstance(atom_data_labels, list):
            raise ValueError("atom_data and atom_data_labels must be lists")
        if len(atom_data) != len(atom_data_labels):
            raise ValueError("Number of atom_data arrays and labels must match")
        for arr in atom_data:
            if arr.shape[0] != natoms:
                raise ValueError("All atom_data arrays must have the same length as coord (natoms)")

        # Open the file for writing (gzip or plain text)
        # ==============================================
        if gz:
            open_func = gzip.open
            file_ext  = '.xyz.gz'
            mode      = 'wt'
        else:
            open_func = open
            file_ext  = '.xyz'
            mode      = 'w'

        # Build Properties string
        # =======================
        prop_str = 'pos:R:3:' + ':'.join([f'{lbl}:R:1' for lbl in atom_data_labels])
        header = (
            f"{natoms}\n"
            f'Lattice="{self._domain_size[0]:.6f} 0.0 0.0 0.0 {self._domain_size[1]:.6f} 0.0 0.0 0.0 {self._domain_size[2]:.6f}" '
            f'Properties={prop_str} Time={simulation_time:.6f}\n'
        )

        # Stack all data for fast formatting
        # ==================================
        all_data = np.column_stack([coord] + atom_data)
        # Format all lines at once using numpy
        lines = [
            " ".join(f"{val:13.6f}" for val in row)
            for row in all_data
        ]

        with open_func(filename + file_ext, mode) as file:
            file.write(header)
            file.write("\n".join(lines))
            file.write("\n")

        if self._verbose:
            tend = time.time()
            print(f'Time to write extended XYZ file {filename + file_ext}: {tend - tstart:.2f} s')

# =====================================================================================

    def read_extended_xyz(self, filename, nfields=0):
        """
        PURPOSE
            Read PFC data in extended XYZ format from file.
        
        INPUT
            filename    Name of the input XYZ file (with or without .xyz/.xyz.gz extension)
            nfields     Number of data fields per atom (beyond its coordinates [x,y,z])

        OUTPUT
            coord       Array of shape (natoms, 3) containing atom coordinates
            domain_size Domain size: [Lx, Ly, Lz]
            time        Simulation time
            partDen     Particle PFC density, [natoms]
            partEne     Particle PFC energy, [natoms]
            partPf      Particle PFC phase field values, [natoms]
        
        Last revision:
        H. Hallberg 2025-09-02
        """
        
        if self._verbose:
            tstart = time.time()

        # Ensure output directory exists
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Determine if file is gzipped and construct full filename
        if filename.endswith('.xyz.gz'):
            full_filename = filename
            is_gzipped = True
        elif filename.endswith('.xyz'):
            full_filename = filename
            is_gzipped = False
        else:
            # Try to find the file with appropriate extension
            import os
            if os.path.exists(filename + '.xyz.gz'):
                full_filename = filename + '.xyz.gz'
                is_gzipped = True
            elif os.path.exists(filename + '.xyz'):
                full_filename = filename + '.xyz'
                is_gzipped = False
            else:
                raise FileNotFoundError(f"Could not find file: {filename}.xyz or {filename}.xyz.gz")
        
        # Open the file (gzip or plain text)
        if is_gzipped:
            open_func = gzip.open
            mode = 'rt'  # Read text mode for gzip
        else:
            open_func = open
            mode = 'r'
        
        with open_func(full_filename, mode) as file:
            lines = file.readlines()
        
        # Parse header
        nAtoms = int(lines[0].strip())
        header_line = lines[1].strip()
        
        # Parse lattice parameters from header
        lattice_match = re.search(r'Lattice="([^"]*)"', header_line)
        if lattice_match:
            lattice_values = lattice_match.group(1).split()
            domain_size = [float(lattice_values[0]), float(lattice_values[4]), float(lattice_values[8])]
        else:
            raise ValueError("Could not parse Lattice information from header")
        
        # Parse time from header
        time_match = re.search(r'Time=([0-9.-]+)', header_line)
        if time_match:
            time = float(time_match.group(1))
        else:
            raise ValueError("Could not parse Time information from header")
        
        # Initialize arrays
        coord = np.zeros((nAtoms, 3))
        atom_data = [np.zeros(nAtoms) for _ in range(nfields)]

        # Parse particle/atom data
        for i in range(nAtoms):
            line_data = lines[i + 2].strip().split()
            if len(line_data) != 3 + nfields:
                raise ValueError(f"Line {i+3} does not contain expected {3 + nfields} values: {lines[i + 2].strip()}")

            coord[i, 0] = float(line_data[0])
            coord[i, 1] = float(line_data[1])
            coord[i, 2] = float(line_data[2])
            for j in range(nfields):
                atom_data[j][i] = float(line_data[3 + j])

        if self._verbose:
            tend = time.time()
            print(f'Time to read extended XYZ file {filename}: {tend - tstart:.2f} s')

        return coord, domain_size, time, atom_data
    
# =====================================================================================

    def write_vtk_points(self, filename, coord, scalarData, scalarDataName, vectorData=None, vectorDataName=None, tensorData=None, tensorDataName=None):
        """
        PURPOSE
            Save 3D point data into a VTK file.
            The data is saved in binary XML format.

        INPUT        
            filename        Output file name
            coord           Array of shape (natoms, 3) containing atom coordinates
            scalarData      List of 3D numpy arrays with scala point data
            scalarDataName  List of labels/names for the data arrays in 'scalarData'
            vectorData      List of 3D numpy arrays, of size [3], with vector point data
            vectorDataName  List of labels/names for the data arrays in 'vectorData'
            tensorData      List of 3D numpy arrays, of size [3x3], with tensor point data
                            3x3 tensors are reshaped according to: (3, 3, nPoints) -> (nPoints, 9) with

                            T = [[T11, T12, T13],
                                [T21, T22, T23],
                                [T31, T32, T33]]

                            becoming

                            T = [T11, T12, T13, T21, T22, T23, T31, T32, T33]

            tensorDataName  List of labels/names for the data arrays in 'tensorData'

        OUTPUT
            Data is written to the file 'filename.vts'

        Last revision:
        H. Hallberg 2025-02-04
        """

        if self._verbose:
            tstart = time.time()

        # Ensure output directory exists
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Ensure that the input points array is 2D with shape (N, 3)
        assert coord.ndim == 2 and coord.shape[1] == 3, 'Points array must be of shape (N, 3)'

        # Create a vtkPoints object and set the data
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(coord, deep=True, array_type=vtk.VTK_FLOAT))

        # Create a vtkPolyData object and set the points
        poly_data = vtk.vtkPolyData()
        poly_data.SetPoints(vtk_points)

        # Convert the scalar numpy array data to VTK arrays and add them to the polydata
        for sd in range(len(scalarData)):
            vtk_array = numpy_to_vtk(scalarData[sd], deep=True, array_type=vtk.VTK_FLOAT)
            vtk_array.SetName(scalarDataName[sd])
            poly_data.GetPointData().AddArray(vtk_array)

        # Convert the vector numpy array data to VTK arrays and add them to the polydata
        if vectorData is not None and vectorDataName is not None:
            for vd in range(len(vectorData)):
                vtk_vector_array = numpy_to_vtk(vectorData[vd], deep=True, array_type=vtk.VTK_FLOAT)
                vtk_vector_array.SetNumberOfComponents(3)  # Ensure the array has 3 components per tuple
                vtk_vector_array.SetName(vectorDataName[vd])
                poly_data.GetPointData().AddArray(vtk_vector_array)

        # Convert the tensor numpy array data to VTK arrays and add them to the polydata
        if tensorData is not None and tensorDataName is not None:
            for td in range(len(tensorData)):
                # Reshape each tensor from (3, 3, nPoints) to (nPoints, 9)
                reshaped_tensor = tensorData[td].reshape(3, 3, -1).transpose(2, 0, 1).reshape(-1, 9)
                vtk_tensor_array = numpy_to_vtk(reshaped_tensor, deep=True, array_type=vtk.VTK_FLOAT)
                vtk_tensor_array.SetNumberOfComponents(9)  # Ensure the array has 9 components per tuple
                vtk_tensor_array.SetName(tensorDataName[td])
                poly_data.GetPointData().AddArray(vtk_tensor_array)

        # Write the vtkPolyData to a file
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename+'.vtp')
        writer.SetInputData(poly_data)
        writer.SetDataModeToBinary()  # Ensure binary format
        writer.Write()

        if self._verbose:
            tend = time.time()
            print(f'Time to write VTK point file {filename}: {tend - tstart:.2f} s')

# =====================================================================================

    def write_vtk_structured_grid(self, filename, arrayData, arrayName):
        """
        PURPOSE
            Save 3D field data to a VTK file.
            The structured-grid data is saved in binary XML format.

        INPUT        
            filename    Output file name
            ndiv        No of grid points along each dimension, [3]
            ddiv        Grid spacing along each axis, [3]
            arrayData   List of 3D numpy data arrays
            arrayName   List oflabels/names for the data arrays in 'arrayData'

        OUTPUT
            Data is written to the file 'filename.vts'

        Last revision:
        H. Hallberg 2025-09-02
        """

        if self._verbose:
            tstart = time.time()

        # Ensure output directory exists
        out_dir = os.path.dirname(filename)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        # Grid
        nx,ny,nz = self._ndiv
        dx,dy,dz = self._ddiv

        # Create a vtkStructuredGrid object
        structured_grid = vtk.vtkStructuredGrid()
        structured_grid.SetDimensions(nx, ny, nz)

        # Create points for the structured grid
        x = np.linspace(0, (nx-1)*dx, nx)
        y = np.linspace(0, (ny-1)*dy, ny)
        z = np.linspace(0, (nz-1)*dz, nz)
        xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
        points = np.column_stack([xv.ravel(order='F'), yv.ravel(order='F'), zv.ravel(order='F')])

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points, deep=True, array_type=vtk.VTK_FLOAT))
        structured_grid.SetPoints(vtk_points)

        # Convert the numpy arrays to VTK arrays and set the scalar fields in the vtkStructuredGrid
        for ad in range(len(arrayData)):
            vtk_array = numpy_to_vtk(arrayData[ad].ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
            vtk_array.SetName(arrayName[ad])
            structured_grid.GetPointData().AddArray(vtk_array)

        # Write the vtkStructuredGrid to a file
        writer = vtk.vtkXMLStructuredGridWriter()
        writer.SetFileName(filename+'.vts')
        writer.SetInputData(structured_grid)
        writer.SetDataModeToBinary()  # Ensure binary format
        writer.Write()

        if self._verbose:
            tend = time.time()
            print(f'Time to write VTK structured grid file {filename}: {tend - tstart:.2f} s')

# =====================================================================================

    def save_pickle(self, filename, data):
        """
        PURPOSE
            Save a list of data objects to a binary pickle file

        INPUT
            filename        Path to the binary file (without extension)
            data            List of data objects to save

        OUTPUT
            Saves data to a file 'filename.pickle'

        Last revision:
        H. Hallberg 2025-09-02
        """

        if self._verbose:
            tstart = time.time()

        try:
            with open(filename + '.pickle', 'wb') as output_file:
                # Save data
                for data_object in data:
                    pickle.dump(data_object, output_file)
        except IOError as e:
            raise IOError(f"An error occurred while writing to the file {filename}.pickle: {e}")

        if self._verbose:
            tend = time.time()
            print(f'Time to write pickle file {filename}: {tend - tstart:.2f} s')


# =====================================================================================

    def load_pickle(self, filename, ndata):
        """
        PURPOSE
            Load data objetcs from a binary pickle file.

        INPUT
            filename        Path to the binary file (without extension)
            ndata           Number of data objects to read

        OUTPUT
            data            List of data objects read from the file 'filename.pickle'

        Last revision:
        H. Hallberg 2025-09-02
        """

        if self._verbose:
            tstart = time.time()

        # Check if file exists
        if not os.path.exists(filename + '.pickle'):
            raise FileNotFoundError(f"The file {filename}.pickle does not exist.")

        # Open file
        with open(filename + '.pickle', 'rb') as input_file:
            # Load data
            data = [pickle.load(input_file) for _ in range(ndata)]

        if self._verbose:
            tend = time.time()
            print(f'Time to read pickle file {filename}: {tend - tstart:.2f} s')

        return data

# =====================================================================================

    def write_info_file(self, filename='pypfc_simulation.txt', output_path=None):
        '''
        PURPOSE
            Write simulation setup information to a file.

        INPUT
            filename      Name of the output file
            output_path   Path to the output file

        OUTPUT

        Last revision:
        H. Hallberg 2025-09-15
        '''

        if output_path is None:
            output_path = os.path.join(os.getcwd(), filename)
        try:
            # Ensure the directory exists
            dir_path = os.path.dirname(output_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
        except Exception as e:
            print(f"Error creating directory for output: {e}")
            return

        try:
            with open(output_path, 'w') as f:

                f.write(f'======================================================\n')
                f.write(f'{self.get_time_stamp()}\n')
                f.write(f'======================================================\n')
                f.write(f"Number of grid divisions:  {self._ndiv}\n")
                f.write(f"   nx:                     {self._nx}\n")
                f.write(f"   ny:                     {self._ny}\n")
                f.write(f"   nz:                     {self._nz}\n")
                f.write(f"Grid spacings:             {self._ddiv}\n")
                f.write(f"   dx:                     {self._dx}\n")
                f.write(f"   dy:                     {self._dy}\n")
                f.write(f"   dz:                     {self._dz}\n")
                f.write(f"Total grid size:           {self._domain_size}\n")
                f.write(f"   Lx:                     {self._Lx}\n")
                f.write(f"   Ly:                     {self._Ly}\n")
                f.write(f"   Lz:                     {self._Lz}\n")
                f.write(f"Time increment:            {self._dtime}\n")
                f.write(f"Crystal structure:         {self._struct}\n")
                f.write(f"Lattice parameter:         {self._alat}\n")
                f.write(f"Effective temperature:     {self._sigma}\n")
                f.write(f"Number of C2 peaks:        {self._npeaks}\n")
                f.write(f"Widths of the peaks:       {self._alpha}\n")
                f.write(f"Density amplitudes:        {self._ampl}\n")
                f.write(f"Liquid/Solid densities:    {self._nlns}\n")
                f.write(f"Phase field Gauss. var.:   {self._pf_gauss_var}\n")
                f.write(f"Normalize phase field:     {self._normalize_pf}\n")
                f.write(f"Update scheme:             {self._update_scheme}\n")
                f.write(f"Update scheme parameters:  {self._update_scheme_params}\n")
                f.write(f"Data type, CPU:            {self._dtype_cpu}\n")
                f.write(f"Data type, GPU:            {self._dtype_gpu}\n")
                f.write(f"Verbose output:            {self._verbose}\n")
                f.write(f"Device type:               {self._device_type}\n")
                if self._device_type.upper() == 'GPU':
                    f.write(f'Device name:               {torch.cuda.get_device_name(self._device_number)}\n')
                    f.write(f"Device number:             {self._device_number}\n")
                    f.write(f"Compute capability:        {torch.cuda.get_device_properties(self._device_number).major}.{torch.cuda.get_device_properties(self._device_number).minor}\n")
                    f.write(f"Total memory:              {round(torch.cuda.get_device_properties(self._device_number).total_memory/1024**3,2)} GB\n")
                    f.write(f"Allocated memory:          {round(torch.cuda.memory_allocated()/1024**3, 2)} GB\n")
                    f.write(f"Cached memory:             {round(torch.cuda.memory_reserved()/1024**3, 2)} GB\n")
                    f.write(f"Reserved memory:           {round(torch.cuda.memory_reserved()/1024**3, 2)} GB\n")
                    f.write(f"Multi processor count:     {torch.cuda.get_device_properties(self._device_number).multi_processor_count}\n")
                if self._device_type.upper() == 'CPU':
                    f.write(f"Number of CPU threads:     {self._set_num_threads}\n")
                    f.write(f"Number of interop threads: {self._set_num_interop_threads}\n") 
                f.write(f'======================================================\n')
                f.write(f'\n')

            # Activate further output to the setup file
            # =========================================
            self._using_setup_file = True
            self._setup_file_path  = output_path

        except Exception as e:
            print(f"Error writing setup to file: {e}")

# =====================================================================================

    def append_to_info_file(self, info, filename='pypfc_simulation.txt', output_path=None):
        '''
        PURPOSE
            Append lines to a text file.

        INPUT
            info         String or list of strings to append
            filename     Name of the output file
            output_path  Path to the output file

        OUTPUT
            Appends info to the file

        Last revision:
        H. Hallberg 2025-09-15
        '''
        if output_path is None:
            output_path = os.path.join(os.getcwd(), filename)
        try:
            with open(output_path, 'a') as f:
                if isinstance(info, list):
                    for line in info:
                        f.write(f"{line}\n")
                else:
                    f.write(f"{info}\n")
        except Exception as e:
            print(f"Error appending to setup file: {e}")

# =====================================================================================