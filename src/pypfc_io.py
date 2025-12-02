# Copyright (C) 2025 Håkan Hallberg
# SPDX-License-Identifier: GPL-3.0-or-later
# See LICENSE file for full license text

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
from typing import Union, List, Optional, Tuple, Dict, Any

class setup_io(setup_pre):
    """
    I/O operations for Phase Field Crystal simulation data.
    
    This class provides comprehensive file I/O functionality for PFC simulations including:
    
    - Extended XYZ format for atomic positions and properties
    - VTK output for visualization in ParaView/VisIt
    - Binary pickle serialization for Python objects
    - Structured grid data export for continuous fields
    - Point data export for atomic/particle systems
    - Simulation metadata and setup information files
    
    The class supports both text and binary formats, with optional compression
    for large datasets. All methods are designed to handle typical PFC simulation
    outputs including atomic positions, density fields, energy fields and
    phase field data.
        
    Notes
    -----
    The extended XYZ format follows the convention used in molecular dynamics
    and materials simulation communities, allowing storage of arbitrary per-atom
    properties alongside coordinates. VTK output enables direct visualization
    in scientific visualization software.
    """

    def __init__(self, domain_size: Union[List[float], np.ndarray], ndiv: Union[List[int], np.ndarray], config: Dict[str, Any]) -> None:
        """
        Initialize I/O handler with domain parameters and device configuration.
        
        Parameters
        ----------
        domain_size : ndarray of float, shape (3,)
            Physical size of the simulation domain [Lx, Ly, Lz] in lattice parameter units.
        ndiv : ndarray of int, shape (3,)
            Number of grid divisions [nx, ny, nz]. Must be even numbers for FFT compatibility.
        config : dict
            Configuration parameters as key-value pairs.
            See the [pyPFC overview](core.md) for a complete list of the configuration parameters.
        """

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

    def write_extended_xyz(self, filename: str, coord: np.ndarray, atom_data: List[np.ndarray], atom_data_labels: List[str], simulation_time: float = 0.0, gz: bool = True) -> None:
        """
        Save PFC atomic data in extended XYZ format.
        
        Writes atomic positions and associated properties to an extended XYZ file,
        which is a standard format in molecular dynamics and materials simulation.
        The format includes atomic coordinates plus arbitrary per-atom properties
        such as density, energy, phase field values, etc.
        
        Parameters
        ----------
        filename : str
            Base name of the output XYZ file (without extension).
        coord : ndarray of float, shape (natoms, 3)
            Atomic coordinates [x, y, z] for each atom.
        atom_data : list of ndarray
            List of arrays containing per-atom data. Each array must have
            shape (natoms,) and represent a property for each atom.
        atom_data_labels : list of str
            Labels for each data array in `atom_data`. Must have same length
            as `atom_data` list.
        simulation_time : float, optional
            Simulation time to include in file header.
        gz : bool, optional
            If `True`, compress output using gzip.
            
        Notes
        -----
        The output file will be named `filename.xyz` or `filename.xyz.gz` if
        compression is enabled. The extended XYZ format includes a header with
        the number of atoms, simulation time, and property labels, followed by
        atomic coordinates and properties.
        
        Examples
        --------
        >>> write_extended_xyz('output', coord, [density, energy], 
        ...                   ['density', 'energy'], simulation_time=100.0)
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

    def read_extended_xyz(self, filename: str, nfields: int = 0) -> Tuple[np.ndarray, List[float], float, List[np.ndarray]]:
        """
        Read PFC data from extended XYZ format file.
        
        Reads atomic positions and associated properties from an extended XYZ
        file, which may be compressed with gzip. Automatically detects file
        format and handles both .xyz and .xyz.gz extensions.
        
        Parameters
        ----------
        filename : str
            Name of input XYZ file (with or without .xyz/.xyz.gz extension).
        nfields : int, optional
            Number of data fields per atom beyond coordinates [x,y,z].
            
        Returns
        -------
        coord : ndarray of float, shape (natoms, 3)
            Atomic coordinates.
        domain_size : ndarray of float, shape (3,)
            Domain size [Lx, Ly, Lz] from file header.
        time : float
            Simulation time from file header.
        partDen : ndarray of float, shape (natoms,)
            Particle density values (if available).
        partEne : ndarray of float, shape (natoms,)
            Particle energy values (if available).
        partPf : ndarray of float, shape (natoms,)
            Particle phase field values (if available).
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

    def write_vtk_points(self, filename: str, coord: np.ndarray, scalar_data: List[np.ndarray], scalar_data_name: List[str], vector_data: Optional[List[np.ndarray]] = None, vector_data_name: Optional[List[str]] = None, tensor_data: Optional[List[np.ndarray]] = None, tensor_data_name: Optional[List[str]] = None) -> None:
        """
        Save 3D point data to VTK file for visualization.
        
        Exports atomic positions and associated scalar, vector and tensor data
        to a VTK (Visualization Toolkit) file in binary XML format. This format
        is compatible with ParaView, VisIt and other scientific visualization
        software.
        
        Parameters
        ----------
        filename : str
            Output file name (with or without .vtu extension).
        coord : ndarray of float, shape (natoms, 3)
            3D coordinates of points/atoms.
        scalar_data : list of ndarray
            List of scalar data arrays. Each array should have shape (natoms,).
        scalar_data_name : list of str
            Names/labels for each scalar data array. Must match length of `scalarData`.
        vector_data : list of ndarray, optional
            List of vector data arrays. Each array should have shape (natoms, 3).
        vector_data_name : list of str, optional
            Names/labels for each vector data array. Required if `vectorData` is provided.
        tensor_data : list of ndarray, optional
            List of tensor data arrays. Each array should have shape (natoms, 3, 3).
            Tensors are automatically reshaped to VTK format (natoms, 9).
        tensor_data_name : list of str, optional
            Names/labels for each tensor data array. Required if `tensorData` is provided.
            
        Notes
        -----
        Tensor data is reshaped from (3,3) matrices to 9-component vectors following
        VTK convention:
        
        $$T = \\begin{bmatrix} 
        T_{11} & T_{12} & T_{13} \\\\
        T_{21} & T_{22} & T_{23} \\\\
        T_{31} & T_{32} & T_{33} 
        \\end{bmatrix}$$
        
        becomes: $[T_{11}, T_{12}, T_{13}, T_{21}, T_{22}, T_{23}, T_{31}, T_{32}, T_{33}]$
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
        for sd in range(len(scalar_data)):
            vtk_array = numpy_to_vtk(scalar_data[sd], deep=True, array_type=vtk.VTK_FLOAT)
            vtk_array.SetName(scalar_data_name[sd])
            poly_data.GetPointData().AddArray(vtk_array)

        # Convert the vector numpy array data to VTK arrays and add them to the polydata
        if vector_data is not None and vector_data_name is not None:
            for vd in range(len(vector_data)):
                vtk_vector_array = numpy_to_vtk(vector_data[vd], deep=True, array_type=vtk.VTK_FLOAT)
                vtk_vector_array.SetNumberOfComponents(3)  # Ensure the array has 3 components per tuple
                vtk_vector_array.SetName(vector_data_name[vd])
                poly_data.GetPointData().AddArray(vtk_vector_array)

        # Convert the tensor numpy array data to VTK arrays and add them to the polydata
        if tensor_data is not None and tensor_data_name is not None:
            for td in range(len(tensor_data)):
                # Reshape each tensor from (3, 3, nPoints) to (nPoints, 9)
                reshaped_tensor = tensor_data[td].reshape(3, 3, -1).transpose(2, 0, 1).reshape(-1, 9)
                vtk_tensor_array = numpy_to_vtk(reshaped_tensor, deep=True, array_type=vtk.VTK_FLOAT)
                vtk_tensor_array.SetNumberOfComponents(9)  # Ensure the array has 9 components per tuple
                vtk_tensor_array.SetName(tensor_data_name[td])
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

    def write_vtk_structured_grid(self, filename: str, array_data: List[np.ndarray], array_name: List[str]) -> None:
        """
        Save 3D field data to VTK structured grid file.
        
        Exports 3D field data (such as density, energy, or phase fields) to a 
        VTK structured grid file in binary XML format. This format is ideal for
        visualizing continuous field data in ParaView, VisIt and other 
        scientific visualization software.
        
        Parameters
        ----------
        filename : str
            Output file name (with or without .vts extension).
        array_data : list of ndarray
            List of 3D numpy arrays containing field data. Each array should
            have shape (nx, ny, nz) matching the simulation grid.
        array_name : list of str
            Names/labels for each data array. Must match length of `arrayData`.
            
        Notes
        -----
        The output file will be named `filename.vts` and uses VTK's structured
        grid format with binary encoding for efficient storage. The grid 
        dimensions and spacing are automatically determined from the inherited
        grid setup.
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
        for ad in range(len(array_data)):
            vtk_array = numpy_to_vtk(array_data[ad].ravel(order='F'), deep=True, array_type=vtk.VTK_FLOAT)
            vtk_array.SetName(array_name[ad])
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

    def save_pickle(self, filename: str, data: List[Any]) -> None:
        """Save data objects to a binary pickle file.
        
        Serializes a list of Python objects to a binary pickle file for
        efficient storage and later retrieval. This is useful for saving
        simulation state, configuration parameters or processed results.
        
        Parameters
        ----------
        filename : str
            Path to the output file (without .pickle extension).
        data : list
            List of Python objects to serialize and save.
            
        Notes
        -----
        The output file will be named `filename.pickle`. Pickle files are
        Python-specific binary format that preserves object structure and
        types.
        
        Warning
        -------
        Only load pickle files from trusted sources, as they can execute
        arbitrary code during deserialization.
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

    def load_pickle(self, filename: str, ndata: int) -> List[Any]:
        """
        Load data objects from a binary pickle file.
        
        Deserializes Python objects from a pickle file created with `save_pickle`.
        Reads a specified number of objects from the binary file.
        
        Parameters
        ----------
        filename : str
            Path to input pickle file (without .pickle extension).
        ndata : int
            Number of data objects to read from the file.
            
        Returns
        -------
        data : list
            List of Python objects loaded from `filename.pickle`.
            
        Warning
        -------
        Only load pickle files from trusted sources, as they can execute
        arbitrary code during deserialization.
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

    def write_info_file(self, filename: str = 'pypfc_simulation.txt', output_path: Optional[str] = None) -> None:
        """
        Write simulation setup information to a file.
        
        Creates a text file containing simulation parameters, grid configuration,
        and other setup information for documentation and reproducibility.
        
        Parameters
        ----------
        filename : str, optional
            Name of the output file.
        output_path : str, optional
            Path to the output directory. Uses the current working
            directory as default.
        """

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

    def append_to_info_file(self, info: Union[str, List[str]], filename: str = 'pypfc_simulation.txt', output_path: Optional[str] = None) -> None:
        """
        Append information to a text file.
        
        Adds new content to an existing text file, useful for logging simulation
        progress or adding additional information to setup files.
        
        Parameters
        ----------
        info : str or list of str
            String or list of strings to append to the file.
        filename : str, optional
            Name of the output file.
        output_path : str, optional
            Path to the output directory. Uses the current working
            directory as default.
        """
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