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
import time
import torch
from pypfc_base import setup_base

class setup_pre(setup_base):

    def __init__(self, domain_size, ndiv, config):

        # Initiate the inherited class
        # ============================
        super().__init__(domain_size, ndiv, config=config)

        # Handle input arguments
        # ======================
        nx,ny,nz = self.get_ndiv()

        self._den    = np.zeros((nx, ny, nz), dtype=config['dtype_cpu'])
        self._ene    = np.zeros((nx, ny, nz), dtype=config['dtype_cpu'])
        self._struct = config['struct']
        self._alat   = config['alat']
        self._sigma  = config['sigma']
        self._npeaks = config['npeaks']

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

    def set_ampl(self, ampl):
        ampl         = np.array(ampl, dtype=self._dtype_cpu)
        self._ampl   = ampl
        self._ampl_d = torch.from_numpy(ampl).to(self._device)

    def get_ampl(self):
        return self._ampl

    def set_nlns(self, nlns):
        nlns         = np.array(nlns, dtype=self._dtype_cpu)
        self._nlns   = nlns
        self._nlns_d = torch.from_numpy(nlns).to(self._device)

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

    def do_single_crystal(self, xtalRot=None, params=None, model=0):
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
        H. Hallberg 2025-09-20
        '''

        # Default orientation
        if xtalRot is None:
            xtalRot = np.eye(3, dtype=self._dtype_cpu)

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

    def do_polycrystal(self, xtalRot, params=None, liq_width=0.0, model=0):
        '''
        PURPOSE
            Define a polycrystal in a periodic 3D domain.
    
        INPUT
            xtalRot       Crystal orientations (rotation matrices): [3 x 3 x n_xtal]
            params        List containing parameters for the polycrystal model
            liq_width     Width of the liquid band along the GB
            model         Density field layout:
                            0 = A row of cylindrical seeds along y, with the cylinders extending through z
    
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
        
        # Number of crystals
        n_xtal = xtalRot.shape[2]

        # Generate grid coordinates
        xc = np.linspace(0, (nx-1)*dx, nx)
        yc = np.linspace(0, (ny-1)*dy, ny)
        zc = np.linspace(0, (nz-1)*dz, nz)
        Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')

        # Generate polycrystal        
        if model==0:
            xtalRadius = (Ly - n_xtal*liq_width) / n_xtal / 2
            xcrd       = Lx / 2
            for i in range(n_xtal+1):
                ycrd      = i*liq_width + i*2*xtalRadius
                condition = (np.sqrt((Xc-xcrd)**2 + (Yc-ycrd)**2) <= xtalRadius)
                crd       = np.array([Xc[condition], Yc[condition], Zc[condition]])
                if i<n_xtal:
                    density[condition] = self.generate_density_field(crd, xtalRot[:,:,i].T)
                else:
                    density[condition] = self.generate_density_field(crd, xtalRot[:,:,0].T)
        else:
            raise ValueError(f'Unsupported seed layout: model={model}')

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
        n0   = self._nlns[1]   # Reference density (liquid)

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

    def evaluate_ampl_dens(self):
        '''
        PURPOSE
            Get the amplitudes and densities for different density field expansions.
            For use in XPFC simulations.

        INPUT
            struct      Crystal structure: BCC, FCC
            npeaks      Number of peaks to use in the two-point correlation function
            sigma       Effective temperature in the Debye-Waller factor
            device      Device to allocate the tensors on (CPU or GPU)

        OUTPUT
            ampl        Density field amplitudes, real rank-1 array of size, [npeaks]
            nLnS        Densities in the liquid (nL) and solid (nS) phase, [2]


        Last revision:
        H. Hallberg 2025-09-19
        '''

        if self._struct.upper()=='BCC':
            if self._sigma==0:
                if self._npeaks==2:
                    # Including [110], [200]
                    ampl = np.array([ 0.116548193580713,  0.058162568591367], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.151035610711215, -0.094238426687741], dtype=self._dtype_cpu)
                elif self._npeaks==3:
                    # Including [110], [200], [211]
                    ampl = np.array([ 0.111291217521458,  0.056111205274590, 0.005813371421170], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.158574317081128, -0.108067574994277], dtype=self._dtype_cpu)
                else:
                    raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
            elif self._sigma==0.1:
                if self._npeaks==2:
                    # Including [110], [200]
                    ampl = np.array([ 0.113205280767407,  0.042599977405133], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.106228213129645, -0.055509415103115], dtype=self._dtype_cpu)
                else:
                    raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
            else:
                raise ValueError(f'Unsupported value of sigma={self._sigma}')
        elif self._struct.upper()=='FCC':
            if self._sigma==0:
                if self._npeaks==2:
                    # Including [111], [200]
                    ampl = np.array([ 0.127697395147358,  0.097486643368977], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.127233738562750, -0.065826817872435], dtype=self._dtype_cpu)
                elif self._npeaks==3:
                    # Including [111], [200], [220]
                    ampl = np.array([ 0.125151338544038,  0.097120295466816, 0.009505792832995], dtype=self._dtype_cpu)
                    nLnS = np.array([-0.138357209505865, -0.081227380909546], dtype=self._dtype_cpu)
                else:
                    raise ValueError(f'Unsupported value of npeaks={self._npeaks}')
            else:
                raise ValueError(f'Unsupported value of sigma={self._sigma}')
        else:
            raise ValueError(f'Amplitudes and densities are not set. Unsupported value of struct={self._struct}')

        return ampl, nLnS

# =====================================================================================
