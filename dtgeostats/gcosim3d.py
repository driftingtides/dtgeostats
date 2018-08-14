# -*- coding: utf-8 -*-
""" GCOSIM3D module

    Module for working with GCOSIM3D
"""

import os
import sys
import errno
import subprocess
import numpy as np
from shutil import copyfile
import time, platform
sys.path.append(os.path.join('.', '..'))

# Variogram model ccdes
vtypes = {'Spherical': 1,
          'Exponential': 2,
          'Gaussian': 3,
          'Uniform': 4}

class model():

    def __init__(self,
                 par_fp,
                 variables,
                 vgms,
                 seed=32,
                 nsim=1,
                 ox=0., dx=1., nx=10,
                 oy=0., dy=1., ny=10,
                 oz=0., dz=1., nz=1,
                 ellips_radii=[10, 10, 0.1],
                 max_octant=3,
                 max_data=12,
                 debug_flag=0,
                 dir_cos1=[1.0, 0.0, 0.0],
                 dir_cos2=[0.0, 1.0, 0.0],
                 dir_cos3=[0.0, 0.0, 1.0],
                 batchfname='run_gcosim3d.bat'):

        """Initializes the gcosim3d model class

        -------------- For the *.geo file
        :param par_fp:      str, model directory. Is created if doesn't exist
        :param variables:     list of variable classes
        :param seed:        int, seed for random simulation
        :param nsim:        int, number of fields to simulate
        :param ox, oy, oz:  float, Origin in x,y,z-direction
        :param dx, dy, dz:  float, Dimension of simulation grid cell in x,y,z-direction
        :param nx, ny, nz:  int, Number of simulation grid cell in x,y,z-direction
        :param ellips_radii: list of floats, three semiaxes defining the ellipsoidal search
        :param dir_cos1, dir_cos2, dir_cos3:    list of ints, cosines of the angles forming the ellips_radii semiaxes
        :param max_octant:  int, maximum number of points to search for the pdf. Recommended between 2-8
        :param max_data:    int, maximum number of points to search for the pdf
        :param debug_flag:  int, flag to define amount of data to be stored in the *.dbg file. default 0

        -------------- For the *.var file
        :param vgms: list of lists, nugget sill and range of each (co-) variogram
        :param batchfname:      str, name for the batchfile

        """
        self.par_fp = par_fp
        self.variables = variables
        self.vgms = vgms
        self.seed = seed
        self.nsim = nsim

        self.ox = ox
        self.dx = dx
        self.nx = nx
        self.oy = oy
        self.dy = dy
        self.ny = ny
        self.oz = oz
        self.dz = dz
        self.nz = nz
        self.n_nodes = nx * ny * nz
        self.ellips_radii = ellips_radii
        self.dir_cos1 = dir_cos1
        self.dir_cos2 = dir_cos2
        self.dir_cos3 = dir_cos3
        self.max_octant = max_octant
        self.max_data = max_data
        self.debug_flag = debug_flag

        self.nvar = len(variables)
        self.nvar_sim = len([v for v in self.variables if v.sim is True])
        self.sim_var = [v.index for v in self.variables if v.sim is True]
        self.mean_var = [v.mean for v in self.variables if v.sim is True]
        self.ndata = len(self.variables[0].data)
        self.batchfname = batchfname

        for var in variables:
            pass

        try:
            os.makedirs(self.par_fp)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @staticmethod
    def printime(message='Provide your message as a string'):
        """  Print date time message
        :param message:
        :return: printed message with current time/date
        """
        print(time.strftime("%d-%m %H:%M:%S",
                            time.localtime(time.time())) + ': ' + message)

    def make_geo_file(self, fp=None):
        """ Write *.geo file for gcosim3d in model folder.
        :param fp : str, file path for saving parameter file. Default none
        :return:
            message if file is successfully written
        """
        pl = []
        if fp is None:
            fp = self.par_fp

        pl.append('{} {}'.format(self.seed, self.nsim))
        pl.append('{} {} {}'.format(self.dx, self.dy, self.dz))
        pl.append('{} {} {}'.format(self.ox, self.oy, self.oz))
        pl.append('{} {} {}'.format(self.nx, self.ny, self.nz))
        pl.append('{} {} {}'.format(1, 1, 1)) # todo: (integers) index of the initial cell of the subarea to be simulated
        pl.append('{} {} {}'.format(self.nx, self.ny, self.nz))

        pl.append(' ') # necessary empty line

        pl.append('{} {} {}'.format(self.ellips_radii[0], self.ellips_radii[1], self.ellips_radii[2]))

        pl.append(' ')  # necessary empty line

        pl.append('{} {} {}'.format(self.dir_cos1[0], self.dir_cos1[1], self.dir_cos1[2]))
        pl.append('{} {} {}'.format(self.dir_cos2[0], self.dir_cos2[1], self.dir_cos2[2]))
        pl.append('{} {} {}'.format(self.dir_cos3[0], self.dir_cos3[1], self.dir_cos3[2]))

        pl.append(' ')  # necessary empty line

        pl.append('{} {}'.format(self.max_octant, self.max_data))
        pl.append('{}'.format(self.debug_flag))

        # Join list and write to file
        parfile = '\n'.join(pl)
        with open(os.path.join(fp,'gcosim3d.geo'), 'w') as tfile:
            tfile.write(parfile)

        self.printime(message='Done writing geo file << %s.geo >> ' %os.path.join(fp,'gcosim3d'))

    def make_var_file(self, fp=None):
        """ Write *.var file for gcosim3d in model folder.
        :param fp : str, path for saving parameter file. Default none
        :return:
            message if file is successfully written
        """
        pl = []
        if fp is None:
            fp = self.par_fp

        pl.append('{}'.format(self.nvar))
        pl.append('{}'.format(self.nvar_sim))
        pl.append(' '.join([str(v) for v in self.sim_var]))
        pl.append(' '.join(['{:.5f}'.format(v.mean) for v in self.variables if v.sim is True]))

        pl.append(' ')  # necessary empty line

        # Deal with variograms now:
        # The order is the following:
        # First: params of variogram for variable 1
        # Second: params of co-variogram between variables 1 and 2
        # Third: params of variogram for variable 2
        # Fourth: params of co-variogram between variables 1 and 3
        # Fifth: params of variogram between variables 2 and 3
        # Sixth: params of variogram for variables 3...(and so on)
        for v in self.vgms:
            # Variogram parameters
            pl.append(str(v.nugget))     # nugget
            pl.append(str(v.cmax))
            pl.append(str(v.n_struct))
            pl.append(str(v.type))
            pl.append(str(v.sill))     # Sill
            pl.append(str(v.rx))     # Range in x
            pl.append(str(v.ry))     # Range in y
            pl.append(str(v.rz))      # Range in z
            for di in range(0, 3):                # Append Anisotropy vectors
                pl.append(' '.join([str(aa) for aa in v.aniso[di]]))
            pl.append(' ')  # necessary empty line

        # Join list and write to file
        parfile = '\n'.join(pl)
        with open(os.path.join(fp, 'gcosim3d.var'), 'w') as tfile:
            tfile.write(parfile)

        self.printime(message='Done writing var file << %s.var >>' % os.path.join(fp, 'gcosim3d'))

    def make_dat_file(self, fp=None):
        """ Write the dat file, with coordinates, variable number and variable value.

        :return:
            message if file is successfully written
        """
        if fp is None:
            fp = self.par_fp

        to_write = np.zeros((self.ndata * self.nvar, 5))    # Initialise array
        for i, var in enumerate(self.variables):
            ind = np.arange(0, self.ndata) + i *self.ndata
            to_write[ind, 0:3] = var.coords
            to_write[ind, 3] = var.index
            to_write[ind, 4] = var.data

        np.savetxt(os.path.join(fp, 'gcosim3d.dat'),
                   to_write, fmt='%.3f %.3f %.3f %u {}'.format(var.vform),
                   delimiter='\t',
                   header=str(self.ndata*self.nvar),
                   comments='')

        self.printime(message='Done writing data file << %s.dat >>' % os.path.join(fp, 'gcosim3d'))

    def make_batch(self, fp=None):
        """Write batchfile to call gcosim3d externally.

        :param fp:      str, path for saving parameter file. Default none

        :return:
            output message if file is written successfully
        """
        if fp is None:
            fp = self.par_fp
        with open(os.path.join(fp, "run_gcosim3d.bat"), 'w') as tfile:
            #tfile.write('gcosim3d.exe\npause')
            tfile.write('gcosim3d.exe')
        self.printime(message='Done writing batch file << %s >>' % os.path.join(fp, "run_gcosim3d.bat"))

    def run(self, copyexec=True, gcosim_execpath=os.path.curdir, fp=None, makebatch=True):
        """ execute gcosim3D

        :param copyexec:        bool, copy original exe file to model folder if True
        :param gcosim_execpath: str, model path, where gcosim3d will be run
        :param fp:              str, path for saving parameter file. Default none
        :param makebatch:       bool, write a batch file for calling gcosim3d if True.

        :return:
        output message if model run was successful
        """
        if fp is None:
            fp = self.par_fp
        if makebatch is True:
            self.make_batch(fp=fp)

        if copyexec is True:
            assert os.path.isfile(os.path.join(gcosim_execpath, 'gcosim3d.exe')) is True, 'Exe does not exist!'
            copyfile(os.path.join(gcosim_execpath, 'gcosim3d.exe'), os.path.join(self.par_fp, 'gcosim3d.exe'))
        qq = subprocess.Popen("run_gcosim3d.bat", shell=True, cwd=self.par_fp, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output_qq, error_qq = qq.communicate()
        if platform.system() == 'Windows':
            qq.kill()

        self.printime(message='Finished gcosim3d simulation: << %s >>' % fp)

    def out(self, f=[], fout=None, backtransform=None):
        """
        Routine for creating different outputs

        From p34 of the Spanish manual (using Google translate...):
        "The output file is similar to the output files of the previous programs, it starts with a line of text,
        the next record contains the number of variables that have been simulated (nvar_sim),
        the following nvar_sim records are labels with the order indexes variables.
        Following are the records containing the nsim simulations of the nvar_sim variables.
        Each record contains one column for each simulated variable. The order of printing of the records is:
            one simulation after another,
            within each simulation:
                one layer after another (where layer is associated with the coordinate z),
                    within each layer one row after another (where row is associated with the coordinate y) and
                        within each row one column after another (where column is associated with coordinate x).

        All loops always start with the value that has the lowest coordinates."

        :param f:       list of strings, output type requested
        :param fout     str, filepath for output if requested
        :param backtransform      str, filepath to conditioning data for back-transformation (variables ending with '_nst')

        :return:
            Output type, for selected outputs ('numpy', 'pandas')
        """
        if 'vtr' in f:
            from pyevtk.hl import gridToVTK
            xv = np.arange(self.ox, self.dx * self.nx + self.ox + self.dx, self.dx)
            yv = np.arange(self.oy, self.dy * self.ny + self.oy + self.dy, self.dy)
            zv = np.arange(self.oz, self.dz * self.nz + self.oz + self.dz, self.dz)

        try:
            import h5py
        except ImportError:
            print('h5 output not possible: h5py not installed.')

        """ Get data """

        fgout = os.path.join(self.par_fp, 'gcosim3d.out')   # GCOSIM3D output file
        skiprows = self.nvar + 2

        # rawdata = np.loadtxt(fgout, skiprows=skiprows)
        # data = dict()
        # for i, sv in enumerate(self.variables):         # ASSUMES ALL INPUT VARIABLES ARE SIMULATED!!
        #     data[sv.name] = rawdata[:, i]

        """ Translate data into 3D arrays """
        outdict = {}
        for real in range(0, self.nsim):

            # Realisation wrangling
            realname = 'real{0:03d}'.format(real)
            realpath = os.path.join(fout, realname)
            outdict = dict()                  # Initialise output dictionary of 3D numpy arrays
            try:
                os.makedirs(realpath)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            c = real * self.n_nodes
            lstart = c + skiprows
            lend = lstart + self.n_nodes

            # Extract realisation data from big file
            data = read_out(fgout, lstart, lend, self.variables) # ASSUMES ALL INPUT VARIABLES ARE SIMULATED!!
            dks = ['lnK_geo_nst', 'poros_nst']

            for key in dks:
                keyvar = [i for i in self.variables if i.name == key][0]                       # Get variable class

                # # Back transform if necessary
                # if keyvar.ntransform:
                #     data[key] = np.array(keyvar.backtransform(data[key]))

                if backtransform is not None and key[-3:] == 'nst':
                    """ Back-transformation of simulated variable based on conditioning data input """
                    import pandas as pd
                    from scipy import stats
                    cdata = pd.read_csv(backtransform)

                    # Get non-parametric distribution
                    kde1 = stats.gaussian_kde((cdata[key[:-4]]))             # Set up KDE class
                    kde_mesh = np.linspace(np.min(cdata[key[:-4]]),          # Range of data values
                                           np.max(cdata[key[:-4]]),
                                           1000)
                    kde_pdf = kde1.evaluate(kde_mesh)

                    # Get cumulative probability and normal-score values
                    cdf = np.cumsum(kde_pdf)
                    cdf /= np.max(cdf)
                    cdf[-1] -= np.diff(cdf[-2:])/2              # Make sure last value in cdf is not 1
                    normalscore = stats.norm(0, 1).ppf(cdf)     # Get normal scores for CDF (not data)

                    # Apply to data
                    dd = pd.DataFrame(data=data[key], columns=[key])
                    dd.sort_values(by=[key], inplace=True)                            # Get sorted  indices of lnK_geo
                    quants = np.interp(dd[key].values, normalscore, cdf)                                    # Get quantiles for points in data set
                    dd[key.replace('_nst', '')] = np.interp(quants, cdf, kde_mesh)                    # Transform to distribution  of original data set
                    dd.sort_index(inplace=True)
                    data[key] = dd[key.replace('_nst', '')].values
                    data[key.replace('_nst', '')] = data.pop(key)

                if keyvar.minval is not None:
                     data[key.replace('_nst', '')][data[key.replace('_nst', '')] < keyvar.minval] = keyvar.minval

            """ Put into 3D array """
            outdict = {key.replace('_nst', ''): np.zeros((self.nx, self.ny, self.nz)) for key in data.keys()}    # Initialise dictionary

            di = 0  # Data index
            for iz in range(0, self.nz):
                for iy in range(0, self.ny):
                    for ix in range(0, self.nx):
                        for key in data.keys():
                            outdict[key][ix, iy, iz] = data[key][di]
                        di += 1

            if 'h5' in f:
                with h5py.File(os.path.join(realpath, realname + '.h5'), 'w') as hf:
                    for key in data.keys():
                     hf.create_dataset(key, data=outdict[key], compression=True)
                self.printime(message='HDF5 export complete: << %s >>' % fout)

            if 'vtr' in f:
                gridToVTK(os.path.join(realpath, realname), xv, yv, zv, cellData=outdict)
                self.printime(message='VTR output complete: << %s >>' % fout)

        # Return one realisation only
        return outdict


    def rshp_hyvr_totecplot(self, outdict):
        """ Reshape three dimensional arrays generated by hyvr for tecplot. Does not support reshaping variable ktensor
        :param outdict: hyvr dictionary
        :return: reshaped dictionary
        """
        # Create support arrays:
        nx_el, ny_el, nz_el=  self.nx, self.ny, self.nz
        vartemp = np.empty([ny_el, nx_el])
        varbasket = np.zeros([1])
        rshp_outdict = {}
        for key, value in outdict.items():
            if 'tensor' not in key:
                for ii in range(0, nz_el):
                    vartemp[0:ny_el, 0:nx_el] = value[0:nx_el, 0:ny_el, ii].T
                    vartempshaped = np.reshape(vartemp, ny_el * nx_el, order='A')
                    varbasket = np.r_[(varbasket, vartempshaped)]  # This array has the order needed in HGS

                rshp_outdict[key] = varbasket[1:]

        return  rshp_outdict


class variable():

    def __init__(self,
                 name,
                 index,
                 finput,
                 fincol=None,
                 vform='%.5e',
                 sim=True,
                 ntransform=False,
                 minval=None,
                 maxval=None,
                 ):
        """

        :param name: (str) Name of variable
        :param index: (int) Index for gcosim3D
        :param finput:  (str) File input
        :param fincol:  (int) data column in input file
        :param vform:   (str) Format of variable
        :param sim:     bool, should variable be simulated
        :param ntransform:  Perform normal score transform
        :param minval       Assign mininmum value is simulated data is lower than this
        :param maxval       Assign maximum value is simulated data is lower than this


        """

        self.name = name
        self.index = index
        self.finput = finput
        self.sim = sim
        self.vform = vform
        self.ntransform = ntransform
        self.minval = minval
        self.maxval = maxval

        # Get data
        with open(finput, 'r') as ff:
            colnames = ff.readline().strip('\n').split(',')

        colxyz = [colnames.index(i) for i in ['x', 'y', 'z']]
        self.coords = np.loadtxt(finput, skiprows=1, usecols=colxyz, delimiter=',')

        if fincol is None:
            fincol = colnames.index(name)

        self.data = np.loadtxt(finput, skiprows=1, usecols=fincol, unpack=True, delimiter=',')

        self.mean = np.mean(self.data)
        self.std = np.std(self.data)


        if self.ntransform is True:
            try:
                import dtgeostats.utils as du
            except:
                import dtgeostats.dtgeostats.utils as du
            self.rawdata = self.data
            self.data, self.cprob, self.nscore_value = du.to_norm(self.rawdata)

    def backtransform(self, simdata):
        try:
            import dtgeostats.utils as du
        except:
            import dtgeostats.dtgeostats.utils as du
        back_data = du.from_norm(simdata, self.cprob, self.nscore_value, self.mean, self.std)

        return back_data


class vmodel():

    def __init__(self,
                 variable,
                 fxml=None,
                 rx=1,
                 ry=1,
                 rz=1,
                 type=1,
                 nugget=0,
                 sill=1,
                 n_struct=1,
                 cmax=1000,
                 aniso=[[1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]]):
        """ Variogram model class
        Class for storing variogram information. Currently only variograms with single structures can be modelled,
        but shouldn't be too difficult to adapt for multiple structures.

        :param variable (str, or list of str)
        :param fxml (str)  file path to xml file (see example in hyvr/main)
        :param rx: (float) range in x direction
        :param ry: (float) range in y direction
        :param rz: (float) range in  direction
        :param type: (int) variogram type
            1: spherical
            2: exponential
            3: Gaussian
            4: Uniform (?)
        :param nugget: (float)
        :param sill: (float)

        """

        self.variable = variable
        if fxml is not None:
            """ Parse XML output files from AR2GEMS """
            self.fxml = fxml
            import xml.etree.ElementTree as ET
            root = ET.parse(fxml).getroot()

            self.nugget = float(root.attrib['nugget'])
            self.n_struct = int(root.attrib['structures_count'])

            for child in root:
                self.sill = float(child.attrib['contribution'])
                self.type = vtypes[child.attrib['type']]
                self.rx = float(child.find('ranges').attrib['max'])

                # If medium / min are zero -> assume isotropic range
                if float(child.find('ranges').attrib['medium']) == 0:
                    self.ry = float(child.find('ranges').attrib['max'])
                else:
                    self.ry = float(child.find('ranges').attrib['medium'])

                if float(child.find('ranges').attrib['medium']) == 0:
                    self.rz = float(child.find('ranges').attrib['max'])
                else:
                    self.rz = float(child.find('ranges').attrib['min'])

        else:
            self.rx = rx
            self.ry = ry
            self.rz = rz
            self.type = type
            self.nugget = nugget
            self.sill = sill
            self.n_struct = n_struct

        self.cmax = cmax
        self.aniso = aniso


def read_out(fbf, lstart, lend, ovars):
    """

    :param fbf:     filepath for big ascii file
    :param lines:   lines to read
    :param ovars:   variables

    :return:
    """
    data = dict()
    for o in ovars:
        data[o.name] = np.empty((lend - lstart,))

    with open(fbf) as f:
        for i, line in enumerate(f):
            if i < lstart:
                # Don't read headers
                continue
            elif i >= lend:
                # Break after max lines read
                break
            else:
                for io, o in enumerate(ovars):
                    data[o.name][i - lstart] = float(line.split()[io])

    return data
