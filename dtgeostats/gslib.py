# -*- coding: utf-8 -*-
""" glsib module for interfacing with GSLIB geostatistical package

    Module for working with multiple-point geostatistical simulators.
        - impala (imp)

"""
import os
import shutil
import subprocess
import numpy as np
import random
import string

gslib_dir = r'E:\Software\gslib\gslib90\Gslib90'
print('Running GSLIB in: {}'.format(gslib_dir))

class gamv():
    """ Create class for running GAMV function in GSLIB """

    def __init__(self,
                 name,
                 data,                     # gslib.data class
                 foutput,                   # File output for variograms
                 vardirs,                    # variogram direction vardir() classes
                 fpar='',
                 trim=[1e-21, 1e21],        # trim values outside of this range
                 nlags=20,          # Number of lags
                 lsep=1,            # Lag separation distance
                 ltol=0.5,            # Lag tolerance
                 std_sills=0,              # standardize sills? (0=no, 1=yes)
                 variograms=[[1, 1, 1]]):          # Tail var., head var., variogram type

        self.name = name
        self.data = data
        self.foutput = foutput
        self.trim = trim
        self.nlags = nlags
        self.lsep = lsep
        self.ltol = ltol
        self.vardirs = vardirs
        self.ndir = len(vardirs)
        self.std_sills = std_sills
        self.variograms = variograms
        self.nvgm = len(variograms)             # Number of variograms
        if len(fpar) == 0:
            self.fpar = os.path.join(os.path.dirname(foutput), 'gamv.par')
        else:
            self.fpar = fpar

    def save_parameter_file(self):
        """ Save parameter file """


        # Write parameters
        p = []
        p.append('START OF PARAMETERS:')
        p.append(os.path.basename(self.data.fpath))
        p.append(' '.join([str(i+1) for i in self.data.xyz_cols]))                      # Columns of x,y,z coordinates
        p.append(' '.join([str(i) for i in [self.data.nvar] + self.data.var_cols]))     # number of variables and column order
        p.append(' '.join([str(i) for i in self.trim]))
        p.append(os.path.basename(self.foutput))
        p.append(str(self.nlags))
        p.append(str(self.lsep))
        p.append(str(self.ltol))
        p.append(str(self.ndir))
        for d in self.vardirs:
            p.append(d.string()) # azm, atol, bandwh, dip, dtol, and bandwd
        p.append(str(self.std_sills))
        p.append(str(self.nvgm))
        for v in self.variograms:
            p.append(' '.join([str(i) for i in v])) # ivtail, ivhead and ivtype

        # Join list and write to file
        parfile = '\n'.join(p)
        with open(self.fpar, 'w') as tfile:
            tfile.write(parfile)

    def run(self):
        """ Run the simulation.

        In this function, the gamv executable is copied to the working directory and then deleted following the
        termination of the GSLIB routine

        """
        newgam = os.path.join(os.path.dirname(self.fpar), 'gamv.exe')
        shutil.copyfile('E:/Software/gslib/gslib90/Gslib90/gamv.exe', newgam)
        subprocess.run([newgam, self.fpar], cwd=os.path.dirname(self.fpar))
        os.remove(newgam)


class sgsim():

    def __init__(self,
                 name,
                 data,
                 mg,
                 fpar='sgsim.par',          # parameter file
                 trim=[1e-21, 1e21],        # trim values outside of this range
                 itrans=False,              #
                 ftrans='sgsim.trn',
                 ismooth=False,
                 fsmooth='',
                 icolvrwt=None,
                 zminmax=[0., 1.0001],
                 ltp=[1, 0.],
                 utp=[1, 0.],
                 debug=0,
                 fdebug='sgsim.dbg',
                 fout='sgsim.out',
                 nreal=1,
                 seed=20160429,
                 ndminmax=[0, 8],
                 ncnode=12,
                 sstrat=0,
                 nmult=None,
                 noct=0,
                 radii=[100., 100., 10.],
                 sang=[0., 0., 0.],
                 ktype=0,
                 rho=None,
                 fsec='',
                 nst=1,
                 nuggete=0.05,


                 ):
        """



        Parameters
        ----------
        name : str
            Name of SG simulation
        data : Data class
            Data for simulation
        mg : Hyvr.Grid.Grid class
            grid for simulation
        trim : list of two floats
            Trim values outside of this range
        itrans : bool
            if set to 0 then no transformation will be performed; the variable is assumed already standard normal
             (the simulation results will also be left unchanged). If itrans=1, transformations are performed.
        ftrans : str
            Output file for the transformation table if transformation is required (igauss=0).
        ismooth : bool
            if set to 0, then the data histogram, possibly with declustering weights is used for transformation,
            if set to 1, then the data are transformed according to the values in another file (perhaps from histogram smoothing).
        fsmooth : str
            file with the values to use for transformation to normal scores (if ismooth is set to 0).
        icolvrwt : list of two ints
            columns in fsmooth for the variable and the declustering weight (set to 1 and 2 if fsmooth is the output from ftrans)
        zminmax : list of two floats
            Minimum and maximum allowable data values. These are used in the back transformation procedure.
        ltp : list of int and float
            specify the back transformation implementation in the lower tail of the distribution:
                ltail=1 implements linear interpolation to the lower limit zmin, and
                ltail=2 implements power model interpolation, with w=ltpar, to the lower limit zmin.
        utp : list of int and float
            specify the back transformation implementation in the upper tail of the distribution:
                utail=1 implements linear interpolation to the upper limit zmax,
                utail=2 implements power model interpolation, with w=utpar, to the upper limit zmax, and
                utail=4 implements hyperbolic model extrapolation with w=utpar.
                The hyperbolic tail extrapolation is limited by zmax.
        debug : int [0,1,2,3]
            The larger the debugging level the more information written out.
        fdebug : str
            Debug file name
        fout : str
            Simulation output file
        nreal : int
            Number of realizations
        seed = int
            Random number seed
        ndminmax : list of two ints
            the minimum and maximum number of original data that should be used to simulate a grid node.
            If there are fewer than ndmin data points the node is not simulated.
        ncnode : int
            maximum number of previously simulated nodes to use for the simulation of another node.
        sstrat : bool
            if set to 0, the data and previously simulated grid nodes are searched separately:
            the data are searched with a super block search and the previously simulated nodes are
            searched with a spiral search (see section II.4).
            If set to 1, the data are relocated to grid nodes and a spiral search is used and the
            parameters ndmin and ndmax are not considered.


        Notes
        -----
        http://www.statios.com/help/sgsim.html


        """
        self.name = name
        self.data = data
        self.mg = mg
        self.fpar = fpar
        self.trim = trim
        self.itrans = itrans
        self.ftrans = ftrans
        self.ismooth = ismooth
        self.fsmooth = fsmooth
        self.icolvrwt = icolvrwt
        self.zminmax = zminmax
        self.ltp = ltp
        self.utp = utp
        self.debug = debug
        self.fdebug = fdebug
        self.fout = fout
        self.nreal = nreal
        self.seed = seed
        self.ndminmax = ndminmax
        self.ncnode = ncnode
        self.sstrat = sstrat


    def save_parameter_file(self):
        """ Save parameter file """


        # Write parameters
        p = []
        p.append('START OF PARAMETERS:')

        # Join list and write to file
        parfile = '\n'.join(p)
        with open(self.fpar, 'w') as tfile:
            tfile.write(parfile)

    def run(self):
        """ Run the simulation.

        In this function, the SGSIM executable is copied to the working directory and then deleted following the
        termination of the GSLIB routine

        """
        newsgsim = os.path.join(os.path.dirname(self.fpar), 'sgsim.exe')
        shutil.copyfile('E:/Software/gslib/gslib90/Gslib90/sgsim.exe', newsgsim)
        subprocess.run([newsgsim, self.fpar], cwd=os.path.dirname(self.fpar))
        os.remove(newsgsim)


class data():

    def __init__(self,
                 name,
                 fpath,             # Data filepath
                 bhdf=None):
        """

        Parameters
        ----------
        fpath : str
            File path to data
        col_names : list of strings
            Names of columns in order
        """
        self.name = name
        self.fpath = fpath

        xyz = ['x', 'y', 'z']

        if bhdf is not None:
            # Save to GEOEAS format
            df2geoeas(bhdf, self.fpath, head_note='Virtual boreholes')
            self.col_names = bhdf.columns.values.tolist()
            self.ncol = len(self.col_names)
            self.xyz_cols = [self.col_names.index(i) for i in xyz]
            self.variables = [i for i in self.col_names if i not in xyz]
            self.var_cols = [self.col_names.index(i) for i in self.variables]
            self.nvar = len(self.variables)

    def nscore(self,
               variable,
               trim=[1e-21, 1e21],
               fsmooth=None):
        """
        Normal-score transformation

        """

        """ Write parameter file"""
        fout = '.'.split(os.path.basename(self.fpath))[0]
        # Write parameters
        p = []
        p.append('START OF PARAMETERS:')
        p.append(self.fpath)
        p.append()                      # the column numbers for the variable and the weight. If icolwt less than or equal to 0, equal weighting is considered.
        p.append(trim)
        if fsmooth is not None:
            p.append('1')
            p.append(fsmooth)
            p.append()                  # icolvr and icolwt: the column numbers for the variable and the weight in smoothfl
        else:
            p.append('0')
        p.append(fout)
        p.append(ftrans)

        # Join list and write to file
        parfile = '\n'.join(p)
        with open(self.fpar, 'w') as tfile:
            tfile.write(parfile)

        """ Run transformation """


class nscore():

    def __init__(self,
                 variable,
                 rawdata,
                 trim=[1e-21, 1e21],
                 fsmooth=None,
                 filename=None,
                 gslib_outputs=False):

        """
        Normal-score transformation wish GSLIB

        """

        self.variable = variable
        self.rawdata = rawdata
        self.trim = trim
        self.fsmooth = fsmooth
        self.gslib_outputs = gslib_outputs

        """ Write data array """
        if filename is None:
            filename = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
        self.filename = filename
        ndarray2GSLIB(rawdata, '{}.dat'.format(self.filename), self.variable)

        """ Write parameter file"""
        # Write parameters
        p = []
        p.append('START OF PARAMETERS:')
        p.append('{}.dat'.format(self.filename))
        p.append('1     0')                      # the column numbers for the variable and the weight. If icolwt less than or equal to 0, equal weighting is considered.
        p.append('{0[0]}  {0[1]}'.format(trim))
        if fsmooth is not None:
            p.append('1')
            p.append(fsmooth)
            p.append('1   2')                  # icolvr and icolwt: the column numbers for the variable and the weight in smoothfl
        else:
            p.append('0')
            p.append('None provided')
            p.append('1  2')
        p.append('{}.out'.format(self.filename))
        p.append('{}.trn'.format(self.filename))

        # Join list and write to file
        parfile = '\n'.join(p)
        with open('{}.par'.format(self.filename), 'w') as tfile:
            tfile.write(parfile)

        """ Execute GSLIB function and return value"""
        subprocess.call('{} {}.par'.format(os.path.join(gslib_dir, 'nscore.exe'), self.filename))
        #y, name = GSLIB2ndarray('nscore.out', 1, nx, ny)


        """ Delete GSLIB files """
        if self.gslib_outputs is False:
            for suf in ['dat', 'out', 'par', 'trn']:
                os.remove('{}.{}'.format(self.filename, suf))

        return

class vardir():
    """ Variogram direction """

    def __init__(self,
                 azm=90.,
                 atol=10.,
                 bandwh=10.,
                 dip=0.,
                 dtol=10.,
                 bandwd=10.):
        """
        
        Parameters
        ----------
        azm : float
            azimuth angle in degrees from clockwise north
        atol : float
            half window azimuth tolerance
         bandwh : float
            azimuth bandwidth
         dip : float
            dip angle in negative degrees down from horizontal
         dtol : float
            half window dip tolerance
         bandwd : float
            dip bandwidth
        """

        self.azm = azm
        self.atol = atol
        self.bandwh = bandwh
        self.dip = dip
        self.dtol = dtol
        self.bandwd = bandwd

    def string(self):
        """ Create variogram direction string output """
        return ' '.join([str(i) for i in [self.azm, self.atol, self.bandwh, self.dip, self.dtol, self.bandwd]])


def df2geoeas(bh_df, file_out, parameter=[], head_note='', type='dat'):
    """ Convert pandas dataframe into GEOEAS-format file

    Parameters
    ----------
        bh_df (pandas dataframe):
        file_out (str):             File output path
        parameter (str):            Which parameter to calculate
        head_note (str):               Header
        type : str
            Type of data

    """

    """ Wrangle data """
    bh_data = bh_df.as_matrix()

    if type == 'tp':
        """ Write transitional probabilities """
        head_note += ': Transitional probabilities'
        # Get unique values in dataframe
        unival = bh_df[parameter].unique()
        unival.sort()

        # Need to change values from integers to probabilities at each point
        # Create matrix assigning probabilities from each parameter value
        tp_vals = bh_df[parameter].astype('int').as_matrix()
        tp_matrix = np.zeros([bh_df.__len__(), len(unival)])
        tp_matrix[np.arange(0, 1100), tp_vals] = 1
        data_out = np.concatenate((bh_data, tp_matrix), axis=1)
        header = [head_note, str(bh_df.columns.__len__()), 'X', 'Y', 'Z']

        for uvi in unival:
            header.append(str(uvi))

        dformat = '%.5f'


    elif type == 'dat':
        """ Write data """
        dcols = list(bh_df.columns)             # Dataframe columns
        ncols = len(dcols)                      # Number of columns
        data_out = bh_data                      # Data to write

        # Set up header
        head_note += ': Borehole data'
        header = [head_note, str(ncols)]
        header.extend(dcols)

        # Sort out formatting
        coord_format = '%.2f'
        angle_format = '%.2f'

        fmt_dict = {'x': coord_format,
                    'y': coord_format,
                    'z': coord_format,
                    'azim': angle_format,
                    'dip': angle_format,
                    'anirat': '%.2f',
                    'k_iso': '%.3e',
                    'log10_K': '%.3e',
                    'y_iso': '%.5f',
                    'poros': '%.3f',
                    'ssm': '%d',
                    'fac': '%d'}

        dformat = ' '.join([fmt_dict[i] for i in header[2::]])

    """ Write to text file """
    wheader = '\n'.join(header)
    np.savetxt(file_out, data_out, delimiter=' ', header=wheader, comments='', fmt=dformat)
    return header[2::]


def run(func):
    """
    Run GSLIB function
    :param func:
    :return:
    """
    newfunc = os.path.join(os.path.dirname(self.fpar), 'sgsim.exe')
    shutil.copyfile('E:/Software/gslib/gslib90/Gslib90/sgsim.exe', newsgsim)
    subprocess.run([newsgsim, self.fpar], cwd=os.path.dirname(self.fpar))
    os.remove(newsgsim)

""" ============================================================
    Stuff from https://github.com/GeostatsGuy/geostatspy/blob/master/Variogram.ipynb

    ==========================================================  """


def ndarray2GSLIB(array, data_file, col_name):
    """
    From https://github.com/GeostatsGuy/geostatspy/blob/master/Variogram.ipynb

    :param array:
    :param data_file:
    :param col_name:
    :return:
    """
    with open(data_file, "w") as file_out:
        file_out.write(data_file + '\n')
        file_out.write('1 \n')
        file_out.write(col_name  + '\n')
        if array.ndim == 2:
            nx = array.shape[0]
            ny = array.shape[1]
            ncol = 1
            for iy in range(0, ny):
                for ix in range(0, nx):
                    file_out.write(str(array[ny-1-iy,ix])+ '\n')
        elif array.ndim == 1:
            nx = len(array)
            for ix in range(0, nx):
                file_out.write(str(array[ix])+ '\n')
        else:
            print("Error: must use a 2D array")
            return


def GSLIB2ndarray(data_file, kcol, nx, ny):

    colArray = []
    if ny > 1:
        array = np.ndarray(shape=(nx, ny), dtype=float, order='F')
    else:
        array = np.zeros(nx)

    with open(data_file) as myfile:   # read first two lines
        head = [next(myfile) for x in range(2)]
        line2 = head[1].split()
        ncol = int(line2[0])          # get the number of columns
        for icol in range(0, ncol):   # read over the column names
            head = [next(myfile) for x in range(1)]
            if icol == kcol:
                col_name = head[0].split()[0]
        for iy in range(0,ny):
            for ix in range(0,nx):
                head = [next(myfile) for x in range(1)]
                array[ny-1-iy][ix] = head[0].split()[kcol]
    return array, col_name

"""=================================
Testing functions
===================================="""
if __name__ == '__main__':

    par_fp = r'E:\Repositories\fidelity\runfiles\small_1\braid_vr_bh25.csv'
    rawdata = np.loadtxt(par_fp, usecols=8, delimiter=',', skiprows=1)

    nscore('lnK', rawdata)
    print('Ada')
