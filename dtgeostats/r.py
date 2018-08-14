# -*- coding: utf-8 -*-
"""
Module for playing with R

"""

import pandas as pd
import time
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import r, pandas2ri
import rpy2.robjects as ro
import pandas.rpy.common as com


from pyevtk.hl import gridToVTK

import hyvr
import dtgeostats.utils as du


# R preliminaries
base = importr('base')
rf = importr('RandomFields')

pandas2ri.activate()



def test_model_try1(mg, gsm, vbh=None):
    """ Gaussian simulation using R

    Parameters
    ----------
    mg : HyVR model grid class
        Model grid for simulation
    gsm : dtgeostats.utils.gsmodel class
        Geostatistical model
    vbh : str (optional)
        File path for virtual borehole information
    """
    ro.r('library(\'RandomFields\')')

    """ Set some variables """
    xy_ratio = gsm.lx / gsm.ly
    xz_ratio = gsm.lx / gsm.lz

    """ Set up conditional simulation """
    if vbh is not None:
        bhdf = pd.read_csv(vbh)

        # Need to 'stretch' the coordinates according to the range ratios
        # bhdf.y = bhdf.y * xy_ratio
        # bhdf.z = bhdf.z * xz_ratio

        # Assign 'data' in R instance
        ro.r('data = read.table("{}", sep=",", header=TRUE)'.format(vbh.replace('\\', '/')))
        ro.r('data$y = data$y * {}'.format(xy_ratio))
        ro.r('data$z = data$z * {}'.format(xz_ratio))

        # Get rid of additional columns in data.frame
        ro.r('keep = c("x", "y", "z", "{}")'.format(gsm.variable))
        ro.r('data = data[keep]')
        if gsm.log is True:
            ro.r('data${} = log(data${})'.format(gsm.variable, gsm.variable))


    """ Run RF simulation in R """
    ro.r('model = RMexp(var={}, scale={})'.format(gsm.sig2, gsm.lx))
    ro.r('x = seq({}, {}, by={})'.format(mg.ox + mg.dx/2, mg.lx, mg.dx))
    ro.r('y = seq({}, {}, len={})'.format(mg.oy + mg.dy/2, mg.ly*xy_ratio, mg.ny))
    ro.r('z = seq({}, {}, len={})'.format(mg.oz + mg.dz/2, mg.lz*xz_ratio, mg.nz))

    print(time.strftime("%d-%m %H:%M:%S", time.localtime(time.time())) + ': Simulating conditional field')
    if vbh is not None:
        # Conditional simulation
        ro.r('simu = RFsimulate(model, x=x, y=y, z=z, data=data)')
    else:
        ro.r('simu = RFsimulate(model, x=x, y=y, z=z)')

    xyz = ro.r('coordinates(simu)')
    condvar = ro.r('as.array(simu)')

    if gsm.log is True:
        condvar = np.exp(condvar)

    """ Export to VTK """
    gvec = mg.vec_node()
    gridToVTK('cond_log', gvec[0], gvec[1], gvec[2], cellData={'ss': condvar})

    return condvar


class dataset():

    def __init__(self,
                 name,
                 data):
        """
        Dataset class for R data inputs

        Parameters
        ----------
        name : str
            Name of datasett
        data : 2-tuple
            data[0]: data format, e.g. 'gslib', 'bhdf'
            data[1]: data  (or path/location)

        """
        self.name = name
        self.dataformat = data[0]
        self.datapath = data[1]

        xyz = ['x', 'y', 'z']

        if data[0] is 'table':
            self.dataframe = pd.read_csv(self.datapath)
            self.col_names = self.dataframe.columns.values.tolist()
            self.ncol = len(self.col_names)
            self.xyz_cols = [self.col_names.index(i) for i in xyz]
            self.variables = [i for i in self.col_names if i not in xyz]
            self.var_cols = [self.col_names.index(i) for i in self.variables]
            self.nvar = len(self.variables)

        self.rdf = com.convert_to_r_dataframe(self.dataframe)

    def variogram(self, props=['k_iso', 'poros']):
        """ Calculate and model variograms using R(gstats) library

        Parameters
        ----------
        props : list of str
            Properties to evaluate

        """

        # Load libraries
        ro.r(r'library(sp)')
        ro.r(r'library(gstat)')

        ro.globalenv['bhdata'] = self.rdf                   # Assign r_dataframe to R engine
        ro.r('coordinates(bhdata) = ~x+y+z')                # Assign coordinates to spatial data frame class

        # Loop over variables
        for ip, pp in enumerate(props):
            if ip == 0:
                g = 'NULL'
            else:
                g = 'g'

            # Direct variogram
            ro.r('g <- gstat({}, id="{}", formula={}~x+y+z, data=bhdata)'.format(g, pp, pp))

            for cv in [i for i in props if i != pp]:
                # Cross variogram
                ro.r('g <- gstat({}, id=c("{}","{}"), formula={}~x+y+z, data=bhdata)'.format(g, pp, cv, pp))

            ro.r('plot(variogram(g))')

        print('break')

    def lmc_fit(self):
        """
        Calculate Linear Model of Coregionalization covariance

        """


class variogram():

    def __init__(self,
                 name,
                 prop,             # Data filepath
                 ranges=[]):

        self.name = name
        self.prop = prop


class gsmodel():
    """ Geostatistical model """

    def __init__(self,
                 name,
                 variable,
                 cmodel,
                 sig2,
                 lx=1,
                 ly=None,
                 lz=None,
                 log=False):
        """

        Parameters
        ----------
        name : str
            Name of model
        variable : str
            Variable to model
        cmodel : str ['Sph', 'Gau', 'Exp']
            Geostatistical covariance model
        sig2 :  float
            Variance of parameter
        lx : float
            correlation length in x
        ly, lz : floats, optional
            correlation length in y, z
        log : bool (default)
            Is variable log distributed?
        """

        self.name = name
        self.variable = variable
        self.cmodel = cmodel
        self.sig2 = sig2
        self.lx = lx
        self.ndim = 1
        if ly is not None:
            self.ly = ly
            self.ndim = 2
        if lz is not None:
            self.lz = lz
            self.ndim = 3
        self.log = log


class nscore():
    """ Normal Score transformations
        Taken from https://msu.edu/~ashton/temp/nscore.R
    """
    def __init__(self,
                 data
                 ):

        self.data = data

        self.scores = ro.r(r'qqnorm(x, plot.it = FALSE)$x')
        self.trn_table = ro.r(r'data.frame(x=sort(x),nscore=sort(nscore))')



    def backtr(self):
        pass

"""=================================
Testing functions
===================================="""
if __name__ == '__main__':
    """ Testing functions"""


    # ini = 'E:\\Repositories\\WP3_effects\\fidelity\\small\\small.ini'
    ini = '..\\..\\fidelity\\runfiles\\small_0\\braid.ini'
    run, mod, strata, hydraulics, flowtrans, elements, mg = hyvr.parameters.model_setup(ini, nodir=True)
    # Create geostatistical model
    gs_bh100 = du.gsmodel('bh25', 'k_iso', 'Exp', 2.9, lx=10, ly=10, lz=0.7, log=True)
    # gs_bh100 = du.gsmodel('bh25', 'poros', 'Exp', 0.002, lx=10, ly=10, lz=0.7)

    vbh = "..\\..\\fidelity\\runfiles\\small_0\\braid_vr_bh50.txt"
    #test_model(mg, gs_bh100)

    test_model_try1(mg, gs_bh100, vbh=vbh)
