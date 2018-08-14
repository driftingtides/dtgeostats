# -*- coding: utf-8 -*-
""" MPS module

    Module for working with multiple-point geostatistical simulators.
        - impala (imp)

"""
import os
import errno
import subprocess
import numpy as np
from shutil import copyfile



class Impala_parameters():

    def __init__(self, name,
                 facies,
                 ti,
                 ox=0., dx=1., nx=10,
                 oy=0., dy=1., ny=10,
                 oz=None, dz=None, nz=None,
                 n_multigrids=4,
                 n_dir_mg=4,
                 mg_pathtype='RANDOM_PATH',
                 auxvar_treatment='MEAN',
                 auxvar_fp=[],
                 localpdfs=[],
                 data_templates=[],
                 dtar=True,
                 int_mg=True,
                 max_data=[-1],
                 turbo='NONE',
                 mask_fp='',
                 zones=[],
                 servo=False,
                 servo_type=None,
                 servo_prop=[],
                 servo_weight=[],
                 window_scan=False,
                 cond_fp=None,
                 cond_conn=False,
                 val_thresh=1,
                 reject_loop=4,
                 processing='NO',
                 nreal=1,
                 seed=None,
                 par_fp='',
                 run_flags={'hd_ext_only': str(0),
                            'wd_only': str(0),
                            'res': str(1),
                            'mean': str(0),
                            'read_list': str(0),
                            'write_list': str(0),
                            'log_file': 'run.log'}
                 ):
        """
        Parameters
        ----------
        name : string
            Name of the parameter file
        facies : list of strings
            Facies to be modelled with impala
        ti : list of training image classes
            Training images to be included in MPS simulation
        ox : float
            Origin in x-direction
        dx : float
            x-dimension of model grid cell
        nx : int
            number of grid cells in x-direction
        oy : float
            Origin in y-direction
        dy : float
            y-dimension of model grid cell
        ny : int
            number of grid cells in y-direction
        oz : float, optional
            Origin in z-direction
        dz : float, optional
            z-dimension of model grid cell
        nz : int, optional
            number of grid cells in x-direction
        n_multigrids : int, optional
            Number of multigrids
        n_dir_mg : list of int, optional
            Number of multigrid levels in each direction
        mg_pathtype: list of strings, optional
            Path type for each multigrid

        auxvar_treatment : string
            Treatment of auxilliary variable
        auxvar_fp : list of strings, optional
            Filepaths of auxilliary variables
        localpdfs : list of strings, optional
            File paths for local PDFs
        data_templates : list of data template classes
            Data template data
        dtar : bool
            Data template automatic reduction for fine multigrids
        int_mg : bool
            intermediate multigrids usage
        max_data : list of int
            maximal number of data (-1 for illimited) for each multigrid (for optimizing search template)
        turbo : str
            Type of turbo ('None', 'Manual', 'Auto')
        mask_fp : string, optional
            File path for mask
        zones : list of Simulation_zones classes
            Simulation zone classes
        servo : bool, optional
            Use servo system
        servo_type : str ['GLOBAL' or 'LOCAL'], optional
            Type of servo to use
        servo_prop : list of floats (if 'GLOBAL' type) or str (if 'LOCAL' type), optional
            Target proportions
        servo_weights : list of floats
            weights for servo system
        window_scan : bool
            Window scan method
        cond_fp : str, optional
            File path for conditional data
        cond_conn : bool, optional
            Is conditional data connected?
        val_thresh : float [0,1]
            Validation threshold
        reject_loop : int, optional
            Rejection loops per multigrid
        processing : str ['NO', 'SYNC'], optional
            Processing type
        nreal : int, optional
            Number of realisations
        seed : int
            Seed number for realizations
        par_fp : str
            Location to save parameter file
        run_flags : dict
            Flags for running impala
            ************** At this stage, no provision for list read/write *****
        """

        self.name = name
        self.ox = ox
        self.dx = dx
        self.nx = nx
        self.oy = oy
        self.dy = dy
        self.ny = ny
        self.oz = oz
        self.dz = dz
        self.nz = nz
        if self.nz is not None:
            self.ndim = 3
        else:
            self.ndim = 2

        # Multigrids and paths
        self.n_multigrids = n_multigrids        # number of multigrids
        self.n_dir_mg = n_dir_mg                # number of multigrid levels in each direction
        self.mg_pathtype = mg_pathtype          # Path type for multigrids

        # Facies
        self.facies = facies
        self.n_facies = len(facies)

        # Auxiliary variables
        if len(auxvar_fp) == 0:
            self.n_auxvar = 0
            self.auxvar_treatment = auxvar_treatment

        # Local PDFs
        self.localpdfs = localpdfs
        self.n_localpdfs = len(localpdfs)

        # Training images
        self.ti = ti
        self.n_ti = len(ti)

        # Data templates
        if len(data_templates) == 0:
            data_templates = [Data_template()]
        self.data_templates = data_templates
        self.n_dt = len(data_templates)
        self.dtar = dtar
        self.int_mg = int_mg
        self.max_data = max_data
        self.turbo = turbo
        self.mask_fp = mask_fp
        self.mask = False
        if len(self.mask_fp) > 0:
            self.mask = True
        self.zones = zones
        self.n_zones = len(zones)
        self.servo = servo
        self.servo_type = servo_type
        self.servo_prop = servo_prop
        self.servo_weight = servo_weight
        self.window_scan = window_scan

        self.cond_fp = cond_fp
        self.cond = False
        if self.cond_fp is not None:
            self.cond = True
        self.cond_conn = cond_conn

        self.val_thresh = val_thresh
        self.reject_loop = reject_loop
        self.processing = processing
        self.nreal = nreal
        self.seed = seed
        self.par_fp = par_fp
        self.run_flags = run_flags


    def make_parameter_file(self, fp=None):
        """
        Parameters
        ----------
        fp : str
            file path for saving parameter file

        """
        if fp is None:
            fp = self.par_fp

        tf = {True: 'ON', False: 'OFF'}
        fl = {True: '1', False: '0'}
        pl = []

        # Simulation grid
        pl.append(str(self.ndim))
        pl.append('{} {} {}'.format(self.ox, self.dx, self.nx))
        pl.append('{} {} {}'.format(self.oy, self.dy, self.ny))
        if self.ndim == 3:
            pl.append('{} {} {}'.format(self.oz, self.dz, self.nz))

        # Multigrids and paths
        pl.append(str(self.n_multigrids))
        pl.append('{} '.format(self.n_dir_mg) * self.ndim)
        for i in range(0, self.n_multigrids):
            pl.append(self.mg_pathtype)

        # Facies
        pl.append(str(self.n_facies))
        #pl.append(' '.join(str(i) for i in self.facies))
        pl.append(' '.join(str(i) for i in range(0, self.n_facies))) # Convert facies codes into zero-indexed integers compatible with vtk outputs

        pl.append(str(self.n_auxvar))
        pl.append(str(self.n_localpdfs))

        # Training images
        pl.append(str(self.n_ti))
        for iti in self.ti:
            pl.append(iti.name)
            pl.append(iti.fp)
            pl.append(str(iti.n_auxvar))

        """ Data templates """
        pl.append(str(self.n_dt))
        for idt in self.data_templates:
            pl.append(str(idt.name))
            pl.append('DATA_TEMPLATE_FILE_{}'.format(tf[idt.fkw]))
            pl.append(idt.type)
            pl.append(str(self.ndim))
            pl.append('{} '.format(idt.axis_spacing) * self.ndim)
            pl.append('{} '.format(idt.halfaxis_size) * self.ndim)
            pl.append('ROTATION_{}'.format(tf[idt.rotation]))

        pl.append('DATA_TEMPLATE_AUTO_REDUCTION_{}'.format(tf[self.dtar]))
        pl.append('INTERMEDIATE_MULTIGRID_{}'.format(tf[self.int_mg]))

        if len(self.max_data) == 1:
            pl.append('{} '.format(str(self.max_data[0])) * self.n_multigrids)
            if self.max_data[0] == -1:
                pl.append('0 ' * self.n_multigrids)
            else:
                pl.append('1 ' * self.n_multigrids)
        else:
            pl.append(' '.join([str(i) for i in self.max_data]))
            max_flag = [int(j) for j in [i == -1 for i in self.max_data]]
            pl.append(' '.join([str(i) for i in max_flag]))

        if self.turbo == 'Manual':
            pl.append('TURBO_MANUAL')
        elif self.turbo == 'Auto':
            pl.append('TURBO_AUTO')
        else:
            pl.append('TURBO_NONE')

        pl.append('MASK_MAP_{}'.format(tf[self.mask]))
        if self.mask:
            pl.append(self.mask_fp)

        """ Simulation Zones """
        if len(self.zones) == 0:
            # If no simulation zones assigned then it is assumed only one zone, training image and data_template
            self.zones = [Simulation_zones(601, self.ti[0], Data_template())]
            self.n_zones = 1

        pl.append(str(self.n_zones))
        if self.zones[0].fp is not None:
            pl.append(' '.join([str(i.name) for i in self.zones]))
            pl.append(self.zones[0].fp)
        for zz in self.zones:
            pl.append('{} '.format(zz.min_reps) * self.n_multigrids)
        for zz in self.zones:
            pl.append(zz.ti.name)
            pl.append('{} '.format(zz.dt.name) * self.n_multigrids)
            pl.append('AFFINITY_{}'.format(tf[zz.affinity]))
            pl.append('ROTATION_{}'.format(tf[zz.rotation]))
            pl.append(fl[zz.auxvar])
            pl.append('COMBIN_PDF_{}'.format(tf[zz.combin_pdf]))

        pl.append('SERVO_SYSTEM_{}'.format(tf[self.servo]))
        if self.servo:
            pl.append(self.servo_type)
            if self.servo_type == 'GLOBAL':
                pl.append(' '.join([str(i) for i in self.servo_prop]))
            elif self.servo_type == 'LOCAL':
                for i in self.servo_prop:
                    pl.append(i)
            pl.append('{} '.format(self.servo_weight) * self.n_multigrids)

        pl.append('WINDOW_SCAN_{}'.format(tf[self.window_scan]))
        pl.append('CONDITIONAL_{}'.format(tf[self.cond]))
        if self.cond:
            pl.append('CONNECTIVITY_{}'.format(tf[self.cond_conn]))
            pl.append(self.cond_fp)
        pl.append('{} '.format(self.val_thresh) * self.n_multigrids)
        pl.append('{} '.format(self.reject_loop) * self.n_multigrids)
        pl.append('{}_PROCESSING'.format(self.processing))

        if self.seed is None:
            pl.append('TIME_SEED')
        else:
            pl.append('EXPLICIT_SEED')
            pl.append(str(self.seed))
        pl.append(str(self.nreal))

        # Join list and write to file
        parfile = '\n'.join(pl)
        with open(fp, 'w') as tfile:
            tfile.write(parfile)

    def make_conditioning_data(self, bhdf, col='fac'):
        """
        Create borehole data suitable for impala simulations
        **** No provision for: non-integer values or connected hydrofacies at the moment ****

        Parameters
        ----------
        bhdf : Pandas Dataframe
            Borehole data
        data : str
            Which data to write

        """
        n_conddata = bhdf.shape[0]
        to_write = np.hstack((bhdf.as_matrix(columns=['x', 'y', 'z', col]), np.zeros((bhdf.shape[0], 1))))
        header = '\n'.join([str(self.ndim), str(n_conddata)])
        np.savetxt(self.cond_fp, to_write, delimiter=' ', header=header, comments='', fmt='%.3f %.3f %.3f %i %i')

    def run(self):
        """
        Run impala

        """
        run_args = ['impala',                           # call routine
                    self.par_fp,                        # input parameters
                    self.name,                          # Prefix for output files - must be string
                    self.run_flags['hd_ext_only'],
                    self.run_flags['wd_only'],
                    self.run_flags['res'],
                    self.run_flags['mean'],
                    self.run_flags['read_list'],
                    self.run_flags['write_list'],
                    self.run_flags['log_file']]

        subprocess.call(run_args)

class Training_image():

    def __init__(self, name, fp, auxvar_fp=[]):
        """
        Training image class

        Parameters
        ----------
        name : int
            name of training image - MUST BE INTEGER!
        fp : str
            File path to training image
        auxvar_fp : list of strings, optional
            File paths of auxiliary variables

        """

        self.name = name
        self.fp = fp
        self.auxvar_fp = auxvar_fp
        self.n_auxvar = len(auxvar_fp)

class Data_template():

    def __init__(self,
                 name=501,      # - MUST BE INTEGER!
                 fkw=False,
                 type='ELLIPTIC',
                 axis_spacing=1.0,
                 halfaxis_size=6.2,
                 rotation=False):

        self.name = name        # - MUST BE INTEGER!
        self.fkw = fkw
        self.type = type
        self.axis_spacing = axis_spacing
        self.halfaxis_size = halfaxis_size
        self.rotation = rotation

class Simulation_zones():

    def __init__(self,
                 name,
                 ti,
                 dt,
                 fp=None,
                 affinity=False,
                 rotation=False,
                 auxvar=0,
                 combin_pdf=False,
                 min_reps=1):
        """
        Parameters
        ----------
        name:
        fp : str, optional
            File path to zone map
        ti : Training Image class
        dt : Data Template class

        :param affinity:
        :param rotation:
        :param auxvar:
        :param combin_pdf:

        """

        self.name = name
        self.ti = ti
        self.fp = fp
        self.dt = dt
        self.affinity = affinity
        self.rotation = rotation
        self.auxvar = auxvar
        self.combin_pdf = combin_pdf
        self.min_reps = min_reps


ds_variable_formats = {'k_iso': '%10.5E',
                       'poros': '%10.5E',
                       'lnK': '%10.5E',
                       'dip': '%10.5E',
                       'azim': '%10.5E',
                       'fac': '%U'}

ds_output_settings = {''}


class deesse_model():

    def __init__(self,
                 name,
                 ti,
                 par_fp,
                 ox=0., dx=1., nx=10,
                 oy=0., dy=1., ny=10,
                 oz=0., dz=1., nz=1,
                 sim_var=[['varName1', 1, 'DEFAULT_FORMAT']],
                 output_settings=['per_real', 'test'],
                 output_report=[True, 'test_report.txt'],
                 data_image_files=[],
                 cond_data_files=[],
                 mask_image=None,
                 homothety=False,
                 rotation=False,
                 cond_cons=-1,
                 norm_type='l',
                 sim_type='4D',
                 path_type='r',
                 tol=0.0,
                 postpro=None,
                 pyramids=0,
                 nreal=1,
                 seed=1234):
        """
        Parameters
        ----------
        name : string
            Name of the parameter file
        facies : list of strings
            Facies to be modelled with impala
        ti : list of training image classes
            Training images to be included in MPS simulation
        ox, oy, oz : float
            Origin in x,y,z-direction
        dx, dy, dz : float
            Dimension of simulation grid cell in x,y,z-direction
        nx, ny, nz : int
            Number of simulation grid cell in x,y,z-direction
        sim_var : list of lists
            Simulation variables
        output_settings : list
            [<option: 'no_file'/'one_file'/'per_var'/'per_real'>, option requirement]
            Key word and required name(s) or prefix, for output of the realizations:
                - OUTPUT_SIM_NO_FILE:
                no file in output,
                - OUTPUT_SIM_ALL_IN_ONE_FILE:
                one file in output,
                requires one file name
                - OUTPUT_SIM_ONE_FILE_PER_VARIABLE:
                one file per variable in output (flagged as 1 above),
                requires as many file name(s) as variable(s) flagged as 1 above
                - OUTPUT_SIM_ONE_FILE_PER_REALIZATION:
                one file per realization,
                requires one prefix (for file name)
        output_report : list [bool, str]

        data_image_files : list of strings
            Data image files

                 data_pointset_files=[],
                 mask_image=None,

         pyramids : 0 or DeeSse_pyramids class




        nreal : int, optional
            Number of realisations
        seed : int
            Seed number for realizations
        par_fp : str
            Location to save parameter file

        """

        self.ti = ti
        self.n_ti = len(ti)
        self.name = name
        self.par_fp = par_fp
        self.ox = ox
        self.dx = dx
        self.nx = nx
        self.oy = oy
        self.dy = dy
        self.ny = ny
        self.oz = oz
        self.dz = dz
        self.nz = nz
        if self.nz is not None:
            self.ndim = 3
        else:
            self.ndim = 2

        self.sim_var = sim_var
        self.output_settings = output_settings
        self.output_report = output_report
        self.data_image_files = data_image_files
        self.cond_data_files = cond_data_files
        self.mask_image = mask_image
        self.homothety = homothety
        self.rotation = rotation
        self.cond_cons = cond_cons
        self.norm_type = norm_type
        self.sim_type = sim_type
        self.path_type = path_type
        self.tol = tol
        self.postpro = postpro
        self.pyramids = pyramids
        self.nreal = nreal
        self.seed = seed

        # Create folder for modelling
        self.basedir = os.path.dirname(par_fp)
        try:
            os.makedirs(self.basedir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        # Copy training images into folder
        for tii in self.ti:
            copied_ti = os.path.join(self.basedir, os.path.split(tii.fpath)[1])

            # Check that variables in ci are same as for simulation
            with open(tii.fpath, 'r') as in_t:
                nc = [i for i in next(in_t).strip().split()]                         # Number of data points
                ti_nv = int(next(in_t).strip())                      # Number of properties
                ti_v = [next(in_t).strip() for v in range(ti_nv)]    # Properties
                sim_props = [v.prop for v in self.sim_var]

                # Check if order of cond data is same as sim_vars
                if not ti_v == sim_props:

                    # Create mapping between in and out point data
                    im = dict()
                    for vv in sim_props:
                        im[vv] = ti_v.index(vv)

                    # Remove the relevant columns from ci and save new
                    with open(copied_ti, 'w') as outt:
                        outt.write(' '.join([str(i) for i in nc]) + '\n')
                        outt.write(str(len(sim_props)) + '\n')
                        outt.write('\n'.join(sim_props) + '\n')

                        # Loop over all lines
                        for ii in range(np.product([int(i) for i in nc[:3]])):
                            linein = next(in_t).strip().split()           # Get data
                            lineout = ' '.join([linein[im[st]] for st in sim_props])
                            outt.write(lineout + '\n')

                else:
                    copyfile(tii.fpath, copied_ti)


        # Copy conditioning data into folder
        for ci in self.cond_data_files:
            copied_ci = os.path.join(self.basedir, os.path.split(ci)[1])
            # Check that variables in ci are same as for simulation
            with open(ci, 'r') as inc:
                nc = int(next(inc).strip())                         # Number of data points
                ci_nv = int(next(inc).strip())                      # Number of properties
                ci_v = [next(inc).strip() for v in range(ci_nv)]    # Properties
                sim_props = [v.prop for v in self.sim_var]
                out_v = ['x', 'y', 'z']
                out_v.extend(sim_props)

                # Check if order of cond data is same as sim_vars
                if not ci_v[3:] == sim_props:

                    # Create mapping between in and out point data
                    im = {'x': 0,
                          'y': 1,
                          'z': 2}
                    for vv in sim_props:
                        im[vv] = ci_v.index(vv)

                    # Remove the relevant columns from ci and save new
                    with open(copied_ci, 'w') as outc:
                        outc.write(str(nc) + '\n')
                        outc.write(str(3 + len(sim_props)) + '\n')
                        outc.write('\n'.join(['x', 'y', 'z']) + '\n')
                        outc.write('\n'.join(sim_props) + '\n')

                        # Loop over all lines
                        for ii in range(nc):
                            linein = next(inc).strip().split()           # Get data
                            lineout = ' '.join([linein[im[st]] for st in out_v])
                            outc.write(lineout + '\n')

                else:
                    copyfile(ci, copied_ci)

    def make_parameter_file(self, fp=None):
        """
        Parameters
        ----------
        fp : str
            file path for saving parameter file

        """
        if fp is None:
            fp = self.par_fp

        tf = {True: 'ON', False: 'OFF'}
        fl = {True: '1', False: '0'}
        pl = []

        # Simulation grid
        pl.append('{} {} {}'.format(self.nx, self.ny, self.nz))
        pl.append('{} {} {}'.format(self.dx, self.dy, self.dz))
        pl.append('{} {} {}'.format(self.ox, self.oy, self.oz))

        # simulation variables
        pl.append(str(len(self.sim_var)))
        for vari in self.sim_var:
            pl.append('{} {} {}'.format(vari.prop, fl[vari.output], vari.format))

        # Output settings
        #[<option: 'no_file'/'one_file'/'per_var'/'per_real'>, option requirement]#
        if self.output_settings[0] == 'no_file':
            pl.append('OUTPUT_SIM_NO_FILE')

        elif self.output_settings[0] == 'one_file':
            pl.append('OUTPUT_SIM_ALL_IN_ONE_FILE')
            pl.append(self.output_settings[1])

        elif self.output_settings[0] == 'per_var':
            pl.append('OUTPUT_SIM_ONE_FILE_PER_VARIABLE')
            for ii in self.output_settings[1:]:
                pl.append(ii)

        elif self.output_settings[0] == 'per_real':
            pl.append('OUTPUT_SIM_ONE_FILE_PER_REALIZATION')
            pl.append(self.output_settings[1])

        # Output report
        pl.append(fl[self.output_report[0]])
        if self.output_report[0] is True:
            pl.append(self.output_report[1])

        # Training images
        pl.append(str(self.n_ti))
        for iti in self.ti:
            pl.append(os.path.split(iti.fpath)[1])

        # Data image files for SG
        pl.append(str(len(self.data_image_files)))
        if len(self.data_image_files) > 0:
            for di in self.data_image_files:
                pl.append(os.path.split(di)[1])

        # Data point set files for SG
        pl.append(str(len(self.cond_data_files)))
        if len(self.cond_data_files) > 0:
            for di in self.cond_data_files:
                pl.append(os.path.split(di)[1])

        # Mask image
        if self.mask_image is not None:
            pl.append('1')
            pl.append(self.mask_image)
        else:
            pl.append('0')

        # Homothety
        if self.homothety is True:
            print('Under construction')
            pass
        else:
            pl.append(fl[self.homothety])

        # Rotation
        if self.rotation is True:
            print('Under construction')
            pass
        else:
            pl.append(fl[self.rotation])

        # CONSISTENCY OF CONDITIONING DATA
        pl.append(str(self.cond_cons))

        # Normalisation type
        normd = {'l': 'LINEAR', 'u': 'UNIFORM', 'n': 'normal'}
        pl.append('NORMALIZING_{}'.format(normd[self.norm_type]))

        # Search neighbourhood parameters
        for vari in self.sim_var:
            pl.append(' '.join([str(i) for i in vari.search_radius]))
            pl.append(' '.join([str(i) for i in vari.search_anirat]))
            pl.append(' '.join([str(i) for i in vari.search_angles]))
            pl.append(str(vari.power))

        for vari in self.sim_var:
            pl.append(str(vari.max_neighbours))

        for vari in self.sim_var:
            pl.append(str(vari.max_dens))

        for vari in self.sim_var:
            pl.append(str(vari.rel_dist))

        for vari in self.sim_var:
            pl.append(str(vari.dist_type))

        for vari in self.sim_var:
            pl.append(str(vari.cond_weight))

        # Simulation and path parameters
        simtd = {'4D': 'ONE_BY_ONE', '3D': 'VARIABLE_VECTOR'}
        pl.append('SIM_{}'.format(simtd[self.sim_type]))

        simtd = {'r': 'RANDOM', '3D': 'UNILATERAL:'}
        pl.append('PATH_{}'.format(simtd[self.path_type]))

        for vari in self.sim_var:
            pl.append(str(vari.dist_thresh))

        for vari in self.sim_var:
            pl.append(str(vari.prob_constraint))

        for vari in self.sim_var:
            pl.append(str(vari.block_data))

        for tii in self.ti:
            pl.append(str(tii.max_scan_fraction))

        pl.append(str(self.tol))

        if self.postpro is None:
            pl.append('1')
            pl.append('POST_PROCESSING_PARAMETERS_DEFAULT')

        # Pyramids
        if self.pyramids == 0:
            pl.append('0')
        else:
            pl.append(str(self.pyramids.nlevel))
            for nn in range(0, self.pyramids.nlevel):
                pl.append(' '.join([str(i) for i in self.pyramids.red_step]))
            pl.append(self.pyramids.sim_mode)
            pl.append(self.pyramids.max_number_factor)
            pl.append(self.pyramids.adapt_dist_thresh)

            for vari in self.sim_var:
                pl.append(str(self.pyramids.nlevel))
                if vari.data_type == 'categorical':
                    pl.append(self.pyramids.var_ptype_cat)
                else:
                    pl.append(self.pyramids.var_ptype_cont)

        pl.append(str(self.seed))
        pl.append('1')                  # Seed increment
        pl.append(str(self.nreal))
        pl.append('END')

        # Join list and write to file
        parfile = '\n'.join(pl)
        with open(fp, 'w') as tfile:
            tfile.write(parfile)

    def run(self, nthreads=-2):
        """
        Run DeeSse

        """

        bindir = 'C:\\Program Files\\Neuchatel\\deesse-windows-20170530\\bin\\'     # DeeSse binaries directory
        inpar = os.path.split(self.par_fp)[-1]
        rpath = os.path.dirname(os.path.realpath(self.par_fp))

        if nthreads == 0:
            run_args = ' '.join([os.path.join(bindir, 'deesse'),                           # call routine
                                 inpar])                        # input parameters
        else:
            run_args = ' '.join([os.path.join(bindir, 'deesseOMP'),
                                 str(nthreads),
                                 inpar])                        # input parameters

        subprocess.call(run_args, cwd=rpath)


class deesse_variable():

    def __init__(self,
                 prop,
                 output=True,
                 search_radius=[-1.0, -1.0, -1.0],
                 search_anirat=[1.0, 1.0, 1.0],
                 search_angles=[0.0, 0.0, 0.0],
                 power=0.0,
                 max_neighbours=24,
                 max_dens=1.0,
                 rel_dist=0,
                 cond_weight=1.0,
                 dist_thresh=0.05,
                 prob_constraint=0,
                 block_data=0,
                 format='%.3e'):

        self.prop = prop
        self.output = output
        self.search_radius = search_radius
        self.search_anirat = search_anirat
        self.search_angles = search_angles
        self.power = power
        self.max_neighbours = max_neighbours
        self.max_dens = max_dens
        self.rel_dist = rel_dist
        self.cond_weight = cond_weight
        self.dist_thresh = dist_thresh
        self.prob_constraint = prob_constraint
        self.block_data = block_data
        self.dformat = ds_variable_formats[self.prop]
        self.format = format

        if prop in ['fac', 'ae', 'ha']:
            self.data_type = 'categorical'
            self.dist_type = 0
        else:
            self.data_type = 'continuous'
            self.dist_type = 2


class deesse_ti():

    def __init__(self,
                 name,
                 fpath,
                 max_scan_fraction=0.25):

        self.name = name
        self.fpath = fpath
        self.max_scan_fraction = max_scan_fraction

class deesse_pyramid():

    def __init__(self,
                 nlevel=2,
                 red_step=[2, 2, 2],
                 sim_mode='hue',
                 max_number_factor='PYRAMID_ADAPTING_FACTOR_DEFAULT',
                 adapt_dist_thresh='PYRAMID_ADAPTING_FACTOR_DEFAULT',
                 var_nlevel=[],
                 var_ptype_cont='PYRAMID_CONTINUOUS',
                 var_ptype_cat='PYRAMID_CATEGORICAL_AUTO'):

        self.nlevel = nlevel
        self.red_step = red_step
        if sim_mode == 'hue':
            self.sim_mode = 'PYRAMID_SIM_HIERARCHICAL_USING_EXPANSION'
        elif sim_mode == 'h':
            self.sim_mode = 'PYRAMID_SIM_HIERARCHICAL'
        elif sim_mode == '1b1':
            self.sim_mode = 'PYRAMID_SIM_ALL_LEVEL_ONE_BY_ONE'
        self.max_number_factor = max_number_factor
        self.adapt_dist_thresh = adapt_dist_thresh
        if len(var_nlevel) == 0:
            self.var_nlevel = nlevel
        else:
            self.var_nlevel = var_nlevel
        self.var_ptype_cat = var_ptype_cat
        self.var_ptype_cont = var_ptype_cont


if __name__ == '__main__':
    """ Testing functions"""

    hyvr_file = '..\\..\\fidelity\\runfiles\\braid001\\braid_vr\\braid_vr.vtr'
    ti1 = Training_image('hyvr', hyvr_file)
    dt1 = Data_template()
    sz1 = Simulation_zones('sz1', ti1, dt1)

    fac = [0, 1, 2]
    imppy = Impala_parameters('i', fac, [ti1], zones=[sz1])
    imppy.make_parameter_file(os.path.join(os.path.dirname(hyvr_file), 'impala.in'))
    print('ff')

