# -*- coding: utf-8 -*-
""" Script for running conditional simulations on HyVR virtual borehole data """

# Mystery modules
import dtgeostats.gcosim3d as dg
import hyvr

inputs = {'ini': 'braid.ini',               # Configuration file
         'fbh': 'braid_vr_bh25.csv',       # Borehole data file
          }

""" Load data """
run, mod, strata, hydraulics, ft, elements, mg = hyvr.parameters.model_setup(inputs['ini'], nodir=True)

kc = dg.variable('lnK', 1, inputs['fbh'], 8, vform='%.3f')
pc = dg.variable('poros', 2, inputs['fbh'], 4, vform='%.3f')

vgm_1 = dg.vmodel(kc.name, rx=10.8, ry=10, rz=2, type=2, sill=6)
vgm_2 = dg.vmodel(pc.name, rx=12, ry=10, rz=2, type=2, sill=0.004)
vgm_1x2 = dg.vmodel([kc.name, pc.name], rx=11.1, ry=10, rz=2, type=2, sill=0.13)

mod = dg.model('test_pygcosim',
               [kc, pc],
               [vgm_1, vgm_1x2, vgm_2],
               nsim=2,
               ellips_radii=[100, 100, 20],
               max_octant=2,
               max_data=12,
               ox=mg.ox, dx=mg.dx, nx=mg.nx,
               oy=mg.oy, dy=mg.dy, ny=mg.ny,
               oz=mg.oz, dz=mg.dz, nz=mg.nz,
               batchfname='run_gcosim3d.bat')

mod.make_geo_file()
mod.make_var_file()
mod.make_dat_file()
execpath = r'D:\Jeremy\IRTG\Repos\dtgeostats\testcases\hyvr\test_pygcosim'
execpath = r'E:\Software\gcosim3dv12'
# mod.run(copyexec=False, gcosim_execpath=execpath, makebatch=True)
dd = mod.out(f=['vtr', 'h5'], fout='test_pygcosim')


print('bb')
