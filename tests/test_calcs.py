# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of deprojected_sersic_models calculations

import os
import shutil
import math

import numpy as np

from deprojected_sersic_models import core, io
from deprojected_sersic_models.utils import calcs as util_calcs
try:
    from deprojected_sersic_models.paper import plot_calcs
    _plot_calcs_loaded = True
except:
    _plot_calcs_loaded = False


# TESTING DIRECTORY
path = os.path.abspath(__file__)
_dir_tests = os.path.dirname(path) + '/'
_dir_tests_output = _dir_tests+'PYTEST_OUTPUT/'

class TestSersic:
    smodel = core.DeprojSersicModel(total_mass=1.e10, Reff=1., n=1., q=0.4, i=90., Upsilon=1.)
    Rarr = np.array([0.,0.5,1.,2.,5.])
    ftol = 1.e-9

    menc_true = np.array([0.0, 1599797392.8157372, 4590528447.530832,
                          8344356142.958505, 9976766033.920475])
    vcirc_true = np.array([0.0, 110.79794643694885, 142.1678809832618,
                           142.50346201160482, 95.55561194815215])
    rho_true = np.array([219587465413.20197, 3190256524.945349, 1018485744.913661,
                         138126431.7048887, 579136.752045775])
    dlnrho_dlnR_true = np.array([0.0, -1.26088233092482404629, -2.12849502914837263035,
                                 -3.82725604392065399395, -8.87834716438393556359])
    surf_dens_true = np.array([11207884540.979828, 4842562450.209616, 2092313763.4437168,
                               390597964.2000569, 2541203.623721638])
    menc_2D_proj_true = np.array([0.0, 2053529393.7195516, 5000000000.0,
                                  8481679755.870425, 9978705784.48509])
    menc_ellip_true = np.array([0.0, 989023978.4512446, 3409221465.774015,
                                7488664919.810521, 9948371392.047262])


    def test_menc(self):
        for i, R in enumerate(self.Rarr):
            assert math.isclose(self.smodel.enclosed_mass(R), self.menc_true[i], rel_tol=self.ftol)

    def test_vcirc(self):
        for i, R in enumerate(self.Rarr):
            assert math.isclose(self.smodel.v_circ(R), self.vcirc_true[i], rel_tol=self.ftol)

    def test_density(self):
        for i, R in enumerate(self.Rarr):
            assert math.isclose(self.smodel.density(R), self.rho_true[i], rel_tol=self.ftol)

    def test_dlnrho_dlnR(self):
        for i, R in enumerate(self.Rarr):
            assert math.isclose(self.smodel.dlnrho_dlnR(R), self.dlnrho_dlnR_true[i], rel_tol=self.ftol)

    def test_surface_density(self):
        for i, R in enumerate(self.Rarr):
            assert math.isclose(self.smodel.surface_density(R), self.surf_dens_true[i], rel_tol=self.ftol)

    def test_projected_enclosed_mass(self):
        for i, R in enumerate(self.Rarr):
            assert math.isclose(self.smodel.projected_enclosed_mass(R),
                                self.menc_2D_proj_true[i], rel_tol=self.ftol)

    def test_enclosed_mass_ellipsoid(self):
        for i, R in enumerate(self.Rarr):
            assert math.isclose(self.smodel.enclosed_mass_ellipsoid(R),
                                self.menc_ellip_true[i], rel_tol=self.ftol)

    def test_table(self):
        table = self.smodel.profile_table(self.Rarr)

        # Save table
        io.save_profile_table(table=table, path=_dir_tests_output, overwrite=True)
        assert os.path.isfile(io._default_table_fname(_dir_tests_output,
                                                      io._sersic_profile_filename_base,
                                                      table['n'], table['invq']))

        for i in range(len(table['R'])):
            assert math.isclose(table['menc3D_sph'][i], self.menc_true[i], rel_tol=self.ftol)
            assert math.isclose(table['vcirc'][i], self.vcirc_true[i], rel_tol=self.ftol)
            assert math.isclose(table['rho'][i], self.rho_true[i], rel_tol=self.ftol)
            assert math.isclose(table['dlnrho_dlnR'][i], self.dlnrho_dlnR_true[i], rel_tol=self.ftol)

        assert math.isclose(table['ktot_Reff'], 2.127933776848163, rel_tol=self.ftol)
        assert math.isclose(table['k3D_sph_Reff'], 0.9768340537083219, rel_tol=self.ftol)
        assert math.isclose(table['rhalf3D_sph'], 1.1090810728920568, rel_tol=self.ftol)




class TestHalos:
    if _plot_calcs_loaded:
        nfw = plot_calcs.NFW(z=2., Mvir=1.e12, conc=4.)
        tph = plot_calcs.TPH(z=2., Mvir=1.e12, conc=4., alpha=0., beta=3.)
        Rarr = np.array([0.,5.,10.,15.,20.])
        ftol = 1.e-9

    def test_rvir(self):
        if _plot_calcs_loaded:
            assert math.isclose(self.nfw.rvir, 99.9142529492978, rel_tol=self.ftol)
            assert math.isclose(self.tph.rvir, 99.9142529492978, rel_tol=self.ftol)

    def test_menc(self):
        if _plot_calcs_loaded:
            menc_true_nfw = np.array([0.0, 19369905672.91485, 62794218888.21235,
                                      117518991877.56961, 177298034319.76498 ])
            menc_true_tph = np.array([0.0, 3616346279.4748654, 20353246885.7532,
                                      50540431345.8649, 91231496358.292 ])
            for i, R in enumerate(self.Rarr):
                assert math.isclose(self.nfw.enclosed_mass(R), menc_true_nfw[i], rel_tol=self.ftol)
                assert math.isclose(self.tph.enclosed_mass(R), menc_true_tph[i], rel_tol=self.ftol)

    def test_vcirc(self):
        if _plot_calcs_loaded:
            vcirc_true_nfw = np.array([0., 129.08010058, 164.33889998, 183.56460474, 195.2618982 ])
            vcirc_true_tph = np.array([0.,  55.77384005,  93.56154714, 120.38001329, 140.06768333])
            for i, R in enumerate(self.Rarr):
                assert math.isclose(self.nfw.v_circ(R), vcirc_true_nfw[i], rel_tol=self.ftol)
                assert math.isclose(self.tph.v_circ(R), vcirc_true_tph[i], rel_tol=self.ftol)
