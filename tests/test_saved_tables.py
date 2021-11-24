# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of sersic_profile_mass_VC pre-computed tables

import os
import shutil
import math

import numpy as np

from sersic_profile_mass_VC import core, io
from sersic_profile_mass_VC.utils import calcs as util_calcs


class TestSersicSavedTable:
    ftol = 3.e-8

    def setup_table_sprof_Rarr(self, n=1., invq=2.5):
        table = io.read_profile_table(n=n, invq=invq)
        sprof = core.DeprojSersicDist(total_mass=table['total_mass'], Reff=table['Reff'],
                                      n=table['n'], q=table['q'])
        Rarr = np.array([0.1, 1., 10.]) * table['Reff']

        return table, sprof, Rarr

    def check_saved_table_n_invq(self, n=None, invq=None, ftol=None):
        if ftol is None:
            ftol = self.ftol
        table, sprof, Rarr = self.setup_table_sprof_Rarr(n=n, invq=invq)
        for R in Rarr:
            i = np.where(table['R'] == R)[0][0]
            mencr = sprof.enclosed_mass(R)
            vcircr = sprof.v_circ(R)
            assert math.isclose(mencr, table['menc3D_sph'][i], rel_tol=ftol)
            assert math.isclose(vcircr, table['vcirc'][i], rel_tol=ftol)
            assert math.isclose(sprof.density(R), table['rho'][i], rel_tol=ftol)
            assert math.isclose(sprof.dlnrho_dlnR(R), table['dlnrho_dlnR'][i], rel_tol=ftol)

            if R == table['Reff']:
                assert math.isclose(util_calcs.virial_coeff_tot(sprof.Reff,
                                    total_mass=sprof.total_mass, vc=vcircr),
                                    table['ktot_Reff'], rel_tol=ftol)
                assert math.isclose(util_calcs.virial_coeff_3D(sprof.Reff,
                                    m3D=mencr, vc=vcircr),
                                    table['k3D_sph_Reff'], rel_tol=ftol)


    def test_saved_table_n1_invq1_medftol(self):
        self.check_saved_table_n_invq(n=1., invq=1., ftol=3.e-8)

    def test_saved_table_n1_invq25_medftol(self):
        self.check_saved_table_n_invq(n=1., invq=2.5, ftol=3.e-8)

    def test_saved_table_n1_invq5_medftol(self):
        self.check_saved_table_n_invq(n=1., invq=5., ftol=3.e-8)

    def test_saved_table_n4_invq1_medftol(self):
        # Higher ftol, bc n>=2 uses cumulative for mass -- small diffs
        self.check_saved_table_n_invq(n=4., invq=1., ftol=5.e-7)

    def test_saved_table_n05_invq4_medftol(self):
        self.check_saved_table_n_invq(n=0.5, invq=4., ftol=3.e-8)
