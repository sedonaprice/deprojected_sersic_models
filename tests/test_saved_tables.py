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
    ftol = 1.e-9
    #ftol_high = 3.e-9

    def setup_table_sprof_rarr(self, n=1., invq=2.5):
        table = io.read_profile_table(n=n, invq=invq)
        sprof = core.DeprojSersicDist(total_mass=table['total_mass'], Reff=table['Reff'],
                                      n=table['n'], q=table['q'])
        rarr = np.array([0.1, 1., 10.]) * table['Reff']

        return table, sprof, rarr

    def check_saved_table_n_invq(self, n=None, invq=None, ftol=None):
        if ftol is None:
            ftol = self.ftol
        table, sprof, rarr = self.setup_table_sprof_rarr(n=n, invq=invq)
        for r in rarr:
            i = np.where(table['r'] == r)[0][0]
            mencr = sprof.enclosed_mass(r)
            vcircr = sprof.v_circ(r)
            assert math.isclose(mencr, table['menc3D_sph'][i], rel_tol=ftol)
            assert math.isclose(vcircr, table['vcirc'][i], rel_tol=ftol)
            assert math.isclose(sprof.density(r), table['rho'][i], rel_tol=ftol)
            assert math.isclose(sprof.dlnrho_dlnr(r), table['dlnrho_dlnr'][i],
                                rel_tol=ftol) #ftol_high

            if r == table['Reff']:
                assert math.isclose(util_calcs.virial_coeff_tot(sprof.Reff,
                                    total_mass=sprof.total_mass, vc=vcircr),
                                    table['ktot_Reff'], rel_tol=ftol)
                assert math.isclose(util_calcs.virial_coeff_3D(sprof.Reff,
                                    m3D=mencr, vc=vcircr),
                                    table['k3D_sph_Reff'], rel_tol=ftol)

    # def test_saved_table_n1_invq1(self):
    #     self.check_saved_table_n_invq(n=1., invq=1., ftol=1.e-9)
    #
    # def test_saved_table_n1_invq25(self):
    #     self.check_saved_table_n_invq(n=1., invq=2.5, ftol=1.e-9)
    #
    # def test_saved_table_n1_invq5(self):
    #     self.check_saved_table_n_invq(n=1., invq=5., ftol=1.e-9)
    #
    # def test_saved_table_n4_invq1(self):
    #     self.check_saved_table_n_invq(n=4., invq=1., ftol=1.e-9)
    #
    # def test_saved_table_n05_invq4(self):
    #     self.check_saved_table_n_invq(n=0.5, invq=4., ftol=1.e-9)



    def test_saved_table_n1_invq1_medftol(self):
        self.check_saved_table_n_invq(n=1., invq=1., ftol=3.e-8)

    def test_saved_table_n1_invq25_medftol(self):
        self.check_saved_table_n_invq(n=1., invq=2.5, ftol=3.e-8)

    def test_saved_table_n1_invq5_medftol(self):
        self.check_saved_table_n_invq(n=1., invq=5., ftol=3.e-8)

    def test_saved_table_n4_invq1_medftol(self):
        self.check_saved_table_n_invq(n=4., invq=1., ftol=3.e-8)

    def test_saved_table_n05_invq4_medftol(self):
        self.check_saved_table_n_invq(n=0.5, invq=4., ftol=3.e-8)


    # def test_saved_table_n1_invq1_highftol(self):
    #     self.check_saved_table_n_invq(n=1., invq=1., ftol=1.e-6)
    #
    # def test_saved_table_n1_invq25_highftol(self):
    #     self.check_saved_table_n_invq(n=1., invq=2.5, ftol=1.e-6)
    #
    # def test_saved_table_n1_invq5_highftol(self):
    #     self.check_saved_table_n_invq(n=1., invq=5., ftol=1.e-6)
    #
    # def test_saved_table_n4_invq1_highftol(self):
    #     self.check_saved_table_n_invq(n=4., invq=1., ftol=1.e-6)
    #
    # def test_saved_table_n05_invq4_highftol(self):
    #     self.check_saved_table_n_invq(n=0.5, invq=4., ftol=1.e-6)
