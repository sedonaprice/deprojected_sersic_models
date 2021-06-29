# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of sersic_profile_mass_VC pre-computed tables

import os
import shutil
import math

import numpy as np

# Supress warnings: Runtime & integration warnings are frequent
import warnings
warnings.filterwarnings("ignore")


from sersic_profile_mass_VC import core, io
from sersic_profile_mass_VC.utils import calcs as util_calcs
#
# def check_all_keys(table_old, table_new, i, ftol=None):
#     keys = ['menc3D_sph', 'vcirc', 'rho', 'dlnrho_dnr']
#     for key in keys:
#         assert math.isclose(table_old[key][i], table_new[key][i], rel_tol=ftol), \
#             '{}, n={}, invq={}, ftol='.format(key,table_new['n'],table_new['invq'],ftol)

#
# def check_all_keys(table_old, table_new, i, ftol=None):
#     keys = ['menc3D_sph', 'vcirc', 'rho', 'dlnrho_dnr']
#     for key in keys:
#         assert math.isclose(table_old[key][i], table_new[key][i], rel_tol=ftol), \
#             '{}, r={:0.2e}, n={:0.1f}, invq={:0.1f}, ftol={:0.1e}'.format(key,
#                     table_new['r'][i], table_new['n'],table_new['invq'],ftol)

_PATH_NEW = BOB

def check_all_keys(table_old, table_new, i, ftol=ftol):
    keys = ['menc3D_sph', 'vcirc', 'rho', 'dlnrho_dnr']
    for key in keys:
        for i, r in enumerate(table_old['r']):
            assert math.isclose(table_old['r'][i], table_new['r'][i], rel_tol=ftol), \
                '{}, r_old={}, r_new={}'.format(key, table_old['r'][i], table_new['r'][i])
            assert math.isclose(table_old[key][i], table_new[key][i], rel_tol=ftol), \
                '{}, r={:0.2e}, n={:0.1f}, invq={:0.1f}, ftol={:0.1e}'.format(key,
                        r, table_new['n'],table_new['invq'],ftol)



class TestSersicSavedTableNewOld:
    ftol = 1.e-9
    ftol_high = 3.e-9

    def setup_tables(self, n=1., invq=2.5, path_new=None):
        table_old = io.read_profile_table(n=n, invq=invq)
        table_new = io.read_profile_table(n=n, invq=invq, path=path_new)
        return table_old, table_new

    def check_saved_tables_n_invq(self, n=None, invq=None, ftol=None, path_new=None):
        if ftol is None:
            ftol = self.ftol
        table_old, table_new = self.setup_table_sprof_rarr(n=n, invq=invq, path_new=path_new)

        check_all_keys(table_old, table_new, i, ftol=ftol)

        assert math.isclose(table_old['ktot_Reff'], table_new['ktot_Reff'], rel_tol=ftol)
        assert math.isclose(table_old['k3D_sph_Reff'],table_new['k3D_sph_Reff'], rel_tol=ftol)

    def test_all_saved_tables(self, path_new=_PATH_NEW):
        # Sersic indices
        n_arr =         np.arange(0.5, 8.1, 0.1)
        # Flattening ratio invq
        invq_arr =      np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 20., 100.,
                                1.11, 1.25, 1.43, 1.67, 2.5, 3.33,
                                0.5, 0.67])
        for n in n_arr:
            for invq in invq_arr:
                self.check_saved_tables_n_invq(n=n, invq=invq, path_new=path_new)
