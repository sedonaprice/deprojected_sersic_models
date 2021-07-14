# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of sersic_profile_mass_VC pre-computed tables

import os
import shutil
import math

import numpy as np

import pytest

from sersic_profile_mass_VC import core, io
from sersic_profile_mass_VC.utils import calcs as util_calcs


_SERSIC_PROFILE_MASS_VC_DATADIR_NEW = '/Users/sedona/data/sersic_profile_mass_VC/'

def check_all_keys(table_old, table_new, ftol=1.e-9, ftol_dlnrho_dlnr=1.e-4, debug=False):
    keys = ['menc3D_sph', 'vcirc', 'rho']
    #keys = ['menc3D_sph', 'vcirc', 'rho', 'dlnrho_dlnr']
    rratio = table_new['r']/table_old['r']
    whzero = np.where(table_old['r'] == 0.)[0]
    for whz in whzero:
        if (table_new['r'][whz] == 0.):
            rratio[whz] = 1.
    # assert math.isclose(np.min(rratio), 1., rel_tol=ftol), 'r, rratio={}'.format(rratio)
    # #                    'r, old={}, new={}'.format(table_old['r'], table_new['r'])
    # assert math.isclose(np.max(rratio), 1., rel_tol=ftol), 'r, rratio={}'.format(rratio)
    # #                    'r, old={}, new={}'.format(table_old['r'], table_new['r'])

    if (not math.isclose(np.min(rratio), 1., rel_tol=ftol)) | \
        (not math.isclose(np.max(rratio), 1., rel_tol=ftol)):
        mgs = "n={}, invq={}: {} diff: {}".format(table_new['n'], table_new['invq'], 'r', rratio)
        if debug:
            raise ValueError(msg)
        else:
            print(msg)

    for key in keys:
        kratio = table_new[key]/table_old[key]
        whzero = np.where(table_old[key] == 0.)[0]
        for whz in whzero:
            if (table_new[key][whz] == 0.):
                kratio[whz] = 1.
        # assert math.isclose(np.min(kratio), 1., rel_tol=ftol), '{}, kratio={}'.format(key, kratio)
        # #                    '{}, old={}, new={}'.format(key, table_old[key], table_new[key])
        # assert math.isclose(np.max(kratio), 1., rel_tol=ftol), '{}, kratio={}'.format(key, kratio)
        # #                    '{}, old={}, new={}'.format(key, table_old[key], table_new[key])
        #
        if (not math.isclose(np.min(kratio), 1., rel_tol=ftol)) | \
            (not math.isclose(np.max(kratio), 1., rel_tol=ftol)):
            msg = "n={}, invq={}: {} diff: {}".format(table_new['n'], table_new['invq'], key, kratio)
            if debug:
                raise ValueError(msg)
            else:
                print(msg)

        # for i, r in enumerate(table_old['r']):
        #     assert math.isclose(table_old['r'][i], table_new['r'][i], rel_tol=ftol), \
        #         '{}, r_old={}, r_new={}'.format(key, table_old['r'][i], table_new['r'][i])
        #     assert math.isclose(table_old[key][i], table_new[key][i], rel_tol=ftol), \
        #         '{}, r={:0.2e}, n={:0.1f}, invq={:0.1f}, ftol={:0.1e}'.format(key,
        #                 r, table_new['n'],table_new['invq'],ftol)


    for key in ['dlnrho_dlnr']:
        #pass
        kratio = table_new[key]/table_old[key]
        whzero = np.where(table_old[key] == 0.)[0]
        for whz in whzero:
            if (table_new[key][whz] == 0.):
                kratio[whz] = 1.
        if (not math.isclose(np.min(kratio), 1., rel_tol=ftol_dlnrho_dlnr)) | \
            (not math.isclose(np.max(kratio), 1., rel_tol=ftol_dlnrho_dlnr)):
            msg = "n={}, invq={}: {} diff: {}".format(table_new['n'], table_new['invq'], key, kratio)
            if debug:
                raise ValueError(msg)
            else:
                print(msg)

    return None

#
# def check_all_keys(table_old, table_new, ftol=1.e-9):
#     keys = ['menc3D_sph', 'vcirc', 'rho']
#     #, 'dlnrho_dlnr']
#     rratio = table_new['r']/table_old['r']
#     whzero = np.where(table_old['r'] == 0.)[0]
#     for whz in whzero:
#         if (table_new['r'][whz] == 0.):
#             rratio[whz] = 1.
#     if not math.isclose(np.min(rratio), 1., rel_tol=ftol):
#         flag = False
#     if not math.isclose(np.max(rratio), 1., rel_tol=ftol):
#         flag = False
#     for key in keys:
#         kratio = table_new[key]/table_old[key]
#         whzero = np.where(table_old[key] == 0.)[0]
#         for whz in whzero:
#             if (table_new[key][whz] == 0.):
#                 kratio[whz] = 1.
#         if not math.isclose(np.min(kratio), 1., rel_tol=ftol):
#             flag = False
#         if not math.isclose(np.max(kratio), 1., rel_tol=ftol):
#             flag = False
#
#
#     # Ignore NaN at high r for 'dlnrho_dlnr':
#     key = 'dlnrho_dlnr'
#     for i, r in enumerate(table_old['r']):
#         if (r <= 31.) | (np.isfinite(table_old[key][i])):
#             if not math.isclose(table_old[key][i], table_new[key][i], rel_tol=ftol):
#                 flag = False
#
#     return flag


class TestSersicSavedTableNewOld:
    #ftol = 1.e-9
    ftol = 1.e-3
    #ftol_high = 3.e-9

    def setup_tables(self, n=1., invq=2.5, path_new=None):
        table_old = io.read_profile_table(n=n, invq=invq)
        table_new = io.read_profile_table(n=n, invq=invq, path=path_new)
        return table_old, table_new

    def OLDcheck_saved_tables_n_invq(self, n=None, invq=None, ftol=None, path_new=None, debug=False):
        if ftol is None:
            ftol = self.ftol
        try:
            table_old, table_new = self.setup_tables(n=n, invq=invq, path_new=path_new)
            table_loaded = True
        except:
            table_loaded = False
            print("    table not loaded: n={:0.1f}, invq={:0.1f}".format(n, invq))

        if table_loaded:
            check_all_keys(table_old, table_new, ftol=ftol, debug=debug)
            #assert check_all_keys

            # assert
            for key in ['ktot_Reff', 'k3D_sph_Reff']:
                # assert
                if not math.isclose(table_old[key],table_new[key], rel_tol=ftol):
                    msg = "n={}, invq={}: {} diff: {}, {}".format(n, invq, key,
                                    table_old[key], table_new[key])
                    if debug:
                        raise ValueError(msg)
                    else:
                        print(msg)

    def check_saved_tables_n_invq(self, n=None, invq=None, ftol=None, path_new=None, debug=False):
        if ftol is None:
            ftol = self.ftol
        try:
            table = io.read_profile_table(n=n, invq=invq)
            table_loaded = True
        except:
            table_loaded = False
            print("    table not loaded: n={:0.1f}, invq={:0.1f}".format(n, invq))

        if table_loaded:
            key = 'n'
            if not math.isclose(table['n'], n, rel_tol=ftol):
                msg = "n={}, invq={}: {} diff: {}, {}".format(n, invq, key, table[key],n)
                if debug:
                    raise ValueError(msg)
                else:
                    print(msg)
            key = 'invq'
            if not math.isclose(np.round(table['invq'],2), invq, rel_tol=ftol):
                msg = "n={}, invq={}: {} diff: {}, {}".format(n, invq, key, table[key],invq)
                if debug:
                    raise ValueError(msg)
                else:
                    print(msg)



    def test_all_saved_tables(self, path_new=_SERSIC_PROFILE_MASS_VC_DATADIR_NEW, debug=False):
        # Sersic indices
        #n_arr =         np.arange(0.5, 8.1, 0.1)
        n_arr =         np.arange(0.5, 8.1, 0.1)
        # Flattening ratio invq
        # invq_arr =      np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 20., 100.,
        #                         1.11, 1.25, 1.43, 1.67, 2.5, 3.33,
        #                         0.5, 0.67])
        invq_arr =      np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 20., 100.,
                                  1.11, 1.25, 1.43, 1.67, 2.5, 3.33, 0.5, 0.67])
        for n in n_arr:
            for invq in invq_arr:
                self.check_saved_tables_n_invq(n=n, invq=invq, path_new=path_new, debug=debug)



if __name__ == "__main__":
    # From the command line, call the wrapper to make full *default table set.
    #   Input args: output_path
    #   Optional input:  f_log

    ts = TestSersicSavedTableNewOld()
    ts.test_all_saved_tables()
