# coding=utf8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
#
# Testing of DYSMALPY model (component) calculations

import os
import shutil

import math

import numpy as np
import astropy.io.fits as fits
import astropy.units as u

from dysmalpy.fitting_wrappers import dysmalpy_make_model
from dysmalpy.fitting_wrappers import utils_io as fw_utils_io

from dysmalpy import galaxy, models, parameters, instrument


# # TESTING DIRECTORY
# path = os.path.abspath(__file__)
# _dir_tests = os.path.dirname(path) + '/'
# _dir_tests_data = _dir_tests+'test_data/'

class TestSersic:

    def test_BLAH(self):
