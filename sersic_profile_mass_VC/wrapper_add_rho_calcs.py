##################################################################################
# sersic_profile_mass_VC/table_generation.py                                     #
#                                                                                #
# Copyright 2018-2021 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import os, sys

import numpy as np
import astropy.constants as apy_con

import logging

from sersic_profile_mass_VC import table_io
from sersic_profile_mass_VC import calcs
from sersic_profile_mass_VC import utils



__all__ = [ 'calculate_sersic_profile_table', 'wrapper_calculate_sersic_profile_tables',
            'wrapper_calculate_full_table_set' ]

# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SersicProfileMassVC')


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


def calculate_sersic_profile_table(n=1., invq=5.,
        Reff=1., total_mass=5.e10,
        fileout=None, fileout_base=None,
        input_path=None,
        output_path=None, overwrite=False,
        logr_min = -2., logr_max = 2., nSteps=101, i=90.,
        cumulative=None):
    """
    Calculate the Sersic profile table for a specific Sersic index n and inverse axis ratio invq.

    Usage:  calculate_sersic_profile_table(n=n, invq=invq, output_path=output_path, **kwargs)

    Input:
        n:                  Sersic index
        invq:               Inverse intrinsic axis ratio
        output_path:        Path to directory where the Sersic profile table will be saved

    Optional input:
        Reff:               Effective radius [kpc]. Default: Reff = 1 kpc
        total_mass:         Total mass of the Sersic profile [Msun]. Default: total_mass = 5.e10 Msun

        logr_min:           Log of minimum radius to calculate, relative to Reff.
                                Default: logr_min = -2. (or r_min = 10^(-2.) * Reff)
        logr_max:           Log of maximum radius to calculate, relative to Reff.
                                Default: logr_max = +2. (or r_max = 10^(+2.) * Reff)
        nSteps:             Number of radii steps to calculate. Default: 101

        i:                  Inclination of model (to determin q_obs relative to the intrinsic axis ratio q.)
                                Default: i = 90 deg

        cumulative:         Shortcut option to only calculate the next annulus, then add to the previous Menc(r-rdelt).
                            Default: Uses cumulative if n >= 2.

        fileout_base:       Base filename to use, when combined with default naming convention:
                                <fileout_base>_nX.X_invqX.XX.fits
                            Default: 'mass_VC_profile_sersic'

        fileout:            Option to override the default filename convention and
                                instead directly specify the file location.

        overwrite:          Option to overwrite the FITS file, if a previous version exists.
                            Default: False (will throw an error if the file already exists).

    Output:                 Saved binary FITS table containing Sersic profile values.

    """

    if fileout is None:
        if output_path is None: raise ValueError("Must set 'output_path' if 'fileout' is not set !")

        # Ensure output path ends in trailing slash:
        if (output_path[-1] != '/'): output_path += '/'
        if (input_path[-1] != '/'):  input_path += '/'

        if fileout_base is None: fileout_base = 'mass_VC_profile_sersic'
        fileout = output_path+fileout_base+'_n{:0.1f}_invq{:0.2f}.fits'.format(n, invq)

        filein =  input_path+fileout_base+'_n{:0.1f}_invq{:0.2f}.fits'.format(n, invq)


    if cumulative is None:
        if n >= 2.:
            cumulative = True
        else:
            cumulative = False

    # ---------------------
    # Read table:


    tabin = table_io.read_profile_table(path=input_path, n=n, invq=invq)

    q = tabin['q']
    Reff = tabin['Reff']
    total_mass = tabin['total_mass']
    rarr = tabin['r']

    # ---------------------
    # Calculate profiles:

    rho =           calcs.rho(rarr, q=q, n=n, total_mass=total_mass, Reff=Reff, i=i)

    dlnrho_dlnr =   calcs.dlnrho_dlnr(rarr, q=q, n=n, total_mass=total_mass, Reff=Reff, i=i)


    # ---------------------
    # Setup table:
    table    = { 'r':                   rarr,
                 'rho':                 rho,
                 'dlnrho_dlnr':         dlnrho_dlnr,
                 'total_mass':          total_mass,
                 'Reff':                Reff,
                 'invq':                invq,
                 'q':                   q,
                 'n':                   n }

    # ---------------------
    # Get pre-calculated profiles:
    keys_copy = ['vcirc', 'menc3D_sph', 'menc3D_ellipsoid',
                'menc3D_sph_Reff', 'menc3D_ellipsoid_Reff',
                'vcirc_Reff', 'ktot_Reff', 'k3D_sph_Reff', 'rhalf3D_sph']
    for key in keys_copy:
        table[key] = tabin[key]

    # ---------------------
    # Check that table calculated correctly:
    status = utils.check_for_inf(table=table)
    if status > 0:
        raise ValueError("Problem in table calculation: n={:0.1f}, invq={:0.2f}: status={}".format(n, invq, status))

    # ---------------------
    # Save table:
    table_io.save_profile_table(table=table, filename=fileout, overwrite=overwrite)

    return None


def wrapper_calculate_sersic_profile_tables(n_arr=None, invq_arr=None,
        Reff=1., total_mass=5.e10,
        fileout_base=None, input_path=None, output_path=None, overwrite=False,
        logr_min = -2., logr_max = 2., nSteps=101, i=90.,
        cumulative=None,
        f_log=None):

    """
    Wrapper function to calculate Sersic profile tables over a range of n and invq values.

    Usage:  wrapper_calculate_sersic_profile_tables(n_arr=n_arr, invq_arr=invq_arr, output_path=output_path, **kwargs)

    Input:
        n_arr:              Array of Sersic indices
        invq_arr:           Array of inverse intrinsic axis ratio
        output_path:        Path to directory where the Sersic profile table will be saved

    Optional input:
        Reff:               Effective radius [kpc]. Default: Reff = 1 kpc
        total_mass:         Total mass of the Sersic profile [Msun]. Default: total_mass = 5.e10 Msun

        logr_min:           Log of minimum radius to calculate, relative to Reff.
                                Default: logr_min = -2. (or r_min = 10^(-2.) * Reff)
        logr_max:           Log of maximum radius to calculate, relative to Reff.
                                Default: logr_max = +2. (or r_max = 10^(+2.) * Reff)
        nSteps:             Number of radii steps to calculate. Default: 101

        i:                  Inclination of model (to determin q_obs relative to the intrinsic axis ratio q.)
                                Default: i = 90 deg

        cumulative:         Shortcut option to only calculate the next annulus, then add to the previous Menc(r-rdelt).
                            Default: Uses cumulative if n >= 2.

        fileout_base:       Base filename to use, for the default naming convention:
                                <fileout_base>_nX.X_invqX.XX.fits
                            Default: 'mass_VC_profile_sersic'

        overwrite:          Option to overwrite the FITS file, if a previous version exists.
                            Default: False (will throw an error if the file already exists).

        f_log:              Filename of log file, to save information output while calculation is in progress.

    Output:                 Saved binary FITS tables containing Sersic profile values.

    """

    # Ensure output path ends in trailing slash:
    if (output_path[-1] != '/'): output_path += '/'
    if fileout_base is None: fileout_base = 'mass_VC_profile_sersic'

    if f_log is not None:
        # Clean up existing log file:
        if (os.path.isfile(f_log)):    os.remove(f_log)

        loggerfile = logging.FileHandler(f_log)
        loggerfile.setLevel(logging.INFO)
        logger.addHandler(loggerfile)


    for n in n_arr:
        logger.info(" Calculating n={:0.1f}".format(n))
        for invq in invq_arr:
            logger.info("      Calculating invq={:0.2f}".format(invq))

            if (not overwrite) & (os.path.isfile(output_path+fileout_base+'_n{:0.1f}_invq{:0.2f}.fits'.format(n, invq))):
                logger.info("       Calculation already completed.")
                continue
            else:
                calculate_sersic_profile_table(n=n, invq=invq,
                    Reff=Reff, total_mass=total_mass,
                    fileout_base=fileout_base, input_path=input_path, output_path=output_path, overwrite=overwrite,
                    logr_min = logr_min, logr_max = logr_max, nSteps=nSteps, i=i,
                    cumulative=cumulative)


    return None


def wrapper_calculate_full_table_set(fileout_base=None, input_path=None, output_path=None, overwrite=False, f_log=None,
        indChunk=None, nChunk=None, invqstart=None, cumulative=None):
    """
    Wrapper function to calculate the full set of Sersic profile tables.

    Usage:  wrapper_calculate_full_table_set(output_path=output_path, **kwargs)

    Input:
        output_path:        Path to directory where the Sersic profile table will be saved

    Optional input:
        fileout_base:       Base filename to use, for the default naming convention:
                                <fileout_base>_nX.X_invqX.XX.fits
                            Default: 'mass_VC_profile_sersic'

        overwrite:          Option to overwrite the FITS file, if a previous version exists.
                            Default: False (will throw an error if the file already exists).

        f_log:              Filename of log file, to save information output while calculation is in progress.

        indChunk:           Index of chunk to run
        nChunk:             Total number of chunks
        invqstart:          Where to start with invq array

        cumulative:         Shortcut option to only calculate the next annulus, then add to the previous Menc(r-rdelt).
                            Default: Uses cumulative if n >= 2.

    Output:                 Saved binary FITS tables containing Sersic profile values.

    """

    # Default settings:
    Reff =          1.      # kpc
    total_mass =    5.e10   # Msun
    logr_min =      -2.     # log10(R[kpc])
    logr_max =      2.      # log10(R[kpc])
    nSteps =        101     #
    i =             90.     # degrees

    # Sersic indices
    n_arr =         np.arange(0.5, 8.1, 0.1)
    # Flattening ratio invq
    invq_arr =      np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 20., 100.,
                            1.11, 1.25, 1.43, 1.67, 2.5, 3.33,
                            0.5, 0.67])
    # invq corresponds to:
    #                q = [1., 0.5, 0.333, 0.25, 0.2, 0.167, 0.142, 0.125, 0.1, 0.05, 0.01,
    #                      0.9, 0.8, 0.7, 0.6, 0.4, 0.3,
    #                      2., 1.5]

    if (indChunk is not None) & (nChunk is not None):
        stepChunk = np.int(np.round(len(n_arr)/(1.*nChunk)))
        n_arr = n_arr[indChunk*stepChunk:(indChunk+1)*stepChunk]

        f_log = output_path+'sersic_table_calc_{}.log'.format(indChunk+1)

        if invqstart is not None:
            whinvq = np.where(invq_arr == invqstart)[0]
            if len(whinvq) == 1:
                invq_arr = invq_arr[whinvq[0]:]

    wrapper_calculate_sersic_profile_tables(n_arr=n_arr, invq_arr=invq_arr,
            Reff=Reff, total_mass=total_mass,
            fileout_base=fileout_base, input_path=input_path, output_path=output_path, overwrite=overwrite,
            logr_min=logr_min, logr_max=logr_max, nSteps=nSteps, i=i,
            f_log=f_log, cumulative=cumulative)

    return None



if __name__ == "__main__":
    # From the command line, call the wrapper to make full *default table set.
    #   Input args: output_path
    #   Optional input:  f_log

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    f_log = None
    indChunk = None
    nChunk = None
    invqstart = None

    if len(sys.argv) == 4:
        f_log = sys.argv[3]
    elif len(sys.argv) == 5:
        indChunk = np.int(sys.argv[3])
        nChunk = np.int(sys.argv[4])
    elif len(sys.argv) >= 6:
        f_log = sys.argv[3]
        indChunk = np.int(sys.argv[4])
        nChunk = np.int(sys.argv[5])

        try:
            invqstart = np.float(sys.argv[6])
        except:
            invqstart = None


    wrapper_calculate_full_table_set(input_path=input_path, output_path=output_path, f_log=f_log,
                indChunk=indChunk, nChunk=nChunk, invqstart=invqstart)
