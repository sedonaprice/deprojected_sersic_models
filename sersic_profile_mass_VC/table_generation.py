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
        fileout=None, fileout_base=None, output_path=None, overwrite=False,
        logr_min = -2., logr_max = 2., nSteps=101,
        include_r0 = True, i=90., cumulative=None):
    """
    Calculate the Sersic profile table for a specific Sersic index n and
    inverse axis ratio invq.

    Parameters
    ----------
        n : float
            Sersic index
        invq : float
            Inverse intrinsic axis ratio
        output_path: str
            Path to directory where the Sersic profile table will be saved

        Reff: float, optional
            Effective radius [kpc]. Default: Reff = 1 kpc
        total_mass: float, optional
            Total mass of the Sersic profile [Msun].
            Default: total_mass = 5.e10 Msun
        logr_min: float, optional
            Log of minimum radius to calculate, relative to Reff.
            Default: logr_min = -2. (or r_min = 10^(-2.) * Reff)
        logr_max: float, optional
            Log of maximum radius to calculate, relative to Reff.
            Default: logr_max = +2. (or r_max = 10^(+2.) * Reff)
        nSteps: int, optional
            Number of radii steps to calculate. Default: 101
        include_r0: bool, optional
            Include r=0 in table or not. Default: True
        i: float, optional
            Inclination of model (to determin q_obs relative to the
            intrinsic axis ratio q.) Default: i = 90 deg
        cumulative: bool, optional
            Shortcut option to only calculate the next annulus,
            then add to the previous Menc(r-rdelt).
            Default: Uses cumulative if n >= 2.
        fileout_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<fileout_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        fileout: str, optional
            Option to override the default filename convention
            and instead directly specify the file location.
        overwrite: bool, optional
            Option to overwrite the FITS file, if a previous version exists.
            Default: False (will throw an error if the file already exists).

    Returns
    -------

    """

    if fileout is None:
        if output_path is None: raise ValueError("Must set 'output_path' if 'fileout' is not set !")

        # Ensure output path ends in trailing slash:
        if (output_path[-1] != '/'): output_path += '/'

        if fileout_base is None: fileout_base = 'mass_VC_profile_sersic'
        fileout = output_path+fileout_base+'_n{:0.1f}_invq{:0.2f}.fits'.format(n, invq)


    if cumulative is None:
        if n >= 2.:
            cumulative = True
        else:
            cumulative = False

    # ---------------------
    # Define q:
    q = 1./invq

    # Catch a few special cases:
    # Rounded to 2 decimal places values of invq, and the corresponding exact q:
    special_invq =  np.array([3.33, 1.67, 1.43, 1.11, 0.67])
    special_q =     np.array([0.3,  0.6,  0.7,  0.9,  1.5])
    wh_match = np.where(np.abs(special_invq-invq))[0]
    if len(wh_match) == 1:
        q = special_q[wh_match[0]]

    # ---------------------
    # Calculate profiles:
    rarr = np.logspace(logr_min, logr_max, num=nSteps)*Reff
    if include_r0:
        rarr = np.append(0., rarr) 

    vcirc =         calcs.v_circ(rarr, q=q, n=n, total_mass=total_mass, Reff=Reff, i=i)
    menc3D_sph =    calcs.M_encl_3D(rarr, q=q, n=n, total_mass=total_mass, Reff=Reff, i=i, cumulative=cumulative)
    menc3D_ellip =  calcs.M_encl_3D_ellip(rarr, q=q, n=n, total_mass=total_mass, Reff=Reff, i=i, cumulative=cumulative)

    rho =           calcs.rho(rarr, q=q, n=n, total_mass=total_mass, Reff=Reff, i=i)

    dlnrho_dlnr =   calcs.dlnrho_dlnr(rarr, q=q, n=n, total_mass=total_mass, Reff=Reff, i=i)

    # ---------------------
    # Setup table:
    table    = { 'r':                   rarr,
                 'vcirc':               vcirc,
                 'menc3D_sph':          menc3D_sph,
                 'menc3D_ellipsoid':    menc3D_ellip,
                 'rho':                 rho,
                 'dlnrho_dlnr':         dlnrho_dlnr,
                 'total_mass':          total_mass,
                 'Reff':                Reff,
                 'invq':                invq,
                 'q':                   q,
                 'n':                   n }

    # ---------------------
    # Calculate selected values at Reff:
    wh_reff = np.where(table['r'] == table['Reff'])[0]

    table['menc3D_sph_Reff'] =          table['menc3D_sph'][wh_reff]
    table['menc3D_ellipsoid_Reff'] =    table['menc3D_ellipsoid'][wh_reff]
    table['vcirc_Reff'] =               table['vcirc'][wh_reff]

    table['ktot_Reff'] = (table['total_mass'] * Msun.cgs.value) * G.cgs.value / \
                              (( table['Reff']*1.e3*pc.cgs.value ) * (table['vcirc_Reff']*1.e5)**2)
    table['k3D_sph_Reff'] = (table['menc3D_sph_Reff'] * Msun.cgs.value) * G.cgs.value / \
                              (( table['Reff']*1.e3*pc.cgs.value ) * (table['vcirc_Reff']*1.e5)**2)

    # ---------------------
    # 3D Spherical half total_mass radius, for reference:
    table['rhalf3D_sph'] = calcs.find_rhalf3D_sphere(r=table['r'], menc3D_sph=table['menc3D_sph'],
                                       total_mass=table['total_mass'])

    # ---------------------
    # Check that table calculated correctly:
    status = utils.check_for_inf(table=table)
    if status > 0:
        raise ValueError("Problem in table calculation: n={:0.1f}, invq={:0.2f}: status={}".format(n, invq, status))

    # ---------------------
    # Save table:
    table_io.save_profile_table(table=table, filename=fileout, overwrite=overwrite)

    # Save extra version if cumulative to clarify which were calculated this way:
    if cumulative:
        fileout_cumul = output_path+fileout_base+'_n{:0.1f}_invq{:0.2f}.cumulative.fits'.format(n, invq)
        table_io.save_profile_table(table=table, filename=fileout_cumul, overwrite=overwrite)

    return None


def wrapper_calculate_sersic_profile_tables(n_arr=None, invq_arr=None,
        Reff=1., total_mass=5.e10,
        fileout_base=None, output_path=None, overwrite=False,
        logr_min = -2., logr_max = 2., nSteps=101, i=90.,
        cumulative=None,
        f_log=None):

    """
    Wrapper function to calculate Sersic profile tables over a range of n and invq values.

    Parameters
    ----------
        n_arr: array_like
            Array of Sersic indices
        invq_arr: array_like
            Array of inverse intrinsic axis ratio
        output_path: str
            Path to directory where the Sersic profile table will be saved

        Reff: float, optional
            Effective radius [kpc]. Default: Reff = 1 kpc
        total_mass: float, optional
            Total mass of the Sersic profile [Msun]. Default: total_mass = 5.e10 Msun
        logr_min: float, optional
            Log of minimum radius to calculate, relative to Reff.
            Default: logr_min = -2. (or r_min = 10^(-2.) * Reff)
        logr_max: float, optional
            Log of maximum radius to calculate, relative to Reff.
            Default: logr_max = +2. (or r_max = 10^(+2.) * Reff)
        nSteps: int, optional
            Number of radii steps to calculate. Default: 101
        i: float, optional
            Inclination of model (to determin q_obs relative to the
            intrinsic axis ratio q.) Default: i = 90 deg
        cumulative: bool, optional
            Shortcut option to only calculate the next annulus,
            then add to the previous Menc(r-rdelt).
            Default: Uses cumulative if n >= 2.
        fileout_base: str, optional
            Base filename to use, for the default naming convention:
            ``<fileout_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        overwrite: bool, optional
            Option to overwrite the FITS file, if a previous version exists.
            Default: False (will throw an error if the file already exists).
        f_log: str, optional
            Filename of log file, to save information output while calculation is in progress.

    Returns
    -------

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
                    fileout_base=fileout_base, output_path=output_path, overwrite=overwrite,
                    logr_min = logr_min, logr_max = logr_max, nSteps=nSteps, i=i,
                    cumulative=cumulative)


    return None


def wrapper_calculate_full_table_set(fileout_base=None, output_path=None,
        overwrite=False, f_log=None,
        indChunk=None, nChunk=None, invqstart=None, cumulative=None):
    """
    Wrapper function to calculate the full set of Sersic profile tables.

    Parameters
    ----------
        output_path: str
            Path to directory where the Sersic profile table will be saved

        fileout_base: str, optional
            Base filename to use, for the default naming convention:
            `<fileout_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`

        overwrite: bool, optional
            Option to overwrite the FITS file, if a previous version exists.
            Default: False (will throw an error if the file already exists).

        f_log: str, optional
            Filename of log file, to save information output while
            calculation is in progress. Default: None

        indChunk: int, optional
            Index of chunk to run. Default: None (only one chunk)

        nChunk: int, optional
            Total number of chunks. Default: None (only one chunk)

        invqstart: int, optional
            Where to start with invq array. Default: None (start at beginning)

        cumulative: bool, optional
            Shortcut option to only calculate the next annulus,
            then add to the previous Menc(r-rdelt).
            Default: Uses cumulative if n >= 2.

    Returns
    -------

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
            fileout_base=fileout_base, output_path=output_path, overwrite=overwrite,
            logr_min=logr_min, logr_max=logr_max, nSteps=nSteps, i=i,
            f_log=f_log, cumulative=cumulative)

    return None


if __name__ == "__main__":
    # From the command line, call the wrapper to make full *default table set.
    #   Input args: output_path
    #   Optional input:  f_log

    output_path = sys.argv[1]

    f_log = None
    indChunk = None
    nChunk = None
    invqstart = None

    if len(sys.argv) == 3:
        f_log = sys.argv[2]
    elif len(sys.argv) == 4:
        indChunk = np.int(sys.argv[2])
        nChunk = np.int(sys.argv[3])
    elif len(sys.argv) >= 5:
        f_log = sys.argv[2]
        indChunk = np.int(sys.argv[3])
        nChunk = np.int(sys.argv[4])

        try:
            invqstart = np.float(sys.argv[5])
        except:
            invqstart = None

    f_log_tmp = output_path+'sersic_table_calc_{}.log'.format(indChunk+1)
    print("Starting chunk: indChunk={}, nChunk={}".format(indChunk, nChunk))
    print("Logfile: {}".format(f_log_tmp))

    wrapper_calculate_full_table_set(output_path=output_path, f_log=f_log,
                indChunk=indChunk, nChunk=nChunk, invqstart=invqstart) #, cumulative=cumulative)
