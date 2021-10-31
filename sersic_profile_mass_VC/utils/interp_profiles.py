##################################################################################
# sersic_profile_mass_VC/utils/interp_profiles.py                                #
#                                                                                #
# Copyright 2018-2021 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os

# Supress warnings: Runtime & integration warnings are frequent
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.interpolate as scp_interp

import logging
import copy

from sersic_profile_mass_VC.io import read_profile_table, _sersic_profile_filename_base

__all__ = [ 'interpolate_entire_table',
            'interpolate_sersic_profile_menc', 'interpolate_sersic_profile_VC',
            'interpolate_sersic_profile_rho', 'interpolate_sersic_profile_dlnrho_dlnR',
            'interpolate_entire_table_nearest',
            'interpolate_sersic_profile_menc_nearest', 'interpolate_sersic_profile_VC_nearest',
            'interpolate_sersic_profile_rho_nearest',
            'interpolate_sersic_profile_dlnrho_dlnR_nearest',
            'interpolate_sersic_profile_dlnrho_dlnR_two_component_nearest',
            'nearest_n_invq']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SersicProfileMassVC')

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# +++++++++++++++++++++++++++++++++++++++++++++++++
# Interpolation functions:

def interpolate_entire_table(R=None, table=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None, interp_type='cubic'):
    """
    Interpolate entire table, returning new profiles sampled at project major axis radius R.

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate table [kpc]

        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]

        n: float, optional
            Sersic index
            Must be specified if `table=None`.
        invq: float, optional
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q
            Must be specified if `table=None`.

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`

        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        table_interp: dict

    """
    if table is None:
        if n is None:
            raise ValueError("Must specify 'n' if 'table' is not set!")
        if invq is None:
            raise ValueError("Must specify 'n' if 'table' is not set!")
        table = read_profile_table(filename=filename, n=n, invq=invq, path=path,
                                   filename_base=filename_base)

    vcirc =         interpolate_sersic_profile_VC(R=R, total_mass=total_mass, Reff=Reff,
                                                  n=table['n'], invq=table['invq'], table=table,
                                                  interp_type=interp_type)
    menc3D_sph =    interpolate_sersic_profile_menc(R=R, total_mass=total_mass, Reff=Reff,
                                                    n=table['n'], invq=table['invq'], table=table,
                                                    interp_type=interp_type)
    menc3D_ellip =  interpolate_sersic_profile_menc(R=R, total_mass=total_mass, Reff=Reff,
                                                    n=table['n'], invq=table['invq'], table=table,
                                                    sphere=False, interp_type=interp_type)
    rho =           interpolate_sersic_profile_rho(R=R, total_mass=total_mass, Reff=Reff,
                                                   n=table['n'], invq=table['invq'], table=table,
                                                   interp_type=interp_type)
    dlnrho_dlnR =   interpolate_sersic_profile_dlnrho_dlnR(R=R, Reff=Reff, n=table['n'],
                                                           invq=table['invq'], table=table,
                                                           interp_type=interp_type)

    # ---------------------
    # Setup table:
    table_interp = { 'R':                   R,
                     'total_mass':          total_mass,
                     'Reff':                Reff,
                     'vcirc':               vcirc,
                     'menc3D_sph':          menc3D_sph,
                     'menc3D_ellipsoid':    menc3D_ellip,
                     'rho':                 rho,
                     'dlnrho_dlnR':         dlnrho_dlnR }


    # Scale at Reff:
    table_interp['menc3D_sph_Reff'] = table['menc3D_sph_Reff'] * \
                                      table_interp['total_mass']/table['total_mass']
    table_interp['menc3D_ellipsoid_Reff'] = table['menc3D_ellipsoid_Reff'] * \
                                  table_interp['total_mass']/table['total_mass']
    table_interp['vcirc_Reff'] = table['vcirc_Reff'] * \
                                 np.sqrt(table_interp['total_mass']/table['total_mass']) * \
                                 np.sqrt(table_interp['Reff']/table['Reff'])

    keys_copy = [ 'invq', 'q', 'n', 'rhalf3D_sph', 'ktot_Reff', 'k3D_sph_Reff']
    for key in keys_copy:
        table_interp[key] = table[key]

    return table_interp


def interpolate_sersic_profile_menc(R=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None, table=None,
        sphere=True, interp_type='cubic'):
    """
    Interpolate Menc3D_sphere(<R=R) at arbitrary radii major axis raidus R,
    for arbitrary Mtot and Reff.

    Uses the saved table of Menc3D_sphere(R) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Mtot and Reff.
    (by mapping the radius using R' = (R/Reff * table_Reff) )

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate profile [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

        sphere: bool, optional
            Flag to calculate enclosed mass in sphere (True) vs spheroid/ellipsoid (False).
            Default: True

    Returns
    -------
        menc_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq, path=path,
                                   filename_base=filename_base)

    if sphere:
        table_menc =    table['menc3D_sph']
    else:
        table_menc =    table['menc3D_ellipsoid']

    table_rad =     table['R']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['R'][0] > 0.:
        table_rad = np.append(0., table_rad)
        table_menc = np.append(0., table_menc)


    # Ensure it's an array:
    if isinstance(R*1., float):
        Rarr = np.array([R])
    else:
        Rarr = np.array(R)
    # Ensure all radii are 0. or positive:
    Rarr = np.abs(Rarr)

    scale_fac = (total_mass / table_mass)

    if interp_type.lower().strip() == 'cubic':
        m_interp = scp_interp.interp1d(table_rad, table_menc, fill_value=np.NaN, bounds_error=False, kind='cubic')
        m_interp_extrap = scp_interp.interp1d(table_rad, table_menc, fill_value='extrapolate', kind='linear')

        menc_interp = np.zeros(len(Rarr))
        wh_in =     np.where((Rarr <= table_rad.max()) & (Rarr >= table_rad.min()))[0]
        wh_extrap = np.where((Rarr > table_rad.max()) | (Rarr < table_rad.min()))[0]
        menc_interp[wh_in] =     (m_interp(Rarr[wh_in] / Reff * table_Reff) * scale_fac )
        menc_interp[wh_extrap] = (m_interp_extrap(Rarr[wh_extrap] / Reff * table_Reff) * scale_fac )

    elif interp_type.lower().strip() == 'linear':
        m_interp = scp_interp.interp1d(table_rad, table_menc, fill_value='extrapolate',
                                       bounds_error=False, kind='linear')

        menc_interp = m_interp(Rarr / Reff * table_Reff) * scale_fac

    else:
        raise ValueError("interp type '{}' unknown!".format(interp_type))

    if (len(Rarr) > 1):
        return menc_interp
    else:
        if isinstance(R*1., float):
            # Float input
            return menc_interp[0]
        else:
            # Length 1 array input
            return menc_interp

def interpolate_sersic_profile_VC(R=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base,
        filename=None, table=None,
        interp_type='cubic'):
    """
    Interpolate vcirc(R) at arbitrary radii R, for arbitrary Mtot and Reff.

    Uses the saved table of vcirc(R) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Mtot and Reff.
    (by mapping the radius using R' = (R/Reff * table_Reff), and scaling for the difference
    between Mtot and table_Mtot)

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate profile [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        vcirc_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq, path=path, filename_base=filename_base)

    table_vcirc =   table['vcirc']
    table_rad =     table['R']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: vcirc=0
    if table['R'][0] > 0.:
        table_rad = np.append(0., table_rad)
        table_vcirc = np.append(0., table_vcirc)

    # Ensure it's an array:
    if isinstance(R*1., float):
        Rarr = np.array([R])
    else:
        Rarr = np.array(R)
    # Ensure all radii are 0. or positive:
    Rarr = np.abs(Rarr)

    scale_fac = np.sqrt(total_mass / table_mass) * np.sqrt(table_Reff / Reff)

    if interp_type.lower().strip() == 'cubic':
        v_interp = scp_interp.interp1d(table_rad, table_vcirc, fill_value=np.NaN, bounds_error=False, kind='cubic')
        v_interp_extrap = scp_interp.interp1d(table_rad, table_vcirc, fill_value='extrapolate', kind='linear')

        vcirc_interp = np.zeros(len(Rarr))
        wh_in =     np.where((Rarr <= table_rad.max()) & (Rarr >= table_rad.min()))[0]
        wh_extrap = np.where((Rarr > table_rad.max()) | (Rarr < table_rad.min()))[0]
        vcirc_interp[wh_in] =     (v_interp(Rarr[wh_in]  / Reff * table_Reff) * scale_fac )
        vcirc_interp[wh_extrap] = (v_interp_extrap(Rarr[wh_extrap]  / Reff * table_Reff) * scale_fac )

    elif interp_type.lower().strip() == 'linear':
        v_interp = scp_interp.interp1d(table_rad, table_vcirc, fill_value='extrapolate',
                                       bounds_error=False, kind='linear')
        vcirc_interp = (v_interp(Rarr  / Reff * table_Reff) * scale_fac )
    else:
        raise ValueError("interp type '{}' unknown!".format(interp_type))

    if (len(Rarr) > 1):
        return vcirc_interp
    else:
        if isinstance(R*1., float):
            # Float input
            return vcirc_interp[0]
        else:
            # Length 1 array input
            return vcirc_interp


def interpolate_sersic_profile_rho(R=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base,
        filename=None, table=None,
        interp_type='cubic'):
    """
    Interpolate Rho(R) at arbitrary projected major axis radii R, for arbitrary Mtot and Reff.

    Uses the saved table of rho(R=R) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Mtot and Reff.
    (by mapping the radius using R' = (R/Reff * table_Reff) )

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate profile [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        rho_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq,  path=path, filename_base=filename_base)

    table_rho =     table['rho']
    table_rad =     table['R']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']


    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['R'][0] > 0.:
        try:
            table_rad = np.append(r.min() * table_Reff/Reff, table_rad)
            table_rho = np.append(rho(r.min()* table_Reff/Reff, total_mass=table_mass,
                        Reff=table_Reff, n=n, q=table['q']), table_rho)
        except:
            pass

    # Clean up if n>1: TECHNICALLY asymptotic at r=0, but replace with large value
    #                  so scipy interpolation works.
    if (n > 1.) & (table['R'][0] == 0.):
        if ~np.isfinite(table_rho[0]):
            table_rho[0] = table_rho[1]**2/table_rho[2] * 1.e3

    # Ensure it's an array:
    if isinstance(R*1., float):
        Rarr = np.array([R])
    else:
        Rarr = np.array(R)
    # Ensure all radii are 0. or positive:
    Rarr = np.abs(Rarr)

    scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**3

    if interp_type.lower().strip() == 'cubic':
        r_interp = scp_interp.interp1d(table_rad, table_rho, fill_value=np.NaN, bounds_error=False, kind='cubic')
        r_interp_extrap = scp_interp.interp1d(table_rad, table_rho, fill_value='extrapolate', kind='linear')

        rho_interp = np.zeros(len(Rarr))
        wh_in =     np.where((Rarr <= table_rad.max()) & (Rarr >= table_rad.min()))[0]
        wh_extrap = np.where((Rarr > table_rad.max()) | (Rarr < table_rad.min()))[0]
        rho_interp[wh_in] =     (r_interp(Rarr[wh_in] / Reff * table_Reff) * scale_fac )
        rho_interp[wh_extrap] = (r_interp_extrap(Rarr[wh_extrap] / Reff * table_Reff) * scale_fac)
    elif interp_type.lower().strip() == 'linear':
        r_interp = scp_interp.interp1d(table_rad, table_rho, fill_value='extrapolate',
                                       bounds_error=False, kind='linear')
        rho_interp =     (r_interp(Rarr / Reff * table_Reff) * scale_fac )

    else:
        raise ValueError("interp type '{}' unknown!".format(interp_type))

    # Back replace inf, if interpolating at r=0 for n>1:
    if (n > 1.) & (table['R'][0] == 0.):
        if (~np.isfinite(table['rho'][0]) & (Rarr.min() == 0.)):
            rho_interp[Rarr==0.] = table['rho'][0]


    if (len(Rarr) > 1):
        return rho_interp
    else:
        if isinstance(R*1., float):
            # Float input
            return rho_interp[0]
        else:
            # Length 1 array input
            return rho_interp


def interpolate_sersic_profile_dlnrho_dlnR(R=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None, table=None,
        interp_type='cubic'):
    """
    Interpolate dlnrho/dlnr at arbitrary projected major axis radii R, for arbitrary Reff.

    Uses the saved table of dlnrho/dlnR(R) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Reff.
    (by mapping the radius using R' = (R/Reff * table_Reff) )

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate profile [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        dlnrho_dlnR_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq,  path=path, filename_base=filename_base)

    table_dlnrho_dlnR =     table['dlnrho_dlnR']
    table_rad =     table['R']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['R'][0] > 0.:
        try:
            table_rad = np.append(r.min() * table_Reff/Reff, table_rad)
            table_dlnrho_dlnR = np.append(dlnrho_dlnR(r.min()* table_Reff/Reff, n=n, total_mass=table_mass,
                                        Reff=table_Reff, q=table['q']), table_dlnrho_dlnR)
        except:
            pass

    # Catch NaNs:
    whfin = np.where(np.isfinite(table_dlnrho_dlnR))
    table_dlnrho_dlnR = table_dlnrho_dlnR[whfin]
    table_rad = table_rad[whfin]

    # Ensure it's an array:
    if isinstance(R*1., float):
        Rarr = np.array([R])
    else:
        Rarr = np.array(R)
    # Ensure all radii are 0. or positive:
    Rarr = np.abs(Rarr)

    if interp_type.lower().strip() == 'cubic':
        r_interp = scp_interp.interp1d(table_rad, table_dlnrho_dlnR, fill_value=np.NaN, bounds_error=False, kind='cubic')
        r_interp_extrap = scp_interp.interp1d(table_rad, table_dlnrho_dlnR, fill_value='extrapolate', kind='linear')

        dlnrho_dlnR_interp = np.zeros(len(Rarr))
        wh_in =     np.where((Rarr <= table_rad.max()) & (Rarr >= table_rad.min()))[0]
        wh_extrap = np.where((Rarr > table_rad.max()) | (Rarr < table_rad.min()))[0]
        dlnrho_dlnR_interp[wh_in] =     (r_interp(Rarr[wh_in] / Reff * table_Reff) )
        dlnrho_dlnR_interp[wh_extrap] = (r_interp_extrap(Rarr[wh_extrap] / Reff * table_Reff) )
    elif interp_type.lower().strip() == 'linear':
        r_interp = scp_interp.interp1d(table_rad, table_dlnrho_dlnR, fill_value='extrapolate',
                                       bounds_error=False, kind='linear')
        dlnrho_dlnR_interp =     (r_interp(Rarr / Reff * table_Reff) )

    else:
        raise ValueError("interp type '{}' unknown!".format(interp_type))

    if (len(Rarr) > 1):
        return dlnrho_dlnR_interp
    else:
        if isinstance(R*1., float):
            # Float input
            return dlnrho_dlnR_interp[0]
        else:
            # Length 1 array input
            return dlnrho_dlnR_interp


# +++++++++++++++++++++++++++++++++++++++++++++++++
# Nearest n, invq values and aliases to the Menc, vcirc interpolation functions

def nearest_n_invq(n=None, invq=None):
    """
    Nearest value of n and invq for which a Sersic profile table
    has been calculated, using the *DEFAULT* array of n and invq which have been used here.

    A similar function can be defined if a different set of Sersic profile tables
    (over n, invq) have been calculated.

    Example
    -------
    >>> nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    Parameters
    ----------
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

    Returns
    -------
        nearest_n: float
            Nearest value of Sersic index n from lookup array
        nearest_invq: float
            Nearest value of flattening invq from lookup array

    """
    # Use the "typical" collection of table values:
    table_n = np.arange(0.5, 8.1, 0.1)   # Sersic indices
    table_invq = np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 20., 100.,
                    1.11, 1.43, 1.67, 3.33, 0.5, 0.67])
    # 1:1, 1:2, 1:3, ... flattening  [also elongated 2:1, 1.5:1]

    nearest_n = table_n[ np.argmin( np.abs(table_n - n) ) ]
    nearest_invq = table_invq[ np.argmin( np.abs( table_invq - invq) ) ]

    return nearest_n, nearest_invq

def interpolate_entire_table_nearest(R=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base,
        filename=None, table=None, interp_type='cubic'):
    """
    Interpolate entire table, returning new profiles sampled at R,
    using the **nearest values** of n and invq that are included
    in the Sersic profile table collection.

    Parameters
    ----------
        R: float or array_like
            Projected major axis radius R at which to interpolate table [kpc]

        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]

        n: float, optional
            Sersic index
            Must be specified if `table=None`.
        invq: float, optional
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q
            Must be specified if `table=None`.

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`

        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        table_interp_nearest: dict

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    table_interp_nearest = interpolate_entire_table(R=R, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename,
                    interp_type=interp_type)

    return table_interp_nearest

def interpolate_sersic_profile_menc_nearest(R=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None,
        interp_type='cubic'):
    """
    Interpolate Menc3D_sphere(<R=R) at arbitrary projected major axis radii R,
    for arbitrary Mtot and Reff, using the **nearest values** of n and invq that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_menc()` with these values.

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate profile [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`

    Returns
    -------
        menc_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    menc_interp_nearest = interpolate_sersic_profile_menc(R=R, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename,
                    interp_type=interp_type)

    return menc_interp_nearest


def interpolate_sersic_profile_VC_nearest(R=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None,
        interp_type='cubic'):
    """
    Interpolate vcirc(r) at arbitrary projected major axis radii R, for arbitrary Mtot and Reff,
    using the **nearest values of n and invq** that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_VC()` with these values.

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate profile [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`

    Returns
    -------
        vcirc_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    vcirc_interp_nearest = interpolate_sersic_profile_VC(R=R, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename,
                    interp_type=interp_type)

    return vcirc_interp_nearest


def interpolate_sersic_profile_rho_nearest(R=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None,
        interp_type='cubic'):
    """
    Interpolate Rho(r) at arbitrary projected major axis radii R, for arbitrary Mtot and Reff,
    using the **nearest values of n and invq** that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_rho()` with these values.

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate profile [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q.

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`

    Returns
    -------
        rho_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    rho_interp_nearest = interpolate_sersic_profile_rho(r=r, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename,
                    interp_type=interp_type)

    return rho_interp_nearest


def interpolate_sersic_profile_dlnrho_dlnR_nearest(R=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None,
        interp_type='cubic'):
    """
    Interpolate dlnrho_g/dlnr at arbitrary projected major axis radii R, for arbitrary Reff,
    using the **nearest values of n and invq** that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_dlnrho_dlnR()` with these values.

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate profile [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q.

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`

    Returns
    -------
        dlnrho_dlnR_interp_nearest: float or array_like

    """
    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    dlnrho_dlnR_interp_nearest = interpolate_sersic_profile_dlnrho_dlnR(R=R, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename,
                    interp_type=interp_type)

    return dlnrho_dlnR_interp_nearest



def interpolate_sersic_profile_dlnrho_dlnR_two_component_nearest(R=None,
        mass_comp1=1.e11, mass_comp2=0.,
        Reff_comp1=None, n_comp1=1., invq_comp1=5.,
        Reff_comp2=1.,   n_comp2=4., invq_comp2=1.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None,
        interp_type='cubic'):
    """
    Interpolate dlnrho_g/dlnR at arbitrary projected major axis radii R, for a composite system.
    Both comp1 and comp2 (e.g., disk and bulge) can have arbitary Reff,
    but this uses the **nearest values of n and invq** that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then returns `dlnrho_dlnR_interp_nearest()` for the total system.

    Parameters
    ----------
        R: float or array_like
            Radius at which to interpolate profile [kpc]

        mass_comp1: float
            Total mass of the first component [Msun]    [Default: 10^11 Msun]
        Reff_comp1: float
            Effective radius of the first component Sersic profile [kpc]
        n_comp1: float
            Sersic index of the first component.  [Default: 1.]
        invq_comp1: float
            Inverse of the intrinsic axis ratio of the first component Sersic profile, invq = 1/q.
            [Default: 5.]
        mass_comp2: float
            Total mass of the second component [Msun]    [Default: 0 Msun]
        Reff_comp2: float
            Effective radius of the second component Sersic profile [kpc]. [Default: 1 kpc]
        n_comp2: float
            Sersic index of the second component. [Default: 4]
        invq_comp2: float
            Inverse of the intrinsic axis ratio of the second component Sersic profile, invq = 1/q.
            [Default: 1.]

        path: str, optional
            Path to directory containing the saved Sersic profile tables.
            If not set, system variable `SERSIC_PROFILE_MASS_VC_DATADIR` must be set.
            Default: system variable `SERSIC_PROFILE_MASS_VC_DATADIR`, if specified.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        interp_type: str, optional
            Default profile interpolation within the table Rarr region.
            (Extrapolation is always linear). Default: `cubic`

    Returns
    -------
        dlnrho_dlnR_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    rho_t = R * 0.
    rho_dlnrho_dlnR_sum = R * 0.
    for n, invq, Reff, M in zip([n_comp1, n_comp2], [invq_comp1, invq_comp2],
            [Reff_comp1, Reff_comp2], [mass_comp1, mass_comp2]):
        nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

        dlnrho_dlnR_interp_nearest = interpolate_sersic_profile_dlnrho_dlnR(R=R, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename,
                    interp_type=interp_type)

        rho_interp_nearest = interpolate_sersic_profile_rho(R=R, total_mass=M,
                Reff=Reff, n=nearest_n, invq=nearest_invq, path=path,
                filename_base=filename_base, filename=filename,
                interp_type=interp_type)

        rho_t += rho_interp_nearest
        rho_dlnrho_dlnR_sum += rho_interp_nearest * dlnrho_dlnR_interp_nearest

    dlnrho_dlnR_interp_total = (1./rho_t) * rho_dlnrho_dlnR_sum

    return dlnrho_dlnR_interp_total
