##################################################################################
# sersic_profile_mass_VC/utils/interp_profiles.py                                #
#                                                                                #
# Copyright 2018-2021 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import numpy as np
import scipy.interpolate as scp_interp

import logging
import copy

from sersic_profile_mass_VC.io import read_profile_table, _sersic_profile_filename_base

__all__ = [ 'interpolate_entire_table',
            'interpolate_sersic_profile_menc', 'interpolate_sersic_profile_VC',
            'interpolate_sersic_profile_rho', 'interpolate_sersic_profile_dlnrho_dlnr',
            'interpolate_entire_table_nearest',
            'interpolate_sersic_profile_menc_nearest', 'interpolate_sersic_profile_VC_nearest',
            'interpolate_sersic_profile_rho_nearest',
            'interpolate_sersic_profile_dlnrho_dlnr_nearest',
            'interpolate_sersic_profile_dlnrho_dlnr_bulge_disk_nearest',
            'nearest_n_invq']


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SersicProfileMassVC')

# +++++++++++++++++++++++++++++++++++++++++++++++++
# Interpolation functions:


def interpolate_entire_table(r=None, table=None,
        total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base,
        filename=None):
    """
    Interpolate entire table, returning new profiles sampled at r.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]

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
        table = read_profile_table(filename=filename, n=n, invq=invq, path=path, filename_base=filename_base)

    vcirc =         interpolate_sersic_profile_VC(r=r, total_mass=total_mass, Reff=Reff,
                                                  n=table['n'], invq=table['invq'], table=table)
    menc3D_sph =    interpolate_sersic_profile_menc(r=r, total_mass=total_mass, Reff=Reff,
                                                    n=table['n'], invq=table['invq'], table=table)
    menc3D_ellip =  interpolate_sersic_profile_menc(r=r, total_mass=total_mass, Reff=Reff,
                                                    n=table['n'], invq=table['invq'], table=table,
                                                    sphere=False)
    rho =           interpolate_sersic_profile_rho(r=r, total_mass=total_mass, Reff=Reff,
                                                   n=table['n'], invq=table['invq'], table=table)
    dlnrho_dlnr =   interpolate_sersic_profile_dlnrho_dlnr(r=r, Reff=Reff, n=table['n'],
                                                           invq=table['invq'], table=table)

    # ---------------------
    # Setup table:
    table_interp = { 'r':                   r,
                     'total_mass':          total_mass,
                     'Reff':                Reff,
                     'vcirc':               vcirc,
                     'menc3D_sph':          menc3D_sph,
                     'menc3D_ellipsoid':    menc3D_ellip,
                     'rho':                 rho,
                     'dlnrho_dlnr':         dlnrho_dlnr }


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


def interpolate_sersic_profile_menc(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base,
        filename=None, table=None,
        sphere=True):
    """
    Interpolate Menc3D_sphere(r) at arbitrary radii r, for arbitrary Mtot and Reff.

    Uses the saved table of Menc3D_sphere(r) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Mtot and Reff.
    (by mapping the radius using r' = (r/Reff * table_Reff) )

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
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
        table = read_profile_table(filename=filename, n=n, invq=invq, path=path, filename_base=filename_base)

    if sphere:
        table_menc =    table['menc3D_sph']
    else:
        table_menc =    table['menc3D_ellipsoid']

    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['r'][0] > 0.:
        table_rad = np.append(0., table_rad)
        table_menc = np.append(0., table_menc)

    m_interp = scp_interp.interp1d(table_rad, table_menc, fill_value=np.NaN, bounds_error=False, kind='cubic')
    m_interp_extrap = scp_interp.interp1d(table_rad, table_menc, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    menc_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    menc_interp[wh_in] =     (m_interp(rarr[wh_in] / Reff * table_Reff) * (total_mass / table_mass) )
    menc_interp[wh_extrap] = (m_interp_extrap(rarr[wh_extrap] / Reff * table_Reff) * (total_mass / table_mass) )

    if (len(rarr) > 1):
        return menc_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return menc_interp[0]
        else:
            # Length 1 array input
            return menc_interp

def interpolate_sersic_profile_VC(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base,
        filename=None, table=None):
    """
    Interpolate vcirc(r) at arbitrary radii r, for arbitrary Mtot and Reff.

    Uses the saved table of vcirc(r) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Mtot and Reff.
    (by mapping the radius using r' = (r/Reff * table_Reff), and scaling for the difference
    between Mtot and table_Mtot)

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate vcirc [kpc]
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
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        vcirc_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq, path=path, filename_base=filename_base)

    table_vcirc =   table['vcirc']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: vcirc=0
    if table['r'][0] > 0.:
        table_rad = np.append(0., table_rad)
        table_vcirc = np.append(0., table_vcirc)

    v_interp = scp_interp.interp1d(table_rad, table_vcirc, fill_value=np.NaN, bounds_error=False, kind='cubic')
    v_interp_extrap = scp_interp.interp1d(table_rad, table_vcirc, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    vcirc_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    vcirc_interp[wh_in] =     (v_interp(rarr[wh_in]  / Reff * table_Reff) * np.sqrt(total_mass / table_mass) * np.sqrt(table_Reff / Reff))
    vcirc_interp[wh_extrap] = (v_interp_extrap(rarr[wh_extrap]  / Reff * table_Reff) * np.sqrt(total_mass / table_mass) * np.sqrt(table_Reff / Reff))

    if (len(rarr) > 1):
        return vcirc_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return vcirc_interp[0]
        else:
            # Length 1 array input
            return vcirc_interp


def interpolate_sersic_profile_rho(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base,
        filename=None, table=None):
    """
    Interpolate Rho(r) at arbitrary radii r, for arbitrary Mtot and Reff.

    Uses the saved table of rho(r) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Mtot and Reff.
    (by mapping the radius using r' = (r/Reff * table_Reff) )

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
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
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        rho_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq,  path=path, filename_base=filename_base)

    table_rho =     table['rho']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']


    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['r'][0] > 0.:
        try:
            table_rad = np.append(r.min() * table_Reff/Reff, table_rad)
            table_rho = np.append(rho(r.min()* table_Reff/Reff, total_mass=table_mass, Reff=table_Reff, n=n, q=table['q']), table_rho)
        except:
            pass

    r_interp = scp_interp.interp1d(table_rad, table_rho, fill_value=np.NaN, bounds_error=False, kind='cubic')
    r_interp_extrap = scp_interp.interp1d(table_rad, table_rho, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    rho_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    rho_interp[wh_in] =     (r_interp(rarr[wh_in] / Reff * table_Reff) * (total_mass / table_mass) * (table_Reff / Reff)**3 )
    rho_interp[wh_extrap] = (r_interp_extrap(rarr[wh_extrap] / Reff * table_Reff) * (total_mass / table_mass) * (table_Reff / Reff)**3 )

    if (len(rarr) > 1):
        return rho_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return rho_interp[0]
        else:
            # Length 1 array input
            return rho_interp


def interpolate_sersic_profile_dlnrho_dlnr(r=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None, table=None):
    """
    Interpolate dlnrho/dlnr at arbitrary radii r, for arbitrary Reff.

    Uses the saved table of dlnrho/dlnr(r) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Reff.
    (by mapping the radius using r' = (r/Reff * table_Reff) )

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate vcirc [kpc]
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
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        dlnrho_dlnr_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq,  path=path, filename_base=filename_base)

    table_dlnrho_dlnr =     table['dlnrho_dlnr']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['r'][0] > 0.:
        try:
            table_rad = np.append(r.min() * table_Reff/Reff, table_rad)
            table_dlnrho_dlnr = np.append(dlnrho_dlnr(r.min()* table_Reff/Reff, n=n, total_mass=table_mass,
                                        Reff=table_Reff, q=table['q']), table_dlnrho_dlnr)
        except:
            pass

    # Catch NaNs:
    whfin = np.where(np.isfinite(table_dlnrho_dlnr))
    table_dlnrho_dlnr = table_dlnrho_dlnr[whfin]
    table_rad = table_rad[whfin]

    r_interp = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, fill_value=np.NaN, bounds_error=False, kind='cubic')
    r_interp_extrap = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    dlnrho_dlnr_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    dlnrho_dlnr_interp[wh_in] =     (r_interp(rarr[wh_in] / Reff * table_Reff) )
    dlnrho_dlnr_interp[wh_extrap] = (r_interp_extrap(rarr[wh_extrap] / Reff * table_Reff) )

    if (len(rarr) > 1):
        return dlnrho_dlnr_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return dlnrho_dlnr_interp[0]
        else:
            # Length 1 array input
            return dlnrho_dlnr_interp


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
    # 1:1, 1:2, 1:3, ... flattening  [also prolate 2:1, 1.5:1]

    nearest_n = table_n[ np.argmin( np.abs(table_n - n) ) ]
    nearest_invq = table_invq[ np.argmin( np.abs( table_invq - invq) ) ]

    return nearest_n, nearest_invq

def interpolate_entire_table_nearest(r=None, table=None,
        total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base,
        filename=None):
    """
    Interpolate entire table, returning new profiles sampled at r,
    using the **nearest values** of n and invq that are included
    in the Sersic profile table collection.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]

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

        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        table_interp_nearest: dict

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    table_interp_nearest = interpolate_entire_table(r=r, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

    return table_interp_nearest

def interpolate_sersic_profile_menc_nearest(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None):
    """
    Interpolate Menc3D_sphere(r) at arbitrary radii r, for arbitrary Mtot and Reff,
    using the **nearest values** of n and invq that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_menc()` with these values.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
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

    Returns
    -------
        menc_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    menc_interp_nearest = interpolate_sersic_profile_menc(r=r, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

    return menc_interp_nearest


def interpolate_sersic_profile_VC_nearest(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None):
    """
    Interpolate vcirc(r) at arbitrary radii r, for arbitrary Mtot and Reff,
    using the **nearest values of n and invq** that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_VC()` with these values.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
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

    Returns
    -------
        vcirc_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    vcirc_interp_nearest = interpolate_sersic_profile_VC(r=r, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

    return vcirc_interp_nearest


def interpolate_sersic_profile_rho_nearest(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None):
    """
    Interpolate Rho(r) at arbitrary radii r, for arbitrary Mtot and Reff,
    using the **nearest values of n and invq** that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_rho()` with these values.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
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

    Returns
    -------
        rho_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    rho_interp_nearest = interpolate_sersic_profile_rho(r=r, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

    return rho_interp_nearest


def interpolate_sersic_profile_dlnrho_dlnr_nearest(r=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None):
    """
    Interpolate dlnrho_g/dlnr at arbitrary radii r, for arbitrary Reff,
    using the **nearest values of n and invq** that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_dlnrho_dlnr()` with these values.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
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

    Returns
    -------
        dlnrho_dlnr_interp_nearest: float or array_like

    """
    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    dlnrho_dlnr_interp_nearest = interpolate_sersic_profile_dlnrho_dlnr(r=r, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

    return dlnrho_dlnr_interp_nearest



def interpolate_sersic_profile_dlnrho_dlnr_bulge_disk_nearest(r=None,
        BT=0.,  total_mass=1.e11,
        Reff_disk=None, n_disk=1., invq_disk=5.,
        Reff_bulge=1.,  n_bulge=4., invq_bulge=1.,
        path=None, filename_base=_sersic_profile_filename_base, filename=None):
    """
    Interpolate dlnrho_g/dlnr at arbitrary radii r,
    for a composite DISK+BULGE system. Both disk and bulge can have arbitary Reff,
    but this uses the **nearest values of n and invq** that are included
    in the Sersic profile table collection.

    Finds the nearest n, invq for the "default" table collection,
    then returns `dlnrho_dlnr_interp_nearest()` for the total DISK+BULGE system.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
        total_mass: float
            Total mass of the component [Msun]    [Default: 10^11 Msun]
        BT: float
            Bulge to total ratio (Total = Disk + Bulge)  [Default: 0.]
        Reff_disk: float
            Effective radius of disk Sersic profile [kpc]
        n_disk: float
            Sersic index of disk.
        invq_disk: float
            Inverse of the intrinsic axis ratio of disk Sersic profile, invq = 1/q.
        Reff_bulge: float
            Effective radius of bulge Sersic profile [kpc]
        n_bulge: float
            Sersic index of bulge. [Default: 4]
        invq_bulge: float
            Inverse of the intrinsic axis ratio of bulge Sersic profile, invq = 1/q.
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

    Returns
    -------
        dlnrho_dlnr_interp_nearest: float or array_like

    """

    Mbulge = total_mass * BT
    Mdisk = total_mass * (1.-BT)

    # Use the "typical" collection of table values:
    rho_t = r * 0.
    rho_dlnrho_dlnr_sum = r * 0.
    for n, invq, Reff, M in zip([n_disk, n_bulge], [invq_disk, invq_bulge],
            [Reff_disk, Reff_bulge], [Mdisk, Mbulge]):
        nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

        dlnrho_dlnr_interp_nearest = interpolate_sersic_profile_dlnrho_dlnr(r=r, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

        rho_interp_nearest = interpolate_sersic_profile_rho(r=r, total_mass=M,
                Reff=Reff, n=nearest_n, invq=nearest_invq, path=path,
                filename_base=filename_base, filename=filename)
        rho_t += rho_interp_nearest
        rho_dlnrho_dlnr_sum += rho_interp_nearest * dlnrho_dlnr_interp_nearest

    dlnrho_dlnr_interp_total = (1./rho_t) * rho_dlnrho_dlnr_sum

    return dlnrho_dlnr_interp_total
