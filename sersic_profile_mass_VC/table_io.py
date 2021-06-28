##################################################################################
# sersic_profile_mass_VC/table_io.py                                             #
#                                                                                #
# Copyright 2018-2020 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import numpy as np
from astropy.io import fits
from astropy.table import Table

__all__ = [ 'save_profile_table', 'read_profile_table' ]


def _default_table_fname(path, filename_base, n, invq):
    return path+filename_base+'_n{:0.1f}_invq{:0.2f}.fits'.format(n, invq)

def save_profile_table(table=None,
        path=None, filename_base='mass_VC_profile_sersic',
        filename=None, overwrite=False):
    """
    Save the table of Sersic profile values in a binary FITS table.

    Parameters
    ----------
        table: dict
            The dictionary of the table for a particular n, invq

        path: str or None
            Path to directory containing the saved Sersic profile tables.

        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<filename_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location. (FITS format)
        overwrite: bool, optional
            Option to overwrite the FITS file, if a previous version exists.
            Default: False (will throw an error if the file already exists).

    Notes
    -----
    Saves a binary FITS table containing Sersic profile values.

    The table includes:

        `r`:                array of radii [kpc]

        `invq`:             inverse intrinsic axis ratio

        `q`:                intrinsic axis ratio

        `n`:                Sersic index

        `Reff`:             Effective radius of Sersic profile (the projected 2D half-light radius)

        `total_mass`:       Total mass used for calculation

        `menc3D_sph`:       Mass enclosed within a sphere of radius r

        `vcirc`:            Circular velocity profile at r

        `rho`:              Density at m=r

        `dlnrho_dlnr`:        Derivative of ln(rho) w.r.t. ln(r) at m=r.

        `menc3D_sph_Reff`:    Mass enclosed within a sphere of radius r=Reff

        `vcirc_Reff`:         Circular velocity profile at r=Reff

        `ktot_Reff`:          Total virial coefficient at Reff

        `k3D_sph_Reff`:       Virial coefficient to convert between menc3D and vcirc at Reff

        `rhalf3D_sph`:        3D half mass radius (defined within spherical apertures)

    """

    if table is None:       raise ValueError("Must set 'table'!")
    if filename is None:
        if path is None:    raise ValueError("Must set 'path' if 'filename' is not set !")

        filename = _default_table_fname(path, filename_base, table['n'], table['invq'])

    # Setup FITS recarray:
    fmt_arr = '{}D'.format(len(table['r']))
    fmt_flt = 'D'


    key_list = ['r', 'invq', 'q', 'n', 'total_mass', 'Reff',
                'menc3D_sph', 'menc3D_ellipsoid', 'rho', 'dlnrho_dlnr', 'vcirc',
                'menc3D_sph_Reff', 'menc3D_ellipsoid_Reff', 'vcirc_Reff', 'ktot_Reff', 'k3D_sph_Reff', 'rhalf3D_sph']
    fmt_list = [fmt_arr, fmt_flt, fmt_flt, fmt_flt, fmt_flt, fmt_flt,
                fmt_arr, fmt_arr, fmt_arr, fmt_arr, fmt_arr,
                fmt_flt, fmt_flt, fmt_flt, fmt_flt, fmt_flt, fmt_flt]

    col_stack = []
    for key, fmt in zip(key_list, fmt_list):
        col_stack.append(fits.Column(name=key, format=fmt, array=np.array([table[key]])))

    hdu = fits.BinTableHDU.from_columns(col_stack)
    hdu.writeto(filename, overwrite=overwrite)


def read_profile_table(n=None, invq=None, path=None,
        filename_base='mass_VC_profile_sersic', filename=None):
    """
    Read the table of Sersic profile values from the binary FITS table.

    Parameters
    ----------
        n: float
            Sersic index
        invq: float
            Inverse intrinsic axis ratio
        path: str or None
            Path to directory containing the saved Sersic profile tables.
            If `None`, assumes current directory.

        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<filename_base>_nX.X_invqX.XX.fits`.
            Default: `mass_VC_profile_sersic`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.

    Returns
    -------
        table: AstroPy table
            Slice of AstroPy table containing the Sersic profile curves and values

    """

    if filename is None:
        if path is None:    raise ValueError("Must set 'path' if 'filename' is not set !")
        if n is None:       raise ValueError("Must set 'n' if 'filename' is not set !")
        if invq is None:    raise ValueError("Must set 'invq' if 'filename' is not set !")

        # Ensure output path ends in trailing slash:
        if (path[-1] != '/'): path += '/'

        filename = filename = _default_table_fname(path, filename_base, n, inv)

    t = Table.read(filename)

    return t[0]
