##################################################################################
# sersic_profile_mass_VC/plot/plots.py                                           #
#                                                                                #
# Copyright 2018-2021 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import os
import copy

import dill as pickle

import numpy as np
import scipy.interpolate as scp_interp
import astropy.cosmology as apy_cosmo
import astropy.constants as apy_con
from astropy.table import Table, Row

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, FixedLocator, FixedFormatter, LogLocator
import matplotlib.markers as mks
from matplotlib.legend_handler import HandlerLine2D
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
from matplotlib import colorbar
from matplotlib.patches import Rectangle

from sersic_profile_mass_VC import core


__all__ = [ 'plot_profiles', 'plot_enclosed_mass', 'plot_vcirc',
            'plot_density', 'plot_dlnrho_dlnr',
            'plot_surface_density', 'plot_projected_enclosed_mass']

# ---------------------
# Plot settings

fontsize_leg = 7.
fontsize_labels = 14.

_aliases_profiles_table = {'enclosed_mass': 'menc3D_sph',
                          'v_circ': 'vcirc',
                          'density': 'rho',
                          'dlnrho_dlnr': 'dlnrho_dlnr',
                          'surface_density': None,
                          'projected_enclosed_mass': None}

# Without '$':
_labels_profiles = {'enclosed_mass': r'M_{\mathrm{enc}}',
                    'v_circ': r'v_{\mathrm{circ}}',
                    'density': r'\rho',
                    'dlnrho_dlnr': r'd\ln\rho/d\ln{}r',
                    'surface_density': r'\Sigma',
                    'projected_enclosed_mass': r'M_{\mathrm{2D,proj,enc}}'}


def plot_profiles(sersic_profs, r=np.arange(0., 6., 1.), rlim=None, rlog=True, fileout=None):
    """
    Function to show all six parameter profiles for the Sersic mass distribution(s).

    Parameters
    ----------
        sersic_profs: array_like or ``DeprojSersicDist`` instance or Sersic profile table dictionary.
            Distributions to plot. Can be a list of ``DeprojSersicDist`` instances,
            or a list of Sersic profile table dictionaries, or a single instance of either.

        r: array_like, optional
            Radii over which to show profile. Ignored if sersic_profs contains tables,
            in favor of the table radii `r`.
            Default: np.arange(0., 6., 1.)
        rlim: array_like or None, optional
            If set, spefcify r plot bounds (either linear for rlog=False, or log for rlog=True).
            Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: True

        fileout: str or None, optional
            Filename for saving file. If set to `None`, figure is returned to display.
            Default: None

    """

    nrows = 3
    ncols = 2

    padx = pady = 0.35

    xextra = 0.
    yextra = 0.25

    scale = 3.

    f = plt.figure()
    f.set_size_inches((ncols+(ncols-1)*padx+xextra)*scale, (nrows+pady+yextra)*scale)

    gs = gridspec.GridSpec(nrows, ncols, wspace=padx, hspace=pady)
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            axes.append(plt.subplot(gs[i,j]))

    prof_names = ['enclosed_mass', 'v_circ', 'density',
                  'dlnrho_dlnr', 'surface_density', 'projected_enclosed_mass']

    ylogs = [True, False, True, False, True, True]

    for i in range(nrows):
        for j in range(ncols):
            k = i*ncols + j
            axes[k] = plot_profiles_single_type(sersic_profs, prof_name=prof_names[k],
                                                r=r, rlim=rlim, ylog=ylogs[k], rlog=rlog,
                                                ax=axes[k])

    #############################################################
    if fileout is not None:
        # Save to file:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        # Show plot
        plt.draw()
        plt.show()

    return None


def plot_profiles_single_type(sersic_profs, prof_name='enclosed_mass',
                              r=np.arange(0., 6., 1.), rlim=None, rlog=True, ylog=True,
                              fileout=None, ax=None):
    """
    Base function to plot single parameter profile(s) for the Sersic mass distribution(s).

    Parameters
    ----------
        sersic_profs: array_like or ``DeprojSersicDist`` instance or Sersic profile table dictionary.
            Disributions to plot. Can be a list of ``DeprojSersicDist`` instances,
            or a list of Sersic profile table dictionaries, or a single instance of either.
        prof_name: str
            Name of the profile to plot. Default: `enclosed_mass`

        r: array_like, optional
            Radii over which to show profile. Ignored if sersic_profs contains tables,
            in favor of the table radii `r`.
            Default: np.arange(0., 6., 1.)
        rlim: array_like or None, optional
            If set, spefcify r plot bounds (either linear for rlog=False, or log for rlog=True).
            Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: True
        ylog: bool, optional
            Option to plot log of profile instead of linear. Default: True

        fileout: str or None, optional
            Filename for saving file. If set to `None`, figure is returned to display.
            Default: None
        ax: matplotlib Axes instance or None, optional
            Target plot axis. If not set, assumes only this panel is to be plot and displayed,
            and a new axis is created.
            Default: None

    """

    if ax is None:
        ax_in = False
        # Create axis:
        xextra = 0.
        yextra = 0.
        scale = 3.
        f = plt.figure()
        f.set_size_inches((1+xextra)*scale, (1+yextra)*scale)
        ax = plt.subplot(111)
    else:
        ax_in = True

    # Check if multiple or single:
    if isinstance(sersic_profs, (Table, Row, dict, core.DeprojSersicDist)):
        # Single: coerce into list:
        nProf = 1
        sprof = copy.deepcopy(sersic_profs)
        sersic_profs = [sprof]
    else:
        # Already list-like:
        nProf = len(sersic_profs)

    #############################################################
    # Loop over instances
    for sprof in sersic_profs:
        if isinstance(sprof, core.DeprojSersicDist):
            paramprof = getattr(sprof, prof_name)(r)
            lbl = 'totM={:0.1e}, Reff={:0.1f}, n={:0.1f}, invq={:0.1f}'.format(sprof.total_mass,
                        sprof.Reff, sprof.n, sprof.q)
        else:
            r = sprof['r']
            keyalias = _aliases_profiles_table[prof_name]
            if keyalias is not None:
                paramprof = sprof[keyalias]
            else:
                sprof_tmp = core.DeprojSersicDist(total_mass=sprof['total_mass'],
                        Reff=sprof['Reff'], n=sprof['n'], q=sprof['q'])
                paramprof = getattr(sprof_tmp, prof_name)(r)

            lbl = 'totM={:0.1e}, Reff={:0.1f}, n={:0.1f}, invq={:0.1f}'.format(sprof['total_mass'],
                        sprof['Reff'], sprof['n'], sprof['invq'])


        if ylog:
            paramprof = np.log10(paramprof)
        if rlog:
            rplot = np.log10(r)
        else:
            rplot = r
        # Add bounds:
        if rlim is not None:
            whin = np.where((rplot>= rlim[0]) & (rplot<= rlim[1]))[0]
            rplot = rplot[whin]
            paramprof = paramprof[whin]
        ax.plot(rplot, paramprof, label=lbl, lw=1., ls='-')

    # Add legend:
    legend = ax.legend(frameon=True, numpoints=1,
                scatterpoints=1,
                fontsize=fontsize_leg)

    # Add bounds:
    if rlim is not None:
        ax.set_xlim(rlim)

    # Add axes labels:
    if rlog:
        ax.set_xlabel(r'$\log_{10}(r/\mathrm{kpc})$', fontsize=fontsize_labels)
    else:
        ax.set_xlabel('r [kpc]', fontsize=fontsize_labels)
    if ylog:
        ax.set_ylabel(r'$\log_{10}('+_labels_profiles[prof_name]+r')$', fontsize=fontsize_labels)
    else:
        ax.set_ylabel(r'$'+_labels_profiles[prof_name]+r'$', fontsize=fontsize_labels)

    #############################################################
    # Return axis, if it was input:
    if ax_in:
        return ax

    else:
        if fileout is not None:
            # Save to file:
            plt.savefig(fileout, bbox_inches='tight', dpi=300)
            plt.close()
        else:
            # Show plot
            plt.draw()
            plt.show()

        return None



def plot_enclosed_mass(sersic_profs, r=np.arange(0., 6., 1.), rlim=None, rlog=True, ylog=True,
                       fileout=None, ax=None):
    """
    Function to show enclosed mass profile(s) for the Sersic mass distribution(s).

    Parameters
    ----------
        sersic_profs: array_like or ``DeprojSersicDist`` instance or Sersic profile table dictionary.
            Distributions to plot. Can be a list of ``DeprojSersicDist`` instances,
            or a list of Sersic profile table dictionaries, or a single instance of either.

        r: array_like, optional
            Radii over which to show profile. Ignored if sersic_profs contains tables,
            in favor of the table radii `r`.
            Default: np.arange(0., 6., 1.)
        rlim: array_like or None, optional
            If set, spefcify r plot bounds (either linear for rlog=False, or log for rlog=True).
            Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: True
        ylog: bool, optional
            Option to plot log of profile instead of linear. Default: True

        fileout: str or None, optional
            Filename for saving file. If set to `None`, figure is returned to display.
            Default: None
        ax: matplotlib Axes instance or None, optional
            Target plot axis. If not set, assumes only this panel is to be plot and displayed,
            and a new axis is created.
            Default: None

    """
    plot_profiles_single_type(sersic_profs, prof_name='enclosed_mass',
                              r=r, rlim=rlim, rlog=rlog, ylog=ylog, fileout=fileout, ax=ax)

def plot_vcirc(sersic_profs, r=np.arange(0., 6., 1.), rlim=None, rlog=True, ylog=False,
               fileout=None, ax=None):
    """
    Function to show circular velocity profile(s) for the Sersic mass distribution(s).

    Parameters
    ----------
        sersic_profs: array_like or ``DeprojSersicDist`` instance or Sersic profile table dictionary.
            Distributions to plot. Can be a list of ``DeprojSersicDist`` instances,
            or a list of Sersic profile table dictionaries, or a single instance of either.

        r: array_like, optional
            Radii over which to show profile. Ignored if sersic_profs contains tables,
            in favor of the table radii `r`.
            Default: np.arange(0., 6., 1.)
        rlim: array_like or None, optional
            If set, spefcify r plot bounds (either linear for rlog=False, or log for rlog=True).
            Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: True
        ylog: bool, optional
            Option to plot log of profile instead of linear. Default: False

        fileout: str or None, optional
            Filename for saving file. If set to `None`, figure is returned to display.
            Default: None
        ax: matplotlib Axes instance or None, optional
            Target plot axis. If not set, assumes only this panel is to be plot and displayed,
            and a new axis is created.
            Default: None

    """
    plot_profiles_single_type(sersic_profs, prof_name='v_circ',
                              r=r, rlim=rlim, rlog=rlog, ylog=ylog, fileout=fileout, ax=ax)


def plot_density(sersic_profs, r=np.arange(0., 6., 1.), rlim=None, rlog=True, ylog=True,
                 fileout=None, ax=None):
    """
    Function to show mass density profile(s) for the Sersic mass distribution(s).

    Parameters
    ----------
        sersic_profs: array_like or ``DeprojSersicDist`` instance or Sersic profile table dictionary.
            Distributions to plot. Can be a list of ``DeprojSersicDist`` instances,
            or a list of Sersic profile table dictionaries, or a single instance of either.

        r: array_like, optional
            Radii over which to show profile. Ignored if sersic_profs contains tables,
            in favor of the table radii `r`.
            Default: np.arange(0., 6., 1.)
        rlim: array_like or None, optional
            If set, spefcify r plot bounds (either linear for rlog=False, or log for rlog=True).
            Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: True
        ylog: bool, optional
            Option to plot log of profile instead of linear. Default: True

        fileout: str or None, optional
            Filename for saving file. If set to `None`, figure is returned to display.
            Default: None
        ax: matplotlib Axes instance or None, optional
            Target plot axis. If not set, assumes only this panel is to be plot and displayed,
            and a new axis is created.
            Default: None

    """
    plot_profiles_single_type(sersic_profs, prof_name='density',
                              r=r, rlim=rlim, rlog=rlog, ylog=ylog, fileout=fileout, ax=ax)


def plot_dlnrho_dlnr(sersic_profs, r=np.arange(0., 6., 1.), rlim=None, rlog=True, ylog=False,
                     fileout=None, ax=None):
    """
    Function to show dlnrho/dlnr profile(s) for the Sersic mass distribution(s).

    Parameters
    ----------
        sersic_profs: array_like or ``DeprojSersicDist`` instance or Sersic profile table dictionary.
            Distributions to plot. Can be a list of ``DeprojSersicDist`` instances,
            or a list of Sersic profile table dictionaries, or a single instance of either.

        r: array_like, optional
            Radii over which to show profile. Ignored if sersic_profs contains tables,
            in favor of the table radii `r`.
            Default: np.arange(0., 6., 1.)
        rlim: array_like or None, optional
            If set, spefcify r plot bounds (either linear for rlog=False, or log for rlog=True).
            Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: True
        ylog: bool, optional
            Option to plot log of profile instead of linear. Default: True

        fileout: str or None, optional
            Filename for saving file. If set to `None`, figure is returned to display.
            Default: None
        ax: matplotlib Axes instance or None, optional
            Target plot axis. If not set, assumes only this panel is to be plot and displayed,
            and a new axis is created.
            Default: None

    """
    plot_profiles_single_type(sersic_profs, prof_name='dlnrho_dlnr',
                              r=r, rlim=rlim, rlog=rlog, ylog=ylog, fileout=fileout, ax=ax)

def plot_surface_density(sersic_profs, r=np.arange(0., 6., 1.), rlim=None, rlog=True, ylog=True,
                         fileout=None, ax=None):
    """
    Function to show surface density profile(s) for the Sersic mass distribution(s).

    Parameters
    ----------
        sersic_profs: array_like or ``DeprojSersicDist`` instance or Sersic profile table dictionary.
            Distributions to plot. Can be a list of ``DeprojSersicDist`` instances,
            or a list of Sersic profile table dictionaries, or a single instance of either.

        r: array_like, optional
            Radii over which to show profile. Ignored if sersic_profs contains tables,
            in favor of the table radii `r`.
            Default: np.arange(0., 6., 1.)
        rlim: array_like or None, optional
            If set, spefcify r plot bounds (either linear for rlog=False, or log for rlog=True).
            Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: True
        ylog: bool, optional
            Option to plot log of profile instead of linear. Default: True

        fileout: str or None, optional
            Filename for saving file. If set to `None`, figure is returned to display.
            Default: None
        ax: matplotlib Axes instance or None, optional
            Target plot axis. If not set, assumes only this panel is to be plot and displayed,
            and a new axis is created.
            Default: None

    """
    plot_profiles_single_type(sersic_profs, prof_name='surface_density',
                              r=r, rlim=rlim, rlog=rlog, ylog=ylog, fileout=fileout, ax=ax)

def plot_projected_enclosed_mass(sersic_profs, r=np.arange(0., 6., 1.),
                                 rlim=None, rlog=True, ylog=True,
                                 fileout=None, ax=None):
    """
    Function to show projected (2D) enclosed mass profile(s) for the Sersic mass distribution(s).

    Parameters
    ----------
        sersic_profs: array_like or ``DeprojSersicDist`` instance or Sersic profile table dictionary.
            Distributions to plot. Can be a list of ``DeprojSersicDist`` instances,
            or a list of Sersic profile table dictionaries, or a single instance of either.

        r: array_like, optional
            Radii over which to show profile. Ignored if sersic_profs contains tables,
            in favor of the table radii `r`.
            Default: np.arange(0., 6., 1.)
        rlim: array_like or None, optional
            If set, spefcify r plot bounds (either linear for rlog=False, or log for rlog=True).
            Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: True
        ylog: bool, optional
            Option to plot log of profile instead of linear. Default: True

        fileout: str or None, optional
            Filename for saving file. If set to `None`, figure is returned to display.
            Default: None
        ax: matplotlib Axes instance or None, optional
            Target plot axis. If not set, assumes only this panel is to be plot and displayed,
            and a new axis is created.
            Default: None

    """
    plot_profiles_single_type(sersic_profs, prof_name='projected_enclosed_mass',
                              r=r, rlim=rlim, rlog=rlog, ylog=ylog, fileout=fileout, ax=ax)
