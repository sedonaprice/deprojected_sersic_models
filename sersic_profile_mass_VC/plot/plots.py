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
from astropy.table import Table, Row

import matplotlib as mpl
import matplotlib.pyplot as plt

from sersic_profile_mass_VC import core

__all__ = [ 'plot_profiles', 'plot_enclosed_mass', 'plot_vcirc',
            'plot_density', 'plot_dlnrho_dlnr',
            'plot_surface_density', 'plot_projected_enclosed_mass']

# ---------------------
# Plot settings

_aliases_profiles_table = {'enclosed_mass': 'menc3D_sph',
                          'v_circ': 'vcirc',
                          'density': 'rho',
                          'dlnrho_dlnr': 'dlnrho_dlnr',
                          'surface_density': None,
                          'projected_enclosed_mass': None}

_labels_profiles = {'enclosed_mass': r'Enclosed mass [$M_{\odot}$]',
                    'v_circ': r'Circular velocity [km/s]',
                    'density': r'Mass density [$M_{\odot}/\mathrm{kpc}^3$]',
                    'dlnrho_dlnr': r'Log density slope',
                    'surface_density': r'Projected mass surface density [$M_{\odot}/\mathrm{kpc}^2$]',
                    'projected_enclosed_mass': r'Projected enclosed mass [$M_{\odot}$]'}

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def plot_profiles(sersic_profs, r=np.arange(0., 6., 1.),
                  rlim=None, rlog=True, ylog=None,
                  prof_names=None, plot_kwargs=None, fig_kwargs=None, fileout=None):
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
            If set, spefcify r plot bounds. Default: None (default bounds)
        rlog: array_like or bool, optional
            Option to plot log of radius instead of linear. Default: True
        ylog: array_like or bool, optional
            Option to plot log of profile instead of linear. Default: Specified per panel
        prof_names: array_like, optional
            Specify which profilies to plot.
            Default: None (all six standard profiiles)
        plot_kwargs: array_like or dict, optional
            Option to pass plotting keyword arguments.
            Should include either dict or list of dicts, to match number of input profiles.
            Default: None
        fig_kwargs: dict, optional
            Option to pass other arguments to figure.
            Default: None

        fileout: str or None, optional
            Filename for saving file. If set to `None`, figure is returned to display.
            Default: None

    """
    prof_names_defaults = ['v_circ', 'enclosed_mass', 'density',
                           'dlnrho_dlnr', 'surface_density', 'projected_enclosed_mass']
    ylogs_defaults = {}
    for pn, yl in zip(prof_names_defaults, [False, False, True, False, True, False]):
        ylogs_defaults[pn] = yl

    if prof_names is None:
         prof_names = prof_names_defaults
    nProfs = len(prof_names)

    ncols = 2
    nrows = np.int(np.ceil(nProfs/(1.*ncols)))
    fac = 4.
    figsize_default = (fac*ncols,8./9.*fac*nrows)
    if fig_kwargs is not None:
        figsize = fig_kwargs.get('figsize', figsize_default)
    else:
        figsize = figsize_default
    f = plt.figure(figsize=figsize)
    axes = []
    for i in range(nrows):
        for j in range(ncols):
            axes.append(plt.subplot(nrows,ncols,i*ncols+j+1))

    # Coerce rlog, ylog into array:
    if isinstance(rlog, bool):
        rlogs = [rlog]*nProfs
    else:
        if len(rlog) != nProfs:
            raise ValueError("If specifying 'ylog' per profile, must use {}-element array".format(nProfs))
        rlogs = rlog

    if isinstance(ylog, (bool, type(None))):
        if ylog is None:
            ylogs = []
            for pn in prof_names:
                ylogs.append(ylogs_defaults[pn])
        else:
            ylogs = [ylog]*nProfs
    else:
        if len(ylog) != nProfs:
            raise ValueError("If specifying 'ylog' per profile, must use 6-element array")
        ylogs = ylog

    for i in range(nrows):
        for j in range(ncols):
            k = i*ncols + j
            if k >= nProfs:
                axes[k].set_axis_off()
            else:
                axes[k] = plot_profiles_single_type(sersic_profs, prof_name=prof_names[k],
                                                    r=r, rlim=rlim, rlog=rlogs[k], ylog=ylogs[k],
                                                    plot_kwargs=plot_kwargs,
                                                    fig_kwargs=fig_kwargs, ax=axes[k])

    #############################################################
    if fileout is not None:
        # Save to file:
        plt.savefig(fileout, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        # Show plot
        plt.tight_layout()
        plt.draw()
        plt.show()

    return None


def plot_profiles_single_type(sersic_profs, prof_name='enclosed_mass',
                              r=np.arange(0., 6., 1.), rlim=None, ylim=None, rlog=False, ylog=False,
                              plot_kwargs=None, fig_kwargs=None, fileout=None, ax=None):
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
            If set, spefcify r plot bounds. Default: None (default bounds)
        ylim: array_like or None, optional
            If set, spefcify y plot bounds. Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: False
        ylog: bool, optional
            Option to plot log of profile instead of linear. Default: False
        plot_kwargs: array_like or dict, optional
            Option to pass plotting keyword arguments.
            Should include either dict or list of dicts, to match number of input profiles.
            Default: None
        fig_kwargs: dict, optional
            Option to pass other arguments to figure.
            Default: None

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
        fac = 4.
        figsize_default = (fac,8./9.*fac)
        if fig_kwargs is not None:
            figsize = fig_kwargs.get('figsize', figsize_default)
        else:
            figsize = figsize_default
        f = plt.figure(figsize=figsize)
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

    # Also handle plot kwargs:
    if plot_kwargs is not None:
        if isinstance(plot_kwargs, dict):
            plot_kwargs = [plot_kwargs] * nProf

    #############################################################
    # Loop over instances
    rs, paramprofs = [], []
    for i, sprof in enumerate(sersic_profs):
        if isinstance(sprof, core.DeprojSersicDist):
            paramprof = getattr(sprof, prof_name)(r)
            total_mass, Reff, n, q = sprof.total_mass, sprof.Reff, sprof.n, sprof.q
            rplot = r
        else:
            rplot = sprof['r']
            keyalias = _aliases_profiles_table[prof_name]
            if keyalias is not None:
                paramprof = sprof[keyalias]
            else:
                paramprof = sprof.get(prof_name, None)
                if paramprof is None:
                    sprof_tmp = core.DeprojSersicDist(total_mass=sprof['total_mass'],
                            Reff=sprof['Reff'], n=sprof['n'], q=sprof['q'])
                    paramprof = getattr(sprof_tmp, prof_name)(rplot)
            total_mass, Reff, n, q = sprof['total_mass'], sprof['Reff'], sprof['n'], sprof['invq']

        rs.append(rplot)
        paramprofs.append(paramprof)

        lbl = 'totM={:0.1e}, Reff={:0.1f}, n={:0.1f}, invq={:0.1f}'.format(total_mass, Reff, n, q)
        if plot_kwargs is not None:
            if 'label' not in plot_kwargs[i].keys():
                plot_kwargs[i]['label'] = lbl
            ax.plot(rplot, paramprof, **plot_kwargs[i])
        else:
            ax.plot(rplot, paramprof, label=lbl, lw=1., ls='-')

    # Add legend:
    legend_kwargs = {'frameon': True, 'numpoints': 1, 'scatterpoints': 1}
    if fig_kwargs is not None:
        for kw in fig_kwargs.keys():
            if 'legend_' in kw:
                kwsplit = kw.split('legend_')
                legend_kwargs[kwsplit[1]] = fig_kwargs.get(kw, None)
    legend = ax.legend(**legend_kwargs)

    # Set scale
    if rlog:
        ax.set_xscale('log')
    if ylog:
        ax.set_yscale('log')

    # Add bounds:
    if rlim is not None:
        ax.set_xlim(rlim)
        # Add ylim to plot bound, if ylog=True: otherwise can have very squished plots
        if (ylim is None) & ylog:
            ylo = 1.e100
            yhi = -1.e100
            for rplot, paramprof in zip(rs, paramprofs):
                whin = np.where((rplot >= rlim[0]) & (rplot <= rlim[1]))[0]
                paramtrim = paramprof[whin]
                if paramtrim.min() < ylo:
                    ylo = paramtrim.min()
                if paramtrim.max() > yhi:
                    yhi = paramtrim.max()
            ylim = [ylo, yhi]

    if ylim is not None:
        ax.set_ylim(ylim)

    # Add axes labels:
    ax.set_xlabel('Radius [kpc]')
    ax.set_ylabel(_labels_profiles[prof_name])

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
            plt.tight_layout()
            plt.draw()
            plt.show()
        return None


def plot_enclosed_mass(sersic_profs, r=np.arange(0., 6., 1.), rlim=None, rlog=False, ylog=False,
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
            If set, spefcify r plot bounds. Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: False
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
    plot_profiles_single_type(sersic_profs, prof_name='enclosed_mass',
                              r=r, rlim=rlim, rlog=rlog, ylog=ylog, fileout=fileout, ax=ax)

def plot_vcirc(sersic_profs, r=np.arange(0., 6., 1.), rlim=None, rlog=False, ylog=False,
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
            If set, spefcify r plot bounds. Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: False
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
            If set, spefcify r plot bounds. Default: None (default bounds)
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


def plot_dlnrho_dlnr(sersic_profs, r=np.arange(0., 6., 1.), rlim=None, rlog=False, ylog=False,
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
            If set, spefcify r plot bounds. Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: False
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
            If set, spefcify r plot bounds.
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
                                 rlim=None, rlog=False, ylog=False,
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
            If set, spefcify r plot bounds.
            Default: None (default bounds)
        rlog: bool, optional
            Option to plot log of radius instead of linear. Default: False
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
    plot_profiles_single_type(sersic_profs, prof_name='projected_enclosed_mass',
                              r=r, rlim=rlim, rlog=rlog, ylog=ylog, fileout=fileout, ax=ax)
