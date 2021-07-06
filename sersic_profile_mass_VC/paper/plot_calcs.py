##################################################################################
# sersic_profile_mass_VC/paper/plot_calcs.py                                     #
#                                                                                #
# Copyright 2018-2021 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import os
import numpy as np
import scipy.special as scp_spec
import astropy.units as u
import astropy.cosmology as apy_cosmo
import astropy.constants as apy_con

import logging

from sersic_profile_mass_VC.utils import calcs as util_calcs

# DEFAULT COSMOLOGY
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc
deg2rad = np.pi/180.

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SersicProfileMassVC')



# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Halos, for use in plot.paper_plots

class TPH:
    """
    Class that specifies a generalized NFW halo ("two-power halo")

    Parameters
    ----------
        z: float
            Redshift
        Mvir: float
            Total halo mass (within virial radius=R200) [Msun]
        conc: float
            Halo concentration.

        alpha: float, optional
            Halo inner slope. Default: 1. (NFW)
        beta: flaot, optional
            Halo outer slope. Default: 3. (NFW)
        cosmo: AstroPy cosmology instance, optional
            Default: flat LCDM, Om0=0.3, H0=70.

    """

    def __init__(self, z=2., Mvir=1.e12, conc=4., alpha=1., beta=3.,
                cosmo=_default_cosmo):
        self.z = z
        self.Mvir = Mvir
        self.conc = conc
        self.alpha = alpha
        self.beta = beta
        self.cosmo = cosmo

    @property
    def rvir(self):
        """
        Calculate the halo virial radius at a given redshift

        Returns
        -------
            rvir: float
                Halo virial radius [kpc]

        """
        g_new_unit = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value
        hz = self.cosmo.H(self.z).value
        rvir = ((self.Mvir * (g_new_unit * 1e-3) / (10 * hz * 1e-3) ** 2) ** (1. / 3.))

        return rvir


    def enclosed_mass(self, r):
        """
        Calculate the enclosed mass of the halo as a function of radius

        Paramaters
        ----------
            r: float or array_like
                Radi[us/i] at which to calculate the enclosed mass  [kpc]

        Returns
        -------
            mhalo_enc: float or array_like
                Enclosed halo mass profile as as a function of radius   [Msun]

        """
        rs = self.rvir/self.conc
        aa = self.Mvir*(r/self.rvir)**(3 - self.alpha)
        bb = (scp_spec.hyp2f1(3-self.alpha, self.beta-self.alpha, 4-self.alpha, -r/rs) /
              scp_spec.hyp2f1(3-self.alpha, self.beta-self.alpha, 4-self.alpha, -self.conc))

        return aa*bb


    def v_circ(self, r):
        """
        Determine vcirc for the halo, assuming spherical symmetry:

        .. math::

            v_{\mathrm{circ}}(r) = \sqrt{\\frac{G M_{\mathrm{enc,halo}}(r)}{r}}

        Parameters
        -----------
            r: float or array_like
                Radi[us/i] at which to calculate the circular velocity [kpc]

        Returns
        -------
            vcirc_halo: float or array_like
                Halo circular velocity as a function of radius  [km/s]

        """
        mass_enc = self.enclosed_mass(r)
        return util_calcs.vcirc_spherical_symmetry(r=r, menc=mass_enc)

class NFW(TPH):
    """
    Class that specifies a NFW halo

    Parameters
    ----------
        z: float
            Redshift
        Mvir: float
            Total halo mass (within virial radius=R200) [Msun]
        conc: float
            Halo concentration.
        cosmo: AstroPy cosmology instance, optional
            Default: flat LCDM, Om0=0.3, H0=70.

    """

    def __init__(self, z=2., Mvir=1.e12, conc=4., cosmo=_default_cosmo):
        super(NFW, self).__init__(z=z, Mvir=Mvir, conc=conc, alpha=1., beta=3., cosmo=cosmo)

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Calculation helper functions: Halos

def _Mhalo_from_fDMv_NFW(vcirc_baryons=None, fDMv=None, r=None, conc=None, z=None):

    vsq_bar = vcirc_baryons**2
    vsqr_dm_target = vsq_bar / (1./fDMv - 1.)

    mtest = np.arange(-5, 50, 1.0)
    vtest = np.array([_minfunc_vdm_NFW(m, vsqr_dm_target, conc, z, r) for m in mtest])

    a = mtest[vtest < 0][-1]
    b = mtest[vtest > 0][0]

    lmhalo = scp_opt.brentq(_minfunc_vdm_NFW, a, b, args=(vsqr_dm_target, conc, z, r))
    Mhalo = np.power(10., lmhalo)
    return Mhalo

def _minfunc_vdm_NFW(lmass, vtarget, conc, z, r):
        halo_vcirc = calcs.NFW_halo_vcirc(r=r, Mvirial=10.**lmass, conc=conc, z=z)
        return halo_vcirc ** 2 - vtarget

#
def _Mhalo_from_fDMv_TPH(vcirc_baryons=None, fDMv=None, r=None, conc=None, z=None, alpha=None):

    vsq_bar = vcirc_baryons**2
    vsqr_dm_target = vsq_bar / (1./fDMv - 1.)

    mtest = np.arange(-5, 50, 1.0)
    vtest = np.array([_minfunc_vdm_TPH(m, vsqr_dm_target, conc, z, r, alpha) for m in mtest])

    a = mtest[vtest < 0][-1]
    b = mtest[vtest > 0][0]

    lmhalo = scp_opt.brentq(_minfunc_vdm_TPH, a, b, args=(vsqr_dm_target, conc, z, r, alpha))
    Mhalo = np.power(10., lmhalo)
    return Mhalo

def _minfunc_vdm_TPH(lMvir, vtarget, conc, z, r, alpha):
        halo_vcirc = calcs.TPH_halo_vcirc(r=r, Mvirial=10.**lMvir, conc=conc, z=z, alpha=alpha)
        return halo_vcirc ** 2 - vtarget


def _alpha_from_fDMv_TPH(vcirc_baryons=None, fDMv=None, r=None, conc=None, z=None, Mhalo=None):

    vsq_bar = vcirc_baryons**2
    vsqr_dm_target = vsq_bar / (1./fDMv - 1.)

    alphtest = np.arange(-50, 50, 1.)
    vtest = np.array([_minfunc_vdm_TPH_alpha(alph, vsqr_dm_target, np.log10(Mhalo), conc, z, r) for alph in alphtest])

    try:
        a = alphtest[vtest < 0][-1]
        try:
            b = alphtest[vtest > 0][0]
        except:
            a = alphtest[-2] # Even if not perfect, force in case of no convergence...
            b = alphtest[-1]
    except:
        a = alphtest[0]    # Even if not perfect, force in case of no convergence...
        b = alphtest[1]

    alpha = scp_opt.brentq(_minfunc_vdm_TPH_alpha, a, b, args=(vsqr_dm_target, np.log10(Mhalo), conc, z, r))

    return alpha

def _minfunc_vdm_TPH_alpha(alpha, vtarget, lMvir, conc, z, r):
    halo_vcirc = calcs.TPH_halo_vcirc(r=r, Mvirial=10.**lMvir, conc=conc, z=z, alpha=alpha)
    return halo_vcirc ** 2 - vtarget


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Calculation helper functions: for Mbar/Mstar/fgas


def _solver_lmstar_fgas_mbar(z=None, Mbar=None):

    lmbar_targ = np.log10(Mbar)

    lm_test = np.arange(6., 12., 0.2)
    vtest = np.array([_minfunc_lmstar_fgas_mbar(lm, z, lmbar_targ) for lm in lm_test])

    try:
        a = lm_test[vtest < 0][-1]
        try:
            b = lm_test[vtest > 0][0]
        except:
            a = lm_test[-2] # Even if not perfect, force in case of no convergence...
            b = lm_test[-1]
    except:
        a = lm_test[0]    # Even if not perfect, force in case of no convergence...
        b = lm_test[1]

    lmstar = scp_opt.brentq(_minfunc_lmstar_fgas_mbar, a, b, args=(z, lmbar_targ))

    return lmstar

def _minfunc_lmstar_fgas_mbar(lmstar_test, z, lmbar_targ):

    fgas = _fgas_scaling_relation_MS(z=z, lmstar=lmstar_test)

    lmbar_test = lmstar_test - np.log10(1.-fgas)

    return lmbar_test - lmbar_targ



# def _sigr_toy(r, sig01, sig0, rsig):
#     sigr = np.sqrt((sig01*np.exp(-r/rsig))**2 + sig0**2)
#     return sigr
#
# def _alpha_sigr_toy(r, sig01, sig0, rsig):
#     sigr = _sigr_toy(r, sig01, sig0, rsig)
#     dlnsigsq_dlnr = -2.*r/rsig * (sig01*np.exp(-r/rsig))**2 / sigr**2
#     return -dlnsigsq_dlnr
