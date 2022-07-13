##################################################################################
# deprojected_sersic_models/paper/plot_calcs.py                                  #
#                                                                                #
# Copyright 2018-2022 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import os
import numpy as np
import scipy.special as scp_spec
import astropy.units as u
import astropy.cosmology as apy_cosmo
import astropy.constants as apy_con

import logging

from deprojected_sersic_models.utils import calcs as util_calcs

# DEFAULT COSMOLOGY
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc
deg2rad = np.pi/180.

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DeprojectedSersicModels')


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# FREEMAN DISK (Infinitely thin exponential disk) VCIRC

def freeman_vcirc(rarr, total_mass, Reff):
    n = 1.
    Ie = util_calcs.get_Ie(total_mass=total_mass, Reff=Reff, n=n, q=1., i=0., Upsilon=1.)
    Sig0 = util_calcs.Ikap(0., Reff=Reff, n=n, Ie=Ie)
    bn = util_calcs.bn_func(n)
    Rd = Reff/bn

    y = rarr/(2.*Rd)

    arg =  (y**2) * ( scp_spec.i0(y)*scp_spec.k0(y) - scp_spec.i1(y)*scp_spec.k1(y) )

    # Convert to cgs, add prefactors
    vsq_cgs = 4*np.pi*G.cgs.value*Msun.cgs.value/(1000.*pc.cgs.value) * Sig0 * Rd * arg

    # Return km/s
    return np.sqrt(vsq_cgs)/1.e5

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
                cosmo=_default_cosmo, scale_fac=1.):
        self.z = z
        self.Mvir = Mvir
        self.conc = conc
        self.alpha = alpha
        self.beta = beta
        self.cosmo = cosmo
        self.scale_fac = 1.

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
        return util_calcs.vcirc_spherical_symmetry(R=r, menc=mass_enc)

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

    def __init__(self, z=2., Mvir=1.e12, conc=4., cosmo=_default_cosmo, scale_fac=1.):
        super(NFW, self).__init__(z=z, Mvir=Mvir, conc=conc, alpha=1., beta=3., cosmo=cosmo, scale_fac=scale_fac)

    def calc_rho0(self, rvirial=None):
        r"""
        Normalization of the density distribution

        Returns
        -------
        rho0 : float
            Mass density normalization in :math:`M_{\odot}/\rm{kpc}^3`
        """
        if rvirial is None:
            rvirial = self.rvir
        aa = self.Mvir/(4.*np.pi*rvirial**3)*self.conc**3
        bb = 1./(np.log(1.+self.conc) - (self.conc/(1.+self.conc)))

        return aa * bb * self.scale_fac


    def density(self, r):
        rvirial = self.rvir
        rho0 = self.calc_rho0(rvirial=rvirial)
        rs = rvirial/self.conc

        x = r/rs
        return rho0 / (x * (1+x)**2)


    def force_z(self, R, z):
        # Analytic
        # Phi = -4*pi*G*rho0*rs^2* (ln(1+x)/x), x =r/rs
        # dPhi/dr = -4*pi*G*rho0*rs^3/r^2 * ( x/(1+x) - ln(1+x) )
        # dPhi/dz = -4*pi*G*rho0*z/x^3 * ( x/(1+x) - ln(1+x) )
        # Limit of dPhi/dr as r->0, applicable for R=0, dPhi/dz as z->0:
        # -4*pi*G*rho0* (-1/2 * rs) = 2*pi*G*rho0*rs
        # as 1/x^2 * ( x/(1+x) - ln(1+x) ) -> -1/2.

        conv_fac = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value) # cmtokpc * Msuntog * kmtocm^2

        rvirial = self.rvir
        rho0 = self.calc_rho0(rvirial=rvirial)
        rs = rvirial/self.conc

        x = np.sqrt(R**2 + z**2) / rs
        gzterm = (z/x**3) * (x/(1.+x) - np.log(1.+x))
        try:
            gzterm[x==0.] = -0.5 * rs
        except:
            if (x == 0.):
                gzterm = -0.5 * rs  # limit as z=r->0 for R=0
        gz = 4*np.pi * G.cgs.value * conv_fac * rho0 * gzterm

        return gz
