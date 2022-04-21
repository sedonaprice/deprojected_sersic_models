##################################################################################
# sersic_profile_mass_VC/core.py                                                 #
#                                                                                #
# Copyright 2018-2021 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import astropy.constants as apy_con

import logging
import copy

from sersic_profile_mass_VC.utils import calcs as util_calcs

from sersic_profile_mass_VC.utils.interp_profiles import interpolate_sersic_profile_rho_function

__all__ = [ 'DeprojSersicDist' ]

# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SersicProfileMassVC')


# TEST:
from scipy.special import expi
import scipy.integrate as scp_integrate

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

class _SersicDistBase:
    """
    Base class that specifies a Sersic distribution.

    Parameters
    ----------
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        i: float
            Inclination of system [deg]

        Upsilon: float, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

        invq: float, derived
            Flattening of Sersic profile; invq=1/q. (Derived)
        Ie: float, derived
            Normalization of Sersic intensity profile at kap = Reff. (Derived)

    """

    def __init__(self, total_mass=1., Reff=1., n=1., q=0.4, i=90., Upsilon=1.):
        self._total_mass    = total_mass
        self._Reff          = Reff
        self._n             = n
        self._q             = q
        self._i             = i
        self._Upsilon       = Upsilon

        self._set_derived()

    def copy(self):
        return copy.deepcopy(self)

    def _set_derived(self):
        """ Set derived quantities from the basic Sersic profile quantities """
        self._set_invq()
        self._set_qobs()
        self._set_bn()
        self._set_Ie()

    def _set_invq(self):
        """ Set invq from intrinsic axis ratio q """
        self._invq = 1./self.q

    def _set_qobs(self):
        """
        Function to calculate the observed axis ratio for an inclined system.
        Uses intrinsic axis ratio `q` and inclination `i`.

        Stores value as `self._qobs = sqrt(q^2 + (1-q^2)*cos(i))``

        """
        self._qobs = util_calcs.qobs_func(q=self.q, i=self.i)


    def _set_bn(self):
        """
        Function to set bn(n) for a Sersic profile.

        Stores value as `self._bn`

        Notes
        -----
        The constant :math:`b_n` satisfies :math:`\Gamma(2n) = 2\gamma (2n, b_n)`

        """
        self._bn = util_calcs.bn_func(self.n)

    def _set_Ie(self):
        """
        Evalutation of Ie, normalization of the Sersic intensity profile at kap = Reff,
        using the total mass (to infinity) and assuming a constant M/L ratio Upsilon.

        Uses the closed-form solution for the total luminosity of the
        2D projected Sersic intensity profile I(kap).

        Stores value as `self._Ie = I(kap=Reff)`

        """
        self._Ie = util_calcs.get_Ie(total_mass=self.total_mass, Reff=self.Reff, n=self.n,
                                q=self.q, i=self.i, Upsilon=self.Upsilon)


    @property
    def total_mass(self):
        return self._total_mass

    @total_mass.setter
    def total_mass(self, value):
        if value <= 0:
            raise ValueError("Total mass must be positive!")
        self._total_mass = value

        # Reset derived values
        self._set_derived()

    @property
    def Reff(self):
        return self._Reff

    @Reff.setter
    def Reff(self, value):
        if value <= 0:
            raise ValueError("Reff must be positive!")
        self._Reff = value

        # Reset derived values
        self._set_derived()

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, value):
        if value <= 0:
            raise ValueError("Sersic index n must be positive!")
        self._n = value

        # Reset derived values
        self._set_derived()

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if value <= 0:
            raise ValueError("Intrinsic axis ratio q must be positive!")
        self._q = value

        # Reset derived values
        self._set_derived()

    @property
    def i(self):
        return self._i

    @i.setter
    def i(self, value):
        self._i = value

        # Reset derived values
        self._set_derived()

    @property
    def Upsilon(self):
        return self._Upsilon

    @Upsilon.setter
    def Upsilon(self, value):
        if value <= 0:
            raise ValueError("Mass-to-light ratio Upsilon must be positive!")
        self._Upsilon = value

        # Reset derived values
        self._set_derived()

    @property
    def invq(self):
        return self._invq

    @property
    def qobs(self):
        return self._qobs

    @property
    def bn(self):
        return self._bn

    @property
    def Ie(self):
        return self._Ie


class DeprojSersicDist(_SersicDistBase):
    """
    Deprojected Sersic mass distribution, with arbitrary flattening (or elongation).

    Parameters
    ----------
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile (c/a)
        i: float
            Inclination of system [deg]

        Upsilon: float, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

        invq: float, derived
            Flattening of Sersic profile; invq=1/q. (Derived)
        Ie: float, derived
            Normalization of Sersic intensity profile at kap = Reff. (Derived)

    """

    def __init__(self, total_mass=1., Reff=1., n=1., q=0.4, i=90., Upsilon=1.):
        super(DeprojSersicDist, self).__init__(total_mass=total_mass, Reff=Reff,
                                               n=n, q=q, i=i, Upsilon=Upsilon)



    def enclosed_mass(self, R, cumulative=False):
        """
        Enclosed 3D mass within a sphere of radius r=R,
        assuming a constant M/L ratio Upsilon.

        Parameters
        ----------
            R: float or array_like
                Major axis radius within which to determine total enclosed 2D projected mass [kpc]

            cumulative: bool, optional
                Shortcut option to only calculate the next annulus,
                then add to the previous Menc(r-rdelt). Default: False

        Returns
        -------
            Menc3D_sphere: float or array_like

        """
        # Calculate fractional enclosed mass, to avoid numerical problems:
        Ie = util_calcs.get_Ie(total_mass=1., Reff=self.Reff, n=self.n,
                                q=self.q, i=self.i, Upsilon=self.Upsilon)

        try:
            if len(R) > 0:
                menc = np.zeros(len(R))
                for j in range(len(R)):
                    if cumulative:
                        # Only calculate annulus, use previous Menc as shortcut:
                        if j > 0:
                            menc_ann =  util_calcs.total_mass3D_integral(R[j], Reff=self.Reff,
                                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                                            Rinner=R[j-1], Upsilon=self.Upsilon)
                            menc[j] = menc[j-1] + menc_ann
                        else:
                            menc[j] = util_calcs.total_mass3D_integral(R[j], Reff=self.Reff,
                                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                                            Rinner=0, Upsilon=self.Upsilon)
                    else:
                        # Direct calculation for every radius
                        menc[j] = util_calcs.total_mass3D_integral(R[j], Reff=self.Reff,
                                        n=self.n, q=self.q, Ie=Ie, i=self.i,
                                        Rinner=0., Upsilon=self.Upsilon)

                # Perform some garbage collection, in the case of very small q
                #     and smaller n. Sometimes integrated to greater than 1,
                #     and then fell off to small numbers.....
                if (self.q <= 0.01) & (self.n <= 1.0):
                    # Use a tolerance of 1.e-9:
                    whgtone = np.where((menc-1.) > 1.e-9)[0]
                    if whgtone[0] < (len(menc) - 1):
                        menc[whgtone[0]:] = 1.

            else:
                menc = util_calcs.total_mass3D_integral(R[0], Reff=self.Reff,
                                n=self.n, q=self.q, Ie=Ie, i=self.i,
                                Rinner=0., Upsilon=self.Upsilon)
        except:
            menc = util_calcs.total_mass3D_integral(R, Reff=self.Reff,
                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                            Rinner=0., Upsilon=self.Upsilon)

        # Return enclosed mass: fractional menc * total mass
        return menc*self.total_mass

    def v_circ(self, R):
        """
        Circular velocity in the midplane of the deprojected Sersic mass distribution.

        Parameters
        ----------
            R: float or array_like
                Midplane radius at which to evaluate the circular velocity [kpc]

        Returns
        -------
            vcirc: float or array_like
                Circular velocity in the midplane at r [km/s]

        """
        cnst = 4*np.pi*G.cgs.value*Msun.cgs.value*self.q/(1000.*pc.cgs.value)

        try:
            if len(R) > 0:
                vcsq = np.zeros(len(R))
                for j in range(len(R)):
                    vcsq[j] = cnst*util_calcs.vel_integral(R[j], Reff=self.Reff, n=self.n, q=self.q,
                                                           Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
                    if R[j] == 0:
                        vcsq[j] = 0.
            else:
                vcsq = cnst*util_calcs.vel_integral(R[0], Reff=self.Reff, n=self.n, q=self.q,
                                                    Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
                if R == 0:
                    vcsq = 0.
        except:
            vcsq = cnst*util_calcs.vel_integral(R, Reff=self.Reff, n=self.n, q=self.q,
                                                Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            if R == 0:
                vcsq = 0.

        return np.sqrt(vcsq)/1.e5



    def density(self, R):
        """
        Density profile at :math:`m=R` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            R: float or array_like
                Distance at which to evaluate the circular velocity [kpc]

        Returns
        -------
            rho_arR: float or array_like
                Density profile at m [Msun / kpc^3]

        """
        try:
            if len(R) > 0:
                rho_arr = np.zeros(len(R))
                for j in range(len(R)):
                    rho_arr[j] = util_calcs.rho_m(R[j], Reff=self.Reff, n=self.n, q=self.q,
                                                  Ie=self.Ie, i=self.i, Upsilon=self.Upsilon,
                                                  replace_asymptote=True)
            else:
                rho_arr = util_calcs.rho_m(R[0], Reff=self.Reff, n=self.n, q=self.q,
                                           Ie=self.Ie, i=self.i, Upsilon=self.Upsilon,
                                           replace_asymptote=True)
        except:
            rho_arr = util_calcs.rho_m(R, Reff=self.Reff, n=self.n, q=self.q,
                                       Ie=self.Ie, i=self.i, Upsilon=self.Upsilon,
                                       replace_asymptote=True)

        return rho_arr


    def drho_dR(self, R):
        """
        Derivative of the density profile, :math:`d\\rho/dR`,
        at distance :math:`m=R` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            R: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            drho_dR_arR: float or array_like
                Derivative of density profile at m=R

        """
        try:
            if len(R) > 0:
                drho_dR_arr = np.zeros(len(R))
                for j in range(len(R)):
                    drho_dR_arr[j] = util_calcs.drhom_dm_multimethod(R[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                drho_dR_arr = util_calcs.drhom_dm_multimethod(R[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            drho_dR_arr = util_calcs.drhom_dm_multimethod(R, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return drho_dR_arr


    def dlnrho_dlnR(self, R):
        """
        Slope of the log density profile, :math:`d\\ln\\rho/d\\ln{}R`,
        in the midplane at radius :math:`m=R` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            R: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            dlnrho_dlnR_arR: float or array_like
                Derivative of log density profile at r=m

        """
        try:
            if len(R) > 0:
                dlnrho_dlnR_arr = np.zeros(len(R))
                for j in range(len(R)):
                    dlnrho_dlnR_arr[j] = util_calcs.dlnrhom_dlnm_multimethod(R[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                dlnrho_dlnR_arr = util_calcs.dlnrhom_dlnm_multimethod(R[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            dlnrho_dlnR_arr = util_calcs.dlnrhom_dlnm_multimethod(R, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return dlnrho_dlnR_arr

    ############################################################################################
    ############################################################################################
    ############################################################################################
    # vvvvvvvv REMOVE THESE LATER vvvvvvvv

    def _drho_dR_leibniz(self, R):
        """
        Derivative of the density profile, :math:`d\\rho/dR`,
        at distance :math:`m=R` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            R: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            drho_dR_arR: float or array_like
                Derivative of density profile at r=m

        """
        try:
            if len(R) > 0:
                drho_dR_arr = np.zeros(len(R))
                for j in range(len(R)):
                    drho_dR_arr[j] = util_calcs.drhom_dm_leibniz(R[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                drho_dR_arr = util_calcs.drhom_dm_leibniz(R[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            drho_dR_arr = util_calcs.drhom_dm_leibniz(R, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return drho_dR_arr


    def _dlnrho_dlnR_leibniz(self, R):
        """
        Slope of the log density profile, :math:`d\\ln\\rho/d\\ln{}R`,
        in the midplane at radius :math:`m=R` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            R: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            dlnrho_dlnR_arR: float or array_like
                Derivative of log density profile at m=R

        """
        try:
            if len(R) > 0:
                dlnrho_dlnR_arr = np.zeros(len(R))
                for j in range(len(R)):
                    dlnrho_dlnR_arr[j] = util_calcs.dlnrhom_dlnm_leibniz(R[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                dlnrho_dlnR_arr = util_calcs.dlnrhom_dlnm_leibniz(R[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            dlnrho_dlnR_arr = util_calcs.dlnrhom_dlnm_leibniz(R, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return dlnrho_dlnR_arr



    ############################################################################################
    ############################################################################################
    ############################################################################################

    def _drho_dR_scipy(self, R):
        """
        Derivative of the density profile, :math:`d\\rho/dR`,
        at distance :math:`m=R` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            R: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            drho_dR_arR: float or array_like
                Derivative of density profile at m=R

        """
        try:
            if len(R) > 0:
                drho_dR_arr = np.zeros(len(R))
                for j in range(len(R)):
                    drho_dR_arr[j] = util_calcs.drhom_dm_scipy_derivative(R[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                drho_dR_arr = util_calcs.drhom_dm_scipy_derivative(R[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            drho_dR_arr = util_calcs.drhom_dm_scipy_derivative(R, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return drho_dR_arr


    def _dlnrho_dlnR_scipy(self, R):
        """
        Slope of the log density profile, :math:`d\\ln\\rho/d\\ln{}R`,
        in the midplane at radius :math:`m=R` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            R: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            dlnrho_dlnR_arR: float or array_like
                Derivative of log density profile at m=R

        """
        try:
            if len(R) > 0:
                dlnrho_dlnR_arr = np.zeros(len(R))
                for j in range(len(R)):
                    dlnrho_dlnR_arr[j] = util_calcs.dlnrhom_dlnm_scipy_derivative(R[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                dlnrho_dlnR_arr = util_calcs.dlnrhom_dlnm_scipy_derivative(R[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            dlnrho_dlnR_arr = util_calcs.dlnrhom_dlnm_scipy_derivative(R, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return dlnrho_dlnR_arr


    # ^^^^^^^^ REMOVE THESE LATER ^^^^^^^^
    ############################################################################################
    ############################################################################################
    ############################################################################################


    def surface_density(self, R):
        """
        Surface density distribution for a Sersic profile, assuming a M/L ratio Upsilon.

        Parameters
        ----------
            R: float or array_like
                Major axis radius within which to determine surface density distribution [kpc]

        Returns
        -------
            sigma_arR: float or array_like
                Surface density of Sersic profile at R

        """
        return self.Upsilon * util_calcs.Ikap(R, Reff=self.Reff, n=self.n, Ie=self.Ie)

    def projected_enclosed_mass(self, R):
        """
        Projected 2D mass enclosed within an ellipse
        (or elliptical shell), assuming a constant M/L ratio Upsilon.

        Parameters
        ----------
            R: float or array_like
                Major axis radius within which to determine total enclosed 2D projected mass [kpc]

        Returns
        -------
            Menc2D_ellipse: float or array_like

        """
        return util_calcs.total_mass2D_direct(R, total_mass=self.total_mass,
                            Reff=self.Reff, n=self.n, q=self.q, i=self.i)



    def enclosed_mass_ellipsoid(self, R, cumulative=False):
        """
        Enclosed 3D mass within an ellpsoid of
        major axis radius r and intrinsic axis ratio q
        (e.g. the same as the Sersic profile isodensity contours),
        assuming a constant M/L ratio Upsilon.

        Parameters
        ----------
            R: float or array_like
                Major axis radius within which to determine total enclosed 2D projected mass [kpc]

            cumulative: bool, optional
                Shortcut option to only calculate the next annulus,
                then add to the previous Menc(r-rdelt). Default: False

        Returns
        -------
            Menc3D_ellip: float or array_like

        """
        # Calculate fractional enclosed mass, to avoid numerical problems:
        Ie = util_calcs.get_Ie(total_mass=1., Reff=self.Reff, n=self.n, q=self.q, i=self.i)

        try:
            if len(R) > 0:
                menc = np.zeros(len(R))
                for j in range(len(R)):
                    if cumulative:
                        # Only calculate annulus, use previous Menc as shortcut:
                        if j > 0:
                            menc_ann =  util_calcs.total_mass3D_integral_ellipsoid(R[j], Reff=self.Reff,
                                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                                            Rinner=R[j-1], Upsilon=self.Upsilon)
                            menc[j] = menc[j-1] + menc_ann
                        else:
                            menc[j] = util_calcs.total_mass3D_integral_ellipsoid(R[j], Reff=self.Reff,
                                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                                            Rinner=0., Upsilon=self.Upsilon)
                    else:
                        # Direct calculation for every radius
                        menc[j] = util_calcs.total_mass3D_integral_ellipsoid(R[j], Reff=self.Reff,
                                        n=self.n, q=self.q, Ie=Ie, i=self.i,
                                        Rinner=0., Upsilon=self.Upsilon)
            else:
                menc = util_calcs.total_mass3D_integral_ellipsoid(R[0], Reff=self.Reff,
                                n=self.n, q=self.q, Ie=Ie, i=self.i,
                                Rinner=0., Upsilon=self.Upsilon)
        except:
            menc = util_calcs.total_mass3D_integral_ellipsoid(R, Reff=self.Reff,
                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                            Rinner=0., Upsilon=self.Upsilon)

        # Return enclosed mass: fractional menc * total mass
        return menc*self.total_mass


    def virial_coeff_tot(self, R, vc=None):
        """
        The "total" virial coefficient ktot, which satisfies

        .. math::

            M_{\mathrm{tot}} = k_{\mathrm{tot}}(R) \\frac{v_{\mathrm{circ}}(R)^2 r}{ G },

        to convert between the circular velocity at any given radius and the total system mass.

        Parameters
        ----------
            R: float or array_like
                Major axis radius within which to determine total enclosed 2D projected mass [kpc]

            vc: float or array_like, optional
                Pre-calculated evaluation of vcirc(R)
                (saves time to avoid recalculating vcirc(R))  [km/s]

        Returns
        -------
            ktot: float or array_like
                ktot = Mtot * G / (vcirc(R)^2 * r)

        """
        if vc is None:
            vc = self.v_circ(R)

        return util_calcs.virial_coeff_tot(R, total_mass=self.total_mass, vc=vc)



    def virial_coeff_3D(self, R, m3D=None, vc=None):
        """
        The "3D" virial coefficient k3D, which satisfies

        .. math::

            M_{\mathrm{3D,sphere}}(R) = k_{\mathrm{3D}}(R) \\frac{v_{\mathrm{circ}}(R)^2 R}{ G },

        to convert between the circular velocity at any given radius
        and the mass enclosed within a sphere of radius R.

        Parameters
        ----------
            R: float or array_like
                Major axis radius within which to determine total enclosed 2D projected mass [kpc]

            m3D: float or array_like, optional
                Pre-calculated evaluation of Menc3D_sphere(R)
                (saves time to avoid recalculating Menc3D_sphere(R)) [Msun]
            vc: float or array_like, optional
                Pre-calculated evaluation of vcirc(R)
                (saves time to avoid recalculating vcirc(R))  [km/s]

        Returns
        -------
            k3D: float or array_like
                k3D = Menc3D_sphere(R) * G / (vcirc(R)^2 * R)

        """
        if m3D is None:
            m3D = self.enclosed_mass(R)

        if vc is None:
            vc = self.v_circ(R)

        return util_calcs.virial_coeff_3D(R, m3D=m3D, vc=vc)



    def force_R(self, R, z, table=None, func_logrho=None):
        if func_logrho is None:
            if table is not None:
                func_logrho = interpolate_sersic_profile_logrho_function(n=self.n, invq=self.invq,
                                                                   table=table)
        return util_calcs.force_R(R, z, total_mass=self.total_mass, Reff=self.Reff,
                                  n=self.n, q=self.q, Ie=self.Ie, i=self.i,
                                  Upsilon=self.Upsilon, logrhom_interp=func_logrho,
                                  table=table)


    def force_z(self, R, z, table=None, func_logrho=None):
        if func_logrho is None:
            if table is not None:
                func_logrho = interpolate_sersic_profile_logrho_function(n=self.n, invq=self.invq,
                                                                   table=sersic_table)
        return util_calcs.force_z(R, z, total_mass=self.total_mass, Reff=self.Reff,
                                  n=self.n, q=self.q, Ie=self.Ie, i=self.i,
                                  Upsilon=self.Upsilon, logrhom_interp=func_logrho,
                                  table=table)


    def profile_table(self, R, cumulative=None, add_reff_table_values=True):
        """
        Create a set of profiles as a dictionary, calculated over the specified radii.
        Also includes constants for the specified Sersic profile parameters,
        and information about values at Reff or the 3D half mass radius.

        Parameters
        ----------
            R: float or array_like
                Radius array, in kpc

            cumulative: bool, optional
                Shortcut option to only calculate the next annulus,
                then add to the previous Menc(r-rdelt).
                Default: Uses cumulative if n >= 2.
            add_reff_table_values: bool, optional
                Add select values at Reff to the table. Requires Reff to be in `R`
                Default: True

        Returns
        -------
            table: dict
                Dictionary containing the various profiles & values.
        """
        # ---------------------
        # Set default cumulative behavior:
        if cumulative is None:
            if self.n >= 2.:
                cumulative = True
            else:
                cumulative = False

        # ---------------------
        # Calculate profiles:
        vcirc =         self.v_circ(R)
        menc3D_sph =    self.enclosed_mass(R, cumulative=cumulative)
        menc3D_ellip =  self.enclosed_mass_ellipsoid(R, cumulative=cumulative)
        rho =           self.density(R)
        dlnrho_dlnR =   self.dlnrho_dlnR(R)

        # ---------------------
        # Setup table:
        table    = { 'R':                   R,
                     'vcirc':               vcirc,
                     'menc3D_sph':          menc3D_sph,
                     'menc3D_ellipsoid':    menc3D_ellip,
                     'rho':                 rho,
                     'dlnrho_dlnR':         dlnrho_dlnR,
                     'total_mass':          self.total_mass,
                     'Reff':                self.Reff,
                     'invq':                self.invq,
                     'q':                   self.q,
                     'n':                   self.n }

        # ---------------------
        if add_reff_table_values:
            # Calculate selected values at Reff:
            try:
                wh_reff = np.where(table['R'] == table['Reff'])[0][0]
            except:
                raise ValueError("Must include 'Reff' in 'R' if using 'add_reff_table_values=True'!")

            table['menc3D_sph_Reff'] =          table['menc3D_sph'][wh_reff]
            table['menc3D_ellipsoid_Reff'] =    table['menc3D_ellipsoid'][wh_reff]
            table['vcirc_Reff'] =               table['vcirc'][wh_reff]

            table['ktot_Reff'] = util_calcs.virial_coeff_tot(table['Reff'],
                            total_mass=table['total_mass'], vc=table['vcirc_Reff'])

            table['k3D_sph_Reff'] = util_calcs.virial_coeff_3D(table['Reff'],
                            m3D=table['menc3D_sph_Reff'], vc=table['vcirc_Reff'])

        # ---------------------
        # 3D Spherical half total_mass radius, for reference:
        table['rhalf3D_sph'] = util_calcs.find_rhalf3D_sphere(R=table['R'], menc3D_sph=table['menc3D_sph'],
                                           total_mass=table['total_mass'])

        # ---------------------
        # Check that table calculated correctly:
        status = util_calcs.check_for_inf(table=table)
        if status > 0:
            raise ValueError("Problem in table calculation: n={:0.1f}, invq={:0.2f}: status={}".format(self.n,
                            self.invq, status))


        return table







class DeprojSersicDistTestRho(DeprojSersicDist):
    """
    Deprojected Sersic mass distribution, with arbitrary flattening (or elongation).

    Parameters
    ----------
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile (c/a)
        i: float
            Inclination of system [deg]

        Upsilon: float, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

        invq: float, derived
            Flattening of Sersic profile; invq=1/q. (Derived)
        Ie: float, derived
            Normalization of Sersic intensity profile at kap = Reff. (Derived)

    """

    def __init__(self, total_mass=1., Reff=1., n=1., q=0.4, i=90., Upsilon=1.,
                 rho0=1.e9, rs=1., a=1.e-1):

        # self.rho0 = 1.e9
        # self.rs = 1.
        # self.a = 1.e-3 #1

        self.rho0 = rho0
        self.rs = rs
        self.a = a

        super(DeprojSersicDistTestRho, self).__init__(total_mass=total_mass, Reff=Reff,
                                               n=n, q=q, i=i, Upsilon=Upsilon)




    def density(self, R):
        """
        Density profile at :math:`m=R` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            R: float or array_like
                Distance at which to evaluate the circular velocity [kpc]

        Returns
        -------
            rho_arR: float or array_like
                Density profile at m [Msun / kpc^3]

        """

        try:
            if len(R) >= 0:
                Rarr = np.abs(R)
        except:
            Rarr = [np.abs(R)]

        Rarr = np.array(Rarr)


        rho_arr = Rarr*0.
        rho_arr[Rarr <= self.rs] = self.rho0
        rho_arr[Rarr > self.rs] = self.rho0*np.exp((self.rs - Rarr[Rarr>self.rs])/self.a)

        # return rho_arr
        try:
            if len(R) >= 0:
                return rho_arr
        except:
            return rho_arr[0]

    def Menc(self, R):

        try:
            if len(R) >= 0:
                Rarr = np.abs(R)
        except:
            Rarr = [np.abs(R)]

        M_arr = Rarr * 0.

        M_arr[Rarr<=self.rs] = 4*np.pi/3. * self.rho0 * Rarr[Rarr<=self.rs]**3
        M0 = 4*np.pi/3. * self.rho0 * self.rs**3
        Me = M0 + 4*np.pi*self.rho0 * self.a * (self.rs**2 + 2.*self.a*self.rs + 2.*self.a**2)
        M_arr[Rarr>self.rs] = Me - 4*np.pi*self.rho0 * self.a * np.exp(-(Rarr[Rarr>self.rs]-self.rs)/self.a) \
                * (Rarr[Rarr>self.rs]**2 + 2.*self.a*Rarr[Rarr>self.rs] + 2.*self.a**2)
        try:
            if len(R) >= 0:
                return M_arr
        except:
            return M_arr[0]

    def pressure(self, R):

        try:
            if len(R) >= 0:
                Rarr = np.abs(R)
        except:
            Rarr = [np.abs(R)]

        cnst = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value)
        Gval = G.cgs.value * cnst

        P_arr = Rarr * 0.

        M0 = 4*np.pi/4.*self.rho0 * self.rs**3
        Me = M0 + 4*np.pi*self.rho0 * self.a * (self.rs**2 + 2.*self.a*self.rs + 2.*self.a**2)

        P_arr[Rarr>self.rs] = Gval*Me*self.rho0/Rarr[Rarr>self.rs] *np.exp(-(Rarr[Rarr>self.rs]-self.rs)/self.a) \
                        + Gval*Me*self.rho0/self.a * np.exp(self.rs/self.a) * expi(-Rarr[Rarr>self.rs]/self.a) \
                        - 2.*np.pi*Gval*self.rho0**2 * self.a**2 * np.exp(-2.*(Rarr[Rarr>self.rs]-self.rs)/self.a) * (1.+ 4*self.a/Rarr[Rarr>self.rs]) \
                        - 8.*np.pi*Gval*self.rho0**2 * self.a**2 * np.exp(2.*self.rs/self.a) * expi(-2.*Rarr[Rarr>self.rs]/self.a)


        P_rs = Gval * Me * self.rho0 / self.rs + Gval * Me * self.rho0 / self.a * np.exp(self.rs/self.a) * expi(-self.rs/self.a) \
                    - 2*np.pi*Gval*self.rho0**2 * self.a**2 *(1.+ 4.*self.a/self.rs) \
                    - 8*np.pi*Gval*self.rho0**2 * self.a**2 * np.exp(2.*self.rs/self.a) * expi(-2.*self.rs/self.a)

        P_arr[Rarr==self.rs] = P_rs
        P_arr[Rarr<self.rs] = P_rs + 2.*np.pi / 3. * Gval * self.rho0**2 * (self.rs**2 - Rarr[Rarr<self.rs]**2)


        try:
            if len(R) >= 0:
                return P_arr
        except:
            return P_arr[0]


    def force_z_analytic(self, R, z):
        m = np.sqrt(R**2 + z**2)


        cnst = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value) # cmtokpc * Msuntog * kmtocm^2

        return - G.cgs.value * cnst * self.Menc(m) * z / (m**3)


    def drho_dR(self, R):
        raise ValueError


    def dlnrho_dlnR(self, R):
        raise ValueERror

    ############################################################################################
    ############################################################################################
    ############################################################################################
    # vvvvvvvv REMOVE THESE LATER vvvvvvvv

    def _drho_dR_leibniz(self, R):
        raise ValueError


    def _dlnrho_dlnR_leibniz(self, R):
        raise ValueError



    ############################################################################################
    ############################################################################################
    ############################################################################################

    def _drho_dR_scipy(self, R):
        raise ValueError


    def _dlnrho_dlnR_scipy(self, R):
        raise ValueError


    # ^^^^^^^^ REMOVE THESE LATER ^^^^^^^^
    ############################################################################################
    ############################################################################################
    ############################################################################################


    def surface_density(self, R):
        raise ValueError

    def projected_enclosed_mass(self, R):
        raise ValueError



    def enclosed_mass_ellipsoid(self, R, cumulative=False):
        raise ValueError


    def virial_coeff_tot(self, R, vc=None):
        raise ValueError



    def virial_coeff_3D(self, R, m3D=None, vc=None):
        raise ValueError



    def force_R(self, R, z, table=None, func_logrho=None):
        if func_logrho is None:
            if table is not None:
                raise ValueError
        return util_calcs.force_R(R, z, total_mass=self.total_mass, Reff=self.Reff,
                                  n=self.n, q=self.q, Ie=self.Ie, i=self.i,
                                  Upsilon=self.Upsilon, logrhom_interp=func_logrho,
                                  table=table)


    def force_z(self, R, z, table=None, func_logrho=None):
        if func_logrho is None:
            raise ValueError
            if table is not None:
                pass
        return util_calcs.force_z(R, z, total_mass=self.total_mass, Reff=self.Reff,
                                  n=self.n, q=self.q, Ie=self.Ie, i=self.i,
                                  Upsilon=self.Upsilon, logrhom_interp=func_logrho,
                                  table=table)


    def _force_z_integrand_analytic_rho(self, tau, R, z):
        m = np.sqrt(R**2/(tau+1.) + z**2/(tau+self.q**2))
        rho = self.density(m)
        integrand = rho / ( (tau+1.) * np.power((tau + self.q**2), 3./2.) )

        return integrand

    def force_z_analytic_rho(self, R, z):
        cnst = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value) # cmtokpc * Msuntog * kmtocm^2

        # Check case at R=z=0: Must have force = 0 in this case
        if ((R**2 + (z/self.q)**2)==0.):
            int_force_z = 0.
        else:
            int_force_z, _ = scp_integrate.quad(self._force_z_integrand_analytic_rho, 0, np.inf,
                                                args=(R, z))

        return -2.*np.pi*G.cgs.value*self.q*z* cnst * int_force_z

    def profile_table(self, R):
        """
        Create a set of profiles as a dictionary, calculated over the specified radii.
        Also includes constants for the specified Sersic profile parameters,
        and information about values at Reff or the 3D half mass radius.

        Parameters
        ----------
            R: float or array_like
                Radius array, in kpc

            cumulative: bool, optional
                Shortcut option to only calculate the next annulus,
                then add to the previous Menc(r-rdelt).
                Default: Uses cumulative if n >= 2.
            add_reff_table_values: bool, optional
                Add select values at Reff to the table. Requires Reff to be in `R`
                Default: True

        Returns
        -------
            table: dict
                Dictionary containing the various profiles & values.
        """

        # ---------------------
        # Calculate profiles:
        vcirc =         R * 0. + -99.
        menc3D_sph =    R * 0. + -99.
        menc3D_ellip =  R * 0. + -99.
        rho =           self.density(R)
        dlnrho_dlnR =   R * 0. + -99.

        # ---------------------
        # Setup table:
        table    = { 'R':                   R,
                     'vcirc':               vcirc,
                     'menc3D_sph':          menc3D_sph,
                     'menc3D_ellipsoid':    menc3D_ellip,
                     'rho':                 rho,
                     'dlnrho_dlnR':         dlnrho_dlnR,
                     'total_mass':          self.total_mass,
                     'Reff':                self.Reff,
                     'invq':                self.invq,
                     'q':                   self.q,
                     'n':                   self.n }

        return table
