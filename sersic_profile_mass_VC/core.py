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

__all__ = [ 'DeprojSersicDist' ]

# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SersicProfileMassVC')

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
    Deprojected Sersic mass distribution, with arbitrary flattening (or prolateness).

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
        super(DeprojSersicDist, self).__init__(total_mass=total_mass, Reff=Reff,
                                               n=n, q=q, i=i, Upsilon=Upsilon)



    def enclosed_mass(self, r, cumulative=False):
        """
        Enclosed 3D mass within a sphere of radius r,
        assuming a constant M/L ratio Upsilon.

        Parameters
        ----------
            r: float or array_like
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
            if len(r) > 0:
                menc = np.zeros(len(r))
                for j in range(len(r)):
                    if cumulative:
                        # Only calculate annulus, use previous Menc as shortcut:
                        if j > 0:
                            menc_ann =  util_calcs.total_mass3D_integral(r[j], Reff=self.Reff,
                                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                                            rinner=r[j-1], Upsilon=self.Upsilon)
                            menc[j] = menc[j-1] + menc_ann
                        else:
                            menc[j] = util_calcs.total_mass3D_integral(r[j], Reff=self.Reff,
                                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                                            rinner=0, Upsilon=self.Upsilon)
                    else:
                        # Direct calculation for every radius
                        menc[j] = util_calcs.total_mass3D_integral(r[j], Reff=self.Reff,
                                        n=self.n, q=self.q, Ie=Ie, i=self.i,
                                        rinner=0., Upsilon=self.Upsilon)

                # Perform some garbage collection, in the case of very small q
                #     and smaller n. Sometimes integrated to greater than 1,
                #     and then fell off to small numbers.....
                if (self.q <= 0.01) & (self.n <= 1.0):
                    # Use a tolerance of 1.e-9:
                    whgtone = np.where((menc-1.) > 1.e-9)[0]
                    if whgtone[0] < (len(menc) - 1):
                        menc[whgtone[0]:] = 1.

            else:
                menc = util_calcs.total_mass3D_integral(r[0], Reff=self.Reff,
                                n=self.n, q=self.q, Ie=Ie, i=self.i,
                                rinner=0., Upsilon=self.Upsilon)
        except:
            menc = util_calcs.total_mass3D_integral(r, Reff=self.Reff,
                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                            rinner=0., Upsilon=self.Upsilon)

        # Return enclosed mass: fractional menc * total mass
        return menc*self.total_mass

    def v_circ(self, r):
        """
        Circular velocity in the midplane of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r: float or array_like
                Midplane radius at which to evaluate the circular velocity [kpc]

        Returns
        -------
            vcirc: float or array_like
                Circular velocity in the midplane at r [km/s]

        """
        cnst = 4*np.pi*G.cgs.value*Msun.cgs.value*self.q/(1000.*pc.cgs.value)

        try:
            if len(r) > 0:
                vcsq = np.zeros(len(r))
                for j in range(len(r)):
                    vcsq[j] = cnst*util_calcs.vel_integral(r[j], Reff=self.Reff, n=self.n, q=self.q,
                                                           Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
                    if r[j] == 0:
                        vcsq[j] = 0.
            else:
                vcsq = cnst*util_calcs.vel_integral(r[0], Reff=self.Reff, n=self.n, q=self.q,
                                                    Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
                if r == 0:
                    vcsq = 0.
        except:
            vcsq = cnst*util_calcs.vel_integral(r, Reff=self.Reff, n=self.n, q=self.q,
                                                Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            if r == 0:
                vcsq = 0.

        return np.sqrt(vcsq)/1.e5


    def density(self, r):
        """
        Density profile at :math:`m=r` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r: float or array_like
                Distance at which to evaluate the circular velocity [kpc]

        Returns
        -------
            rho_arr: float or array_like
                Density profile at m [Msun / kpc^3]

        """
        try:
            if len(r) > 0:
                rho_arr = np.zeros(len(r))
                for j in range(len(r)):
                    rho_arr[j] = util_calcs.rho_m(r[j], Reff=self.Reff, n=self.n, q=self.q,
                                                  Ie=self.Ie, i=self.i, Upsilon=self.Upsilon,
                                                  replace_asymptote=True)
            else:
                rho_arr = util_calcs.rho_m(r[0], Reff=self.Reff, n=self.n, q=self.q,
                                           Ie=self.Ie, i=self.i, Upsilon=self.Upsilon,
                                           replace_asymptote=True)
        except:
            rho_arr = util_calcs.rho_m(r, Reff=self.Reff, n=self.n, q=self.q,
                                       Ie=self.Ie, i=self.i, Upsilon=self.Upsilon,
                                       replace_asymptote=True)

        return rho_arr


    def drho_dr(self, r):
        """
        Derivative of the density profile, :math:`d\\rho/dr`,
        at distance :math:`m=r` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            drho_dr_arr: float or array_like
                Derivative of density profile at r=m

        """
        try:
            if len(r) > 0:
                drho_dr_arr = np.zeros(len(r))
                for j in range(len(r)):
                    drho_dr_arr[j] = util_calcs.drhom_dm_multimethod(r[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                drho_dr_arr = util_calcs.drhom_dm_multimethod(r[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            drho_dr_arr = util_calcs.drhom_dm_multimethod(r, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return drho_dr_arr


    def dlnrho_dlnr(self, r):
        """
        Slope of the log density profile, :math:`d\\ln\\rho/d\\ln{}r`,
        in the midplane at radius :math:`m=r` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            dlnrho_dlnr_arr: float or array_like
                Derivative of log density profile at r=m

        """
        try:
            if len(r) > 0:
                dlnrho_dlnr_arr = np.zeros(len(r))
                for j in range(len(r)):
                    dlnrho_dlnr_arr[j] = util_calcs.dlnrhom_dlnm_multimethod(r[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                dlnrho_dlnr_arr = util_calcs.dlnrhom_dlnm_multimethod(r[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            dlnrho_dlnr_arr = util_calcs.dlnrhom_dlnm_multimethod(r, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return dlnrho_dlnr_arr

    ############################################################################################
    ############################################################################################
    ############################################################################################
    # vvvvvvvv REMOVE THESE LATER vvvvvvvv

    def _drho_dr_leibniz(self, r):
        """
        Derivative of the density profile, :math:`d\\rho/dr`,
        at distance :math:`m=r` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            drho_dr_arr: float or array_like
                Derivative of density profile at r=m

        """
        try:
            if len(r) > 0:
                drho_dr_arr = np.zeros(len(r))
                for j in range(len(r)):
                    drho_dr_arr[j] = util_calcs.drhom_dm_leibniz(r[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                drho_dr_arr = util_calcs.drhom_dm_leibniz(r[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            drho_dr_arr = util_calcs.drhom_dm_leibniz(r, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return drho_dr_arr


    def _dlnrho_dlnr_leibniz(self, r):
        """
        Slope of the log density profile, :math:`d\\ln\\rho/d\\ln{}r`,
        in the midplane at radius :math:`m=r` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            dlnrho_dlnr_arr: float or array_like
                Derivative of log density profile at r=m

        """
        try:
            if len(r) > 0:
                dlnrho_dlnr_arr = np.zeros(len(r))
                for j in range(len(r)):
                    dlnrho_dlnr_arr[j] = util_calcs.dlnrhom_dlnm_leibniz(r[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                dlnrho_dlnr_arr = util_calcs.dlnrhom_dlnm_leibniz(r[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            dlnrho_dlnr_arr = util_calcs.dlnrhom_dlnm_leibniz(r, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return dlnrho_dlnr_arr



    ############################################################################################
    ############################################################################################
    ############################################################################################

    def _drho_dr_scipy(self, r):
        """
        Derivative of the density profile, :math:`d\\rho/dr`,
        at distance :math:`m=r` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            drho_dr_arr: float or array_like
                Derivative of density profile at r=m

        """
        try:
            if len(r) > 0:
                drho_dr_arr = np.zeros(len(r))
                for j in range(len(r)):
                    drho_dr_arr[j] = util_calcs.drhom_dm_scipy_derivative(r[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                drho_dr_arr = util_calcs.drhom_dm_scipy_derivative(r[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            drho_dr_arr = util_calcs.drhom_dm_scipy_derivative(r, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return drho_dr_arr


    def _dlnrho_dlnr_scipy(self, r):
        """
        Slope of the log density profile, :math:`d\\ln\\rho/d\\ln{}r`,
        in the midplane at radius :math:`m=r` of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r: float or array_like
                Midplane radius at which to evaluate the log density profile slope [kpc]

        Returns
        -------
            dlnrho_dlnr_arr: float or array_like
                Derivative of log density profile at r=m

        """
        try:
            if len(r) > 0:
                dlnrho_dlnr_arr = np.zeros(len(r))
                for j in range(len(r)):
                    dlnrho_dlnr_arr[j] = util_calcs.dlnrhom_dlnm_scipy_derivative(r[j], Reff=self.Reff, n=self.n,
                                    q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
            else:
                dlnrho_dlnr_arr = util_calcs.dlnrhom_dlnm_scipy_derivative(r[0], Reff=self.Reff, n=self.n,
                                q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)
        except:
            dlnrho_dlnr_arr = util_calcs.dlnrhom_dlnm_scipy_derivative(r, Reff=self.Reff, n=self.n,
                            q=self.q, Ie=self.Ie, i=self.i, Upsilon=self.Upsilon)

        return dlnrho_dlnr_arr


    # ^^^^^^^^ REMOVE THESE LATER ^^^^^^^^
    ############################################################################################
    ############################################################################################
    ############################################################################################


    def surface_density(self, r):
        """
        Surface density distribution for a Sersic profile, assuming a M/L ratio Upsilon.

        Parameters
        ----------
            r: float or array_like
                Major axis radius within which to determine surface density distribution [kpc]

        Returns
        -------
            sigma_arr: float or array_like
                Surface density of Sersic profile at r

        """
        return self.Upsilon * util_calcs.Ikap(r, Reff=self.Reff, n=self.n, Ie=self.Ie)

    def projected_enclosed_mass(self, r):
        """
        Projected 2D mass enclosed within an ellipse
        (or elliptical shell), assuming a constant M/L ratio Upsilon.

        Parameters
        ----------
            r: float or array_like
                Major axis radius within which to determine total enclosed 2D projected mass [kpc]

        Returns
        -------
            Menc2D_ellipse: float or array_like

        """
        return util_calcs.total_mass2D_direct(r, total_mass=self.total_mass,
                            Reff=self.Reff, n=self.n, q=self.q, i=self.i)



    def enclosed_mass_ellipsoid(self, r, cumulative=False):
        """
        Enclosed 3D mass within an ellpsoid of
        major axis radius r and intrinsic axis ratio q
        (e.g. the same as the Sersic profile isodensity contours),
        assuming a constant M/L ratio Upsilon.

        Parameters
        ----------
            r: float or array_like
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
            if len(r) > 0:
                menc = np.zeros(len(r))
                for j in range(len(r)):
                    if cumulative:
                        # Only calculate annulus, use previous Menc as shortcut:
                        if j > 0:
                            menc_ann =  util_calcs.total_mass3D_integral_ellipsoid(r[j], Reff=self.Reff,
                                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                                            rinner=r[j-1], Upsilon=self.Upsilon)
                            menc[j] = menc[j-1] + menc_ann
                        else:
                            menc[j] = util_calcs.total_mass3D_integral_ellipsoid(r[j], Reff=self.Reff,
                                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                                            rinner=0., Upsilon=self.Upsilon)
                    else:
                        # Direct calculation for every radius
                        menc[j] = util_calcs.total_mass3D_integral_ellipsoid(r[j], Reff=self.Reff,
                                        n=self.n, q=self.q, Ie=Ie, i=self.i,
                                        rinner=0., Upsilon=self.Upsilon)
            else:
                menc = util_calcs.total_mass3D_integral_ellipsoid(r[0], Reff=self.Reff,
                                n=self.n, q=self.q, Ie=Ie, i=self.i,
                                rinner=0., Upsilon=self.Upsilon)
        except:
            menc = util_calcs.total_mass3D_integral_ellipsoid(r, Reff=self.Reff,
                            n=self.n, q=self.q, Ie=Ie, i=self.i,
                            rinner=0., Upsilon=self.Upsilon)

        # Return enclosed mass: fractional menc * total mass
        return menc*self.total_mass


    def virial_coeff_tot(self, r, vc=None):
        """
        The "total" virial coefficient ktot, which satisfies

        .. math::

            M_{\mathrm{tot}} = k_{\mathrm{tot}}(r) \\frac{v_{\mathrm{circ}}(r)^2 r}{ G },

        to convert between the circular velocity at any given radius and the total system mass.

        Parameters
        ----------
            r: float or array_like
                Major axis radius within which to determine total enclosed 2D projected mass [kpc]

            vc: float or array_like, optional
                Pre-calculated evaluation of vcirc(r)
                (saves time to avoid recalculating vcirc(r))  [km/s]

        Returns
        -------
            ktot: float or array_like
                ktot = Mtot * G / (vcirc(r)^2 * r)

        """
        if vc is None:
            vc = self.v_circ(r)

        return util_calcs.virial_coeff_tot(r, total_mass=self.total_mass, vc=vc)



    def virial_coeff_3D(self, r, m3D=None, vc=None):
        """
        The "3D" virial coefficient k3D, which satisfies

        .. math::

            M_{\mathrm{3D,sphere}}(r) = k_{\mathrm{3D}}(r) \\frac{v_{\mathrm{circ}}(r)^2 r}{ G },

        to convert between the circular velocity at any given radius
        and the mass enclosed within a sphere of radius r.

        Parameters
        ----------
            r: float or array_like
                Major axis radius within which to determine total enclosed 2D projected mass [kpc]

            m3D: float or array_like, optional
                Pre-calculated evaluation of Menc3D_sphere(r)
                (saves time to avoid recalculating Menc3D_sphere(r)) [Msun]
            vc: float or array_like, optional
                Pre-calculated evaluation of vcirc(r)
                (saves time to avoid recalculating vcirc(r))  [km/s]

        Returns
        -------
            k3D: float or array_like
                k3D = Menc3D_sphere(r) * G / (vcirc(r)^2 * r)

        """
        if m3D is None:
            m3D = self.enclosed_mass(r)

        if vc is None:
            vc = self.v_circ(r)

        return util_calcs.virial_coeff_3D(r, m3D=m3D, vc=vc)


    ############################################################################################
    ############################################################################################
    ############################################################################################
    # vvvvvvvv REMOVE THESE LATER ????? vvvvvvvv


    def _force_r(self, r, z):
        """
        Radial force of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r, z: float, float
                Midplane radius and vertical height at which to evaluate the radial force [kpc]

        Returns
        -------
            force_r: float
                Radial force at (r,z) [km^2/s^2/kpc]

        """
        return util_calcs.force_r(r, z, Reff=self.Reff, n=self.n, q=self.q, Ie=self.Ie,
                                  i=self.i, Upsilon=self.Upsilon)

    def _force_z(self, r, z):
        """
        Vertical force of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r, z: float, float
                Midplane radius and vertical height at which to evaluate the vertical force [kpc]

        Returns
        -------
            force_z: float
                Vertical force at (r,z) [km^2/s^2/kpc]

        """
        return util_calcs.force_z(r, z, Reff=self.Reff, n=self.n, q=self.q, Ie=self.Ie,
                                  i=self.i, Upsilon=self.Upsilon)



    def _sigma_z(self, r, z, sersic_table=None):
        """
        Vertical velocity dispersion of the deprojected Sersic mass distribution.

        Parameters
        ----------
            r, z: float, float
                Midplane radius and vertical height at which to evaluate the dispersion [kpc]

            sersic_table: dictionary, optional
                Use pre-computed table to create an interpolation function
                that is used for this calculation.

        Returns
        -------
            sigz: float
                Vertical velocity dispersion at (r,z) [km/s]

        """
        return np.sqrt(util_calcs.sigmaz_sq(r, z, total_mass=self.total_mass, Reff=self.Reff,
                                            n=self.n, q=self.q, Ie=self.Ie,
                                            i=self.i, Upsilon=self.Upsilon,
                                            sersic_table=sersic_table))




    # ^^^^^^^^ REMOVE THESE LATER ????? ^^^^^^^^
    ############################################################################################
    ############################################################################################
    ############################################################################################



    def profile_table(self, r, cumulative=None, add_reff_table_values=True):
        """
        Create a set of profiles as a dictionary, calculated over the specified radii.
        Also includes constants for the specified Sersic profile parameters,
        and information about values at Reff or the 3D half mass radius.

        Parameters
        ----------
            r: float or array_like
                Radius array, in kpc

            cumulative: bool, optional
                Shortcut option to only calculate the next annulus,
                then add to the previous Menc(r-rdelt).
                Default: Uses cumulative if n >= 2.
            add_reff_table_values: bool, optional
                Add select values at Reff to the table. Requires Reff to be in `r`
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
        vcirc =         self.v_circ(r)
        menc3D_sph =    self.enclosed_mass(r, cumulative=cumulative)
        menc3D_ellip =  self.enclosed_mass_ellipsoid(r, cumulative=cumulative)
        rho =           self.density(r)
        dlnrho_dlnr =   self.dlnrho_dlnr(r)

        # ---------------------
        # Setup table:
        table    = { 'r':                   r,
                     'vcirc':               vcirc,
                     'menc3D_sph':          menc3D_sph,
                     'menc3D_ellipsoid':    menc3D_ellip,
                     'rho':                 rho,
                     'dlnrho_dlnr':         dlnrho_dlnr,
                     'total_mass':          self.total_mass,
                     'Reff':                self.Reff,
                     'invq':                self.invq,
                     'q':                   self.q,
                     'n':                   self.n }

        # ---------------------
        if add_reff_table_values:
            # Calculate selected values at Reff:
            try:
                wh_reff = np.where(table['r'] == table['Reff'])[0][0]
            except:
                raise ValueError("Must include 'Reff' in 'r' if using 'add_reff_table_values=True'!")

            table['menc3D_sph_Reff'] =          table['menc3D_sph'][wh_reff]
            table['menc3D_ellipsoid_Reff'] =    table['menc3D_ellipsoid'][wh_reff]
            table['vcirc_Reff'] =               table['vcirc'][wh_reff]

            table['ktot_Reff'] = util_calcs.virial_coeff_tot(table['Reff'],
                            total_mass=table['total_mass'], vc=table['vcirc_Reff'])

            table['k3D_sph_Reff'] = util_calcs.virial_coeff_3D(table['Reff'],
                            m3D=table['menc3D_sph_Reff'], vc=table['vcirc_Reff'])

        # ---------------------
        # 3D Spherical half total_mass radius, for reference:
        table['rhalf3D_sph'] = util_calcs.find_rhalf3D_sphere(r=table['r'], menc3D_sph=table['menc3D_sph'],
                                           total_mass=table['total_mass'])

        # ---------------------
        # Check that table calculated correctly:
        status = util_calcs.check_for_inf(table=table)
        if status > 0:
            raise ValueError("Problem in table calculation: n={:0.1f}, invq={:0.2f}: status={}".format(self.n,
                            self.invq, status))


        return table
