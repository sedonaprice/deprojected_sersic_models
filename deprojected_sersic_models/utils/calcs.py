##################################################################################
# deprojected_sersic_models/utils/calcs.py                                       #
#                                                                                #
# Copyright 2018-2022 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import os
import copy

# Supress warnings: Runtime & integration warnings are frequent
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.integrate as scp_integrate
import scipy.misc as scp_misc
import scipy.special as scp_spec
import scipy.interpolate as scp_interp
import astropy.units as u
import astropy.constants as apy_con

import logging

from deprojected_sersic_models.utils.interp_profiles import interpolate_sersic_profile_logrho_function
from deprojected_sersic_models.utils.interp_profiles import interpolate_sersic_profile_dlnrho_dlnR_function
from deprojected_sersic_models.utils.interp_profiles import InterpFunc


__all__ = [ 'vcirc_spherical_symmetry', 'menc_spherical_symmetry',
            'virial_coeff_tot', 'virial_coeff_3D',
            'find_rhalf3D_sphere',
            'check_for_inf' ]

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

def check_for_inf(table=None):
    r"""
    Check core table quantities for non-finite entries, to determine if numerical errors occured.
    """
    status = 0

    keys = ['vcirc', 'menc3D_sph', 'menc3D_ellipsoid', 'rho', 'dlnrho_dlnR']

    for i, R in enumerate(table['R']):
        for key in keys:
            if not np.isfinite(table[key][i]):
                # Check special case: dlnrho_dlnR -- Leibniz uses r/rho*drho/dr,
                #                     so ignore NaN if rho=0.
                if (key == 'dlnrho_dlnR'):
                    if (table['rho'][i] == 0.):
                        pass
                    else:
                        status += 1
                elif (key == 'rho'):
                    if (R == 0.):
                        pass
                    else:
                        status += 1
                else:
                    status += 1

    return status


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Calculation helper functions: General for mass distributions

def vcirc_spherical_symmetry(R=None, menc=None):
    r"""
    Determine vcirc for a spherically symmetric mass distribution:

    .. math::

        v_{\mathrm{circ}}(R) = \sqrt{\\frac{G M_{\mathrm{enc}}(R)}{R}}

    Parameters
    ----------
        R: float or array_like
            Radi[us/i] at which to calculate the circular velocity [kpc]
        menc: float or array_like
            Enclosed mass at the given radii  [Msun]

    Returns
    -------
        vcirc: float or array_like
            Circular velocity as a function of radius  [km/s]

    """
    vcirc = np.sqrt(G.cgs.value * menc * Msun.cgs.value / (R * 1000. * pc.cgs.value)) / 1.e5

    # -------------------------
    # Test for 0:
    try:
        if len(R) >= 1:
            vcirc[np.array(R) == 0.] = 0.
    except:
        if R == 0.:
            vcirc = 0.
    # -------------------------

    return vcirc

def menc_spherical_symmetry(R=None, vcirc=None):
    r"""
    Determine Menc for a spherically symmetric mass distribution, given vcirc:
        Menc(R) = vcirc(R)^2 * R / G

    .. math::

        M_{\mathrm{enc}}(R) = \\frac{v_{\mathrm{circ}}(R)^2 R}{G}

    Parameters
    ----------
        R: float or array_like
            Radi[us/i] at which to calculate the enclosed mass [kpc]
        vcirc: float or array_like
            Circular velocity at the given radii  [km/s]

    Returns
    -------
        menc: float or array_like
            Enclosed mass as a function of radius  [Msun]

    """
    menc = ((vcirc*1.e5)**2.*(R*1000.*pc.cgs.value) / (G.cgs.value * Msun.cgs.value))
    return menc



# +++++++++++++++++++++++++++++++++++++++++++++++++

def virial_coeff_tot(R, total_mass=1., vc=None):
    r"""
    Evalutation of the "total" virial coefficient ktot, which satisfies

    .. math::

        M_{\mathrm{tot}} = k_{\mathrm{tot}}(R) \\frac{v_{\mathrm{circ}}(R)^2 R}{ G },

    to convert between the circular velocity at any given radius and the total system mass.

    Parameters
    ----------
        R: float or array_like
            Major axis radius at which to evaluate virial coefficient [kpc]
        total_mass: float
            Total mass of the component [Msun]
        vc: float or array_like
            Pre-calculated evaluation of vcirc(R)
            (saves time to avoid recalculating vcirc(R))  [km/s]

    Returns
    -------
        ktot: float or array_like
            ktot = Mtot * G / (vcirc(R)^2 * R)

    """

    # need to convert to cgs:
    # units: Mass: msun
    #        r:    kpc
    #        v:    km/s
    ktot = (total_mass * Msun.cgs.value) * G.cgs.value / (( R*1.e3*pc.cgs.value ) * (vc*1.e5)**2)

    return ktot


def virial_coeff_3D(R, m3D=None, vc=None):
    r"""
    Evalutation of the "total" virial coefficient ktot, which satisfies

    .. math::

        M_{\mathrm{3D,sphere}} = k_{\mathrm{3D}}(R) \\frac{v_{\mathrm{circ}}(R)^2 R}{ G },

    to convert between the circular velocity at any given radius
    and the mass enclosed within a sphere of radius r=R.

    Parameters
    ----------
        R: float or array_like
            Major axis radius at which to evaluate virial coefficient [kpc]
        m3D: float or array_like
            Pre-calculated evaluation of Menc3D_sphere(R)
            (saves time to avoid recalculating Menc3D_sphere(R)) [Msun]
        vc: float or array_like
            Pre-calculated evaluation of vcirc(R)
            (saves time to avoid recalculating vcirc(R))  [km/s]

    Returns
    -------
        k3D: float or array_like
            k3D = Menc3D_sphere(R) * G / (vcirc(R)^2 * R)

    """
    k3D = (m3D * Msun.cgs.value) * G.cgs.value / (( R*1.e3*pc.cgs.value ) * (vc*1.e5)**2)

    return k3D


# +++++++++++++++++++++++++++++++++++++++++++++++++

def find_rhalf3D_sphere(R=None, menc3D_sph=None, total_mass=None):
    r"""
    Evalutation of the radius corresponding to the sphere that
    encloses half of the total mass for a Sersic profile of a given
    intrinsic axis ratio, effective radius, and Sersic index.

    This is a utility function, where the Menc3D_sphere must have been pre-calculated.

    Performs an interpolation to find the appropriate rhalf_sph,
    given arrays R and menc3D_sph.

    Parameters
    ----------
        R: array_like
            Radii at which menc3D_sph is evaluated [kpc]
        menc3D_sph: array_like
            Mass enclosed within a sphere (evaluated over the radii in r) [Msun]
        total_mass: float
            Total mass of the component [Msun]

    Returns
    -------
        rhalf_sph: float
            3D radius enclosing half the total Sersic profile mass. [kpc]

    """

    R_interp = scp_interp.interp1d(menc3D_sph, R, fill_value=np.NaN, bounds_error=False, kind='slinear')
    rhalf_sph = R_interp( 0.5 * total_mass )
    # Coerce it into returning a constant:
    return np.float64(rhalf_sph)




# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Calculation helper functions: Seric profiles, mass distributions

def bn_func(n):
    r"""
    Function to calculate bn(n) for a Sersic profile

    Parameters
    ----------
        n: float
            Sersic index

    Returns
    -------
        bn: float
            bn, satisfying gamma(2*n, bn) = 0.5 * Gamma(2*n)

    Notes
    -----
    The constant :math:`b_n` satisfies :math:`\Gamma(2n) = 2\gamma (2n, b_n)`
    """
    # bn(n) satisfies:
    #    gamma(2*n, bn) = 0.5 * Gamma(2*n)
    # Scipy:
    # gammaincinv(a, y) = x such that
    #   y = P(a, x), where
    #     P(a, x) = 1/Gammma(a) * gamma(a, x)
    #  So for a = 2*n,  y = 0.5, we have
    #  0.5 = 1/Gamma(2*n) * gamma(2*n, bn) = P(2*n, bn)
    #   so bn = gammaincinv(2*n, 0.5)
    return scp_spec.gammaincinv(2. * n, 0.5)

def qobs_func(q=None, i=None):
    r"""
    Function to calculate the observed axis ratio for an inclined system.

    Parameters
    ----------
        q: float
            Intrinsic axis ratio of Sersic profile
        i: float
            Inclination of system [deg]

    Returns
    -------
        qobs: float
            qobs = sqrt(q^2 + (1-q^2)*cos(i))

    """
    return np.sqrt(q**2 + (1-q**2)*np.cos(i*deg2rad)**2)

def get_Ie(total_mass=1., Reff=1., n=1., q=0.4, i=90., Upsilon=1.):
    r"""
    Evalutation of Ie, normalization of the Sersic intensity profile at kap = Reff,
    using the total mass (to infinity) and assuming a constant M/L ratio Upsilon.

    Uses the closed-form solution for the total luminosity of the
    2D projected Sersic intensity profile I(kap).

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

        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        Ie: float
            Ie = I(kap=Reff)

    """
    bn = bn_func(n)
    qobs = qobs_func(q=q, i=i)

    # This is Ie, using Upsilon = 1 [cnst M/L].
    Ie = (total_mass * np.power(bn, 2.*n)) / \
            ( Upsilon * 2.*np.pi* qobs * Reff**2 * n * np.exp(bn) * scp_spec.gamma(2.*n) )

    return Ie


def Ikap(kap, Reff=1., n=1., Ie=1.):
    r"""
    Intensity(kappa) for a Sersic profile

    Parameters
    ----------
        kap: float or array_like
            Radius for calculation of Sersic profile (along the major axis)
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff

    Returns
    -------
        I: float or array_like
            Intensity of Sersic profile at kap

    """
    # Using definition of Sersic projected 2D intensity
    #   with respect to normalization at Reff, using Ie = I(Reff)

    bn = bn_func(n)
    I = Ie * np.exp( -bn * (np.power(kap/Reff, 1./n) - 1.) )

    return I

def dIdkap(kap, Reff=1., n=1., Ie=1.):
    r"""
    Derivative d(Intensity(kappa))/dkappa for a Sersic profile

    Parameters
    ----------
        kap: float or array_like
            radius for calculation of Sersic profile (along the major axis)
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff

    Returns
    -------
        dIdk: float or array_like
            Derivative of intensity of Sersic profile at kap

    """
    bn = bn_func(n)
    dIdk = - (Ie*bn)/(n*Reff) * np.exp( -bn * (np.power(kap/Reff, 1./n) - 1.) ) * np.power(kap/Reff, (1./n) - 1.)

    return dIdk

def rho_m_integrand(kap, m, Reff, n, Ie):
    r"""
    Integrand dI/dkap * 1/sqrt(kap^2 - m^2) as part of
    numerical integration to find rho(m)

    Parameters
    ----------
        kap: float or array_like
            independent variable (radius)
        m: float
            Radius at which to evaluate rho(m)
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff


    Returns
    -------
        integrand: float or array_like
            Integrand dI/dkap * 1/sqrt(kap^2 - m^2)

    """
    integ = dIdkap(kap, n=n, Ie=Ie, Reff=Reff) * 1./np.sqrt(kap**2 - m**2)
    return integ

def rho_m(m, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1., replace_asymptote=False):
    r"""
    Evalutation of deprojected Sersic density profile at distance m.

    Parameters
    ----------
        m: float
            Distance at which to evaluate rho(m)
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        rhom: float or array_like
            rho(m)  -- 3D density of Sersic profile at radius m.

    """

    # Handle divergent asymptotic m=0 behavior for n>1:
    if ((replace_asymptote) & (m==0.) & (n>1.)):
        return np.inf
    else:
        qobstoqint = np.sqrt(np.sin(i*deg2rad)**2 + 1./q**2 * np.cos(i*deg2rad)**2 )

        # Evalutate inner integral:
        #   Int_(kap=m)^(infinity) dkap [dI/dkap * 1/sqrt(kap^2 - m^2)]
        int_rho_m_inner, _ = scp_integrate.quad(rho_m_integrand, m, np.inf, args=(m, Reff, n, Ie))

        rhom = -(Upsilon/np.pi)*( qobstoqint ) * int_rho_m_inner

        return rhom




def vel_integrand(m, R, Reff, n, q, Ie, i, Upsilon):
    r"""
    Integrand rho(m) * m^2 / sqrt(R^2 - (1-qint^2) * m^2) as part of numerical integration to find vcirc(R)

    Parameters
    ----------
        m: float
            independent variable (radial)
        R: float
            Radius at which to find vcirc(R)
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]
        Upsilon: float
            Mass-to-light ratio

    Returns
    -------
        integrand: float or array_like
            rho(m) * m^2 / sqrt(R^2 - (1-qint^2) * m^2)

    """
    integ = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon) * \
                ( m**2 / np.sqrt(R**2 - m**2 * (1.- q**2)) )
    return integ

def vel_integral(R, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.):
    r"""
    Evalutation of integrand rho(m) * m^2 / sqrt(R^2 - (1-qint^2) * m^2) from m=0 to R,
    as part of numerical integration to find vcirc(R)

    Parameters
    ----------
        R: float or array_like
            Radius at which to find vcirc(R)
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Upsilon: float, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        integral: float or array_like
            Int_(m=0)^(R) dm [rho(m) * m^2 / sqrt(R^2 - (1-qint^2) * m^2)]

    """
    # integrate outer from m=0 to r
    intgrl, _ = scp_integrate.quad(vel_integrand, 0, R, args=(R, Reff, n, q, Ie, i, Upsilon))
    return intgrl

def total_mass3D_integrand_ellipsoid(m, Reff, n, q, Ie, i, Upsilon):
    r"""
    Integrand m^2 * rho(m)  as part of numerical integration to find M_3D,ellipsoid(<r=R)

    Parameters
    ----------
        m: float or array_like
            independent variable (radial)
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]
        Upsilon: float
            Mass-to-light ratio

    Returns
    -------
        integrand: float or array_like
            m^2 * rho(m)

    """

    integ =  m**2 * rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
    return integ


def total_mass3D_integral_ellipsoid(R, Reff=1., n=1., q=0.4, Ie=1.,i=90.,
                                    Rinner=0., Upsilon=1.):
    r"""
    Evalutation of integrand m^2 * rho(m) from m=0 to R,
    as part of numerical integration to find M_3D_ellipsoid(<r=R)

    Parameters
    ----------
        R: float
            Radius [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Rinner: float, optional
            Calculate radius in annulus instead of sphere, using Rinner>0. [kpc]. Default: 0.
        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        integral: float or array_like
            Int_(m=0)^(R) dm [m^2 * rho(m)]

    """
    ## In ellipsoids:
    if R == 0.:
        return 0.
    else:
        # integrate from m=0 to R
        intgrl, _ = scp_integrate.quad(total_mass3D_integrand_ellipsoid, Rinner, R,
                                       args=(Reff, n, q, Ie, i, Upsilon))
        return 4.*np.pi*q*intgrl


def total_mass3D_integrand_sph_z(z, m, Reff, n, q, Ie, i, Upsilon):
    r"""
    Integrand rho(sqrt(m^2 + (z/qintr)^2) [the arc of a circle with
    cylindrical coords (m, z)] as part of numerical integration
    to find mass enclosed in sphere.

    Parameters
    ----------
        z: float or array_like
            independent variable (height; cylindrical coords)
        m:  float or array_like
            cylindrical coordinates radius m
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]
        Upsilon: float, optional
            Mass-to-light ratio.

    Returns
    -------
        integrand: float or array_like
            rho(sqrt(m^2 + (z/qintr)^2)

    """

    mm = np.sqrt(m**2 + z**2/q**2)
    integ =  rho_m(mm, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
    return integ

def total_mass3D_integral_z(m, R=None, Reff=1., n=1., q=0.4, Ie=1., i=90.,
                            Rinner=None, Upsilon=1.):
    r"""
    Evalutation of integrand 2 * rho(sqrt(m^2 + (z/qintr)^2) from z=0 to sqrt(R^2-m^2), [eg both pos and neg z]
    as part of numerical integration to find mass enclosed in sphere
    (or over the shell corresponding to Rinner...)

    Parameters
    ----------
        m: float or array_like
            Radius at which to evaluate integrand; cylindrical coordinate radius
        R: float or array_like
            Radius of sphere over which to be calculating total enclosed mass [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Rinner: float, optional
            Inner radius of total spherical shell, if only calculating mass
            in a spherical shell. Default: Rinner = 0. (eg the entire sphere out to r)
        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        integral: float or array_like
            Int_(z=0)^(sqrt(R^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)]

    """
    lim = np.sqrt(R**2 - m**2)
    if Rinner > 0.:
        if m < Rinner:
            lim_inner = np.sqrt(Rinner**2 - m**2)
        else:
            # this is the part of the vertical slice where m is greater than Rinner, outside the inner shell part,
            #       so it goes from z=0 to z=sqrt(R^2-m^2).
            lim_inner = 0.
    else:
        lim_inner = 0.
    # symmetric about z:
    intgrl, abserr = scp_integrate.quad(total_mass3D_integrand_sph_z, lim_inner, lim,
                                        args=(m, Reff, n, q, Ie, i, Upsilon))

    # ---------------------
    # Catch some numerical integration errors which happen at very small values of m --
    #       set these to 0., as this is roughly correct
    if intgrl < 0.:
        if np.abs(m) > 1.e-6:
            print('m={}, r={}, zlim={}'.format(m, R, lim))
            raise ValueError
        else:
            # Numerical error:
            intgrl = 0.

    # ---------------------
    # Integral is symmetric about z, so return times 2 for pos and neg z.
    return 2.*intgrl

def total_mass3D_integrand_r(m, R, Reff, n, q, Ie, i, Rinner, Upsilon):
    r"""
    Integrand m * [ Int_(z=0)^(sqrt(R^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]
    as part of numerical integration to find mass enclosed in sphere.

    Parameters
    ----------
        m: float or array_like
            cylindrical coordinates radius m
        R: float or array_like
            Radius of sphere over which to be calculating total enclosed mass [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]
        Rinner: float
            Inner radius of total spherical shell, if only calculating mass in a spherical shell [kpc]
        Upsilon: float
            Mass-to-light ratio.

    Returns
    -------
        integrand: float or array_like
            m * [ Int_(z=0)^(sqrt(R^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]

    """

    integ = total_mass3D_integral_z(m, R=R, Reff=Reff, n=n, q=q, Ie=Ie, i=i,
                                    Rinner=Rinner, Upsilon=Upsilon)
    return m * integ


def total_mass3D_integral(R, Reff=1., n=1., q=0.4, Ie=1., i=90., Rinner=0., Upsilon=1.):
    r"""
    Evalutation of integrand 2 * pi * m * [ Int_(z=0)^(sqrt(R^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]
    from m=Rinner to R, as part of numerical integration to find mass enclosed in sphere
    (or over the shell corresponding to Rinner...)

    Parameters
    ----------
        R: float
            Radius of sphere over which to be calculating total enclosed mass [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Rinner: float, optional
            Inner radius of total spherical shell, if only calculating mass
            in a spherical shell. Default: Rinner = 0. (eg the entire sphere out to R)
        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        integral: float or array_like
            Int_(m=0)^(R) dm * 2 * pi * m * [ Int_(z=0)^(sqrt(R^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]

    """
    # in *SPHERE*
    if R == 0.:
        return 0.
    else:
        intgrl, abserr = scp_integrate.quad(total_mass3D_integrand_r, 0., R,
                                            args=(R, Reff, n, q, Ie, i, Rinner, Upsilon))
        return 2*np.pi*intgrl


def total_mass2D_direct(R, total_mass=1., Reff=1., n=1., q=0.4, i=90., Rinner=0.):
    r"""
    Evalutation of the 2D projected mass enclosed within an ellipse,
    assuming a constant M/L ratio Upsilon.

    Parameters
    ----------
        R: float or array_like
            Major axis radius within which to determine total enclosed
            2D projected mass [kpc]
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

        Rinner: float, optional
            Inner radius of total spherical shell, if only calculating mass
            in a spherical shell. Default: Rinner = 0. (eg the entire sphere out to R)

    Returns
    -------
        Menc2D_ellipse: float or array_like

    """
    #### 2D projected mass within ellipses of axis ratio ####

    ## Within self-similar ellipses of ratio qobs

    # The projected 2D mass w/in ellipses of ratio qobs is just:
    #  M2D_ellip(<R) = Mtot * gamma(2n, x) / Gamma(2n),
    #   where x = bn * (R/Reff)^(1/n).
    # (note that qobs is folded in from the full defition of L(<R) together with the definition of Ie.)
    #
    # Scipy definition of gammainc:  gammainc(a, x) = 1/gamma(a) * int_0^x[t^(a-1)e^-t]dt
    #  so the scp_spec.gammainc(2*n,x) IS NORMALIZED relative to gamma(2n)
    #
    # So we have just M2D(<R) = Mtot * scp_spec.gammainc(2*n, x)

    bn = bn_func(n)
    integ = scp_spec.gammainc(2 * n, bn * np.power(R / Reff, 1./n) )
    if Rinner > 0.:
        integinner = scp_spec.gammainc(2 * n, bn * np.power(Rinner / Reff, 1./n) )
        integ -= integinner


    return total_mass*integ



def lnrho_m(lnm, Reff, n, q, Ie, i, Upsilon):
    r"""
    Log density profile, :math:`\ln\rho`
    at distance :math:`m` of a deprojected Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        lnm: float
            Ln distance [ln kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic profile
        i: float
            Inclination of system [deg]
        Upsilon: float
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        lnrhom: float
            Log density profile at m

    """
    rhom = rho_m(np.exp(lnm), Reff=Reff, n=n, q=q, Ie=Ie, i=i)
    return np.log(rhom)


def dlnrhom_dlnm_scipy_derivative(m, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1., dx=1.e-5, order=3):
    r"""
    Evalutation of the slope of the density profile, :math:`d\ln\rho/d\ln{}m`,
    at distance :math:`m` of a deprojected Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        m: float
            Distance at which to evaluate the profile [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic profile
        i: float
            Inclination of system [deg]

        Upsilon: float, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        dlnrho_dlnm: float
            Derivative of log density profile at m

    """
    deriv = scp_misc.derivative(lnrho_m, np.log(m), args=(Reff, n, q, Ie, i, Upsilon),
                                dx=dx, n=1, order=order)
    return deriv




def drhom_dm_scipy_derivative(m, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1., dx=1.e-5, order=3):
    r"""
    Numerical evalutation of derivative of the density profile, :math:`d\rho/dm`,
    at distance :math:`m` of a deprojected Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        m: float
            Distance at which to evaluate the profile [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic profile
        i: float
            Inclination of system [deg]

        Upsilon: float, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        dlnrho_dlnm: float
            Derivative of log density profile at m

    """
    deriv = scp_misc.derivative(rho_m, m, args=(Reff, n, q, Ie, i, Upsilon),
                                dx=dx, n=1, order=order)
    return deriv

def drhoI_du_integrand(x, u, n, bn):
    r"""
    Integrand for :math:`d\rho(m)/dm` -- as a function of x, u. Will integrate over x.
    """

    v = np.sqrt(x**2 + u**2)

    aa = np.exp(-bn*(np.power(v, 1./n)-1.))
    bb = np.power(v, (1./n - 4.))
    cc = (1./n - 2. - bn/n * np.power(v, 1./n))

    drhoIdu_intgrnd = aa * bb * cc

    return drhoIdu_intgrnd


def drhom_dm_leibniz(m, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.):
    r"""
    Evalutation of derivative :math:`d\rho(m)/dm` of deprojected Sersic density profile
    at radius :math:`m`.

    Parameters
    ----------
        m: float
            Radius at which to evaluate d(rho(m))/dm
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        drhomdm: float or array_like
            d(rho(m))/dm  -- Derivative of 3D density of Sersic profile at radius m.

    """
    qobstoqint = np.sqrt(np.sin(i*deg2rad)**2 + 1./q**2 * np.cos(i*deg2rad)**2 )

    bn = bn_func(n)

    # Integrate over Int (x=0)^(infinity) to get drhoI(u)/du at u=m/Reff
    drhoIdu_int_intgrl, _ = scp_integrate.quad(drhoI_du_integrand, 0, np.inf,
                                       args=(m/Reff, n, bn))

    drhoIdu_int = (m / Reff**2) * drhoIdu_int_intgrl

    drhomdm = (Upsilon/np.pi)*(Ie*bn/(n*Reff))*(qobstoqint) * drhoIdu_int

    return drhomdm


def dlnrhom_dlnm_leibniz(m, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1., dx=1.e-5, order=3):
    r"""
    Evalutation of the slope of the density profile, :math:`d\ln\rho/d\ln{}m`,
    at distance :math:`m` of a deprojected Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        m: float
            Distance at which to evaluate the profile [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic profile
        i: float
            Inclination of system [deg]

        Upsilon: float, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        dlnrho_dlnm: float
            Derivative of log density profile at m

    """
    rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
    drho_dm = drhom_dm_leibniz(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
    return (m/rho)* drho_dm


def _multimethod_classifier(m, Reff=1., n=1.):
    if ((m/Reff) < 1.e-4):
        method = 'scipy'
    elif (n <= 0.5) & ((m/Reff) > 4.):
        method = 'scipy'
    elif (n>0.5) & (n < 0.7) & ((m/Reff) > 5.):
        method = 'scipy'
    elif (n>=0.7) & (n<0.9) & ((m/Reff) > 6.):
        method='scipy'
    elif (n>=0.9) & ((m/Reff) > 8.):
        method = 'scipy'
    else:
        method = 'leibniz'

    return method


def drhom_dm_multimethod(m, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.):
    r"""
    Evalutation of derivative :math:`d\rho(m)/dm` of deprojected Sersic density profile
    at radius :math:`m`.

    Parameters
    ----------
        m: float
            Radius at which to evaluate d(rho(m))/dm
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        drhomdm: float or array_like
            d(rho(m))/dm  -- Derivative of 3D density of Sersic profile at radius m.

    """

    method = _multimethod_classifier(m, Reff=Reff, n=n)

    # Handle asymptotic m=0 behavior for n>1:
    if ((m==0.) & (n>1.)):
        return np.inf
    else:
        if (method == 'leibniz'):
            drhomdm = drhom_dm_leibniz(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
        elif (method == 'scipy'):
            drhomdm = drhom_dm_scipy_derivative(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)

        return drhomdm


def dlnrhom_dlnm_multimethod(m, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1., dx=1.e-5, order=3):
    r"""
    Evalutation of the slope of the density profile, :math:`d\ln\rho/d\ln{}m`,
    at distance :math:`m` of a deprojected Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        m: float
            Distance at which to evaluate the profile [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic profile
        i: float
            Inclination of system [deg]

        Upsilon: float, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)

    Returns
    -------
        dlnrho_dlnm: float
            Derivative of log density profile at m

    """

    method = _multimethod_classifier(m, Reff=Reff, n=n)

    # Handle asymptotic m=0 behavior for n>1:
    if ((m==0.) & (n>=1.)):
        return (1./n) - 1.
    # Seems to asymptote to 0 for <=1?
    else:
        if (method == 'leibniz'):
            dlnrho_dlnm = dlnrhom_dlnm_leibniz(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
        elif (method == 'scipy'):
            dlnrho_dlnm = dlnrhom_dlnm_scipy_derivative(m, Reff=Reff, n=n, q=q,
                                                        Ie=Ie, i=i, Upsilon=Upsilon)

        return dlnrho_dlnm


def BBBBBBBBBBBBBBBBBBBBBBBBBBB():

    return None

def force_R_integrand(tau, R, z, Reff, n, q, Ie, i, Upsilon, logrhom_interp, table, total_mass):
    """
    Integrand :math:`\frac{\rho(m)}{(\tau+1)^2 \sqrt{tau+q^2}}`
    as part of numerical integration to find :math:`g_r(R,z)`

    Parameters
    ----------
        tau: float
            independent variable
        R: float
            Midplane radius [kpc]
        z: float
            Height above midplane [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]
        Upsilon: float
            Mass-to-light ratio
        logrhom_interp: function, optional
            Shortcut to use an interpolation function (from a lookup table)
            instead of recalculating rho(m)
        table: Sersic profile table, optional
        total_mass: log total mass [Msun], optional

    Returns
    -------
        integrand: float

    """
    m = np.sqrt(R**2/(tau+1.) + z**2/(tau+q**2))
    if logrhom_interp is not None:
        table_Reff =    table['Reff']
        table_mass =    table['total_mass']

        scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**3
        #rho = rhom_interp(m / Reff * table_Reff) * scale_fac
        # logrho = logrhom_interp(m / Reff * table_Reff)

        logrho = logrhom_interp(m, Reff)

        rho = np.power(10., logrho) * scale_fac

        # Back replace inf, if interpolating at r=0 for n>1:
        if (table['n'] >= 1.) & (table['R'][0] == 0.):
            if (~np.isfinite(table['rho'][0]) & (m == 0.)):
                rho = table['rho'][0]
    else:
        rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)

    integrand = rho / ( (tau+1.)**2 * np.sqrt(tau + q**2) )

    return integrand




def force_z_integrand(tau, R, z, Reff, n, q, Ie, i, Upsilon, logrhom_interp, table, total_mass):
    """
    Integrand :math:`\frac{\rho(m)}{(\tau+1)(tau+q^2)^{3/2}}`
    as part of numerical integration to find :math:`g_z(R,z)`

    Parameters
    ----------
        tau: float
            independent variable
        R: float
            Midplane radius [kpc]
        z: float
            Height above midplane [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]
        Upsilon: float
            Mass-to-light ratio
        logrhom_interp: function, optional
            Shortcut to use an interpolation function (from a lookup table)
            instead of recalculating rho(m)
        table: Sersic profile table, optional
    Returns
    -------
        integrand: float

    """
    m = np.sqrt(R**2/(tau+1.) + z**2/(tau+q**2))
    if logrhom_interp is not None:
        table_Reff =    table['Reff']
        table_mass =    table['total_mass']

        scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**3
        # rho = rhom_interp(m / Reff * table_Reff) * scale_fac
        # logrho = logrhom_interp(m / Reff * table_Reff)

        logrho = logrhom_interp(m, Reff)

        rho = np.power(10., logrho) * scale_fac

        # Back replace inf, if interpolating at r=0 for n>1:
        if (table['n'] >= 1.) & (table['R'][0] == 0.):
            if (~np.isfinite(table['rho'][0]) & (m == 0.)):
                rho = table['rho'][0]

        # ## DEBUG:
        # rho_direct = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
        # if (np.abs(np.log10(rho)-np.log10(rho_direct))>1.):
        #     raise ValueError
    else:
        rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)

    integrand = rho / ( (tau+1.) * np.power((tau + q**2), 3./2.) )

    return integrand



def force_R(R, z, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.,
            logrhom_interp=None, table=None, total_mass=None):
    """
    Evalutation of gravitational force in the radial direction
    :math:`g_R=-\partial\Phi/\partial R`,
    of a deprojected Sersic density profile at (R,z), by numerically evalutating

    .. math::

        g_R(R,z) = - 2\pi GqR \int_0^{\infty} d\tau \frac{\rho(m)}{(\tau+1)^2 \sqrt{tau+q^2}},
        m = \frac{r^2}{\tau+1} + \frac{z^2}{\tau+q^2}

    Parameters
    ----------
        R: float
            Midplane radius [kpc]
        z: float
            Height above midplane [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)
        logrhom_interp: function, optional
            Shortcut to use an interpolation function (from a lookup table)
            instead of recalculating rho(m)
        table: Sersic profile table, optional

    Returns
    -------
        g_R: float
            g_R(R,z)  -- gravitational force in the radial direction; units [km^2/s^2/kpc]

    """

    cnst = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value) # cmtokpc * Msuntog * kmtocm^2
    # Check case at R=z=0: Must have force = 0 in this case
    if (R==0.) & (z==0.):
        int_force_R = 0.
    else:
        int_force_R, _ = scp_integrate.quad(force_R_integrand, 0, np.inf,
                                            args=(R, z, Reff, n, q, Ie, i, Upsilon,
                                             logrhom_interp, table, total_mass))
    return -2.*np.pi*G.cgs.value*q*cnst * R * int_force_R



def force_z(R, z, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.,
            logrhom_interp=None, table=None, total_mass=None):
    """
    Evalutation of gravitational force in the vertical direction
    :math:`g_z=-\partial\Phi/\partial z`,
    of a deprojected Sersic density profile at (R,z), by numerically evalutating

    .. math::

        g_z(R,z) = - 2\pi Gqz \int_0^{\infty} d\tau \frac{\rho(m)}{(\tau+1)(tau+q^2)^{3/2}},
        m = \frac{R^2}{\tau+1} + \frac{z^2}{\tau+q^2}

    Parameters
    ----------
        R: float
            Midplane radius [kpc]
        z: float
            Height above midplane [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)
        logrhom_interp: function, optional
            Shortcut to use an interpolation function (from a lookup table)
            instead of recalculating rho(m)
        table: Sersic profile table, optional

    Returns
    -------
        g_z: float
            g_z(r,z)  -- gravitational force in the vertial direction; units [km^2/s^2/kpc]

    """

    cnst = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value) # cmtokpc * Msuntog * kmtocm^2

    # Check case at R=z=0: Must have force = 0 in this case
    if ((R**2 + (z/q)**2)==0.):
        int_force_z = 0.
    else:
        int_force_z, _ = scp_integrate.quad(force_z_integrand, 0, np.inf,
                                            args=(R, z, Reff, n, q, Ie, i, Upsilon,
                                             logrhom_interp, table, total_mass))

    return -2.*np.pi*G.cgs.value*q*z* cnst * int_force_z


# def BBBBBBBBBBBBBBBBBBBBBBBBBBB():
#
#     return None
#
# def sigsq_z_integrand(z, R, total_mass, Reff, n, q, Ie, i, Upsilon,
#                       sersic_table, logrhom_interp,
#                       table_forcez, forcez_interp, halo):
#     """
#     Integrand as part of numerical integration to find :math:`\sigma^2_z(r,z)`
#
#     Parameters
#     ----------
#         z: float
#             Height above midplane [kpc]
#         R: float
#             Midplane radius [kpc]
#         total_mass: float
#             Total mass of the Sersic mass component [Msun]
#         Reff: float
#             Effective radius of Sersic profile [kpc]
#         n: float
#             Sersic index
#         q: float
#             Intrinsic axis ratio
#         Ie: float
#             Normalization of Sersic intensity profile at kap = Reff
#         i: float
#             Inclination of system [deg]
#         Upsilon: float
#             Mass-to-light ratio
#         logrhom_interp: dictionary or None
#             Use pre-computed table to create an interpolation function
#             that is used for this calculation.
#         halo: Halo instance or None
#             Optionally include force from a halo component.
#
#     Returns
#     -------
#         integrand: float
#
#     """
#     m = np.sqrt(R**2 + (z/q)**2)
#     if logrhom_interp is not None:
#         table_Reff =    sersic_table['Reff']
#         table_mass =    sersic_table['total_mass']
#
#         scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**3
#         # rho = rhom_interp(m / Reff * table_Reff) * scale_fac
#         logrho = logrhom_interp(m / Reff * table_Reff)
#         rho = np.power(10., logrho) * scale_fac
#     else:
#         rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
#
#     if forcez_interp is not None:
#         table_Reff =    table_forcez['Reff']
#         table_mass =    table_forcez['total_mass']
#
#         scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**2
#         gz = forcez_interp(z / Reff * table_Reff) * scale_fac
#     else:
#         gz = force_z(R, z,  Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon,
#                      logrhom_interp=logrhom_interp, table=sersic_table,
#                      total_mass=total_mass)
#
#     if halo is not None:
#         gz += halo.force_z(R, z)
#
#     integrand = rho * gz
#     return integrand
#
#
#
#
# def sigmaz_sq(R, z, total_mass=1., Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.,
#               sersic_table=None, table_forcez=None,
#               halo=None):
#     """
#     Evalutation of vertical velocity dispersion of a deprojected Sersic density profile at (R,z).
#
#     .. math::
#
#         \sigma^2_z(R,z)=\frac{1}{\rho}\int \rho g_z(R,z)dz
#
#     Parameters
#     ----------
#         R: float
#             Midplane radius [kpc]
#         z: float
#             Height above midplane [kpc]
#
#         total_mass: float
#             Total mass of the Sersic mass component [Msun]
#         Reff: float
#             Effective radius of Sersic profile [kpc]
#         n: float
#             Sersic index
#         q: float
#             Intrinsic axis ratio of Sersic profile
#         Ie: float
#             Normalization of Sersic intensity profile at kap = Reff
#         i: float
#             Inclination of system [deg]
#
#         Upsilon: float or array_like, optional
#             Mass-to-light ratio. Default: 1. (i.e., constant ratio)
#         sersic_table: dictionary, optional
#             Use pre-computed table to create an interpolation function
#             that is used for this calculation.
#         halo: Halo instance, optional
#             Optionally include force from a halo component.
#
#     Returns
#     -------
#         sigzsq: float
#             Vertical velocity dispersion direction; units [km^2/s^2]
#
#     """
#     m = np.sqrt(R**2 + (z/q)**2)
#     if sersic_table is not None:
#         # rho = interpolate_sersic_profile_rho(R=m, total_mass=total_mass, Reff=Reff, n=n, invq=1./q,
#         #                                      table=sersic_table)
#
#         logrhom_interp = interpolate_sersic_profile_logrho_function(n=n, invq=1./q,
#                                                               table=sersic_table)
#         table_Reff =    sersic_table['Reff']
#         table_mass =    sersic_table['total_mass']
#
#         scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**3
#         # rho = rhom_interp(m / Reff * table_Reff) * scale_fac
#         logrho = logrhom_interp(m / Reff * table_Reff)
#         rho = np.power(10., logrho) * scale_fac
#     else:
#         rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
#         logrhom_interp = None
#
#     if table_forcez is not None:
#         table_z =       table_forcez['z']
#         table_Reff =    table_forcez['Reff']
#         # table_mass =    table_forcez['total_mass']
#
#         # Will need to match exact R:
#         whm = np.where(table_forcez['R'] == (R / Reff * table_Reff))[0]
#         if len(whm) == 0:
#             raise ValueError("No matching 'R' in table")
#         table_gz = table_forcez['gz_R_{}'.format(whm[0])]
#
#         # scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**2
#         forcez_interp = scp_interp.interp1d(table_z, table_gz,
#                         fill_value='extrapolate',  kind='cubic')
#     else:
#         forcez_interp = None
#
#     # int_sigsq_z, _ = scp_integrate.quad(sigsq_z_integrand, 0, z,
#     #                                     args=(R, total_mass, Reff, n, q, Ie, i,
#     #                                           Upsilon, sersic_table, logrhom_interp,
#     #                                           table_forcez, forcez_interp))
#     # # WRONG: ///// No negatives, because delPotl = -gz, so already in there
#     # return - 1./rho * int_sigsq_z
#
#
#     int_sigsq_z, _ = scp_integrate.quad(sigsq_z_integrand, z, np.inf,
#                                         args=(R, total_mass, Reff, n, q, Ie, i,
#                                               Upsilon, sersic_table, logrhom_interp,
#                                               table_forcez, forcez_interp,
#                                               halo))
#
#     return -1./rho * int_sigsq_z
#
#

def BBBBBBBBBBBBBBBBBBBBBBBBBBB():

    return None


def _sprof_get_rhog(sprof, m):
    if sprof.isgas:
        if sprof.logrhom_interp is not None:
            table_Reff = sprof.table['Reff']
            table_mass = sprof.table['total_mass']

            scale_fac = (sprof.total_mass / table_mass) * (table_Reff / sprof.Reff)**3
            # logrho = logrhom_interp(m / sprof.Reff * table_Reff)

            logrho = sprof.logrhom_interp(m, sprof.Reff)

            rho_g = np.power(10., logrho) * scale_fac
        else:
            rho_g = sprof.density(m)
    else:
        rho_g = m * 0.

    return rho_g

def _preprocess_sprof_dPdz_calc(sprof, R, z):
    # Note R and z are both scalars

    m = np.sqrt(R**2 + (z/sprof.q)**2)

    rho_g = _sprof_get_rhog(sprof, m)

    if sprof.forcez_interp is not None:
        # gz = sprof.forcez_interp(z / sprof.Reff * sprof.table['Reff']) * sprof.forcez_scale_fac
        gz = sprof.forcez_interp(z, sprof.Reff) * sprof.forcez_scale_fac

        # Catch nans: set to 0.
        if ~np.isfinite(gz):
            gz = 0.
    else:
        gz = sprof.force_z(R, z, table=sprof.table, func_logrho=sprof.logrhom_interp)

    return gz, rho_g


def _dPdz(z, P_z, R, sprof_list, halo):
    """
    Vertical pressure gradient of a deprojected Sersic density profile at (R,z),
    based on hydrostatic equilibrium in the vertical direction

    .. math::

        d P_z / dz = d(\rho \sigma^2_z(R,z))/dz= \rho g_z(R,z)

    Parameters
    ----------
        z: float
            Height above midplane [kpc]

        P_z: 1d-array
            Pressure in the z direction.
            (Necessary for scipy.integrate.solve_ivp(), but not actually used.)

        R: float
            Midplane radius [kpc]

        sprof_list: list
            Set of deprojected Sersic model instances

        halo: Halo instance or None
            Optionally include force from a halo component.

    Returns
    -------
        integrand: float

    """
    rho_g = 0.*P_z
    gz = 0.*P_z

    for sprof in sprof_list:
        gz_i, rho_g_i = _preprocess_sprof_dPdz_calc(sprof, R, z)
        gz[0] += gz_i
        rho_g[0] += rho_g_i

    if halo is not None:
        gz[0] += halo.force_z(R, z)


    dPdz = rho_g * gz



    if np.abs(dPdz) == 0.:
        return np.abs(dPdz)
    else:
        return dPdz

def _dPdz_int_quad(z, R, sprof_list, halo):
    """
    Vertical pressure gradient of a deprojected Sersic density profile at (R,z),
    based on hydrostatic equilibrium in the vertical direction

    .. math::

        d P_z / dz = d(\rho \sigma^2_z(R,z))/dz= \rho g_z(R,z)

    Parameters
    ----------
        z: float
            Height above midplane [kpc]

        R: float
            Midplane radius [kpc]

        sprof_list: list
            Set of deprojected Sersic model instances

        halo: Halo instance or None
            Optionally include force from a halo component.

    Returns
    -------
        integrand: float

    """
    rho_g = 0.*z
    gz = 0.*z

    for sprof in sprof_list:
        gz_i, rho_g_i = _preprocess_sprof_dPdz_calc(sprof, R, z)
        gz += gz_i
        rho_g += rho_g_i

    if halo is not None:
        gz += halo.force_z(R, z)

    # # Set to rough minimum, nonzero feasible value
    # if (rho_g[0] == 0.) & (np.isfinite(z)):
    #     rho_g[0] = np.power(10., -320)


    dPdz = rho_g * gz


    if np.abs(dPdz) == 0.:
        return np.abs(dPdz)
    else:
        return dPdz


def _dlnPdz(z, lnP_z, R, sprof_list, halo):
    """
    Vertical pressure gradient of a deprojected Sersic density profile at (R,z),
    based on hydrostatic equilibrium in the vertical direction

    .. math::

        d ln P_z / dz = dln(\rho \sigma^2_z(R,z))/dz= \rho g_z(R,z) / P_z

    Parameters
    ----------
        z: float
            Height above midplane [kpc]

        lnP_z: 1d-array
            Log pressure in the z direction.

        R: float
            Midplane radius [kpc]

        sprof_list: list
            Set of deprojected Sersic model instances

        halo: Halo instance or None
            Optionally include force from a halo component.

    Returns
    -------
        integrand: float

    """
    rho_g = 0.*lnP_z
    gz = 0.*lnP_z

    for sprof in sprof_list:
        gz_i, rho_g_i = _preprocess_sprof_dPdz_calc(sprof, R, z)
        gz[0] += gz_i
        rho_g[0] += rho_g_i

    if halo is not None:
        gz[0] += halo.force_z(R, z)

    # # Set to rough minimum, nonzero feasible value
    # if (rho_g[0] == 0.) & (np.isfinite(z)):
    #     raise ValueError
    #     rho_g[0] = np.power(10., -320)


    # SIGN ERROR??
    #dlnPdz = -rho_g * gz / (np.exp(lnP_z))

    dlnPdz = rho_g * gz / (np.exp(lnP_z))

    # # Set to rough minimum, nonzero feasible value
    # if (rho_g[0] == 0.) & (np.isfinite(z)):
    #     dlnPdz[0] = np.power(10., -320)

    return dlnPdz



################################

def _dPdu(u, P_z, R, sprof_list, halo):
    """
    Vertical pressure gradient of a deprojected Sersic density profile at (R,u=1/z),
    based on hydrostatic equilibrium in the vertical direction with the
    substitution u = 1/z (to better handle the z->inf boundary condition)

    .. math::

        d P_z / du = d(\rho \sigma^2_z)/du= - \rho g_z  / u^2

    Parameters
    ----------
        u: float
            Inverse height above midplane, u = 1/z [kpc]

        P_z: 1d-array
            Pressure in the z direction.
            (Necessary for scipy.integrate.solve_ivp(), but not actually used.)

        R: float
            Midplane radius [kpc]

        sprof_list: list
            Set of deprojected Sersic model instances

        halo: Halo instance or None
            Optionally include force from a halo component.

    Returns
    -------
        integrand: float

    """
    rho_g = 0.*P_z
    gz = 0.*P_z

    if np.abs(u) > 0:
        z = 1./u
    else:
        z = np.inf

    for sprof in sprof_list:
        gz_i, rho_g_i = _preprocess_sprof_dPdz_calc(sprof, R, z)
        gz[0] += gz_i
        rho_g[0] += rho_g_i

    if halo is not None:
        gz[0] += halo.force_z(R, z)

    # # Set to rough minimum, nonzero feasible value
    # if (rho_g[0] == 0.) & (np.isfinite(z)):
    #     rho_g[0] = np.power(10., -320)


    dPdu = - rho_g * gz / (u**2)


    # Set to rough minimum, nonzero feasible value
    if (rho_g[0] == 0.) & (np.isfinite(z)):
        #dPdu[0] = np.power(10., -320)
        dPdu[0] = 0.

    if np.abs(dPdu) == 0.:
        return np.abs(dPdu)
    else:
        return dPdu

def _dlnPdu(u, lnP_z, R, sprof_list, halo):
    """
    Vertical pressure gradient of a deprojected Sersic density profile at (R,u=1/z),
    based on hydrostatic equilibrium in the vertical direction with the
    substitution u = 1/z (to better handle the z->inf boundary condition)

    .. math::

        d ln P_z / du = dln(\rho \sigma^2_z)/du= - \rho g_z  / (u^2 P_z)

    Parameters
    ----------
        u: float
            Inverse height above midplane, u = 1/z [kpc]

        lnP_z: 1d-array
            Log pressure in the z direction.

        R: float
            Midplane radius [kpc]

        sprof_list: list
            Set of deprojected Sersic model instances

        halo: Halo instance or None
            Optionally include force from a halo component.

    Returns
    -------
        integrand: float

    """
    rho_g = 0.*lnP_z
    gz = 0.*lnP_z

    if np.abs(u) > 0:
        z = 1./u
    else:
        z = np.inf

    for sprof in sprof_list:
        gz_i, rho_g_i = _preprocess_sprof_dPdz_calc(sprof, R, z)
        gz[0] += gz_i
        rho_g[0] += rho_g_i

    if halo is not None:
        gz[0] += halo.force_z(R, z)

    # # Set to rough minimum, nonzero feasible value
    # if (rho_g[0] == 0.) & (np.isfinite(z)):
    #     raise ValueError
    #     rho_g[0] = np.power(10., -320)


    dlnPdu = - rho_g * gz / (u**2 * np.exp(lnP_z))

    # raise ValueError

    # # Set to rough minimum, -inf not a feasible value
    # if (rho_g[0] == 0.) & (np.isfinite(z)):
    #     # dlnPdu[0] = -np.power(10.,308)
    #     #dlnPdu[0] = -np.power(10.,307.5)
    #     dlnPdu[0] = -np.power(10.,300)

    # Does it go to 0???
    if (rho_g[0] == 0.) & (np.isfinite(z)):
        dlnPdu[0] = np.power(10., -320)
        # dlnPdu[0] = 0.

    return dlnPdu

################################


def _preprocess_sprof_dsigzsqdz_calc(sprof, R, z):
    # Note R and z are both scalars

    m = np.sqrt(R**2 + (z/sprof.q)**2)


    rho_g = _sprof_get_rhog(sprof, m)
    if sprof.isgas:
        if sprof.dlnrhodlnm_interp is not None:
            # dlnrhodlnm = sprof.dlnrhodlnm_interp(m / sprof.Reff * sprof.table['Reff'])
            dlnrhodlnm = sprof.dlnrhodlnm_interp(m, sprof.Reff)
        else:
            dlnrhodlnm = sprof.dlnrho_dlnR(m)
        dlnrhogdz = (z / (sprof.q**2 * R**2 + z**2)) * dlnrhodlnm
    else:
        dlnrhogdz = m * 0.


    if sprof.forcez_interp is not None:
        # gz = sprof.forcez_interp(z / sprof.Reff * sprof.table['Reff']) * sprof.forcez_scale_fac
        gz = sprof.forcez_interp(z, sprof.Reff) * sprof.forcez_scale_fac

        # Catch nans: set to 0.
        if ~np.isfinite(gz):
            gz = 0.
    else:
        gz = sprof.force_z(R, z , table=sprof.table, func_logrho=sprof.logrhom_interp)

    return gz, rho_g, dlnrhogdz



def _dsigzsqdz(z, sigzsq, R, sprof_list, halo):
    """
    Vertical pressure gradient of a deprojected Sersic density profile at (R,z),
    based on hydrostatic equilibrium in the vertical direction

    .. math::

        d sig_z^2 / dz = g_z(R,z) - \sigma_z^z(R,z) d\ln\rho/dz

    Parameters
    ----------
        z: float
            Height above midplane [kpc]

        sigzsq: 1d-array
            Square of the dispersion in the z direction in the z direction.

        R: float
            Midplane radius [kpc]

        sprof_list: list
            Set of deprojected Sersic model instances

        halo: Halo instance or None
            Optionally include force from a halo component.

    Returns
    -------
        integrand: float

    """
    rho_g = 0.*sigzsq
    gz = 0.*sigzsq
    drhogdz = 0.*sigzsq

    for sprof in sprof_list:
        gz_i, rho_g_i, dlnrhogdz_i = _preprocess_sprof_dsigzsqdz_calc(sprof, R, z)
        if ~np.isfinite(dlnrhogdz_i):
            raise ValueError("dlnrhogdz_i is not finite!")

        # # Hack for calculation: Set to rough minimum, nonzero feasible value
        # if sprof.isgas & (rho_g_i == 0):
        #     rho_g_i = np.power(10., -320)

        gz[0] += gz_i
        rho_g[0] += rho_g_i
        drhogdz[0] += rho_g_i * dlnrhogdz_i

    if halo is not None:
        gz[0] += halo.force_z(R, z)

    # Set to rough minimum, nonzero feasible value
    if (rho_g[0] == 0.) & (np.isfinite(z)):
        rho_g[0] = np.power(10., -320)


    dlnrhogdz = drhogdz / rho_g

    # SIGN ERROR???
    #dsigzsqdz = -sigzsq * dlnrhogdz - gz

    dsigzsqdz = gz - sigzsq * dlnrhogdz

    return dsigzsqdz


def _preprocess_sprof_dsigzsqdz_calcBVP(sprof, R, z):

    m = np.sqrt(R**2 + (z/sprof.q)**2)


    rho_g = _sprof_get_rhog(sprof, m)
    if sprof.isgas:
        if sprof.dlnrhodlnm_interp is not None:
            # dlnrhodlnm = sprof.dlnrhodlnm_interp(m / sprof.Reff * sprof.table['Reff'])
            dlnrhodlnm = sprof.dlnrhodlnm_interp(m, sprof.Reff)
        else:
            dlnrhodlnm = sprof.dlnrho_dlnR(m)
        dlnrhogdz = (z / (sprof.q**2 * R**2 + z**2)) * dlnrhodlnm
    else:
        dlnrhogdz = m * 0.

    if (R==0):
        dlnrhogdz[z==0] = np.log(np.power(10., -320))


    if sprof.forcez_interp is not None:
        # gz = sprof.forcez_interp(z / sprof.Reff * sprof.table['Reff']) * sprof.forcez_scale_fac
        gz = sprof.forcez_interp(z, sprof.Reff) * sprof.forcez_scale_fac

        # Catch nans: set to 0.
        # if ~np.isfinite(gz):
        #     gz = 0.
        if np.any(~np.isfinite(gz)):
            gz[~np.isfinite(gz)] = 0.
    else:
        gz = sprof.force_z(R, z , table=sprof.table, func_logrho=sprof.logrhom_interp)

    # ## Test
    # gz_recalc = 0.*z
    # for i, zz in enumerate(z):
    #     gz_recalc[i] = sprof.force_z(R, zz , table=sprof.table, func_logrho=sprof.logrhom_interp)
    # raise ValueError

    return gz, rho_g, dlnrhogdz

def _dsigzsqdzBVP(z, sigzsq, R, sprof_list, halo):
    """
    Vertical pressure gradient of a deprojected Sersic density profile at (R,z),
    based on hydrostatic equilibrium in the vertical direction

    .. math::

        d sig_z^2 / dz = g_z(R,z) - \sigma_z^2(R,z) d\ln\rho/dz

    Parameters
    ----------
        z: float or array
            Height above midplane [kpc]

        sigzsq: 1d-array
            Square of the dispersion in the z direction in the z direction.

        R: float
            Midplane radius [kpc]

        sprof_list: list
            Set of deprojected Sersic model instances

        halo: Halo instance or None
            Optionally include force from a halo component.

    Returns
    -------
        integrand: float

    """
    rho_g = 0.*sigzsq
    gz = 0.*sigzsq
    drhogdz = 0.*sigzsq

    for sprof in sprof_list:
        gz_i, rho_g_i, dlnrhogdz_i = _preprocess_sprof_dsigzsqdz_calcBVP(sprof, R, z)
        # if ~np.isfinite(dlnrhogdz_i):
        #     raise ValueError("dlnrhogdz_i is not finite!")
        if np.any(~np.isfinite(dlnrhogdz_i)):
            raise ValueError("dlnrhogdz_i is not finite!")

        # # Hack for calculation: Set to rough minimum, nonzero feasible value
        # if sprof.isgas & (rho_g_i == 0):
        #     rho_g_i = np.power(10., -320)

        gz += gz_i
        rho_g += rho_g_i
        drhogdz += rho_g_i * dlnrhogdz_i

    if halo is not None:
        gz += halo.force_z(R, z)

    # Set to rough minimum, nonzero feasible value
    #if (np.any(rho_g == 0.)) & (np.isfinite(z)):
    if (np.any(rho_g == 0.)):
        rho_g[((rho_g == 0.) & np.isfinite(z))] = np.power(10., -320)


    dlnrhogdz = drhogdz / rho_g

    # SIGN ERROR???
    #dsigzsqdz = -sigzsq * dlnrhogdz - gz

    # SHOULD BE RIGHT ANSWER
    dsigzsqdz = gz - sigzsq * dlnrhogdz

    # raise ValueError

    # # dsigzsqdz = -gz - sigzsq * dlnrhogdz
    #
    # # dsigzsqdz = gz + sigzsq * dlnrhogdz
    #
    # dsigzsqdz = -gz + sigzsq * dlnrhogdz

    return dsigzsqdz

def _dsigzdzBVP(z, sigz, R, sprof_list, halo):
    """
    Vertical pressure gradient of a deprojected Sersic density profile at (R,z),
    based on hydrostatic equilibrium in the vertical direction

    .. math::

        d sig_z / dz = g_z(R,z)/(2\sigma_z) - \sigma_z(R,z)/2 d\ln\rho/dz

    Parameters
    ----------
        z: float or array
            Height above midplane [kpc]

        sigzsq: 1d-array
            Square of the dispersion in the z direction in the z direction.

        R: float
            Midplane radius [kpc]

        sprof_list: list
            Set of deprojected Sersic model instances

        halo: Halo instance or None
            Optionally include force from a halo component.

    Returns
    -------
        integrand: float

    """
    rho_g = 0.*sigz
    gz = 0.*sigz
    drhogdz = 0.*sigz

    for sprof in sprof_list:
        gz_i, rho_g_i, dlnrhogdz_i = _preprocess_sprof_dsigzsqdz_calcBVP(sprof, R, z)
        # if ~np.isfinite(dlnrhogdz_i):
        #     raise ValueError("dlnrhogdz_i is not finite!")
        if np.any(~np.isfinite(dlnrhogdz_i)):
            raise ValueError("dlnrhogdz_i is not finite!")

        # # Hack for calculation: Set to rough minimum, nonzero feasible value
        # if sprof.isgas & (rho_g_i == 0):
        #     rho_g_i = np.power(10., -320)

        gz += gz_i
        rho_g += rho_g_i
        drhogdz += rho_g_i * dlnrhogdz_i

    if halo is not None:
        gz += halo.force_z(R, z)

    # Set to rough minimum, nonzero feasible value
    #if (np.any(rho_g == 0.)) & (np.isfinite(z)):
    if (np.any(rho_g == 0.)):
        rho_g[((rho_g == 0.) & np.isfinite(z))] = np.power(10., -320)


    dlnrhogdz = drhogdz / rho_g

    # # SIGN ERROR???
    # #dsigzsqdz = -sigzsq * dlnrhogdz - gz
    #
    # #dsigzsqdz = gz - sigzsq * dlnrhogdz
    #
    # # dsigzsqdz = -gz - sigzsq * dlnrhogdz
    #
    # dsigzsqdz = gz + sigzsq * dlnrhogdz

    # SIGN???
    dsigzdz = gz/(2.*sigz) - (sigz/2.) * dlnrhogdz

    return dsigzdz

def _preprocess_sprof_Pz_calc(sprof, R, z, return_rho=False,
                              interp_dlnrhodlnm=False):

    # Assume component is gas by default
    if 'isgas' not in sprof.__dict__.keys():
        sprof.isgas = True
    else:
        print("Already set: sprof.isgas={}".format(sprof.isgas))

    logrhom_interp = None
    dlnrhodlnm_interp = None
    if 'table' in sprof.__dict__.keys():
        if sprof.table is not None:
            logrhom_interp = interpolate_sersic_profile_logrho_function(n=sprof.n, invq=sprof.invq,
                                                                        table=sprof.table)
            table_Reff = sprof.table['Reff']
            table_mass = sprof.table['total_mass']

            sprof.rhom_scale_fac = (sprof.total_mass / table_mass) * (table_Reff / sprof.Reff)**3

            if interp_dlnrhodlnm:
                dlnrhodlnm_interp = interpolate_sersic_profile_dlnrho_dlnR_function(n=sprof.n, invq=sprof.invq,
                                                                                    table=sprof.table)

    forcez_interp = None
    if 'table_forcez' in sprof.__dict__.keys():
        if sprof.table_forcez is not None:
            table_z =    sprof.table_forcez['z']
            table_Reff = sprof.table_forcez['Reff']

            # Will need to match exact R:
            tol = 1.e-3
            whm = np.where(np.abs(sprof.table_forcez['R']-(R / sprof.Reff * table_Reff))<=tol)[0]
            if len(whm) == 0:
                raise ValueError("No matching 'R' in table")
            table_gz = sprof.table_forcez['gz_R_{}'.format(whm[0])]

            # forcez_interp = scp_interp.interp1d(table_z, table_gz,
            #                 fill_value='extrapolate',  kind='cubic')

            # # LINEAR EXTRAP
            # fz_interp = scp_interp.interp1d(table_z, table_gz, fill_value=np.NaN, bounds_error=False, kind='cubic')
            # fz_extrap = scp_interp.interp1d(table_z, table_gz, fill_value='extrapolate',
            #                                            bounds_error=False, kind='linear')

            # CUBIC EXTRAP
            fz_interp = scp_interp.interp1d(table_z, table_gz, fill_value='extrapolate',
                                            bounds_error=False, kind='cubic')
            fz_extrap = None

            forcez_interp = InterpFunc(f_interp=fz_interp, f_extrap=fz_extrap, table_Rad=table_z, table_Reff=table_Reff)

            table_mass = sprof.table_forcez['total_mass']

            sprof.forcez_scale_fac = (sprof.total_mass / table_mass) * (table_Reff / sprof.Reff)**2


    # # Only replace if not already in the dict:
    # if 'logrhom_interp' not in sprof.__dict__.keys():
    #     sprof.logrhom_interp = logrhom_interp
    # if 'forcez_interp' not in sprof.__dict__.keys():
    #     sprof.forcez_interp = forcez_interp
    # if 'dlnrhodlnm_interp' not in sprof.__dict__.keys():
    #     sprof.dlnrhodlnm_interp = dlnrhodlnm_interp

    sprof.logrhom_interp = logrhom_interp
    sprof.forcez_interp = forcez_interp
    sprof.dlnrhodlnm_interp = dlnrhodlnm_interp

    if not return_rho:
        return sprof

    else:
        # Also calculate rho:
        m = np.sqrt(R**2 + (z/sprof.q)**2)
        rho_g = _sprof_get_rhog(sprof, m)

        return sprof, rho_g




def Pz(R, z, sprof=None, halo=None, return_sol=False, method=None):
    """
    Evalutation of vertical pressure of a deprojected Sersic density profile at (R,z),
    by solving the following differential equation:

    .. math::

        d P_z / dz = d(\rho \sigma^2_z(R,z))/dz= - \rho g_z(R,z)

    Parameters
    ----------
        R: float
            Midplane radius [kpc]
        z: float or array
            Height above midplane [kpc]


        sprof: Deprojected Sersic model instance or array of intances

        sprof.table: table of (standard) precomputed values, optional
            Use pre-computed table to create an interpolation function
            that is used for this calculation.
        sprof.table_forcez: table of precomputed gz, optional

        halo: Halo instance, optional
            Optionally include force from a halo component.

        return_sol: bool
            Return scipy IVP solution Bunch object or not. (Default: False)

    Returns
    -------
        Pzvals: float or array
            Vertical velocity dispersion direction; units [km^2/s^2]

    """
    if method is None:
        # method = 'dPdz'
        # method = 'dlnPdz'
        # method = 'dsigsqdz'

        # Attempt invert:
        # method = 'dPdu'
        # method = 'dlnPdu'

        # # CRAZY TEST:
        # method = 'dsigsqdz_BVP'
        # # method = 'dsigdz_BVP'

        method = 'dPdz_integrate'


    # # Override method if R=0:
    # if ((method == 'dsigsqdz_BVP') & (R==0.)):
    #     method = 'dlnPdz'


    print("Pz: method={}".format(method))

    if (method.lower().strip() != 'dpdz_integrate'):
        raise ValueError("Should use method='dPdz_integrate' !!!")

    try:
        tmp = sprof[0]
        sprof_list_in = sprof
    except:
        sprof_list_in = [sprof]

    # Cast z into an array and invert for evaluation:
    try:
        tmp = z[0]
        zeval = z[::-1]
    except:
        zeval = np.array([z])

    if method.strip().lower() in ['dsigsqdz', 'dsigsqdz_bvp', 'dsigdz_bvp']:
        rho_g = np.sqrt(R**2 + z**2) * 0.

    sprof_list = []

    # for sprof in sprof_list_in:
    #     sprof = _preprocess_sprof_Pz_calc(sprof, R, z)
    #     sprof_list.append(sprof)


    for sprof in sprof_list_in:
        if method.strip().lower() in ['dsigsqdz', 'dsigsqdz_bvp', 'dsigdz_bvp']:
            sprof, rho_g_i = _preprocess_sprof_Pz_calc(sprof, R, z, return_rho=True)
            rho_g += rho_g_i
        else:
            sprof = _preprocess_sprof_Pz_calc(sprof, R, z)
        sprof_list.append(sprof)


    if method.strip().lower() == 'dpdz':
        # t_span = [np.inf,0.]
        # y0 = np.array([0.])

        # Solver doesn't like t0 = inf....
        # t_span = [100.*sprof.q,0.]
        # t_span = [200.*sprof.q,0.] # orig fiducial
        # t_span = [500.*sprof.q,0.]
        t_span = [1000.*sprof.q,0.]


        sigz_guess = 1.e-6 #1.
        mguess = np.sqrt(R**2 + (t_span[0]/sprof.q)**2)
        Pz_guess = 0.
        for sprof in sprof_list:
            Pz_guess += _sprof_get_rhog(sprof, mguess) * (sigz_guess**2)

        y0 = np.array([Pz_guess])
        # if ~np.isfinite(y0[0]):
        #     raise ValueError("Initial guess is not finite!")

        if y0 == 0.:
            # Set to rough minimum, nonzero feasible value
            y0 = np.array([np.power(10., -320)])

        sol = scp_integrate.solve_ivp(_dPdz, t_span, y0, t_eval=zeval,
                                      args=(R, sprof_list, halo),
                                      # method='RK45',  # use for non-stiff problems
                                      method='Radau', # use for stiff problems
                                      # method='LSODA', # Auto switcher for stiffness
                                      dense_output=False )
        # Used t_eval = z[::-1], in inverse order, so invert output:
        Pzvals = sol.y[0][::-1]

        raise ValueError

    # elif method.strip().lower() == 'dlnpdz':
    #     # t_span = [np.inf,0.]
    #     # y0 = np.array([-np.inf])
    #
    #     # Solver doesn't like t0 = inf....
    #     # t_span = [100.*sprof.q,0.]
    #     # t_span = [200.*sprof.q,0.]  # orig fiducial
    #     # t_span = [300.*sprof.q,0.]
    #     # t_span = [500.*sprof.q,0.]
    #     t_span = [1000.*sprof.q,0.]
    #
    #     # sigz_guess = 1.e-2
    #     sigz_guess = 1.e-6 #1.
    #     mguess = np.sqrt(R**2 + (t_span[0]/sprof.q)**2)
    #     Pz_guess = 0.
    #     for sprof in sprof_list:
    #         Pz_guess += _sprof_get_rhog(sprof, mguess) * (sigz_guess**2)
    #
    #     y0 = np.array([np.log(Pz_guess)])
    #     if ~np.isfinite(y0[0]):
    #         #raise ValueError("Initial guess is not finite!")
    #         # Set to rough minimum feasible value
    #         y0 = np.array([np.log(np.power(10., -320))])
    #
    #     # raise ValueError
    #
    #     sol = scp_integrate.solve_ivp(_dlnPdz, t_span, y0, t_eval=zeval,
    #                                   args=(R, sprof_list, halo),
    #                                   # method='RK45',  # use for non-stiff problems
    #                                   method='Radau', # use for stiff problems
    #                                   # method='LSODA', # Auto switcher for stiffness
    #                                   dense_output=False )
    #     # Used t_eval = z[::-1], in inverse order, so invert output:
    #     Pzvals = np.exp(sol.y[0][::-1])
    #
    #     # raise ValueError


    elif method.strip().lower() == 'dlnpdz':
        # t_span = [np.inf,0.]
        # y0 = np.array([-np.inf])

        # Solver doesn't like t0 = inf....
        # t_span = [100.*sprof.q,0.]
        # t_span = [200.*sprof.q,0.]  # orig fiducial
        # t_span = [300.*sprof.q,0.]
        # t_span = [500.*sprof.q,0.]
        t_span = [1000.*sprof.q,0.]

        # sigz_guess = 1.e-2
        sigz_guess = 1.e-6 #1.
        mguess = np.sqrt(R**2 + (t_span[0]/sprof.q)**2)
        Pz_guess = 0.
        for sprof in sprof_list:
            Pz_guess += _sprof_get_rhog(sprof, mguess) * (sigz_guess**2)

        y0 = np.array([np.log(Pz_guess)])
        if ~np.isfinite(y0[0]):
            #raise ValueError("Initial guess is not finite!")
            # Set to rough minimum feasible value
            y0 = np.array([np.log(np.power(10., -320))])

        # raise ValueError

        sol = scp_integrate.solve_ivp(_dlnPdz, t_span, y0, t_eval=zeval,
                                      args=(R, sprof_list, halo),
                                      # method='RK45',  # use for non-stiff problems
                                      method='Radau', # use for stiff problems
                                      # method='LSODA', # Auto switcher for stiffness
                                      dense_output=False )
        # Used t_eval = z[::-1], in inverse order, so invert output:
        Pzvals = np.exp(sol.y[0][::-1])

        # raise ValueError

    ##################################################
    ##################################################
    elif method.strip().lower() == 'dpdu':
        # t_span = [np.inf,0.]
        # y0 = np.array([0.])

        # Have zeval = z[::-1] (high -> low)
        # -- so just tak 1/zeval to get ascending u
        ueval = 1./zeval


        # t_span = [0., np.inf]     # u limits
        # Nor a non-finite upper limit?
        t_span = [1./(1000.*sprof.q),np.max([ueval[np.isfinite(ueval)].max(),1000.*sprof.q])] #1000.*sprof.q]
        y0 = np.array([0.])   # Pz(u=ulim[0]).
        # As ulim[0]=0., or z=inf, this is Pz(z->inf)=0.

        # sigz_guess = 1.e-6 #1.
        # mguess = np.sqrt(R**2 + (t_span[0]/sprof.q)**2)
        # Pz_guess = 0.
        # for sprof in sprof_list:
        #     Pz_guess += _sprof_get_rhog(sprof, mguess) * (sigz_guess**2)
        #
        # y0 = np.array([Pz_guess])
        # # if ~np.isfinite(y0[0]):
        # #     raise ValueError("Initial guess is not finite!")
        #
        # if y0 == 0.:
        #     # Set to rough minimum, nonzero feasible value
        #     y0 = np.array([np.power(10., -320)])

        sol = scp_integrate.solve_ivp(_dPdu, t_span, y0, t_eval=ueval,
                                      args=(R, sprof_list, halo),
                                      # method='RK45',  # use for non-stiff problems
                                      method='Radau', # use for stiff problems
                                      # method='LSODA', # Auto switcher for stiffness
                                      dense_output=False )
        # Used t_eval = 1/z[::-1], in inverse order, so invert output:
        Pzvals = sol.y[0][::-1]

        raise ValueError

    elif method.strip().lower() == 'dlnpdu':
        # t_span = [0., np.inf]     # u limits
        # y0 = np.array([-np.inf])   # lnPz(u=ulim[0]).
        # # As ulim[0]=0., or z=inf, and this is lnPz, is -inf (as Pz(z->inf)=0.)

        # Have zeval = z[::-1] (high -> low)
        # -- so just tak 1/zeval to get ascending u
        ueval = 1./zeval

        # It doesn't like non-finite y0 either. Okayyyy
        #t_span = [1./(1000.*sprof.q),np.inf]
        # Nor a non-finite upper limit?
        # t_span = [1./(1000.*sprof.q),np.max([ueval[np.isfinite(ueval)].max(),1000.*sprof.q])] #1000.*sprof.q]

        t_span = [1./(200.*sprof.q),np.max([ueval[np.isfinite(ueval)].max(),1000.*sprof.q])] #1000.*sprof.q]

        sigz_guess = 1.e-2
        # sigz_guess = 1.e-6 #1.
        mguess = np.sqrt(R**2 + ((1./t_span[0])/sprof.q)**2)
        Pz_guess = 0.
        for sprof in sprof_list:
            Pz_guess += _sprof_get_rhog(sprof, mguess) * (sigz_guess**2)

        y0 = np.array([np.log(Pz_guess)])
        if ~np.isfinite(y0[0]):
            #raise ValueError("Initial guess is not finite!")
            # Set to rough minimum feasible value
            y0 = np.array([np.log(np.power(10., -320))])


        # test = _dlnPdu(t_span[0], y0, R, sprof_list, halo)
        #
        # raise ValueError

        sol = scp_integrate.solve_ivp(_dlnPdu, t_span, y0, t_eval=ueval,
                                      args=(R, sprof_list, halo),
                                      # method='RK45',  # use for non-stiff problems
                                      method='Radau', # use for stiff problems
                                      # method='LSODA', # Auto switcher for stiffness
                                      dense_output=False )
        # Used t_eval = 1/z[::-1], in inverse order, so invert output:
        Pzvals = np.exp(sol.y[0][::-1])

        # raise ValueError

    elif method.strip().lower() == 'dpdz_integrate':
        # INTEGRATE

        # First check if z is array or not:

        try:
            tmp = z[0]
            Pzvals = 0.*z
            zarr = z
            isarr=True
        except:
            zarr = [z]
            Pzvals = np.array([0.])
            isarr = False

        # _dPdz(z, P_z, R, sprof_list, halo)

        # z0 = sprof.q*sprof.Reff
        # z0 = np.inf

        for jj, zz in enumerate(zarr):
            # Pzval_C, _ = scp_integrate.quad(_dPdz_int_quad, z0, zz,
            #                                args=(R, sprof_list, halo))


            Pzval_C, _ = scp_integrate.quad(_dPdz_int_quad, np.inf, zz,
                                           args=(R, sprof_list, halo))

            Pzvals[jj] = Pzval_C

        sol = None

    ##################################################
    ##################################################

    elif method.strip().lower() == 'dsigsqdz':
        if return_sol:
            sigzsq, sol = sigmaz_sq(R, z, sprof=sprof_list, halo=halo,
                                    method=method, return_sol=True)
        else:
            sigzsq = sigmaz_sq(R, z, sprof=sprof_list, halo=halo, method=method)
        # if len(sol.t[::-1]
        Pzvals = rho_g * sigzsq


    elif method.strip().lower() == 'dsigsqdz_bvp':
        if return_sol:
            sigzsq, sol = sigmaz_sq(R, z, sprof=sprof_list, halo=halo,
                                    method=method, return_sol=True)
        else:
            sigzsq = sigmaz_sq(R, z, sprof=sprof_list, halo=halo, method=method)
        # if len(sol.t[::-1]
        Pzvals = rho_g * sigzsq

        # raise ValueError
    elif method.strip().lower() == 'dsigdz_bvp':
        if return_sol:
            sigzsq, sol = sigmaz_sq(R, z, sprof=sprof_list, halo=halo,
                                    method=method, return_sol=True)
        else:
            sigzsq = sigmaz_sq(R, z, sprof=sprof_list, halo=halo, method=method)
        # if len(sol.t[::-1]
        Pzvals = rho_g * sigzsq

        # raise ValueError
    else:
        raise ValueError("Method={}".format(method))


    # raise ValueError

    # Recast output into same shape as z input:
    try:
        tmp = z[0]
        if return_sol:
            return Pzvals, sol
        else:
            return Pzvals
    except:
        if return_sol:
            return Pzvals[0], sol
        else:
            return Pzvals[0]


def sigmaz_sq(R, z, sprof=None, halo=None, method=None, return_sol=False):
    """
    Evalutation of vertical velocity dispersion of a deprojected Sersic density profile at (R,z).

    .. math::

        \sigma^2_z(R,z)=\frac{1}{\rho}\int \rho g_z(R,z)dz

    Parameters
    ----------
        R: float
            Midplane radius [kpc]
        z: float or array
            Height above midplane [kpc]


        sprof: Deprojected Sersic model instance or array of intances

        sprof.table: table of (standard) precomputed values, optional
            Use pre-computed table to create an interpolation function
            that is used for this calculation.
        sprof.table_forcez: table of precomputed gz, optional

        halo: Halo instance, optional
            Optionally include force from a halo component.

    Returns
    -------
        sigzsq: float
            Square vertical velocity dispersion; units [km^2/s^2]

    """
    if method is None:
        # method = 'dPdz'
        # method = 'dlnPdz'
        # method = 'dsigsqdz'

        # Attempt invert:
        # method = 'dPdu'
        # method = 'dlnPdu'

        # # CRAZY TEST:
        # method = 'dsigsqdz_BVP'
        # # method = 'dsigdz_BVP'


        method = 'dPdz_integrate'

    # # Override method if R=0:
    # if ((method == 'dsigsqdz_BVP') & (R==0.)):
    #     method = 'dlnPdz'


    print("sigmaz_sq: method={}".format(method))

    if (method.lower().strip() != 'dpdz_integrate'):
        raise ValueError("Should use method='dPdz_integrate' !!!")

    try:
        tmp = sprof[0]
        sprof_list_in = sprof
    except:
        sprof_list_in = [sprof]

    rho_g = np.sqrt(R**2 + z**2) * 0.
    sprof_list = []

    for sprof in sprof_list_in:
        sprof, rho_g_i = _preprocess_sprof_Pz_calc(sprof, R, z, return_rho=True,
                                                   interp_dlnrhodlnm=True)
        rho_g += rho_g_i
        sprof_list.append(sprof)

    # try:
    #     # Set to rough minimum feasible value
    #     rho_g[rho_g==0] = np.power(10., -320)
    # except:
    #     if (rho_g==0):
    #         rho_g = np.power(10., -320)


    if method.strip().lower() in ['dpdz', 'dlnpdz', 'dpdu', 'dlnpdu', 'dpdz_integrate']:
        if return_sol:
            Pzvals, sol= Pz(R, z, sprof=sprof_list_in, halo=halo, method=method, return_sol=return_sol)
            # raise ValueError
            sigsqz = Pzvals / rho_g

            # Backpaste the tiny values:
            try:
                sigsqz[Pzvals==np.power(10.,-320)] = np.power(10.,-320)
            except:
                if Pzvals==np.power(10.,-320):
                    sigsqz = np.power(10.,-320)

            # # EXTRAPOLATE TO BACKPASTE:
            # try:
            #     if np.any(Pzvals==np.power(10.,-320)):
            #         ztmp = z[Pzvals!=np.power(10.,-320)]
            #         sigsqztmp = sigsqz[Pzvals!=np.power(10.,-320)]
            #         interptmp = scp_interp.interp1d(ztmp, sigsqztmp,
            #                         fill_value='extrapolate', kind='linear') #kind='cubic')
            #         sigsqz[Pzvals==np.power(10.,-320)] = interptmp(z[Pzvals==np.power(10.,-320)])
            # except:
            #     # Backpaste a tiny value:
            #     if Pzvals==np.power(10.,-320):
            #         sigsqz = np.power(10.,-320)




            return sigsqz, sol
        else:
            Pzvals = Pz(R, z, sprof=sprof_list_in, halo=halo, method=method)
            sigsqz = Pzvals / rho_g
            return sigsqz


    elif method.strip().lower() in ['dsigsqdz']:
        # Cast z into an array and invert for evaluation:
        try:
            tmp = z[0]
            zeval = z[::-1]
        except:
            zeval = np.array([z])

        # ####
        # t_span = [np.inf,0.]
        # y0 = np.array([0.])

        # Solver doesn't like t0 = inf....
        # t_span = [100.*sprof.q,0.]
        # t_span = [200.*sprof.q,0.]  # orig fiducial
        # t_span = [500.*sprof.q,0.]
        t_span = [1000.*sprof.q,0.]


        #sigz_guess = 1.e-6 #0.  # RANDOM GUESS
        sigz_guess = 1.e-2 #0.  # RANDOM GUESS

        y0 = np.array([sigz_guess])


        if ~np.isfinite(y0[0]):
            raise ValueError("Initial guess is not finite!")
            # # Set to rough minimum, nonzero feasible value
            # y0 = np.array([np.power(10., -320)])

        sol = scp_integrate.solve_ivp(_dsigzsqdz, t_span, y0, t_eval=zeval,
                                      args=(R, sprof_list, halo),
                                      # method='RK45',  # use for non-stiff problems
                                      method='Radau', # use for stiff problems
                                      # method='LSODA', # Auto switcher for stiffness
                                      dense_output=False )
        # Used t_eval = z[::-1], in inverse order, so invert output:
        sigzsqvals = sol.y[0][::-1]

        # return sigzsqvals

        raise ValueError

        # Recast output into same shape as z input:
        try:
            tmp = z[0]
            if return_sol:
                return sigzsqvals, sol
            else:
                return sigzsqvals
        except:
            if return_sol:
                return sigzsqvals[0], sol
            else:
                return sigzsqvals[0]



    elif method.strip().lower() in ['dsigsqdz_bvp']:
        # Cast z into an array and invert for evaluation:
        try:
            tmp = z[0]
            zeval = z[:]
        except:
            zeval = np.array([z])

        # ####
        # t_span = [np.inf,0.]
        # y0 = np.array([0.])

        # Solver doesn't like t0 = inf....
        # t_span = [100.*sprof.q,0.]
        # t_span = [200.*sprof.q,0.]  # orig fiducial
        # t_span = [500.*sprof.q,0.]
        #t_span = [1000.*sprof.q,0.]


        # sigz_guess = 1.e-6 #0.  # RANDOM GUESS


        # BCs: for the min, max zeval.
        # if zeval[0] != 0.: raise ValueError
        # if zeval[-1] < 200.: raise ValueError

        # xmesh = np.arange([0.,200.5, 0.5])
        #xmesh = np.append(np.array([0.]), np.logspace(-3.,2.5, 201))
        # xmesh = np.append(np.array([0.]), np.logspace(-3.,3.5, 201))

        xmesh = np.append(np.array([0.]), np.logspace(-2.,3.5, 201))
        # xmesh = np.append(np.array([0.]), np.logspace(-1.,3., 201))
        xmesh = np.append(np.array([0.]), np.logspace(-1.,2., 201))

        xmesh = np.append(np.array([0.]), np.logspace(-3.,4., 201))
        xmesh = np.append(np.array([0.]), np.logspace(-4.,3., 201))
        # xmesh = np.append(np.array([0.]), np.logspace(-10.,3., 201))  # FAIL


        #xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 201))

        xmesh = np.append(np.array([0.]), np.logspace(-4.,3., 51))
        xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 51))
        xmesh = np.append(np.array([0.]), np.logspace(-4.,3., 15))
        xmesh = np.append(np.array([0.]), np.logspace(-4.,3., 7))


        xmesh = np.append(np.array([0.]), np.logspace(-2.,3., 7))
        xmesh = np.append(np.array([0.]), np.logspace(-2.,3., 15))
        xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 15))
        # xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 9))


        # xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 19))
        xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 25))
        xmesh = np.append(np.array([0.]), np.logspace(-6.,3.5, 25))
        xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 25))  # Shape good, normalization WHACK

        xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 25)) # Shape whack
        xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 35))
        xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 45))
        xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 101))
        xmesh = np.append(np.array([0.]), np.logspace(-3.5,3.5, 101))
        xmesh = np.append(np.array([0.]), np.logspace(-3.5,3.5, 15))
        xmesh = np.append(np.array([0.]), np.logspace(-2.5,3.5, 15))
        xmesh = np.append(np.array([0.]), np.logspace(-2.,3.5, 31))
        xmesh = np.append(np.array([0.]), np.logspace(-2.,3.5, 11))



        xmesh = np.append(np.array([0.]), np.logspace(-4.,3.5, 11))
        xmesh = np.append(np.array([0.]), np.logspace(-4.,4., 11))
        xmesh = np.append(np.array([0.]), np.logspace(-4.,4., 25))
        xmesh = np.append(np.array([0.]), np.logspace(-6.,4., 25))  # Shape good, normalization WHACK


        # xmesh = np.append(np.array([0.]), np.logspace(-6.,4., 15)) # Shape good, norm still high


        # xmesh = np.append(np.array([0.]), np.logspace(-6.,3.5, 15)) #

        # xmesh = np.append(np.array([0.]), np.logspace(-2.,2., 25))


        xmesh = np.append(np.array([0.]), np.logspace(-4.,3., 7))
        xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 15))
        xmesh = np.append(np.array([0.]), np.logspace(-4.,3.5, 11))
        xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 101))
        xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 51))

        # xmesh = np.append(np.array([0.]), np.logspace(-3.,3.5, 51))
        # xmesh = np.append(np.array([0.]), np.logspace(-3.,3.5, 15))


        # # xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 11))  # Singularity in jacobian
        # xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 25))  # Shape good, normalization WHACK
        # xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 19)) # Singularity in jacobian
        # xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 24)) # Singularity in jacobian


        # # WTFFF WHY IS OTHER NOT WORKING
        # xmesh = np.append(np.array([0.]), np.logspace(-9.,3., 51))  # WTFFF WHY IS OTHER NOT WORKING
        # xmesh = np.append(np.array([0.]), np.logspace(-6.,3.5, 15))
        # xmesh = np.append(np.array([0.]), np.logspace(-3.,3., 15))
        # xmesh = np.append(np.array([0.]), np.logspace(-3.,2., 9))
        # xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 101))

        # bcs = np.array([100.**2, 1.e-2 **2 ])
        # bcs = np.array([100.**2, 1.e-10 **2 ])
        # bcs = np.array([100.**2, 0. ])
        # bcs = np.array([np.NaN, 0. ])
        # bcs = np.array([np.NaN, 1.e-6**2 ])
        bcs = np.array([np.NaN, 0. ])

        # bcs = np.array([np.NaN, 1.e-2**2 ])

        # bcs = np.array([np.NaN, 1.e-10**2 ])

        # max_nodes = 1000  # default
        max_nodes = 1e6

        ymesh = np.zeros((1,len(xmesh)))

        sprof_list_normed = []
        mass_norm = 0.
        for sprof in sprof_list_in:
            if sprof.total_mass > mass_norm:
                mass_max = sprof.total_mass
        mass_target = 1.e5
        # mass_target = 1.e3
        # mass_target = 1.e1
        # mass_target = 1.
        mass_norm_fac = mass_target / mass_max
        for sprof in sprof_list_in:
            sprof_new = copy.deepcopy(sprof)
            sprof_new.total_mass = sprof.total_mass * mass_norm_fac
            sprof_new = _preprocess_sprof_Pz_calc(sprof_new, R, z,
                                                  interp_dlnrhodlnm=True)
            sprof_list_normed.append(sprof_new)

        # Norm the halo:
        halo_normed = copy.deepcopy(halo)
        halo_normed.scale_fac = mass_norm_fac

        print(sprof_list[0].total_mass)
        print(sprof_list_normed[0].total_mass)

        # raise ValueError

        # sprof_list = None

        def _fun_BC_sigzsqdz(ya, yb):
            #return np.array([ya[0]-bcs[0], yb[0]-bcs[1]])
            # return np.array([ya[0]-bcs[0]])
            return np.array([yb[0]-bcs[1]])


        def _fun_tmp_sigzsqdz(z, sigsq):
            # return _dsigzsqdzBVP(z, sigsq, R, sprof_list, halo)
            return _dsigzsqdzBVP(z, sigsq, R, sprof_list_normed, halo_normed)


        sol = scp_integrate.solve_bvp(_fun_tmp_sigzsqdz, _fun_BC_sigzsqdz, xmesh, ymesh,
                                      max_nodes=max_nodes)

        # sigzsqvals = sol.sol(zeval)[0]
        # sigzsqvals = -sol.sol(zeval)[0]
        sigzsqvals = sol.sol(zeval)[0] / mass_norm_fac


        # ## TEST:
        # sigzsqvals = - sol.sol(zeval)[0] / mass_norm_fac

        # # TEST
        # sigzsqvals = np.abs(sigzsqvals)

        # return sigzsqvals

        # Recast output into same shape as z input:
        try:
            tmp = z[0]
            if return_sol:
                return sigzsqvals, sol
            else:
                return sigzsqvals
        except:
            if return_sol:
                return sigzsqvals[0], sol
            else:
                return sigzsqvals[0]


    #
    # elif method.strip().lower() in ['dsigdz_bvp']:
    #     # Cast z into an array and invert for evaluation:
    #     try:
    #         tmp = z[0]
    #         zeval = z[:]
    #     except:
    #         zeval = np.array([z])
    #
    #     # ####
    #     # t_span = [np.inf,0.]
    #     # y0 = np.array([0.])
    #
    #     # Solver doesn't like t0 = inf....
    #     # t_span = [100.*sprof.q,0.]
    #     # t_span = [200.*sprof.q,0.]  # orig fiducial
    #     # t_span = [500.*sprof.q,0.]
    #     #t_span = [1000.*sprof.q,0.]
    #
    #
    #     # sigz_guess = 1.e-6 #0.  # RANDOM GUESS
    #
    #
    #     # BCs: for the min, max zeval.
    #     # if zeval[0] != 0.: raise ValueError
    #     # if zeval[-1] < 200.: raise ValueError
    #
    #     # xmesh = np.arange([0.,200.5, 0.5])
    #     #xmesh = np.append(np.array([0.]), np.logspace(-3.,2.5, 201))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-3.,3.5, 201))
    #
    #     xmesh = np.append(np.array([0.]), np.logspace(-2.,3.5, 201))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-1.,3., 201))
    #     xmesh = np.append(np.array([0.]), np.logspace(-1.,2., 201))
    #
    #     xmesh = np.append(np.array([0.]), np.logspace(-3.,4., 201))
    #     xmesh = np.append(np.array([0.]), np.logspace(-4.,3., 201))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-10.,3., 201))  # FAIL
    #
    #
    #     #xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 201))
    #
    #     xmesh = np.append(np.array([0.]), np.logspace(-4.,3., 51))
    #     xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 51))
    #     xmesh = np.append(np.array([0.]), np.logspace(-4.,3., 15))
    #     xmesh = np.append(np.array([0.]), np.logspace(-4.,3., 7))
    #
    #
    #     xmesh = np.append(np.array([0.]), np.logspace(-2.,3., 7))
    #     xmesh = np.append(np.array([0.]), np.logspace(-2.,3., 15))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 15))
    #     # # xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 9))
    #     #
    #     #
    #     # # xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 19))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.,3., 25))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.,3.5, 25))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 25))  # Shape good, normalization WHACK
    #     #
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 25)) # Shape whack
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 35))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 45))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.5,3.5, 101))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-3.5,3.5, 101))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-3.5,3.5, 15))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-2.5,3.5, 15))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-2.,3.5, 31))
    #     # xmesh = np.append(np.array([0.]), np.logspace(-2.,3.5, 11))
    #     #
    #     #
    #     # # xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 11))  # Singularity in jacobian
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 25))  # Shape good, normalization WHACK
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 19)) # Singularity in jacobian
    #     # xmesh = np.append(np.array([0.]), np.logspace(-6.,4, 24)) # Singularity in jacobian
    #
    #     # bcs = np.array([100.**2, 1.e-2 **2 ])
    #     # bcs = np.array([100.**2, 1.e-10 **2 ])
    #     # bcs = np.array([100.**2, 0. ])
    #     # bcs = np.array([np.NaN, 0. ])
    #     # bcs = np.array([np.NaN, 1.e-6**2 ])
    #     bcs = np.array([np.NaN, 0. ])
    #     # bcs = np.array([np.NaN, 1.e-2**2 ])
    #
    #     # max_nodes = 1000  # default
    #     max_nodes = 1e6
    #
    #     ymesh = np.zeros((1,len(xmesh)))
    #
    #     def _fun_BC_sigzdz(ya, yb):
    #         #return np.array([ya[0]-bcs[0], yb[0]-bcs[1]])
    #         # return np.array([ya[0]-bcs[0]])
    #         return np.array([yb[0]-bcs[1]])
    #
    #
    #     def _fun_tmp_sigzdz(z, sigsq):
    #         return _dsigzdzBVP(z, sigsq, R, sprof_list, halo)
    #
    #     sol = scp_integrate.solve_bvp(_fun_tmp_sigzdz, _fun_BC_sigzdz, xmesh, ymesh,
    #                                   max_nodes=max_nodes)
    #
    #     # sigzsqvals = sol.sol(zeval)[0]
    #     # sigzsqvals = -sol.sol(zeval)[0]
    #     sigzvals = sol.sol(zeval)[0]
    #     sigzsqvals = sigzvals**2
    #
    #     # return sigzsqvals
    #
    #     # Recast output into same shape as z input:
    #     try:
    #         tmp = z[0]
    #         if return_sol:
    #             return sigzsqvals, sol
    #         else:
    #             return sigzsqvals
    #     except:
    #         if return_sol:
    #             return sigzsqvals[0], sol
    #         else:
    #             return sigzsqvals[0]

    else:
        raise ValueError("method={}".format(method))


def BBBBBBBBBBBBBBBBBBBBBBBBBBB():

    return None
#
# def _preprocess_sprof_sigz_integrand_calc(sprof, R, z):
#     m = np.sqrt(R**2 + (z/sprof.q)**2)
#
#     if sprof.isgas:
#         if sprof.logrhom_interp is not None:
#             table_Reff =    sprof.table['Reff']
#             table_mass =    sprof.table['total_mass']
#
#             scale_fac = (sprof.total_mass / table_mass) * (table_Reff / sprof.Reff)**3
#             logrho = sprof.logrhom_interp(m / sprof.Reff * table_Reff)
#             rho_g = np.power(10., logrho) * scale_fac
#         else:
#             rho_g = sprof.density(m)
#     else:
#         rho_g = m * 0.
#
#     if sprof.forcez_interp is not None:
#         table_Reff =    sprof.table_forcez['Reff']
#         table_mass =    sprof.table_forcez['total_mass']
#
#         scale_fac = (sprof.total_mass / table_mass) * (table_Reff / sprof.Reff)**2
#         gz = sprof.forcez_interp(z / sprof.Reff * table_Reff) * scale_fac
#     else:
#         gz = sprof.force_z(R, z , table=sprof.table, func_logrho=sprof.logrhom_interp)
#
#
#     return gz, rho_g
#
#
# def sigsq_z_integrand(z, R, sprof_list, halo):
#     """
#     Integrand as part of numerical integration to find :math:`\sigma^2_z(r,z)`
#
#     Parameters
#     ----------
#         z: float
#             Height above midplane [kpc]
#         R: float
#             Midplane radius [kpc]
#
#         sprof_list: Set of deprojected Sersic model instances, list
#
#         halo: Halo instance or None
#             Optionally include force from a halo component.
#
#     Returns
#     -------
#         integrand: float
#
#     """
#     rho_g = 0.
#     gz = 0.
#
#     for sprof in sprof_list:
#         gz_i, rho_g_i = _preprocess_sprof_sigz_integrand_calc(sprof, R, z)
#         gz += gz_i
#         rho_g += rho_g_i
#
#     if halo is not None:
#         gz += halo.force_z(R, z)
#
#     integrand = rho_g * gz
#
#     return integrand
#
#
# def _preprocess_sprof_sigz_calc(sprof, R, z):
#
#     m = np.sqrt(R**2 + (z/sprof.q)**2)
#
#     # Assume component is gas by default
#     if 'isgas' not in sprof.__dict__.keys():
#         sprof.isgas = True
#
#     logrhom_interp = None
#     if 'table' in sprof.__dict__.keys():
#         if sprof.table is not None:
#             logrhom_interp = interpolate_sersic_profile_logrho_function(n=sprof.n, invq=sprof.invq,
#                                                                   table=sprof.table)
#
#     if sprof.isgas:
#         if logrhom_interp is not None:
#             table_Reff =    sprof.table['Reff']
#             table_mass =    sprof.table['total_mass']
#
#             scale_fac = (sprof.total_mass / table_mass) * (table_Reff / sprof.Reff)**3
#             logrho = logrhom_interp(m / sprof.Reff * table_Reff)
#             rho_g = np.power(10., logrho) * scale_fac
#         else:
#             rho_g = sprof.density(m)
#     else:
#         rho_g = m * 0.
#
#     forcez_interp = None
#     if 'table_forcez' in sprof.__dict__.keys():
#         if sprof.table_forcez is not None:
#             table_z =       sprof.table_forcez['z']
#             table_Reff =    sprof.table_forcez['Reff']
#
#             # Will need to match exact R:
#             # whm = np.where(sprof.table_forcez['R'] == (R / sprof.Reff * table_Reff))[0]
#             tol = 1.e-3
#             whm = np.where(np.abs(sprof.table_forcez['R']-(R / sprof.Reff * table_Reff))<=tol)[0]
#             if len(whm) == 0:
#                 raise ValueError("No matching 'R' in table")
#             table_gz = sprof.table_forcez['gz_R_{}'.format(whm[0])]
#
#             forcez_interp = scp_interp.interp1d(table_z, table_gz,
#                             fill_value='extrapolate',  kind='cubic')
#
#         # #
#         # if (sprof.n == 4.) & (np.abs(R-5.)<1.e-3) & (np.abs(z-5.)<1.e-3):
#         #     print("DEBUG TEST")
#         #     forcez_interp = None
#
#
#     sprof.logrhom_interp = logrhom_interp
#     sprof.forcez_interp = forcez_interp
#
#     return sprof, rho_g
#
# def sigmaz_sq(R, z, sprof=None, halo=None):
#     """
#     Evalutation of vertical velocity dispersion of a deprojected Sersic density profile at (R,z).
#
#     .. math::
#
#         \sigma^2_z(R,z)=\frac{1}{\rho}\int \rho g_z(R,z)dz
#
#     Parameters
#     ----------
#         R: float
#             Midplane radius [kpc]
#         z: float
#             Height above midplane [kpc]
#
#
#         sprof: Deprojected Sersic model instance or array of intances
#
#         sprof.table: table of (standard) precomputed values, optional
#             Use pre-computed table to create an interpolation function
#             that is used for this calculation.
#         sprof.table_forcez: table of precomputed gz, optional
#
#         halo: Halo instance, optional
#             Optionally include force from a halo component.
#
#     Returns
#     -------
#         sigzsq: float
#             Vertical velocity dispersion direction; units [km^2/s^2]
#
#     """
#
#     try:
#         tmp = sprof[0]
#         sprof_list_in = sprof
#     except:
#         sprof_list_in = [sprof]
#
#
#     rho_g = np.sqrt(R**2 + z**2) * 0.
#     sprof_list = []
#
#     for sprof in sprof_list_in:
#         sprof, rho_g_i = _preprocess_sprof_sigz_calc(sprof, R, z)
#         rho_g += rho_g_i
#         sprof_list.append(sprof)
#
#     # int_sigsq_z, _ = scp_integrate.quad(sigsq_z_integrand, 0, z,
#     #                                     args=(R, sprof_list, halo))
#     # # WRONG: ///// No negatives, because delPotl = -gz, so already in there
#     # return - 1./rho_g * int_sigsq_z
#
#
#     int_sigsq_z, _ = scp_integrate.quad(sigsq_z_integrand, z, np.inf,
#                                         args=(R, sprof_list, halo))
#
#     return -1./rho_g * int_sigsq_z
#
#
# def rho_sigmaz_sq(R, z, sprof=None, halo=None):
#     """
#     Evalutation of vertical velocity dispersion of a deprojected Sersic density profile at (R,z).
#
#     .. math::
#
#         \rho \sigma^2_z(R,z)=\int \rho g_z(R,z)dz
#
#     Parameters
#     ----------
#         R: float
#             Midplane radius [kpc]
#         z: float
#             Height above midplane [kpc]
#
#
#         sprof: Deprojected Sersic model instance or array of intances
#
#         sprof.table: table of (standard) precomputed values, optional
#             Use pre-computed table to create an interpolation function
#             that is used for this calculation.
#         sprof.table_forcez: table of precomputed gz, optional
#
#         halo: Halo instance, optional
#             Optionally include force from a halo component.
#
#     Returns
#     -------
#         rho_sigzsq: float
#             Vertical velocity dispersion direction; units [km^2/s^2]
#
#     """
#
#     try:
#         tmp = sprof[0]
#         sprof_list_in = sprof
#     except:
#         sprof_list_in = [sprof]
#
#
#     rho_g = np.sqrt(R**2 + z**2) * 0.
#     sprof_list = []
#
#     for sprof in sprof_list_in:
#         sprof, rho_g_i = _preprocess_sprof_sigz_calc(sprof, R, z)
#         rho_g += rho_g_i
#         sprof_list.append(sprof)
#
#     # int_sigsq_z, _ = scp_integrate.quad(sigsq_z_integrand, 0, z,
#     #                                     args=(R, sprof_list, halo))
#     # # WRONG: ///// No negatives, because delPotl = -gz, so already in there
#     # return - int_sigsq_z
#
#
#     int_sigsq_z, _ = scp_integrate.quad(sigsq_z_integrand, z, np.inf,
#                                         args=(R, sprof_list, halo))
#
#     return - int_sigsq_z


def BBBBBBBBBBBBBBBBBBBBBBBBBBB():

    return None

def force_R_integrand_z0(m, R, z, Reff, n, q, Ie, i, Upsilon, logrhom_interp, table, total_mass):
    """
    Integrand :math:`\frac{\rho(m)}{(\tau+1)^2 \sqrt{tau+q^2}}`
    as part of numerical integration to find :math:`g_r(R,z)`

    Parameters
    ----------
        tau: float
            independent variable
        R: float
            Midplane radius [kpc]
        z: float
            Height above midplane [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]
        Upsilon: float
            Mass-to-light ratio
        logrhom_interp: function, optional
            Shortcut to use an interpolation function (from a lookup table)
            instead of recalculating rho(m)
        table: Sersic profile table, optional
        total_mass: log total mass [Msun], optional

    Returns
    -------
        integrand: float

    """
    if logrhom_interp is not None:
        table_Reff =    table['Reff']
        table_mass =    table['total_mass']

        scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**3
        # # rho = rhom_interp(m / Reff * table_Reff) * scale_fac
        # logrho = logrhom_interp(m / Reff * table_Reff)

        logrho = logrhom_interp(m, Reff)

        rho = np.power(10., logrho) * scale_fac

        # Back replace inf, if interpolating at r=0 for n>1:
        if (table['n'] >= 1.) & (table['R'][0] == 0.):
            if (~np.isfinite(table['rho'][0]) & (m == 0.)):
                rho = table['rho'][0]
    else:
        rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)

    integrand = (rho * m**2) / np.sqrt(R**2 - m**2 * (1.-q**2) )

    return integrand

def force_z_integrand_R0(m, R, z, Reff, n, q, Ie, i, Upsilon, logrhom_interp, table, total_mass):
    """
    Integrand :math:`\frac{\rho(m)}{(\tau+1)(tau+q^2)^{3/2}}`
    as part of numerical integration to find :math:`g_z(R,z)`

    Parameters
    ----------
        tau: float
            independent variable
        R: float
            Midplane radius [kpc]
        z: float
            Height above midplane [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]
        Upsilon: float
            Mass-to-light ratio
        logrhom_interp: function, optional
            Shortcut to use an interpolation function (from a lookup table)
            instead of recalculating rho(m)
        table: Sersic profile table, optional
    Returns
    -------
        integrand: float

    """
    if logrhom_interp is not None:
        table_Reff =    table['Reff']
        table_mass =    table['total_mass']

        scale_fac = (total_mass / table_mass) * (table_Reff / Reff)**3
        # rho = rhom_interp(m / Reff * table_Reff) * scale_fac
        # logrho = logrhom_interp(m / Reff * table_Reff)

        logrho = logrhom_interp(m, Reff)

        rho = np.power(10., logrho) * scale_fac

        # Back replace inf, if interpolating at r=0 for n>1:
        if (table['n'] >= 1.) & (table['R'][0] == 0.):
            if (~np.isfinite(table['rho'][0]) & (m == 0.)):
                rho = table['rho'][0]
    else:
        rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)

    integrand = (rho * m**2) / (z**2 - m**2 * (q**2-1.))

    return integrand

def force_R_z0(R, z, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.,
            logrhom_interp=None, table=None, total_mass=None):
    """
    Evalutation of gravitational force in the radial direction
    :math:`g_R=-\partial\Phi/\partial R`,
    of a deprojected Sersic density profile at (R,z), by numerically evalutating

    .. math::

        g_R(R,z) = - 2\pi GqR \int_0^{\infty} d\tau \frac{\rho(m)}{(\tau+1)^2 \sqrt{tau+q^2}},
        m = \frac{r^2}{\tau+1} + \frac{z^2}{\tau+q^2}

    Parameters
    ----------
        R: float
            Midplane radius [kpc]
        z: float
            Height above midplane [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)
        logrhom_interp: function, optional
            Shortcut to use an interpolation function (from a lookup table)
            instead of recalculating rho(m)
        table: Sersic profile table, optional

    Returns
    -------
        g_R: float
            g_R(R,z)  -- gravitational force in the radial direction; units [km^2/s^2/kpc]

    """

    cnst = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value) # cmtokpc * Msuntog * kmtocm^2
    int_force_R, _ = scp_integrate.quad(force_R_integrand_z0, 0, R,
                                       args=(R, 0., Reff, n, q, Ie, i, Upsilon,
                                             logrhom_interp, table, total_mass))
    forceRz0 = (-4.*np.pi*G.cgs.value*q*cnst / R) * int_force_R
    if R == 0:
        forceRz0 = 0.
    return forceRz0


def force_z_R0(R, z, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.,
               logrhom_interp=None, table=None, total_mass=None):
    """
    Evalutation of gravitational force in the vertical direction
    :math:`g_z=-\partial\Phi/\partial z`,
    of a deprojected Sersic density profile at (R,z), by numerically evalutating

    .. math::

        g_z(R,z) = - 2\pi Gqz \int_0^{\infty} d\tau \frac{\rho(m)}{(\tau+1)(tau+q^2)^{3/2}},
        m = \frac{R^2}{\tau+1} + \frac{z^2}{\tau+q^2}

    Parameters
    ----------
        R: float
            Midplane radius [kpc]
        z: float
            Height above midplane [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        q: float
            Intrinsic axis ratio of Sersic profile
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

        Upsilon: float or array_like, optional
            Mass-to-light ratio. Default: 1. (i.e., constant ratio)
        logrhom_interp: function, optional
            Shortcut to use an interpolation function (from a lookup table)
            instead of recalculating rho(m)
        table: Sersic profile table, optional

    Returns
    -------
        g_z: float
            g_z(r,z)  -- gravitational force in the vertial direction; units [km^2/s^2/kpc]

    """

    cnst = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value) # cmtokpc * Msuntog * kmtocm^2
    int_force_z, _ = scp_integrate.quad(force_z_integrand_R0, 0, z/(q*1.),
                                       args=(0., z, Reff, n, q, Ie, i, Upsilon,
                                             logrhom_interp, table, total_mass))
    # if z == 0.:
    #     int_force_z = 0.

    return -4.*np.pi*G.cgs.value*q* cnst * int_force_z
