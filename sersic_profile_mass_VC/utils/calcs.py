##################################################################################
# sersic_profile_mass_VC/utils/calcs.py                                          #
#                                                                                #
# Copyright 2018-2021 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import os

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

from sersic_profile_mass_VC.utils.interp_profiles import interpolate_sersic_profile_rho

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
logger = logging.getLogger('SersicProfileMassVC')

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

def check_for_inf(table=None):
    """
    Check core table quantities for non-finite entries, to determine if numerical errors occured.
    """
    status = 0

    keys = ['vcirc', 'menc3D_sph', 'menc3D_ellipsoid', 'rho', 'dlnrho_dlnR']

    for i, R in enumerate(table['R']):
        for key in keys:
            if not np.isfinite(table[key][i]):
                # Check special case: dlnrho_dlnR -- Leibniz uses r/rho*drho/dr, so ignore NaN if rho=0.
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
    """
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
    """
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
    """
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
    """
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
    """
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
    return np.float(rhalf_sph)




# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Calculation helper functions: Seric profiles, mass distributions

def bn_func(n):
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
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
    """
    Evalutation of the 2D projected mass enclosed within an ellipse
    (or elliptical shell), assuming a constant M/L ratio Upsilon.

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
    """
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
    """
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
    """
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
    """
    Integrand for :math:`d\rho(m)/dm` -- as a function of x, u. Will integrate over x.

    """

    v = np.sqrt(x**2 + u**2)

    aa = np.exp(-bn*(np.power(v, 1./n)-1.))
    bb = np.power(v, (1./n - 4.))
    cc = (1./n - 2. - bn/n * np.power(v, 1./n))

    drhoIdu_intgrnd = aa * bb * cc

    return drhoIdu_intgrnd


def drhom_dm_leibniz(m, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.):
    """
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
    """
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
    """
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
    """
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



def force_R_integrand(tau, R, z, Reff, n, q, Ie, i, Upsilon):
    """
    Integrand :math:`\frac{\rho(m)R}{(\tau+1)^2 \sqrt{tau+q^2}}`
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

    Returns
    -------
        integrand: float

    """
    m = np.sqrt(R**2/(tau+1.) + z**2/(tau+q**2))
    rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)

    integrand = rho * R / ( (tau+1.)**2 * np.sqrt(tau + q**2) )

    return integrand


def force_z_integrand(tau, R, z, Reff, n, q, Ie, i, Upsilon):
    """
    Integrand :math:`\frac{\rho(m)z}{(\tau+1)(tau+q^2)^{3/2}}`
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

    Returns
    -------
        integrand: float

    """
    m = np.sqrt(R**2/(tau+1.) + z**2/(tau+q**2))
    rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)

    integrand = rho * z / ( (tau+1.) * np.power((tau + q**2), 3./2.) )

    return integrand


def force_R(R, z, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.):
    """
    Evalutation of gravitational force in the radial direction
    :math:`g_R=-\partial\Phi/\partial R`,
    of a deprojected Sersic density profile at (R,z), by numerically evalutating

    .. math::

        g_R(R,z) = - 2\pi Gq\int_0^{\infty} d\tau \frac{\rho(m)R}{(\tau+1)^2 \sqrt{tau+q^2}},
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

    Returns
    -------
        g_R: float
            g_R(R,z)  -- gravitational force in the radial direction; units [km^2/s^2/kpc]

    """

    cnst = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value) # cmtokpc * Msuntog * kmtocm^2
    int_force_R, _ = scp_integrate.quad(force_R_integrand, 0, np.inf,
                                       args=(R, z, Reff, n, q, Ie, i, Upsilon))
    return -2.*np.pi*G.cgs.value*q*cnst * int_force_R

def force_z(R, z, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.):
    """
    Evalutation of gravitational force in the vertical direction
    :math:`g_z=-\partial\Phi/\partial z`,
    of a deprojected Sersic density profile at (R,z), by numerically evalutating

    .. math::

        g_z(R,z) = - 2\pi Gq\int_0^{\infty} d\tau \frac{\rho(m)z}{(\tau+1)(tau+q^2)^{3/2}},
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

    Returns
    -------
        g_z: float
            g_z(r,z)  -- gravitational force in the vertial direction; units [km^2/s^2/kpc]

    """

    cnst = Msun.cgs.value*1.e-10/(1000.*pc.cgs.value) # cmtokpc * Msuntog * kmtocm^2
    int_force_z, _ = scp_integrate.quad(force_z_integrand, 0, np.inf,
                                       args=(R, z, Reff, n, q, Ie, i, Upsilon))
    return -2.*np.pi*G.cgs.value*q*cnst * int_force_z


def sigsq_z_integrand(z, R, total_mass, Reff, n, q, Ie, i, Upsilon, sersic_table):
    """
    Integrand as part of numerical integration to find :math:`\sigma^2_z(r,z)`

    Parameters
    ----------
        z: float
            Height above midplane [kpc]
        R: float
            Midplane radius [kpc]
        total_mass: float
            Total mass of the Sersic mass component [Msun]
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
        sersic_table: dictionary or None
            Use pre-computed table to create an interpolation function
            that is used for this calculation.

    Returns
    -------
        integrand: float

    """
    m = np.sqrt(R**2 + (z/q)**2)
    if sersic_table is not None:
        rho = interpolate_sersic_profile_rho(R=m, total_mass=total_mass, Reff=Reff, n=n, invq=1./q,
                                             table=sersic_table)
    else:
        rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
    gz = force_z(R, z,  Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
    integrand = rho * gz
    return integrand


def sigmaz_sq(R, z, total_mass=1., Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon=1.,
              sersic_table=None):
    """
    Evalutation of vertical velocity dispersion of a deprojected Sersic density profile at (R,z).

    .. math::

        \sigma^2_z(R,z)=-\frac{1}{\rho}\int \rho g_z(R,z)dz

    Parameters
    ----------
        R: float
            Midplane radius [kpc]
        z: float
            Height above midplane [kpc]
        total_mass: float
            Total mass of the Sersic mass component [Msun]
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
        sersic_table: dictionary, optional
            Use pre-computed table to create an interpolation function
            that is used for this calculation.

    Returns
    -------
        sigzsq: float
            Vertical velocity dispersion direction; units [km^2/s^2]

    """
    m = np.sqrt(R**2 + (z/q)**2)
    if sersic_table is not None:
        rho = interpolate_sersic_profile_rho(R=m, total_mass=total_mass, Reff=Reff, n=n, invq=1./q,
                                             table=sersic_table)
    else:
        rho = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i, Upsilon=Upsilon)
    int_sigsq_z, _ = scp_integrate.quad(sigsq_z_integrand, 0, z,
                                        args=(R, total_mass, Reff, n, q, Ie, i,
                                              Upsilon, sersic_table))
    return - 1./rho * int_sigsq_z
