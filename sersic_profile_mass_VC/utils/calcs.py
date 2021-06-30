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
import astropy.cosmology as apy_cosmo

import logging

__all__ = [ 'vcirc_spherical_symmetry', 'menc_spherical_symmetry',
            'virial_coeff_tot', 'virial_coeff_3D',
            'find_rhalf3D_sphere',
            'check_for_inf' ]

# CONSTANTS
G = apy_con.G
Msun = apy_con.M_sun
pc = apy_con.pc
deg2rad = np.pi/180.

# DEFAULT COSMOLOGY
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)

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

    for i, r in enumerate(table['r']):
        if not np.isfinite(table['vcirc'][i]): status += 1
        if not np.isfinite(table['menc3D_sph'][i]): status += 1
        if not np.isfinite(table['menc3D_ellipsoid'][i]): status += 1

    return status


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# Calculation helper functions: General for mass distributions




def vcirc_spherical_symmetry(r=None, menc=None):
    """
    Determine vcirc for a spherically symmetric mass distribution:

    .. math::

        v_{\mathrm{circ}}(r) = \sqrt{\\frac{G M_{\mathrm{enc}}(r)}{r}}

    Parameters
    ----------
        r: float or array_like
            Radi[us/i] at which to calculate the circular velocity [kpc]
        menc: float or array_like
            Enclosed mass at the given radii  [Msun]

    Returns
    -------
        vcirc: float or array_like
            Circular velocity as a function of radius  [km/s]

    """
    vcirc = np.sqrt(G.cgs.value * menc * Msun.cgs.value / (r * 1000. * pc.cgs.value)) / 1.e5

    # -------------------------
    # Test for 0:
    try:
        if len(r) >= 1:
            vcirc[np.array(r) == 0.] = 0.
    except:
        if r == 0.:
            vcirc = 0.
    # -------------------------

    return vcirc

def menc_spherical_symmetry(r=None, vcirc=None):
    """
    Determine Menc for a spherically symmetric mass distribution, given vcirc:
        Menc(r) = vcirc(r)^2 * r / G

    .. math::

        M_{\mathrm{enc}}(r) = \\frac{v_{\mathrm{circ}}(r)^2 r}{G}

    Parameters
    ----------
        r: float or array_like
            Radi[us/i] at which to calculate the circular velocity [kpc]
        vcirc: float or array_like
            Circular velocity at the given radii  [km/s]

    Returns
    -------
        menc: float or array_like
            Enclosed mass as a function of radius  [Msun]

    """
    menc = ((vcirc*1.e5)**2.*(r*1000.*pc.cgs.value) / (G.cgs.value * Msun.cgs.value))
    return menc



# +++++++++++++++++++++++++++++++++++++++++++++++++

def virial_coeff_tot(r, total_mass=1., vc=None):
    """
    Evalutation of the "total" virial coefficient ktot, which satisfies

    .. math::

        M_{\mathrm{tot}} = k_{\mathrm{tot}}(r) \\frac{v_{\mathrm{circ}}(r)^2 r}{ G },

    to convert between the circular velocity at any given radius and the total system mass.

    Parameters
    ----------
        r: float or array_like
            Major axis radius within which to determine total enclosed 2D projected mass [kpc]
        total_mass: float
            Total mass of the component [Msun]
        vc: float or array_like
            Pre-calculated evaluation of vcirc(r)
            (saves time to avoid recalculating vcirc(r))  [km/s]

    Returns
    -------
        ktot: float or array_like
            ktot = Mtot * G / (vcirc(r)^2 * r)

    """

    # need to convert to cgs:
    # units: Mass: msun
    #        r:    kpc
    #        v:    km/s
    ktot = (total_mass * Msun.cgs.value) * G.cgs.value / (( r*1.e3*pc.cgs.value ) * (vc*1.e5)**2)

    return ktot


def virial_coeff_3D(r, m3D=None, vc=None):
    """
    Evalutation of the "total" virial coefficient ktot, which satisfies

    .. math::

        M_{\mathrm{3D,sphere}} = k_{\mathrm{3D}}(r) \\frac{v_{\mathrm{circ}}(r)^2 r}{ G },

    to convert between the circular velocity at any given radius
    and the mass enclosed within a sphere of radius r.

    Parameters
    ----------
        r: float or array_like
            Major axis radius within which to determine total enclosed 2D projected mass [kpc]
        m3D: float or array_like
            Pre-calculated evaluation of Menc3D_sphere(r)
            (saves time to avoid recalculating Menc3D_sphere(r)) [Msun]
        vc: float or array_like
            Pre-calculated evaluation of vcirc(r)
            (saves time to avoid recalculating vcirc(r))  [km/s]

    Returns
    -------
        k3D: float or array_like
            k3D = Menc3D_sphere(r) * G / (vcirc(r)^2 * r)

    """
    k3D = (m3D * Msun.cgs.value) * G.cgs.value / (( r*1.e3*pc.cgs.value ) * (vc*1.e5)**2)

    return k3D


# +++++++++++++++++++++++++++++++++++++++++++++++++

def find_rhalf3D_sphere(r=None, menc3D_sph=None, total_mass=None):
    """
    Evalutation of the radius corresponding to the sphere that
    encloses half of the total mass for a Sersic profile of a given
    intrinsic axis ratio, effective radius, and Sersic index.

    This is a utility function, where the Menc3D_sphere must have been pre-calculated.

    Performs an interpolation to find the appropriate rhalf_sph,
    given arrays r and menc3D_sph.

    Parameters
    ----------
        r: array_like
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

    r_interp = scp_interp.interp1d(menc3D_sph, r, fill_value=np.NaN, bounds_error=False, kind='slinear')
    rhalf_sph = r_interp( 0.5 * total_mass )
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

def get_Ie(total_mass=1., Reff=1., n=1., q=0.4, i=90., Upsilon = 1.):
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


def rho_m(m, Reff=1., n=1., q=0.4, Ie=1., i=90., Upsilon = 1.):
    """
    Evalutation of Sersic density profile at radius m.

    Parameters
    ----------
        m: float or array_like
            Radius at which to evaluate rho(m)
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
    qobstoqint = np.sqrt(np.sin(i*deg2rad)**2 + 1./q**2 * np.cos(i*deg2rad)**2 )

    # Evalutate inner integral:
    #   Int_(kap=m)^(inifinty) dkap [dI/dkap * 1/sqrt(kap^2 - m^2)]
    int_rho_m_inner, _ = scp_integrate.quad(rho_m_integrand, m, np.inf, args=(m, Reff, n, Ie))

    rhom = -(Upsilon/np.pi)*( qobstoqint ) * int_rho_m_inner

    return rhom


def vel_integrand(m, r, Reff, n, q, Ie, i):
    """
    Integrand rho(m) * m^2 / sqrt(r^2 - (1-qint^2) * m^2) as part of numerical integration to find vcirc(r)

    Parameters
    ----------
        m: float
            independent variable (radial)
        r: float
            Radius at which to find vcirc(r)
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

    Returns
    -------
        integrand: float or array_like
            rho(m) * m^2 / sqrt(r^2 - (1-qint^2) * m^2)

    """
    integ = rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i) * ( m**2 / np.sqrt(r**2 - m**2 * (1.- q**2)) )
    return integ

def vel_integral(r, Reff=1., n=1., q=0.4, Ie=1., i=90.):
    """
    Evalutation of integrand rho(m) * m^2 / sqrt(r^2 - (1-qint^2) * m^2) from m=0 to r,
    as part of numerical integration to find vcirc(r)

    Parameters
    ----------
        r: float or array_like
            Radius at which to find vcirc(r)
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

    Returns
    -------
        integral: float or array_like
            Int_(m=0)^(r) dm [rho(m) * m^2 / sqrt(r^2 - (1-qint^2) * m^2)]

    """
    # integrate outer from m=0 to r
    intgrl, _ = scp_integrate.quad(vel_integrand, 0, r, args=(r, Reff, n, q, Ie, i))
    return intgrl

def total_mass3D_integrand_ellipsoid(m, Reff, n, q, Ie, i):
    """
    Integrand m^2 * rho(m)  as part of numerical integration to find M_3D,ellipsoid(<r)

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

    Returns
    -------
        integrand: float or array_like
            m^2 * rho(m)

    """

    integ =  m**2 * rho_m(m, Reff=Reff, n=n, q=q, Ie=Ie, i=i)
    return integ


def total_mass3D_integral_ellipsoid(r, Reff=1., n=1., q=0.4, Ie=1.,i=90., rinner=0.):
    """
    Evalutation of integrand m^2 * rho(m) from m=0 to r,
    as part of numerical integration to find M_3D_ellipsoid(<r)

    Parameters
    ----------
        r: float or array_like
            Radius at which to find vcirc(r)
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

        rinner: float, optional
            Calculate radius in annulus instead of sphere, using rinner>0. [kpc]. Default: 0.

    Returns
    -------
        integral: float or array_like
            Int_(m=0)^(r) dm [m^2 * rho(m)]

    """
    ## In ellipsoids:

    # integrate from m=0 to r
    intgrl, _ = scp_integrate.quad(total_mass3D_integrand_ellipsoid, rinner, r,
                                   args=(Reff, n, q, Ie, i))
    return 4.*np.pi*q*intgrl


def total_mass3D_integrand_sph_z(z, m, Reff, n, q, Ie, i):
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

    Returns
    -------
        integrand: float or array_like
            rho(sqrt(m^2 + (z/qintr)^2)

    """

    mm = np.sqrt(m**2 + z**2/q**2)
    integ =  rho_m(mm, Reff=Reff, n=n, q=q, Ie=Ie, i=i)
    return integ

def total_mass3D_integral_z(m, r=None, Reff=1., n=1., q=0.4, Ie=1.,  i=90., rinner=None):
    """
    Evalutation of integrand 2 * rho(sqrt(m^2 + (z/qintr)^2) from z=0 to sqrt(r^2-m^2), [eg both pos and neg z]
    as part of numerical integration to find mass enclosed in sphere
    (or over the shell corresponding to rinner...)

    Parameters
    ----------
        m: float or array_like
            Radius at which to evaluate integrand; cylindrical coordinate radius
        r: float or array_like
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

        rinner: float, optional
            Inner radius of total spherical shell, if only calculating mass
            in a spherical shell. Default: rinner = 0. (eg the entire sphere out to r)

    Returns
    -------
        integral: float or array_like
            Int_(z=0)^(sqrt(r^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)]

    """
    lim = np.sqrt(r**2 - m**2)
    if rinner > 0.:
        if m < rinner:
            lim_inner = np.sqrt(rinner**2 - m**2)
        else:
            # this is the part of the vertical slice where m is greater than rinner, outside the inner shell part,
            #       so it goes from z=0 to z=sqrt(r^2-m^2).
            lim_inner = 0.
    else:
        lim_inner = 0.
    # symmetric about z:
    intgrl, abserr = scp_integrate.quad(total_mass3D_integrand_sph_z, lim_inner, lim,
                                        args=(m, Reff, n, q, Ie, i))

    # ---------------------
    # Catch some numerical integration errors which happen at very small values of m --
    #       set these to 0., as this is roughly correct
    if intgrl < 0.:
        if np.abs(m) > 1.e-6:
            print('m={}, r={}, zlim={}'.format(m, r, lim))
            raise ValueError
        else:
            # Numerical error:
            intgrl = 0.

    # ---------------------
    # Integral is symmetric about z, so return times 2 for pos and neg z.
    return 2.*intgrl

def total_mass3D_integrand_r(m, r, Reff, n, q, Ie, i, rinner):
    """
    Integrand m * [ Int_(z=0)^(sqrt(r^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]
    as part of numerical integration to find mass enclosed in sphere.

    Parameters
    ----------
        m: float or array_like
            cylindrical coordinates radius m
        r: float or array_like
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
        rinner: float
            Inner radius of total spherical shell, if only calculating mass in a spherical shell [kpc]

    Returns
    -------
        integrand: float or array_like
            m * [ Int_(z=0)^(sqrt(r^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]

    """

    integ = total_mass3D_integral_z(m, r=r, Reff=Reff, n=n, q=q, Ie=Ie, i=i, rinner=rinner)
    return m * integ


def total_mass3D_integral(r, Reff=1., n=1., q=0.4, Ie=1., i=90., rinner=0.):
    """
    Evalutation of integrand 2 * pi * m * [ Int_(z=0)^(sqrt(r^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]
    from m=rinner to r, as part of numerical integration to find mass enclosed in sphere
    (or over the shell corresponding to rinner...)

    Parameters
    ----------
        r: float or array_like
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

        rinner: float, optional
            Inner radius of total spherical shell, if only calculating mass
            in a spherical shell. Default: rinner = 0. (eg the entire sphere out to r)

    Returns
    -------
        integral: float or array_like
            Int_(m=0)^(r) dm * 2 * pi * m * [ Int_(z=0)^(sqrt(r^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]

    """
    # in *SPHERE*
    intgrl, abserr = scp_integrate.quad(total_mass3D_integrand_r, 0., r,
                                        args=(r, Reff, n, q, Ie, i, rinner))
    return 2*np.pi*intgrl




def total_mass2D_direct(r, total_mass=1., Reff=1., n=1., q=0.4, i=90., rinner=0.):
    """
    Evalutation of the 2D projected mass enclosed within an ellipse
    (or elliptical shell), assuming a constant M/L ratio Upsilon.

    Parameters
    ----------
        r: float or array_like
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

        rinner: float, optional
            Inner radius of total spherical shell, if only calculating mass
            in a spherical shell. Default: rinner = 0. (eg the entire sphere out to r)

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
    integ = scp_spec.gammainc(2 * n, bn * np.power(r / Reff, 1./n) )
    if rinner > 0.:
        integinner = scp_spec.gammainc(2 * n, bn * np.power(rinner / Reff, 1./n) )
        integ -= integinner

    return total_mass*integ

####################


####################

def lnrho_m(lnm, Reff, n, q, Ie, i):
    """
    Log density profile, :math:`\ln\rho`
    at distance :math:`r=m` of a Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        lnm: float
            Ln midplane radius at which to evaluate the circular velocity [ln kpc]
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

    Returns
    -------
        lnrhom: float
            Log density profile at r=m

    """
    rhom = rho_m(np.exp(lnm), Reff=Reff, n=n, q=q, Ie=Ie, i=i)
    return np.log(rhom)

def dlnrhom_dlnr(lnm, Reff=1., n=1., q=0.4, Ie=1., i=90., dx=1.e-5, order=3):
    """
    Evalutation of the slope of the density profile, :math:`d\ln\rho/d\ln{}r`,
    at distance :math:`r=m` of a Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        lnm: float
            Ln of midplane radius at which to evaluate the circular velocity [kpc]
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

    Returns
    -------
        dlnrho_dlnr: float
            Derivative of log density profile at r=m

    """
    deriv = scp_misc.derivative(lnrho_m, lnm, args=(Reff, n, q, Ie, i), dx=dx, n=1, order=order)
    return deriv
