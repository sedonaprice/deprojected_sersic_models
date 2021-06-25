##################################################################################
# sersic_profile_mass_VC/calcs.py                                                #
#                                                                                #
# Copyright 2018-2021 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import numpy as np
import scipy.integrate as scp_integrate
import scipy.misc as scp_misc
import scipy.special as scp_spec
import scipy.interpolate as scp_interp
import astropy.constants as apy_con
import astropy.units as u
import astropy.cosmology as apy_cosmo

import logging

from sersic_profile_mass_VC.table_io import read_profile_table

__all__ = [ 'v_circ', 'M_encl_2D', 'M_encl_3D', 'M_encl_3D_ellip',
            'virial_coeff_tot', 'virial_coeff_3D',
            'interpolate_sersic_profile_menc', 'interpolate_sersic_profile_VC',
            'interpolate_sersic_profile_rho', 'interpolate_sersic_profile_alpha',
            'interpolate_sersic_profile_menc_nearest', 'interpolate_sersic_profile_VC_nearest',
            'interpolate_sersic_profile_alpha_nearest',
            'interpolate_sersic_profile_alpha_bulge_disk_nearest',
            'find_rhalf3D_sphere', 'nearest_n_invq', 'qobs_func']

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

def Ikap(kap, n=1.,  Ie=1., Reff=1.):
    """
    Intensity(kappa) for a Sersic profile

    Parameters
    ----------
        kap: float or array_like
            Radius for calculation of Sersic profile (along the major axis)

        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        Reff: float
            Effective radius of Sersic profile [kpc]

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

def dIdkap(kap, n=1., Ie=1., Reff=1.):
    """
    Derivative d(Intensity(kappa))/dkappa for a Sersic profile

    Parameters
    ----------
        kap: float or array_like
            radius for calculation of Sersic profile (along the major axis)

        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        Reff: float
            Effective radius of Sersic profile [kpc]

    Returns
    -------
        dIdk: float or array_like
            Derivative of intensity of Sersic profile at kap

    """
    bn = bn_func(n)
    dIdk = - (Ie*bn)/(n*Reff) * np.exp( -bn * (np.power(kap/Reff, 1./n) - 1.) ) * np.power(kap/Reff, (1./n) - 1.)

    return dIdk

def rho_m_integrand(kap, m, n, Ie, Reff):
    """
    Integrand dI/dkap * 1/sqrt(kap^2 - m^2) as part of
    numerical integration to find rho(m)

    Parameters
    ----------
        kap: float or array_like
            independent variable (radius)
        m: float
            Radius at which to evaluate rho(m)
        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        Reff: float
            Effective radius of Sersic profile [kpc]

    Returns
    -------
        integrand: float or array_like
            Integrand dI/dkap * 1/sqrt(kap^2 - m^2)

    """
    integ = dIdkap(kap, n=n, Ie=Ie, Reff=Reff) * 1./np.sqrt(kap**2 - m**2)
    return integ

def rho_m_integral(m, Reff=1., n=1., Ie=1.):
    """
    Evalutation of integrand dI/dkap * 1/sqrt(kap^2 - m^2) from kap=m to infinity,
    as part of numerical integration to find rho(m)

    Parameters
    ----------
        m: float or array_like
            Radius at which to evaluate rho(m)

        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        Reff: float
            Effective radius of Sersic profile [kpc]

    Returns
    -------
        integral: float or array_like
            Int_(kap=m)^(inifinty) dkap [dI/dkap * 1/sqrt(kap^2 - m^2)]

    """
    # integrate inner from kap=m to inf
    intgrl, abserr = scp_integrate.quad(rho_m_integrand, m, np.inf, args=(m, n, Ie, Reff))
    return intgrl

def rho_m(m, n=1., Ie=1., Reff=1., q=0.4, i=90., Upsilon = 1.):
    """
    Evalutation of Sersic density profile at radius m.

    Parameters
    ----------
        m: float or array_like
            Radius at which to evaluate rho(m)

        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        Reff: float
            Effective radius of Sersic profile [kpc]
        q: float
            Intrinsic axis ratio of Sersic profile
        i: float
            Inclination of system [deg]

    Returns
    -------
        rhom: float or array_like
            rho(m)  -- 3D density of Sersic profile at radius m.

    """
    # uses rho_m_integrand
    qobstoqint = np.sqrt(np.sin(i*deg2rad)**2 + 1./q**2 * np.cos(i*deg2rad)**2 )
    rhom = -(Upsilon/np.pi)*( qobstoqint ) * rho_m_integral(m, Reff=Reff, n=n, Ie=Ie)
    return rhom

def vel_integrand(m, r, n, Ie, Reff, q, i):
    """
    Integrand rho(m) * m^2 / sqrt(r^2 - (1-qint^2) * m^2) as part of numerical integration to find vcirc(r)

    Parameters
    ----------
        m: float
            independent variable (radial)
        r: float
            Radius at which to find vcirc(r)
        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        Reff: float
            Effective radius of Sersic profile [kpc]
        q: float
            Intrinsic axis ratio of Sersic profile
        i: float
            Inclination of system [deg]

    Returns
    -------
        integrand: float or array_like
            rho(m) * m^2 / sqrt(r^2 - (1-qint^2) * m^2)

    """

    integ = rho_m(m, Reff=Reff, n=n, Ie=Ie, i=i, q=q) * ( m**2 / np.sqrt(r**2 - m**2 * (1.- q**2)) )
    return integ

def vel_integral(r, q=0.4, Reff=1., n=1., Ie=1., i=90.):
    """
    Evalutation of integrand rho(m) * m^2 / sqrt(r^2 - (1-qint^2) * m^2) from m=0 to r,
    as part of numerical integration to find vcirc(r)

    Parameters
    ----------
        r: float or array_like
            Radius at which to find vcirc(r)

        q: float
            Intrinsic axis ratio of Sersic profile
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
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
    intgrl, abserr = scp_integrate.quad(vel_integrand, 0, r, args=(r, n, Ie, Reff, q, i))
    return intgrl

def total_mass3D_integrand_ellipsoid(m, n, Ie, Reff, q, i):
    """
    Integrand m^2 * rho(m)  as part of numerical integration to find M_3D,ellipsoid(<r)

    Parameters
    ----------
        m: float or array_like
            independent variable (radial)
        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        Reff: float
            Effective radius of Sersic profile [kpc]
        q: float
            Intrinsic axis ratio of Sersic profile
        i: float
            Inclination of system [deg]

    Returns
    -------
        integrand: float or array_like
            m^2 * rho(m)

    """

    integ =  m**2 * rho_m(m, n=n,  Ie=Ie, Reff=Reff, q=q, i=i)
    return integ


def total_mass3D_integral_ellipsoid(r, rinner=0., q=0.4, n=1., Reff=1., Ie=1.,i=90.):
    """
    Evalutation of integrand m^2 * rho(m) from m=0 to r,
    as part of numerical integration to find M_3D_ellipsoid(<r)

    Parameters
    ----------
        r: float or array_like
            Radius at which to find vcirc(r)

        q: float
            Intrinsic axis ratio of Sersic profile
        n: float
            Sersic index
        Reff: float
            Effective radius of Sersic profile [kpc]
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        i: float
            Inclination of system [deg]

    Returns
    -------
        integral: float or array_like
            Int_(m=0)^(r) dm [m^2 * rho(m)]

    """
    ## In ellipsoids:

    # integrate from m=0 to r
    intgrl, abserr = scp_integrate.quad(total_mass3D_integrand_ellipsoid, rinner, r, args=(n, Ie, Reff, q, i))
    return 4.*np.pi*q*intgrl


def total_mass3D_integrand_sph_z(z, m, n, Ie, Reff, q, i):
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
        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        Reff: float
            Effective radius of Sersic profile [kpc]
        q: float
            Intrinsic axis ratio of Sersic profile
        i: float
            Inclination of system [deg]

    Returns
    -------
        integrand: float or array_like
            rho(sqrt(m^2 + (z/qintr)^2)

    """

    mm = np.sqrt(m**2 + z**2/q**2)
    integ =  rho_m(mm, n=n,  Ie=Ie, Reff=Reff, q=q, i=i)
    return integ

def total_mass3D_integral_z(m, r=None, q=0.4, n=1., Reff=1., Ie=1.,  i=90., rinner=None):
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
        q: float
            Intrinsic axis ratio of Sersic profile
        n: float
            Sersic index
        Reff: float
            Effective radius of Sersic profile [kcp]
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
    intgrl, abserr = scp_integrate.quad(total_mass3D_integrand_sph_z, lim_inner, lim, args=(m, n, Ie, Reff, q, i))

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

def total_mass3D_integrand_r(m, r, n, Ie, Reff, q, i, rinner):
    """
    Integrand m * [ Int_(z=0)^(sqrt(r^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]
    as part of numerical integration to find mass enclosed in sphere.

    Parameters
    ----------
        m: float or array_like
            cylindrical coordinates radius m
        r: float or array_like
            Radius of sphere over which to be calculating total enclosed mass [kpc]
        n: float
            Sersic index
        Ie: float
            Normalization of Sersic intensity profile at kap = Reff
        Reff: float
            Effective radius of Sersic profile [kpc]
        q: float
            Intrinsic axis ratio of Sersic profile
        i: float
            Inclination of system [deg]
        rinner: float
            Inner radius of total spherical shell, if only calculating mass in a spherical shell [kpc]

    Returns
    -------
        integrand: float or array_like
            m * [ Int_(z=0)^(sqrt(r^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]

    """

    integ = total_mass3D_integral_z(m, r=r , n=n,  Ie=Ie, Reff=Reff, q=q, i=i, rinner=rinner)
    return m * integ


def total_mass3D_integral(r, rinner=0., q=0.4, n=1., Reff=1., Ie=1., i=90.):
    """
    Evalutation of integrand 2 * pi * m * [ Int_(z=0)^(sqrt(r^2-m^2)) dz * 2 * [rho(sqrt(m^2 + (z/qintr)^2)] ]
    from m=rinner to r, as part of numerical integration to find mass enclosed in sphere
    (or over the shell corresponding to rinner...)

    Parameters
    ----------
        r: float or array_like
            Radius of sphere over which to be calculating total enclosed mass [kpc]

        q: float
            Intrinsic axis ratio of Sersic profile
        n: float
            Sersic index
        Reff: float
            Effective radius of Sersic profile [kpc]
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
    intgrl, abserr = scp_integrate.quad(total_mass3D_integrand_r, 0., r, args=(r, n, Ie, Reff, q, i, rinner))
    return 2*np.pi*intgrl


def get_Ie(total_mass=1., q=0.4, n=1., Reff=1., i=90., Upsilon = 1.):
    """
    Evalutation of Ie, normalization of the Sersic intensity profile at kap = Reff,
    using the total mass (to infinity) and assuming a constant M/L ratio Upsilon.

    Uses the closed-form solution for the total luminosity of the
    2D projected Sersic intensity profile I(kap).

    Parameters
    ----------
        total_mass: float
            Total mass of the component [Msun]
        q: float
            Intrinsic axis ratio of Sersic profile
        n: float
            Sersic index
        Reff: float
            Effective radius of Sersic profile [kpc]
        i: float
            Inclination of system [deg]

    Returns
    -------
        Ie: float
            Ie = I(kap=Reff)

    """

    bn = bn_func(n)
    qobs = qobs_func(q=q, i=i)

    # This is Ie, using Upsilon = 1 [cnst M/L].
    Ie = (total_mass * np.power(bn, 2.*n)) / ( Upsilon * 2.*np.pi* qobs * Reff**2 * n * np.exp(bn) * scp_spec.gamma(2.*n) )

    return Ie


def total_mass2D_direct(r, total_mass=1., q=0.4, n=1., Reff=1., i=90.,rinner=0. ):
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
        q: float
            Intrinsic axis ratio of Sersic profile
        n: float
            Sersic index
        Reff: float
            Effective radius of Sersic profile [kpc]
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

    ## Within self-similar ellipses of ratio qobs = qobs_func(q=q, i=i)

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

####################

def lnrho_m(lnm, Ie, q, Reff, n, i):
    """
    Log density profile, :math:`\ln\rho`
    at distance :math:`r=m` of a Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        lnm: float
            Ln midplane radius at which to evaluate the circular velocity [ln kpc]
        Ie: float
            Normalization of Sersic profile
        q: float
            Intrinsic axis ratio of Sersic profile
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        i: float
            Inclination of system [deg]

    Returns
    -------
        lnrhom: float
            Log density profile at r=m

    """
    rhom = rho_m(np.exp(lnm), q=q, Reff=Reff, Ie=Ie, n=n, i=i)
    return np.log(rhom)

def dlnrhom_dlnr(lnm, Ie=1., q=0.4, Reff=1., n=1., i=90., dx=1.e-5, order=3):
    """
    Evalutation of the slope of the density profile, :math:`d\ln\rho/d\ln{}r`,
    at distance :math:`r=m` of a Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        r: float
            Midplane radius at which to evaluate the circular velocity [kpc]
        Ie: float
            Normalization of Sersic profile
        q: float
            Intrinsic axis ratio of Sersic profile
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        i: float
            Inclination of system [deg]

    Returns
    -------
        dlnrho_dlnr: float
            Derivative of log density profile at r=m

    """
    deriv = scp_misc.derivative(lnrho_m, lnm, args=(Ie, q, Reff, n, i), dx=dx, n=1, order=order)

    return deriv

def dlnrho_dlnr(r, total_mass=1., q=0.4, Reff=1., n=1., i=90.):
    """
    Evalutation of the slope of the density profile, :math:`d\ln\rho/d\ln{}r`,
    at distance :math:`r=m` of a Sersic mass distribution
    with intrinsic axis ratio q.

    Parameters
    ----------
        r: float or array_like
            Midplane radius at which to evaluate the circular velocity [kpc]
        total_mass: float
            Total mass of the component [Msun]
        q: float
            Intrinsic axis ratio of Sersic profile
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        i: float
            Inclination of system [deg]

    Returns
    -------
        dlnrho_dlnr_arr: float or array_like
            Derivative of log density profile at r=m

    """
    lnr = np.log(r)

    Ie = get_Ie(total_mass=total_mass, q=q, n=n, Reff=Reff, i=i)

    try:
        if len(r) > 0:
            dlnrho_dlnr_arr = np.zeros(len(r))
            for j in range(len(r)):
                dlnrho_dlnr_arr[j] = dlnrhom_dlnr(lnr[j], q=q, Reff=Reff, Ie=Ie, n=n, i=i)
        else:
            dlnrho_dlnr_arr = dlnrhom_dlnr(lnr[0], q=q, Reff=Reff, Ie=Ie, n=n, i=i)
    except:
        dlnrho_dlnr_arr = dlnrhom_dlnr(lnr, q=q, Reff=Reff, Ie=Ie, n=n, i=i)

    return dlnrho_dlnr_arr
# +++++++++++++++++++++++++++++++++++++++++++++++++



def rho(r, total_mass=1., q=0.4, Reff=1., n=1., i=90.):
    """
    Evalutation of the density profile (at distance r=m) of a Sersic mass distribution with intrinsic axis ratio q.

    Parameters
    ----------
        r: float or array_like
            Midplane radius at which to evaluate the circular velocity [kpc]
        total_mass: float
            Total mass of the component [Msun]
        q: float
            Intrinsic axis ratio of Sersic profile
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        i: float
            Inclination of system [deg]

    Returns
    -------
        rho_arr: float or array_like
            Density profile at r=m [Msun / kpc^3]

    """

    Ie = get_Ie(total_mass=total_mass, q=q, n=n, Reff=Reff, i=i)

    try:
        if len(r) > 0:
            rho_arr = np.zeros(len(r))
            for j in range(len(r)):
                rho_arr[j] = rho_m(r[j], q=q, Reff=Reff, Ie=Ie, n=n, i=i)
        else:
            rho_arr = rho_m(r[0], q=q, Reff=Reff, Ie=Ie, n=n, i=i)
    except:
        rho_arr = rho_m(r, q=q, Reff=Reff, Ie=Ie, n=n, i=i)

    return rho_arr

def v_circ(r, total_mass=1., q=0.4, Reff=1., n=1.,i=90.):
    """
    Evalutation of the circular velocity in the midplane of a
    Sersic mass distribution with intrinsic axis ratio q.

    Parameters
    ----------
        r: float or array_like
            Midplane radius at which to evaluate the circular velocity [kpc]

        total_mass: float
            Total mass of the component [Msun]
        q: float
            Intrinsic axis ratio of Sersic profile
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        i: float
            Inclination of system [deg]

        rinner: float, optional
            Inner radius of total spherical shell, if only calculating
            mass in a spherical shell [kpc]
            Default: rinner = 0. (eg the entire sphere out to r)

    Returns
    -------
        vcirc: float or array_like
            Circular velocity in the midplane at r [km/s]

    """

    Ie = get_Ie(total_mass=total_mass, q=q, n=n, Reff=Reff, i=i)

    cnst = 4*np.pi*G.cgs.value*Msun.cgs.value*q/(1000.*pc.cgs.value)

    try:
        if len(r) > 0:
            vcsq = np.zeros(len(r))
            for j in range(len(r)):
                vcsq[j] = cnst*vel_integral(r[j], q=q, Reff=Reff, Ie=Ie, n=n, i=i)
                if r[j] == 0:
                    vcsq[j] = 0.
        else:
            vcsq = cnst*vel_integral(r[0], q=q, Reff=Reff, Ie=Ie, n=n, i=i)
            if r == 0:
                vcsq = 0.
    except:
        vcsq = cnst*vel_integral(r, q=q, Reff=Reff, Ie=Ie, n=n, i=i)
        if r == 0:
            vcsq = 0.

    return np.sqrt(vcsq)/1.e5



def M_encl_2D(r, total_mass=1., q=0.4, n=1., Reff=1.,i=90.):
    """
    Evalutation of the 2D projected mass enclosed within an ellipse
    (or elliptical shell), assuming a constant M/L ratio Upsilon.

    Alias for total_mass2D_direct.

    Parameters
    ----------
        r: float or array_like
            Major axis radius within which to determine total enclosed 2D projected mass [kpc]

        total_mass: float
            Total mass of the component [Msun]
        q: float
            Intrinsic axis ratio of Sersic profile
        n: float
            Sersic index
        Reff: float
            Effective radius of Sersic profile [kpc]
        i: float
            Inclination of system [deg]

        rinner: float optional
            Inner radius, if only calculating mass in an elliptical annulus.
            Default: rinner = 0. (eg the entire ellipse out to r)

    Returns
    -------
        Menc2D_ellipse: float or array_like

    """
    return total_mass2D_direct(r, total_mass=total_mass, q=q, i=i, n=n, Reff=Reff)


def M_encl_3D(r, total_mass=1., n=1., Reff=1., q=0.4, i=90., cumulative=False):
    """
    Evalutation of the 3D mass enclosed within a sphere of radius r,
    assuming a constant M/L ratio Upsilon.

    Directly calls total_mass3D_integral, after determining Ie and handling shape of r.

    Parameters
    ----------
        r: float or array_like
            Major axis radius within which to determine total enclosed 2D projected mass [kpc]

        total_mass: float
            Total mass of the component [Msun]
        n: float
            Sersic index
        Reff: float
            Effective radius of Sersic profile [kpc]
        q: float
            Intrinsic axis ratio of Sersic profile
        i: float
            Inclination of system [deg]

        cumulative: bool
            Shortcut option to only calculate the next annulus,
            then add to the previous Menc(r-rdelt). Default: False

    Returns
    -------
        Menc3D_sphere: float or array_like

    """
    # Calculate fractional enclosed mass, to avoid numerical problems:
    Ie = get_Ie(total_mass=1., q=q, n=n, Reff=Reff, i=i)

    try:
        if len(r) > 0:
            menc = np.zeros(len(r))
            for j in range(len(r)):
                if cumulative:
                    # Only calculate annulus, use previous Menc as shortcut:
                    if j > 0:
                        menc_ann =  total_mass3D_integral(r[j], rinner=r[j-1], q=q, n=n, Ie=Ie, Reff=Reff, i=i)
                        menc[j] = menc[j-1] + menc_ann
                    else:
                        menc[j] = total_mass3D_integral(r[j], rinner=0., q=q, n=n, Ie=Ie, Reff=Reff, i=i)
                else:
                    # Direct calculation for every radius
                    menc[j] = total_mass3D_integral(r[j], rinner=0., q=q, n=n, Ie=Ie, Reff=Reff, i=i)
        else:
            menc = total_mass3D_integral(r[0], rinner=0., q=q, n=n, Ie=Ie, Reff=Reff, i=i)
    except:
        menc = total_mass3D_integral(r, rinner=0., q=q, n=n, Ie=Ie, Reff=Reff, i=i)

    # Return enclosed mass: fractional menc * total mass
    return menc*total_mass


def M_encl_3D_ellip(r, total_mass=1., n=1., Reff=1., q=0.4, i=90., cumulative=False):
    """
    Evalutation of the 3D mass enclosed within an ellpsoid of
    major axis radius r and intrinsic axis ratio q
    (e.g. the same as the Sersic profile isodensity contours),
    assuming a constant M/L ratio Upsilon.

    Directly calls total_mass3D_integral_ellipsoid,
    after determining Ie and handling shape of r.

    Parameters
    ----------
        r: float or array_like
            Major axis radius within which to determine total enclosed 2D projected mass [kpc]

        total_mass: float
            Total mass of the component [Msun]
        n: float
            Sersic index
        Reff: float
            Effective radius of Sersic profile [kpc]
        q: float
            Intrinsic axis ratio of Sersic profile
        i: float
            Inclination of system [deg]

        cumulative: bool
            Shortcut option to only calculate the next annulus,
            then add to the previous Menc(r-rdelt). Default: False

    Returns
    -------
        Menc3D_ellip: float or array_like

    """

    # Calculate fractional enclosed mass, to avoid numerical problems:
    Ie = get_Ie(total_mass=1., q=q, n=n, Reff=Reff, i=i)

    try:
        if len(r) > 0:
            menc = np.zeros(len(r))
            for j in range(len(r)):
                if cumulative:
                    # Only calculate annulus, use previous Menc as shortcut:
                    if j > 0:
                        menc_ann =  total_mass3D_integral_ellipsoid(r[j], rinner=r[j-1], q=q, n=n, Ie=Ie, Reff=Reff, i=i)
                        menc[j] = menc[j-1] + menc_ann
                    else:
                        menc[j] = total_mass3D_integral_ellipsoid(r[j], rinner=0., q=q, n=n, Ie=Ie, Reff=Reff, i=i)
                else:
                    # Direct calculation for every radius
                    menc[j] = total_mass3D_integral_ellipsoid(r[j], rinner=0., q=q, n=n, Ie=Ie, Reff=Reff, i=i)
        else:
            menc = total_mass3D_integral_ellipsoid(r[0], rinner=0., q=q, n=n, Ie=Ie, Reff=Reff, i=i)
    except:
        menc = total_mass3D_integral_ellipsoid(r, rinner=0., q=q, n=n, Ie=Ie, Reff=Reff, i=i)

    # Return enclosed mass: fractional menc * total mass
    return menc*total_mass

# +++++++++++++++++++++++++++++++++++++++++++++++++

def virial_coeff_tot(r, total_mass=1., q=0.4, Reff=1., n=1.,  i=90., vc=None):
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
        q: float
            Intrinsic axis ratio of Sersic profile
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        i: float
            Inclination of system [deg]

        vc: float or array_like, optional
            Pre-calculated evaluation of vcirc(r)
            (saves time to avoid recalculating vcirc(r))  [km/s]

    Returns
    -------
        ktot: float or array_like
            ktot = Mtot * G / (vcirc(r)^2 * r)

    """

    # vc: can pass pre-calculated vc to save time.
    if vc is None:
        vc = v_circ(r, q=q, Reff=Reff, n=n, total_mass=total_mass, i=i)

    # need to convert to cgs:
    # units: Mass: msun
    #        r:    kpc
    #        v:    km/s
    ktot = (total_mass * Msun.cgs.value) * G.cgs.value / (( r*1.e3*pc.cgs.value ) * (vc*1.e5)**2)

    return ktot


def virial_coeff_3D(r, total_mass=1., q=1., Reff=1., n=1., i=90., m3D=None, vc=None):
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

        total_mass: float
            Total mass of the component [Msun]
        q: float
            Intrinsic axis ratio of Sersic profile
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        i: float
            Inclination of system [deg]

        vc: float or array_like, optional
            Pre-calculated evaluation of vcirc(r)
            (saves time to avoid recalculating vcirc(r))  [km/s]

        m3D: float or array_like, optional
            Pre-calculated evaluation of Menc3D_sphere(r)
            (saves time to avoid recalculating Menc3D_sphere(r)) [Msun]

    Returns
    -------
        k3D: float or array_like
            k3D = Menc3D_sphere(r) * G / (vcirc(r)^2 * r)

    """

    # vc: can pass pre-calculated vc to save time.
    if vc is None:
        vc = v_circ(r, q=q, Reff=Reff, n=n, total_mass=total_mass, i=i)

    if m3D is None:
        m3D = M_encl_3D(r, q=q, n=n,  total_mass=total_mass, Reff=Reff, i=i)

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

    return rhalf_sph




# +++++++++++++++++++++++++++++++++++++++++++++++++
# Supplementary for comparison / plotting:
# NFW halo profile

def halo_rvir(Mvirial=None, z=None, cosmo=_default_cosmo):
    """
    Calculate the halo virial radius at a given redshift

    Parameters
    ----------
        Mvirial: float
            Halo virial mass (Mvir = M200);  [Msun]
        z: float
            Redshift

        cosmo: AstroPy cosmology instance, optional
            Default: FlatLambdaCDM(H0=70., Om0=0.3)

    Returns
    -------
        rvir: float
            Halo virial radius [kpc]

    """
    g_new_unit = G.to(u.pc / u.Msun * (u.km / u.s) ** 2).value
    hz = cosmo.H(z).value
    rvir = ((Mvirial * (g_new_unit * 1e-3) / (10 * hz * 1e-3) ** 2) ** (1. / 3.))

    return rvir

def NFW_halo_enclosed_mass(r=None, Mvirial=None, conc=None, z=None, cosmo=_default_cosmo):
    """
    Calculate the enclosed mass of a NFW halo as a function of radius

    Paramaters
    ----------
        r: float or array_like
            Radi[us/i] at which to calculate the enclosed mass  [kpc]
        Mvirial: float
            Halo virial mass (Mvir = M200);  [Msun]
        conc: float
            Halo concentration
        z: float
            Redshift

        cosmo: AstroPy cosmology instance, optional
            Default: FlatLambdaCDM(H0=70., Om0=0.3)

    Returns
    -------
        mhalo_enc: float or array_like
            Enclosed halo mass profile as as a function of radius   [Msun]

    """
    # Rvir = R200, from Mo, Mao, White 1998
    #     M_vir = 100*H(z)^2/G * R_vir^3

    rvir = halo_rvir(Mvirial=Mvirial, z=z, cosmo=cosmo)

    rho0 = Mvirial/(4.*np.pi*rvir**3)*conc**3 *  1./(np.log(1.+conc) - (conc/(1.+conc)))
    rs = rvir/conc

    aa = 4.*np.pi*rho0*rvir**3/conc**3
    # For very small r, bb can be negative.
    ##bb = np.log((rs + r)/rs) - r/(rs + r)
    bb = np.abs(np.log((rs + r)/rs) - r/(rs + r))

    return aa*bb

def NFW_halo_vcirc(r=None, Mvirial=None, conc=None, z=None, cosmo=_default_cosmo):
    """
    Determine vcirc for the NFW halo, assuming spherical symmetry:

    .. math::

        v_{\mathrm{circ}}(r) = \sqrt{\\frac{G M_{\mathrm{enc,halo}}(r)}{r}}

    Parameters
    -----------
        r: float or array_like
            Radi[us/i] at which to calculate the circular velocity [kpc]
        Mvirial: float
            Halo virial mass (Mvir = M200);  [Msun]
        conc: float
            Halo concentration
        z: float
            Redshift

        cosmo: AstroPy cosmology instance, optional
            Default: FlatLambdaCDM(H0=70., Om0=0.3)

    Returns
    -------
        vcirc_halo: float or array_like
            Halo circular velocity as a function of radius  [km/s]

    """
    mass_enc = NFW_halo_enclosed_mass(r=r, Mvirial=Mvirial, conc=conc, z=z, cosmo=cosmo)
    vcirc = vcirc_spherical_symmetry(r=r, menc=mass_enc)

    return vcirc

def TPH_halo_enclosed_mass(r=None, Mvirial=None, conc=None, z=None,
    alpha=None, beta=3., cosmo=_default_cosmo):
    """
    Calculate the enclosed mass of a Two-power halo (generalized NFW halo)
    as a function of radius

    Paramaters
    ----------
        r: float or array_like
            Radi[us/i] at which to calculate the enclosed mass  [kpc]
        Mvirial: float
            Halo virial mass (Mvir = M200);  [Msun]
        conc: float
            Halo concentration
        z: float
            Redshift
        alpha: float
            Halo inner slope. (NFW: alpha=1.)
        beta: float
            Halo outer slope. (NFW: beta=3.)

        cosmo: AstroPy cosmology instance, optional
            Default: FlatLambdaCDM(H0=70., Om0=0.3)

    Returns
    -------
        mhalo_enc: float or array_like
            Enclosed halo mass profile as as a function of radius   [Msun]

    """
    rvirial = halo_rvir(Mvirial=Mvirial, z=z, cosmo=cosmo)
    rs = rvirial/conc
    aa = Mvirial*(r/rvirial)**(3 - alpha)
    bb = (scp_spec.hyp2f1(3-alpha, beta-alpha, 4-alpha, -r/rs) /
          scp_spec.hyp2f1(3-alpha, beta-alpha, 4-alpha, -conc))

    return aa*bb


def TPH_halo_vcirc(r=None, Mvirial=None, conc=None, z=None,
    alpha=None, beta=3., cosmo=_default_cosmo):
    """
    Determine vcirc for the TPH halo, assuming spherical symmetry:

    .. math::

        v_{\mathrm{circ}}(r) = \sqrt{\\frac{G M_{\mathrm{enc,halo}}(r)}{r}}

    Parameters
    -----------
        r: float or array_like
            Radi[us/i] at which to calculate the circular velocity [kpc]
        Mvirial: float
            Halo virial mass (Mvir = M200);  [Msun]
        conc: float
            Halo concentration
        z: float
            Redshift
        alpha: float
            Halo inner slope. (NFW: alpha=1.)
        beta: float
            Halo outer slope. (NFW: beta=3.)

        cosmo: AstroPy cosmology instance, optional
            Default: FlatLambdaCDM(H0=70., Om0=0.3)

    Returns
    -------
        vcirc_halo: float or array_like
            Halo circular velocity as a function of radius  [km/s]

    """
    mass_enc = TPH_halo_enclosed_mass(r=r, Mvirial=Mvirial, conc=conc, z=z, cosmo=cosmo, alpha=alpha, beta=beta)
    vcirc = vcirc_spherical_symmetry(r=r, menc=mass_enc)
    return vcirc


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
# Interpolation functions:

def interpolate_sersic_profile_rho(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=None, filename=None, table=None):
    """
    Determine the Rho(r) profile at arbitrary radii r, for arbitrary Mtot and Reff.

    Uses the saved table of rho(r) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Mtot and Reff.
    (by mapping the radius using r' = (r/Reff * table_Reff) )

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str
            Path to directory containing the saved Sersic profile tables.
            Default: None (assumes local directory).
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        rho_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq,  path=path, filename_base=filename_base)

    table_rho =     table['rho']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['r'][0] > 0.:
        try:
            table_rad = np.append(r.min() * table_Reff/Reff, table_rad)
            table_rho = np.append(rho(r.min()* table_Reff/Reff, n=n, total_mass=table_mass, Reff=table_Reff, q=table['q']), table_rho)
        except:
            pass

    r_interp = scp_interp.interp1d(table_rad, table_rho, fill_value=np.NaN, bounds_error=False, kind='cubic')
    r_interp_extrap = scp_interp.interp1d(table_rad, table_rho, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    rho_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    rho_interp[wh_in] =     (r_interp(rarr[wh_in] / Reff * table_Reff) * (total_mass / table_mass) * (table_Reff / Reff)**3 )
    rho_interp[wh_extrap] = (r_interp_extrap(rarr[wh_extrap] / Reff * table_Reff) * (total_mass / table_mass) * (table_Reff / Reff)**3 )

    if (len(rarr) > 1):
        return rho_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return rho_interp[0]
        else:
            # Length 1 array input
            return rho_interp


def interpolate_sersic_profile_menc(r=None, total_mass=None, Reff=None, n=1., invq=5., path=None,
        filename_base=None, filename=None, table=None):
    """
    Determine the Menc3D_sphere(r) profile at arbitrary radii r, for arbitrary Mtot and Reff.

    Uses the saved table of Menc3D_sphere(r) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Mtot and Reff.
    (by mapping the radius using r' = (r/Reff * table_Reff) )

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str
            Path to directory containing the saved Sersic profile tables.
            Default: None (assumes current directory).
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        menc_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq,  path=path, filename_base=filename_base)

    table_menc =    table['menc3D_sph']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['r'][0] > 0.:
        table_rad = np.append(0., table_rad)
        table_menc = np.append(0., table_menc)

    m_interp = scp_interp.interp1d(table_rad, table_menc, fill_value=np.NaN, bounds_error=False, kind='cubic')
    m_interp_extrap = scp_interp.interp1d(table_rad, table_menc, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    menc_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    menc_interp[wh_in] =     (m_interp(rarr[wh_in] / Reff * table_Reff) * (total_mass / table_mass) )
    menc_interp[wh_extrap] = (m_interp_extrap(rarr[wh_extrap] / Reff * table_Reff) * (total_mass / table_mass) )

    if (len(rarr) > 1):
        return menc_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return menc_interp[0]
        else:
            # Length 1 array input
            return menc_interp


def interpolate_sersic_profile_VC(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=None, filename=None, table=None):
    """
    Determine the vcirc(r) profile at arbitrary radii r, for arbitrary Mtot and Reff.

    Uses the saved table of vcirc(r) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Mtot and Reff.
    (by mapping the radius using r' = (r/Reff * table_Reff), and scaling for the difference
    between Mtot and table_Mtot)

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate vcirc [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str
            Path to directory containing the saved Sersic profile tables.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        vcirc_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq, path=path, filename_base=filename_base)

    table_vcirc =   table['vcirc']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: vcirc=0
    if table['r'][0] > 0.:
        table_rad = np.append(0., table_rad)
        table_vcirc = np.append(0., table_vcirc)

    v_interp = scp_interp.interp1d(table_rad, table_vcirc, fill_value=np.NaN, bounds_error=False, kind='cubic')
    v_interp_extrap = scp_interp.interp1d(table_rad, table_vcirc, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    vcirc_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    vcirc_interp[wh_in] =     (v_interp(rarr[wh_in]  / Reff * table_Reff) * np.sqrt(total_mass / table_mass) * np.sqrt(table_Reff / Reff))
    vcirc_interp[wh_extrap] = (v_interp_extrap(rarr[wh_extrap]  / Reff * table_Reff) * np.sqrt(total_mass / table_mass) * np.sqrt(table_Reff / Reff))

    if (len(rarr) > 1):
        return vcirc_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return vcirc_interp[0]
        else:
            # Length 1 array input
            return vcirc_interp


def interpolate_sersic_profile_alpha(r=None, Reff=None, n=1., invq=5., path=None,
        filename_base=None, filename=None, table=None):
    """
    Determine the alpha=-dlnrho/dlnr profile at arbitrary radii r, for arbitrary Reff.

    Uses the saved table of rho(r) values for a given Sersic index n and invq,
    and performs scaling and interpolation to map the profile onto the new Reff.
    (by mapping the radius using r' = (r/Reff * table_Reff) )


    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate vcirc [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str
            Path to directory containing the saved Sersic profile tables.
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.
        table: dict, optional
            Option to pass the Sersic profile table, if already loaded.

    Returns
    -------
        alpha_interp: float or array_like

    """
    if table is None:
        table = read_profile_table(filename=filename, n=n, invq=invq,  path=path, filename_base=filename_base)

    table_dlnrho_dlnr =     table['dlnrho_dlnr']
    table_rad =     table['r']
    table_Reff =    table['Reff']
    table_mass =    table['total_mass']

    # Clean up values inside rmin:  Add the value at r=0: menc=0
    if table['r'][0] > 0.:
        try:
            table_rad = np.append(r.min() * table_Reff/Reff, table_rad)
            table_dlnrho_dlnr = np.append(dlnrho_dlnr(r.min()* table_Reff/Reff, n=n, total_mass=table_mass,
                                        Reff=table_Reff, q=table['q']), table_dlnrho_dlnr)
        except:
            pass

    # Catch NaNs:
    whfin = np.where(np.isfinite(table_dlnrho_dlnr))
    table_dlnrho_dlnr = table_dlnrho_dlnr[whfin]
    table_rad = table_rad[whfin]

    r_interp = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, fill_value=np.NaN, bounds_error=False, kind='cubic')
    r_interp_extrap = scp_interp.interp1d(table_rad, table_dlnrho_dlnr, fill_value='extrapolate', kind='linear')

    # Ensure it's an array:
    if isinstance(r*1., float):
        rarr = np.array([r])
    else:
        rarr = np.array(r)
    # Ensure all radii are 0. or positive:
    rarr = np.abs(rarr)

    dlnrho_dlnr_interp = np.zeros(len(rarr))
    wh_in =     np.where((r <= table_rad.max()) & (r >= table_rad.min()))[0]
    wh_extrap = np.where((r > table_rad.max()) | (r < table_rad.min()))[0]
    dlnrho_dlnr_interp[wh_in] =     (r_interp(rarr[wh_in] / Reff * table_Reff) )
    dlnrho_dlnr_interp[wh_extrap] = (r_interp_extrap(rarr[wh_extrap] / Reff * table_Reff) )

    if (len(rarr) > 1):
        return -1. * dlnrho_dlnr_interp
    else:
        if isinstance(r*1., float):
            # Float input
            return -1. * dlnrho_dlnr_interp[0]
        else:
            # Length 1 array input
            return -1. * dlnrho_dlnr_interp


# +++++++++++++++++++++++++++++++++++++++++++++++++
# Nearest n, invq values and aliases to the Menc, vcirc interpolation functions

def nearest_n_invq(n=None, invq=None):
    """
    Function to find the nearest value of n and invq for which a Sersic profile table
    has been calculated, using the *DEFAULT* array of n and invq which have been used here.

    A similar function can be defined if a different set of Sersic profile tables
    (over n, invq) have been calculated.

    Example
    -------
    >>> nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    Parameters
    ----------
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

    Returns
    -------
        nearest_n: float
            Nearest value of Sersic index n from lookup array
        nearest_invq: float
            Nearest value of flattening invq from lookup array

    """
    # Use the "typical" collection of table values:
    table_n = np.arange(0.5, 8.1, 0.1)   # Sersic indices
    table_invq = np.array([1., 2., 3., 4., 5., 6., 7., 8., 10., 20., 100.,
                    1.11, 1.43, 1.67, 3.33, 0.5, 0.67])
    # 1:1, 1:2, 1:3, ... flattening  [also prolate 2:1, 1.5:1]

    nearest_n = table_n[ np.argmin( np.abs(table_n - n) ) ]
    nearest_invq = table_invq[ np.argmin( np.abs( table_invq - invq) ) ]

    return nearest_n, nearest_invq

def interpolate_sersic_profile_menc_nearest(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=None, filename=None):
    """
    Determine the Menc3D_sphere(r) profile at arbitrary radii r, for arbitrary Mtot and Reff,
    **using the nearest values of n and invq that are included
    in the Sersic profile table collection.**

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_menc()` with these values.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str
            Path to directory containing the saved Sersic profile tables.
            Default: None (assumes current directory).
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.

    Returns
    -------
        menc_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    menc_interp_nearest = interpolate_sersic_profile_menc(r=r, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

    return menc_interp_nearest


def interpolate_sersic_profile_VC_nearest(r=None, total_mass=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=None, filename=None):
    """
    Determine the vcirc(r) profile at arbitrary radii r, for arbitrary Mtot and Reff,
    **using the nearest values of n and invq that are included
    in the Sersic profile table collection.**

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_VC()` with these values.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
        total_mass: float
            Total mass of the component [Msun]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q

        path: str
            Path to directory containing the saved Sersic profile tables.
            Default: None (assumes current directory).
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.

    Returns
    -------
        vcirc_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    vcirc_interp_nearest = interpolate_sersic_profile_VC(r=r, total_mass=total_mass, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

    return vcirc_interp_nearest


def interpolate_sersic_profile_alpha_nearest(r=None, Reff=None, n=1., invq=5.,
        path=None, filename_base=None, filename=None):
    """
    Determine the alpha(r)=-dlnrho_g/dlnr profile at arbitrary radii r, for arbitrary Reff,
    **using the nearest values of n and invq that are included
    in the Sersic profile table collection.**

    Finds the nearest n, invq for the "default" table collection,
    then calls `interpolate_sersic_profile_alpha()` with these values.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
        Reff: float
            Effective radius of Sersic profile [kpc]
        n: float
            Sersic index
        invq: float
            Inverse of the intrinsic axis ratio of Sersic profile, invq = 1/q.

        path: str
            Path to directory containing the saved Sersic profile tables.
            Default: None (assumes current directory).
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.

    Returns
    -------
        alpha_interp_nearest: float or array_like

    """

    # Use the "typical" collection of table values:
    nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

    alpha_interp_nearest = interpolate_sersic_profile_alpha(r=r, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

    return alpha_interp_nearest



def interpolate_sersic_profile_alpha_bulge_disk_nearest(r=None,
        BT=0.,  total_mass=1.e11,
        Reff_disk=None, n_disk=1., invq_disk=5.,
        Reff_bulge=1.,  n_bulge=4., invq_bulge=1.,
        path=None, filename_base=None, filename=None):
    """
    Determine the alpha(r)=-dlnrho_g/dlnr profile at arbitrary radii r,
    for a composite DISK+BULGE system. Both disk and bulge can have arbitary Reff,
    but this uses **the nearest values of n and invq that are included
    in the Sersic profile table collection.**

    Finds the nearest n, invq for the "default" table collection,
    then returns `alpha_interp_nearest()` for the total DISK+BULGE system.

    Parameters
    ----------
        r: float or array_like
            Radius at which to interpolate Menc3D_sphere [kpc]
        total_mass: float
            Total mass of the component [Msun]    [Default: 10^11 Msun]
        BT: float
            Bulge to total ratio (Total = Disk + Bulge)  [Default: 0.]
        Reff_disk: float
            Effective radius of disk Sersic profile [kpc]
        n_disk: float
            Sersic index of disk.
        invq_disk: float
            Inverse of the intrinsic axis ratio of disk Sersic profile, invq = 1/q.
        Reff_bulge: float
            Effective radius of bulge Sersic profile [kpc]
        n_bulge: float
            Sersic index of bulge. [Default: 4]
        invq_bulge: float
            Inverse of the intrinsic axis ratio of bulge Sersic profile, invq = 1/q.
            [Default: 1.]

        path: str
            Path to directory containing the saved Sersic profile tables.
            Default: None (assumes current directory).
        filename_base: str, optional
            Base filename to use, when combined with default naming convention:
            `<path>/<filename_base>_nX.X_invqX.XX.fits`.
        filename: str, optional
            Option to override the default filename convention and
            instead directly specify the file location.

    Returns
    -------
        alpha_interp_nearest: float or array_like

    """

    Mbulge = total_mass * BT
    Mdisk = total_mass * (1.-BT)

    # Use the "typical" collection of table values:
    rho_t = r * 0.
    rhoalphasum = r * 0.
    for n, invq, Reff, M in zip([n_disk, n_bulge], [invq_disk, invq_bulge],
            [Reff_disk, Reff_bulge], [Mdisk, Mbulge]):
        nearest_n, nearest_invq = nearest_n_invq(n=n, invq=invq)

        alpha_interp_nearest = interpolate_sersic_profile_alpha(r=r, Reff=Reff,
                    n=nearest_n, invq=nearest_invq,
                    path=path, filename_base=filename_base, filename=filename)

        rho_interp_nearest = interpolate_sersic_profile_rho(r=r, total_mass=M,
                Reff=Reff, n=nearest_n, invq=nearest_invq, path=path,
                filename_base=filename_base, filename=filename)
        rho_t += rho_interp_nearest
        rhoalphasum += rho_interp_nearest * alpha_interp_nearest

    alpha_interp_total = (1./rho_t) * rhoalphasum

    return alpha_interp_total
