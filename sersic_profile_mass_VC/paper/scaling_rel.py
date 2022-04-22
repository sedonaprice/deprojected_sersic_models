##################################################################################
# sersic_profile_mass_VC/paper/scaling_rel.py                                    #
#                                                                                #
# Copyright 2018-2022 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import numpy as np
import scipy.special as scp_spec
import scipy.interpolate as scp_interp
import scipy.integrate as scp_integrate
import scipy.signal as scp_signal
import astropy.cosmology as apy_cosmo

from sersic_profile_mass_VC.utils.calcs import bn_func
from sersic_profile_mass_VC.utils.interp_profiles import interpolate_sersic_profile_VC
from sersic_profile_mass_VC import io

import logging


from astropy.io import ascii
from astropy.table import Table


# DEFAULT COSMOLOGY
_default_cosmo = apy_cosmo.FlatLambdaCDM(H0=70., Om0=0.3)


# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SersicProfileMassVC')

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# ---------------------
# Utility functions:
#   Observed scaling relations, etc

def _mstar_Reff_relation(z=None, lmstar=None, galtype='sf'):
    """
    Stellar mass-R_eff relation from
    van der Wel, A., Franx, M., van Dokkum, P. G., et al. 2014, ApJ, 788, 28

    Using the (1+z) fitting with coefficients from Table 2,
    with interpolation to go between masses.
    Form: Reff/kpc = Bz * (1+z)^betaz

    Parameters
    ----------
        z: float or array
            Redshift

        lmstar: float or array
            Log10 stellar mass

        galtype: 'sf' or 'q', optional
            Selector for which scaling relation to use:
            'sf' (star-forming galaxies) or 'q' (quiescent galaxies)

    Returns
    -------
        Reff: float or array
            Major axis effective radius
    """

    if galtype == 'sf':
        lmass_arr_sf =      np.array([9.25, 9.75, 10.25, 10.75, 11.25])
        logBz_arr_sf =      np.array([0.54, 0.69, 0.74, 0.90, 1.05])
        logbetaz_arr_sf =   np.array([-0.48, -0.63, -0.52, -0.72, -0.80])

        sf_interp_Bz = scp_interp.interp1d(lmass_arr_sf, np.power(10.,logBz_arr_sf),
                        fill_value="extrapolate", kind='linear')
        sf_interp_betaz = scp_interp.interp1d(lmass_arr_sf, logbetaz_arr_sf,
                        fill_value="extrapolate", kind='linear')

        Bz =        sf_interp_Bz(lmstar)
        betaz =     sf_interp_betaz(lmstar)

    elif galtype == 'q':
        lmass_arr_q =       np.array([9.75, 10.25, 10.75, 11.25])
        logBz_arr_q =       np.array([0.29, 0.47, 0.75, 1.05])
        logbetaz_arr_q =    np.array([-0.22, -1.01, -1.24, -1.32])

        q_interp_Bz = scp_interp.interp1d(lmass_arr_q, np.power(10.,logBz_arr_q),
                        fill_value="extrapolate", kind='linear')
        q_interp_betaz = scp_interp.interp1d(lmass_arr_q, logbetaz_arr_q,
                        fill_value="extrapolate", kind='linear')

        Bz =        q_interp_Bz(lmstar)
        betaz =     q_interp_betaz(lmstar)
    else:
        raise ValueError("'galtype'={} not valid! Must be 'sf' or 'q'!".format(galtype))

    Reff = Bz * np.power((1.+z), betaz)

    return Reff


def _mumol_scaling_relation(z=None, lmstar=None, del_lsSFR_MS=None):
    """
    Scaling relation for mu_mol from
    Tacconi, L. J., Genzel, R., & Sternberg, A. 2020, ARA&A, 58, 157

    Table 2b: log10(mu_mol) = log10(M_molgas/Mstar)

    Parameters
    ----------
        z: float or array
            Redshift

        lmstar: float or array
            Log10 stellar mass

        del_lsSFR_MS: float or array
            Offset of Log10(SSFR) relative to main sequence
            (at given redshift, stellar mass)

    Returns
    -------
        log_mu_mol: float or array
            Log10 of mu_mol
    """

    A = 0.06
    B = -3.33
    F = 0.65
    C = 0.51
    D = -0.41

    log_mu_mol = A + B* (np.log10(1.+z)-F)**2 + C* del_lsSFR_MS + D * (lmstar-10.7)

    return log_mu_mol

def _fgas_scaling_relation_MS(z=None, lmstar=None):
    """
    Scaling relation for f_molgas on the main sequence
    calculated using mu_mol scaling relation from
    Tacconi, L. J., Genzel, R., & Sternberg, A. 2020, ARA&A, 58, 157

    Table 2b: log10(mu_mol) = log10(M_molgas/Mstar)

    Parameters
    ----------
        z: float or array
            Redshift

        lmstar: float or array
            Log10 stellar mass

    Returns
    -------
        fgas: float or array
            Molecular gas fraction
    """
    log_mu_mol = _mumol_scaling_relation(z=z, lmstar=lmstar, del_lsSFR_MS=0.)
    mu_mol = np.power(10., log_mu_mol)
    fgas = mu_mol / (1.+mu_mol)
    return fgas

def _invq_disk_lmstar_estimate(z=None, lmstar=None):
    """
    Estimate of galaxy disk flattening inverse q, based on typical values
    adopted in Genzel, R., Price, S. H., Übler, H., et al. 2020, ApJ, 902, 98

    (No dependence on stellar mass assumed)

    Parameters
    ----------
        z: float or array
            Redshift

        lmstar: float or array
            Log10 stellar mass

    Returns
    -------
        invq_disk: float or array
            Inverse q of disk component
    """
    z_arr_ROUGH =      np.array([0., 0.8, 1.2, 1.5, 1.75, 2.])
    invq_arr_ROUGH =   np.array([10., 8., 6., 5., 4.3, 4.])

    fill_value= invq_arr_ROUGH[-1]
    kind = 'quadratic'

    invq_interp_ROUGH = scp_interp.interp1d(z_arr_ROUGH, invq_arr_ROUGH,
                    fill_value=fill_value, kind=kind, bounds_error=False)

    invq_disk =        invq_interp_ROUGH(z)

    if isinstance(z, float):
        return np.float(invq_disk)
    else:
        return invq_disk


def _bt_lmstar_relation(z=None, lmstar=None, galtype='sf'):
    """
    Interpolation of B/T ratio trend with stellar mass based on
    results presented in Figure 1b of
    Lang, P., Wuyts, S., Somerville, R. S., et al. 2014, ApJ, 788, 11

    (Approximate locations)

    (No dependence on redshift assumed)

    Parameters
    ----------
        z: float or array
            Redshift

        lmstar: float or array
            Log10 stellar mass

        galtype: 'sf' or 'q', optional
            Selector for which scaling relation to use:
            'sf' (star-forming galaxies) or 'q' (quiescent galaxies)

    Returns
    -------
        invq_disk: float or array
            Inverse q of disk component
    """

    if galtype == 'sf':
        lmass_arr_sf =      np.array([10.09, 10.29, 10.49, 10.685, 10.88, 11.055, 11.255, 11.45])
        bt_arr_sf =         np.array([0.235, 0.2475, 0.27, 0.29, 0.37, 0.42, 0.47, 0.50])

        sf_interp_bt = scp_interp.interp1d(lmass_arr_sf, bt_arr_sf,
                        fill_value=np.NaN, kind='slinear')
        sf_interp_bt_extrap = scp_interp.interp1d(lmass_arr_sf, bt_arr_sf,
                        fill_value="extrapolate", kind='linear')
        if isinstance(lmstar, float):
            if (lmstar < lmass_arr_sf.min()) | (lmstar > lmass_arr_sf.max()):
                bt =        sf_interp_bt_extrap(lmstar)
            else:
                bt =        sf_interp_bt(lmstar)
        else:
            whout = np.where((lmstar < lmass_arr_sf.min()) | (lmstar > lmass_arr_sf.max()))[0]
            whin = np.where((lmstar >= lmass_arr_sf.min()) & (lmstar <= lmass_arr_sf.max()))[0]
            bt[whout] = sf_interp_bt_extrap(lmstar[whout])
            bt[whin] =    sf_interp_bt(lmstar[whin])

    elif galtype == 'q':
        raise ValueError("Not implemented yet")
    else:
        raise ValueError("'galtype'={} not valid! Must be 'sf' or 'q'!".format(galtype))

    if isinstance(lmstar, float):
        return np.float(bt)
    else:
        return bt


def _smhm_relation(z=None, lmstar=None):
    """
    Stellar mass-halo mass relation from
    Moster, B. P., Naab, T., & White, S. D. 2018, MNRAS, 477, 1822
    based on UPDATED fitting relation (priv. comm., T. Naab, 2020-05-21)
    using the stellar mass binned fitting result (to avoid divergance at high lMstar).

    Parameters
    ----------
        z: float or array
            Redshift

        lmstar: float or array
            Log10 stellar mass

    Returns
    -------
        Mhalo: float or array
            Halo mass (Msun)
    """

    log_m1 = 10.6
    n = np.power(10., (1.507 - 0.124 * (z/(z+1.)) ) )
    b = -0.621 - 0.059 * (z/(z+1.))
    g = 1.055 + 0.838 * (z/(z+1)) - 3.083 * ( ((z/(z+1)))**2 )

    lmhalo = lmstar + np.log10(0.5) + np.log10(n) + np.log10( np.power( np.power(10., (lmstar-log_m1)), b ) + \
                np.power( np.power(10., (lmstar-log_m1)), g ) )

    Mhalo =  np.power(10., lmhalo)

    return Mhalo

def _halo_conc_relation(z=None, lmhalo=None, cosmo=_default_cosmo):
    """
    Halo concentration relation from
    Dutton, A. A., & Macciò, A. V. 2014, MNRAS, 441, 3359

    Eq. 7: log10conc = a + b * log10(Mhalo / (1e12 h^{-1} Msun))

    Using slope and zeropoint for NFW conc200 defined in Eqs 10, 11:
    b = -0.101 + 0.026 * z
    a = 0.520 + (0.905-0.520)*exp(-0.617*z^1.21)

    Parameters
    ----------
        z: float or array
            Redshift

        lmhalo: float or array
            Log10 halo mass

        cosmo: AstroPy cosmology instance, optional
            Cosmology

    Returns
    -------
        conc: float or array
            Halo concentration
    """
    hinv = 100./cosmo.H0.value

    b = -0.101 + 0.026*z
    a = 0.520 + (0.905-0.520)*np.exp(-0.617*np.power(z,1.21))

    log10conc = a + b * (lmhalo - np.log10(1.e12 * hinv))

    return np.power(10.,log10conc)



def _tomczak14_SMF_total_coeffs():
    """
    Coefficients for double Schechter function fit to total stellar mass function from
    Tomczak, A. R., Quadri, R. F., Tran, K.-V. H., et al. 2014, ApJ, 783, 85
    """
    z_bounds = np.array([0.2, 0.5, 0.75, 1., 1.25, 1.5, 2., 2.5, 3.])
    z_mid = 0.5*(z_bounds[:-1]+z_bounds[1:])

    # Use SINGLE SCHECHTER FIT above z>2
    logMstar = np.array([10.78, 10.7, 10.66, 10.54, 10.61, 10.74, 11.13, 11.35])

    alpha1 = np.array([-0.98, -0.39, -0.37, 0.3, -0.12, 0.04, -1.43, -1.74])
    logphistar1 = np.array([-2.54, -2.55, -2.56, -2.72, -2.78, -3.05, -3.59, -4.36])

    alpha2 = np.array([-1.90, -1.53, -1.61, -1.45, -1.56, -1.49, -99., -99.])
    logphistar2 = np.array([-4.29, -3.15, -3.39, -3.17, -3.43, -3.38, -np.inf, -np.inf])

    dict_t14 = {'z_mid':   z_mid,
               'logMstar': logMstar,
               'alpha1': alpha1,
               'alpha2': alpha2,
               'logphistar1': logphistar1,
               'logphistar2': logphistar2}

    return dict_t14


def _tomczak14_SMF_total_coeffs_interp(z, kind='nearest'):
    """
    Interpolation of coefficients for double Schechter function
    fit to total stellar mass function from
    Tomczak, A. R., Quadri, R. F., Tran, K.-V. H., et al. 2014, ApJ, 783, 85
    """
    dict_t14 = _tomczak14_SMF_total_coeffs()

    kindtmp = kind
    if (z >= dict_t14['z_mid'].min()) & (z <= dict_t14['z_mid'].max()):
        pass
    else:
        kindtmp = 'nearest'

    logMstar_func = scp_interp.interp1d(dict_t14['z_mid'], dict_t14['logMstar'],
                        fill_value='extrapolate', bounds_error=False, kind=kindtmp)

    alpha1_func = scp_interp.interp1d(dict_t14['z_mid'], dict_t14['alpha1'],
                        fill_value='extrapolate', bounds_error=False, kind=kindtmp)

    logphistar1_func = scp_interp.interp1d(dict_t14['z_mid'], dict_t14['logphistar1'],
                        fill_value='extrapolate', bounds_error=False, kind=kindtmp)

    alpha2_func = scp_interp.interp1d(dict_t14['z_mid'][:-2], dict_t14['alpha2'][:-2],
                        fill_value='extrapolate', bounds_error=False, kind=kindtmp)

    logphistar2_func = scp_interp.interp1d(dict_t14['z_mid'][:-2], dict_t14['logphistar2'][:-2],
                        fill_value='extrapolate', bounds_error=False, kind=kindtmp)

    d_t14_int = {'z': z,
                'logMstar': logMstar_func(z),
                'alpha1':  alpha1_func(z),
                'alpha2':  alpha2_func(z),
                'logphistar1':  logphistar1_func(z),
                'logphistar2':  logphistar2_func(z)}

    if (z >= 2.):
        d_t14_int['alpha2'] = -99.
        d_t14_int['logphistar2'] = -np.inf

    return d_t14_int



def _num_density_double_schechter_func(lmass_arr, logMstar,
                alpha1, logphistar1, alpha2, logphistar2):
    """
    Double Schechter function from
    Tomczak, A. R., Quadri, R. F., Tran, K.-V. H., et al. 2014, ApJ, 783, 85
    """

    phistar1 = np.power(10,logphistar1)
    phistar2 = np.power(10,logphistar2)

    if ~np.isfinite(logphistar1):
        phistar1 = 0.
    if ~np.isfinite(logphistar2):
        phistar2 = 0.

    x = np.power(10.,((lmass_arr-logMstar)))

    narr = lmass_arr * 0.
    for phistar, alpha in zip([phistar1, phistar2], [alpha1, alpha2]):
        if phistar == 0.:
            narr_comp = lmass_arr * 0.
        else:
            if (alpha+1.) >= 0.:
                gammainc = scp_spec.gammaincc(alpha+1, x)* scp_spec.gamma(alpha+1)
            else:
                # alpha < -1.
                # what is int s.t. alpha+int > -1, or alpha+int+1 > 0?
                s = alpha+1.
                while s < 0:
                    s += 1
                #
                i = 0.
                # Initialize:
                gammainc_n_m_1 = scp_spec.gammaincc(s, x)*scp_spec.gamma(s)
                while (s-i > alpha+1):
                    i += 1.
                    gammainc_n = gammainc_n_m_1.copy()
                    gammainc_n_m_1 = 1./(s-i)*(gammainc_n - np.power(x , s-i)*np.exp(-x))
                gammainc = gammainc_n_m_1


            narr_comp = phistar*gammainc
        narr = narr + narr_comp

    return narr

def _num_density_tomczak14_double_schechter_interp(z, lmass_arr, kind='nearest'):
    """
    Number density interpolation from double Schechter function from
    Tomczak, A. R., Quadri, R. F., Tran, K.-V. H., et al. 2014, ApJ, 783, 85
    """
    d_t14_int = _tomczak14_SMF_total_coeffs_interp(z, kind=kind)

    n_t14_interp = _num_density_double_schechter_func(lmass_arr,
                d_t14_int['logMstar'], d_t14_int['alpha1'], d_t14_int['logphistar1'],
                d_t14_int['alpha2'], d_t14_int['logphistar2'])

    return n_t14_interp

def _interp_mass_from_tomczak14_num_density(lnz, z, kind_cmf_interp='nearest'):
    """
    Interpolation of progenitor mass values from
    Tomczak, A. R., Quadri, R. F., Tran, K.-V. H., et al. 2014, ApJ, 783, 85
    """

    dlmass = 0.05
    lmass_lims = [9., 12]
    lmass_arr = np.arange(lmass_lims[0], lmass_lims[1]+dlmass, dlmass)

    narr = _num_density_tomczak14_double_schechter_interp(z, lmass_arr, kind=kind_cmf_interp)

    minterp_func = scp_interp.interp1d(np.log10(narr), lmass_arr, fill_value=np.NaN,  #'extrapolate',
                        bounds_error=False, kind='cubic')
    minterp = minterp_func(lnz)
    return minterp

def _papovich15_values_MW_M31():
    """
    Progenitor mass values for MW, M31 inferred from
    Papovich, C., Labbé, I., Quadri, R., et al. 2015, ApJ, 803, 26

    Fig. 4: thick dashed / solid lines, which are in turn inferenced from
    models from Moster, B. P., Naab, T., & White, S. D. 2013, MNRAS, 428, 3121
    """

    dmw = {'ln0': -2.9,
           'z': np.array([0., 0.25, 0.5, 0.75,
                          1., 1.25, 1.5, 1.75,
                          2., 2.25, 2.5, 2.75, 3.]),
           'lmass': np.array([10.73,  10.7, 10.64, 10.54,
                            10.4075, 10.27, 10.125, 9.97,
                            9.82, 9.66, 9.51, 9.37, 9.22])}

    dm31 = {'ln0': -3.4,
           'z': np.array([0., 0.25, 0.5, 0.75,
                          1., 1.25, 1.5, 1.75,
                          2., 2.25, 2.5, 2.75, 3.]),
           'lmass': np.array([10.985, 10.975, 10.96, 10.93,
                              10.87, 10.79, 10.69, 10.57,
                              10.445, 10.32, 10.18, 10.05, 9.91])}

    dp15 = {'MW': dmw, 'M31': dm31}

    return dp15

def _interp_mass_from_papovich15_z(ln0, zarr, kind_cmf_interp='quadratic'):
    """
    Interpolation of progenitor mass values for MW, M31 inferred from
    Papovich, C., Labbé, I., Quadri, R., et al. 2015, ApJ, 803, 26
    """
    dp15 = _papovich15_values_MW_M31()

    if ln0 == -2.9:
        key = 'MW'
    elif ln0 == -3.4:
        key = 'M31'

    minterp_func = scp_interp.interp1d(dp15[key]['z'], dp15[key]['lmass'], fill_value='extrapolate',
                        bounds_error=False, kind=kind_cmf_interp)
    minterp = minterp_func(zarr)


    return minterp

def _mass_progenitor_num_density(ln0, zarr, n_evol='const', cmf_source='papovich15',
                                 kind_cmf_interp='quadratic'):
    """
    Calculate the stellar mass of progenitor over a range of redshifts
    based on the log number density at z=0.

    Based on cumulative mass functions from
    Tomczak, A. R., Quadri, R. F., Tran, K.-V. H., et al. 2014, ApJ, 783, 85
    and
    Papovich, C., Labbé, I., Quadri, R., et al. 2015, ApJ, 803, 26

    Parameters
    ----------
        ln0: float
            Log10 number density at z=0

        zarr: array
            Redshift array

        n_evol: str, optional
            Method of calculating number density evolution.
            Default: 'const' (constant number density)

        cmf_source: str, optional
            Cumulative mass function literature source.
            Default: 'papovich15'

        kind_cmf_interp: str, optional
            Interpolation method. Default: 'quadratic'

    Returns
    -------
        lmass_arr: array
            Array of log10Mstar masses of progenitor at each redshift
    """

    cmf_sources = ['tomczak14', 'papovich15']
    if cmf_source not in cmf_sources:
        errmsg = '{} not reconized CMF reference! Must be one of: '
        for i, cmf in enumerate(cmf_sources):
            errmsg += '{}'.format(cmf)
            if i < len(cmf_sources)-1:
                errmsg += ", "
        raise ValueError(errmsg)
    n_evol_types = ['const', 'torrey17_backwards']
    if n_evol not in n_evol_types:
        errmsg = '{} not reconized evolution type for number density! Must be one of: '
        for i, nev in enumerate(n_evol_types):
            errmsg += '{}'.format(nev)
            if i < len(n_evol_types)-1:
                errmsg += ", "
        raise ValueError(errmsg)

    if n_evol == 'const':
        lnzarr = np.ones(len(zarr)) * ln0
    elif n_evol == 'torrey17_backwards':
        raise ValueError("Not yet implemented")


    lmass_arr = np.ones(len(zarr)) * -99.

    if cmf_source == 'papovich15':
        lmass_arr = _interp_mass_from_papovich15_z(ln0, zarr,
                                    kind_cmf_interp=kind_cmf_interp)
    elif cmf_source == 'tomczak14':
        for i in range(len(zarr)):
                lmass_arr[i] = _interp_mass_from_tomczak14_num_density(lnzarr[i], zarr[i],
                                            kind_cmf_interp=kind_cmf_interp)
    return lmass_arr


def _k21_calc_medians_fig4_alpharho(path=None):
    """
    Calculate median values of alpha_rho in bins of R/Reff,
    to reproduce values presented in Fig 4. from
    Kretschmer, M., Dekel, A., Freundlich, J., et al. 2021, MNRAS, 503, 5238

    Based on values from M. Kretschmer (private comm)
    """
    path = '/'.join(path.split('/')[:-2]) + '/kretschmer21_data/'
    fname = path+'kretschmer21_data.csv'
    k21 = ascii.read(fname)

    RtoRe = np.unique(k21['r/r_e'])
    alpha_rho = RtoRe*0. + -99.
    for j, rr in enumerate(RtoRe):
        whm = np.where(k21['r/r_e']==rr)[0]
        alpha_rho[j] = np.median(k21['alpha_rho'][whm])

    cat_out = Table({"RtoRe": RtoRe,
                     "alpha_rho": alpha_rho})

    fname_out = path+'kretschmer21_fig4_alpharho.csv'
    cat_out.write(fname_out)

    return None

def _kretschmer21_fig4_alpharho(path=None):
    """
    Return pre-calculated median values of alpha_rho in bins of R/Reff,
    as presented in Fig 4. from
    Kretschmer, M., Dekel, A., Freundlich, J., et al. 2021, MNRAS, 503, 5238

    Based on values from M. Kretschmer (private comm).
    """

    path = '/'.join(path.split('/')[:-2]) + '/kretschmer21_data/'
    fname = path+'kretschmer21_fig4_alpharho.csv'
    cat = ascii.read(fname)
    return cat


def _dalcanton_stilp10_alpha_n(rtoRe, n):
    """
    Calculate the pressure support correction term alpha(n,R/Reff)
    for a general mass surface density profile following a
    Sersic profile of index n,
    based on Eq 16 of
    Dalcanton, J. J., & Stilp, A. M. 2010, ApJ, 721, 547
    """
    return 0.92 * (bn_func(n)/n) * np.power(rtoRe, 1./n)
