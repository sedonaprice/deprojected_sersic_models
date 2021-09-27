##################################################################################
# sersic_profile_mass_VC/paper/scaling_rel.py                                    #
#                                                                                #
# Copyright 2018-2021 Sedona Price <sedona.price@gmail.com> / MPE IR/Submm Group #
# Licensed under a 3-clause BSD style license - see LICENSE.rst                  #
##################################################################################

import numpy as np
import scipy.special as scp_spec
import scipy.interpolate as scp_interp
import astropy.cosmology as apy_cosmo

from sersic_profile_mass_VC.utils.calcs import bn_func

import logging

# LOGGER SETTINGS
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SersicProfileMassVC')

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# ---------------------
# Utility functions:
#   Observed scaling relations, etc

def _mstar_Reff_relation(z=None, lmstar=None, galtype='sf'):
    # van der Wel + 2014:
    #   From the (1+z) fitting (coefficients form Table 2):
    #       interpolate to go between masses....
    #
    #   Reff/kpc = Bz * (1+z)^betaz
    #   logReff = log(Bz) + betaz*log(1+z)

    ######
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
    ######
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

    Reff = Bz * np.power((1.+z), betaz)

    return Reff


def _mumol_scaling_relation(z=None, lmstar=None, del_lsSFR_MS=None):
    # Using the scaling relation fit from Tacconi, Genzel, Sternberg 2020, ARAA:
    #   Table 2b:
    #   log(mu_mol) = log10(M_molgas/Mstar)

    A = 0.06
    B = -3.33
    F = 0.65
    C = 0.51
    D = -0.41

    log_mu_mol = A + B* (np.log10(1.+z)-F)**2 + C* del_lsSFR_MS + D * (lmstar-10.7)

    return log_mu_mol

def _fgas_scaling_relation_MS(z=None, lmstar=None):
    # Using the log(mu_mol) scaling relation from Tacconi, Genzel, Sternberg 2020, ARAA,
    #    with fgas = mu_mol / (1 + mu_mol)
    #       so just ignoring atomic gas, for expediency

    log_mu_mol = _mumol_scaling_relation(z=z, lmstar=lmstar, del_lsSFR_MS=0.)
    mu_mol = np.power(10., log_mu_mol)
    fgas = mu_mol / (1.+mu_mol)
    return fgas

def _invq_disk_lmstar_estimate(z=None, lmstar=None):
    # Extrapolation, using ROUGH values similar to those that were adopted
    #   for the RC41 fitting for Genzel+2020
    #
    # For now, ignoring any possible stellar mass differences

    ## adding z=1.8, qinv=4.2 point to smooth out curve....
    z_arr_ROUGH =      np.array([0., 0.8, 1.2, 1.5, 1.75, 2.])
    invq_arr_ROUGH =   np.array([10., 8., 6., 5., 4.3, 4.])

    fill_value= invq_arr_ROUGH[-1]
    kind = 'quadratic' # 'linear'

    invq_interp_ROUGH = scp_interp.interp1d(z_arr_ROUGH, invq_arr_ROUGH,
                    fill_value=fill_value, kind=kind, bounds_error=False)

    invq_disk =        invq_interp_ROUGH(z)

    if isinstance(z, float):
        return np.float(invq_disk)
    else:
        return invq_disk


def _bt_lmstar_relation(z=None, lmstar=None, galtype='sf'):
    # Perform a *REDSHIFT INDEPENDENT* lmstar interpolation based on the results from
    #   Lang+2014, Figure 1b
    #  Using approximate point location values for the SF galaxies.


    if galtype == 'sf':
        lmass_arr_sf =      np.array([10.09, 10.29, 10.49, 10.685, 10.88, 11.055, 11.255, 11.45])
        bt_arr_sf =         np.array([0.235, 0.2475, 0.27, 0.29, 0.37, 0.42, 0.47, 0.50])

        sf_interp_bt = scp_interp.interp1d(lmass_arr_sf, bt_arr_sf,
                        fill_value=np.NaN, kind='slinear') #'linear')
        sf_interp_bt_extrap = scp_interp.interp1d(lmass_arr_sf, bt_arr_sf,
                        fill_value="extrapolate", kind='linear')
        #bt =        sf_interp_bt(lmstar)
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

    if isinstance(lmstar, float):
        return np.float(bt)
    else:
        return bt


def _smhm_relation(z=None, lmstar=None):
    # Moster+18 relation
    # From the updated fitting relation from Moster, Naab & White 2018; from email from Thorsten Naab on 2020-05-21
    # From stellar mass binned fitting result (avoids divergance at high lMstar)

    log_m1 = 10.6
    n = np.power(10., (1.507 - 0.124 * (z/(z+1.)) ) )
    b = -0.621 - 0.059 * (z/(z+1.))
    g = 1.055 + 0.838 * (z/(z+1)) - 3.083 * ( ((z/(z+1)))**2 )

    lmhalo = lmstar + np.log10(0.5) + np.log10(n) + np.log10( np.power( np.power(10., (lmstar-log_m1)), b ) + \
                np.power( np.power(10., (lmstar-log_m1)), g ) )

    Mhalo =  np.power(10., lmhalo)

    return Mhalo

def _halo_conc_relation(z=None, lmhalo=None):
    # From Dutton+14
    # Fitting function: Eq 7
    #   log10conc = a + b * log10(Mhalo / (1e12 h^{-1} Msun))
    # Slope, zpt: NFW conc200: Eqs 10, 11:
    #   b = -0.101 + 0.026 * z
    #   a = 0.520 + (0.905-0.520)*exp(-0.617*z^1.21)

    # PLANCK COSMOLOGY
    planck_cosmo = apy_cosmo.FlatLambdaCDM(H0=67.1, Om0=0.3175)
    hinv = 100./planck_cosmo.H(z).value

    b = -0.101 + 0.026*z
    a = 0.520 + (0.905-0.520)*np.exp(-0.617*np.power(z,1.21))

    log10conc = a + b * (lmhalo - np.log10(1.e12 * hinv))

    return np.power(10.,log10conc)


def _int_disp_z_evol_U19(z=None):
    # Fit from Uebler+19.
    # sigma0 = a + b*z
    # Using the "Including upper limts" coefficients, from Table 3
    a = 21.1
    b = 11.3

    sig0 = a + b*z
    return sig0



def _tomczak14_SMF_total_coeffs():

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
    dict_t14 = _tomczak14_SMF_total_coeffs()

    kindtmp = kind #copy.deepcopy(kind)
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


def _single_schecter_func(logM, logMstar, logphistar, alpha):
    return _double_schecter_func(logM, logMstar,
                logphistar, alpha,
                -np.inf, -99.)

def _double_schecter_func(logM, logMstar,
            logphistar1, alpha1,
            logphistar2, alpha2):
    #
    phistar1 = np.power(10,logphistar1)
    phistar2 = np.power(10,logphistar2)
    if ~np.isfinite(logphistar2):
        phistar2 = 0.
    phi = np.log(10.)*\
        np.exp(-np.power(10.,(logM-logMstar)))*\
        np.power(10.,(logM-logMstar))*\
        (phistar1*np.power(10.,((logM-logMstar)*(alpha1))) +\
         phistar2*np.power(10.,((logM-logMstar)*(alpha2))) )
    return np.log10(phi)


def _num_density_single_schechter_func(lmass_arr, logMstar, alpha, logphistar):
    phistar = np.power(10,logphistar)
    if ~np.isfinite(logphistar):
        phistar = 0.

    x = np.power(10.,((lmass_arr-logMstar)))

    if phistar == 0.:
        narr = lmass_arr * 0.
    else:
        if (alpha+1.) >= 0.:
            narr = phistar*scp_spec.gammaincc(alpha+1, x)* scp_spec.gamma(alpha+1)
        else:
            gammainc = 1./(alpha+1.) * ( scp_spec.gammaincc(alpha+2., x)*scp_spec.gamma(alpha+2.) \
                                            - np.power(x , alpha+1.)*np.exp(-x) )
            narr = phistar*gammainc

    return narr


def _num_density_double_schechter_func(lmass_arr, logMstar,
                alpha1, logphistar1, alpha2, logphistar2):
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
                    gammainc_n_m_1 = 1./(s-i)*(gammainc_n   - np.power(x , s-i)*np.exp(-x))
                gammainc = gammainc_n_m_1


            narr_comp = phistar*gammainc
        narr = narr + narr_comp


    return narr


def _num_density_tomczak14_double_schechter_interp(z, lmass_arr, kind='nearest'):

    d_t14_int = _tomczak14_SMF_total_coeffs_interp(z, kind=kind)

    n_t14_interp = _num_density_double_schechter_func(lmass_arr,
                d_t14_int['logMstar'], d_t14_int['alpha1'], d_t14_int['logphistar1'],
                d_t14_int['alpha2'], d_t14_int['logphistar2'])

    return n_t14_interp

def _interp_mass_from_tomczak14_num_density(lnz, z, kind_cmf_interp='nearest'):
    dlmass = 0.05
    lmass_lims = [9., 12]
    lmass_arr = np.arange(lmass_lims[0], lmass_lims[1]+dlmass, dlmass)

    narr = _num_density_tomczak14_double_schechter_interp(z, lmass_arr, kind=kind_cmf_interp)

    minterp_func = scp_interp.interp1d(np.log10(narr), lmass_arr, fill_value=np.NaN,  #'extrapolate',
                        bounds_error=False, kind='cubic')
    minterp = minterp_func(lnz)
    return minterp

def _papovich15_values_MW_M31():
    # Fig 4: Values from the thick dashed/solid lines: inferenced from Moster+13 models

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

def _interp_mass_from_papovich15_z(ln0, zarr):
    dp15 = _papovich15_values_MW_M31()

    if ln0 == -2.9:
        key = 'MW'
    elif ln0 == -3.4:
        key = 'M31'

    minterp_func = scp_interp.interp1d(dp15[key]['z'], dp15[key]['lmass'], fill_value='extrapolate',
                        bounds_error=False, kind='quadratic')
    minterp = minterp_func(zarr)


    return minterp

def _mass_progenitor_num_density(ln0, zarr, n_evol=None, cmf_source=None, kind_cmf_interp='quadratic'):
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
        # LOOK UP VALUES
        lnzarr = None


    lmass_arr = np.ones(len(zarr)) * -99.
    for i in range(len(zarr)):
        if cmf_source == 'tomczak14':
            lmass_arr[i] = _interp_mass_from_tomczak14_num_density(lnzarr[i], zarr[i], kind_cmf_interp=kind_cmf_interp)

    if cmf_source == 'papovich15':
        lmass_arr = _interp_mass_from_papovich15_z(ln0, zarr)

    return lmass_arr


def _kretschmer21_table1_alpha(rtoRe):
    x = rtoRe - 1.
    a, b, c = -0.146, 1.204, 1.475
    return a * x**2 + b * x + c


def _dalcanton_stilp10_alpha_n(rtoRe, n):
    # 0.92 * 1.68 convertion from rd to Re ## ONLY EXP
    #return 1.5456 * rtoRe
    # GENERAL Sersic n
    return 0.92 * (bn_func(n)/n) * np.power(rtoRe, 1./n)

def _consthz_sersic_Sigman_alpha(rtoRe, n):
    return (bn_func(n)/n) * np.power(rtoRe, 1./n)

def _SG_Sersicn_alpha(rtoRe, n):
    return 2.*(bn_func(n)/n) * np.power(rtoRe, 1./n)
