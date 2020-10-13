#################################################################
# Licensed under a 3-clause BSD style license - see LICENSE.rst #
#################################################################


# ---------------------------------------------------------------
try:
    from table_generation import wrapper_calculate_full_table_set, \
                            wrapper_calculate_sersic_profile_tables, \
                            calculate_sersic_profile_table
    from calcs import interpolate_sersic_profile_menc_nearest, \
                            interpolate_sersic_profile_VC_nearest, interpolate_sersic_profile_menc, \
                            interpolate_sersic_profile_VC, \
                            v_circ, M_encl_2D, M_encl_3D, M_encl_3D_ellip, \
                            virial_coeff_tot, virial_coeff_3D, \
                            find_rhalf3D_sphere, qobs_func, \
                            nearest_n_invq
    
except:
    from .table_generation import wrapper_calculate_full_table_set, \
                            wrapper_calculate_sersic_profile_tables, \
                            calculate_sersic_profile_table
    from .calcs import interpolate_sersic_profile_menc_nearest, \
                            interpolate_sersic_profile_VC_nearest, interpolate_sersic_profile_menc, \
                            interpolate_sersic_profile_VC, \
                            v_circ, M_encl_2D, M_encl_3D, M_encl_3D_ellip, \
                            virial_coeff_tot, virial_coeff_3D, \
                            find_rhalf3D_sphere, qobs_func, \
                            nearest_n_invq

# ---------------------------------------------------------------
# Should define tests here


