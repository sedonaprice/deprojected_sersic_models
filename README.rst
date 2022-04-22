***************************************************************************
Non-spherical deprojected Sersic mass profiles and circular velocity curves
***************************************************************************

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge


Description
###########
Package to calculate various profiles for deprojected, flattened (or elongated)
Sersic mass distributions, including:
enclosed mass, circular velocity, density, log density slope, surface density,
and projected enclosed mass.

These calculations follow and extend the derivation of rotation curves for flattened
Sersic bulges presented by `Noordermeer, 2008, MNRAS, 385, 1359`_.
Further details about the calculations included in this package
are described in `Price et al., in prep, 2022`_.

.. _Noordermeer, 2008, MNRAS, 385, 1359: https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract
.. _Price et al., in prep, 2022: LINK_TO_ADS

Please see `the documentation`_ for this package for detailed information about installation,
usage, and to download the set of pre-computed Sersic profile tables.

.. _the documentation: https://sersic_profile_mass_VC.github.io/


Usage
#####

.. code-block:: python

    import os
    import numpy as np
    import sersic_profile_mass_VC as spm
    table_dir = os.getenv('SERSIC_PROFILE_MASS_VC_DATADIR')

    # Sersic profile properties & radius array
    total_mass = 1.e11
    Reff = 5.0
    n = 1.0
    invq = 5.    # Oblate, q = c/a = 0.2

    r = np.arange(0., 30.1, 0.1)

    # Load & interpolate all profiles in saved table:
    table_interp = spm.interpolate_entire_table(r=r, total_mass=total_mass,
                                                Reff=Reff, n=n, invq=invq,
                                                path=table_dir)


Dependencies
###########
* numpy
* scipy
* matplotlib
* astropy
* dill


Acknowledgement
###############
If you use this package or the precomputed profile tables in a publication,
please cite Price et al., 2022, in prep (`ADS`_ | `arXiv`_).

.. _ADS: LINK_TO_ADS
.. _arXiv: LINK_TO_ARXIV



License
###########
This project is Copyright (c) Sedona Price / MPE IR/Submm Group and licensed
under the terms of the BSD 3-Clause license. See the LICENSE.rst for more information.
