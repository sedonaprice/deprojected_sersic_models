Flattened Sersic profile mass and circular velocity curves
-------------------------------------------

.. image:: http://img.shields.io/badge/powered%20by-AstroPy-orange.svg?style=flat
    :target: http://www.astropy.org
    :alt: Powered by Astropy Badge


Description
------------
Package to calculate various profiles for deprojected, flattened (or also prolate)
Sersic mass distributions, including:
enclosed mass, circular velocity, density, log density slope, surface density,
and projected enclosed mass.

These calculations follow and extend the derivation of rotation curves for flattened
Sersic bulges presented by `Noordermeer, 2008, MNRAS, 385, 1359`_.
Further details about the calculations included in this package
are described in `Price et al., in prep 2021`_.

.. _Noordermeer, 2008, MNRAS, 385, 1359: https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract
.. _Price et al., in prep 2021: tofix

Please see `the documentation`_ for this package for detailed information about installation,
usage, and to download the set of **_pre-computed_** Sersic profile tables.

.. _the documentation: https://sersic_profile_mass_VC.github.io/


Usage
------------

.. plot::
    :include-source:

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import sersic_profile_mass_VC as spm
    table_dir = os.getenv('SERSIC_PROFILE_MASS_VC_DATADIR')

    # Sersic profile properties
    total_mass = 1.e11
    Reff = 5.0
    n = 1.0
    r = np.arange(0., 30.1, 0.1)

    # Flattening array (invq = 1/q)
    invq_arr = [1., 2.5, 3.33, 5., 10.]

    # Calculate & plot interpolated circular velocity profiles at r for each invq
    plt.figure(figsize=(4,3.5))
    for invq in invq_arr:
        vc = spm.interpolate_sersic_profile_VC(r=r, total_mass=total_mass, Reff=Reff,
                                           n=n, invq=invq, path=table_dir)
        plt.plot(r, vc, '-', label='q={:0.2f}'.format(1./invq))

    plt.xlabel('Radius [kpc]')
    plt.ylabel('Circular velocity [km/s]')
    plt.legend(title='Intrinic axis ratio')

    plt.tight_layout()
    plt.show()


Dependencies
------------
* numpy
* scipy
* matplotlib
* astropy


License
-------
This project is Copyright (c) Sedona Price / MPE IR/Submm Group and licensed
under the terms of the BSD 3-Clause license. See the LICENSE.rst for more information.
