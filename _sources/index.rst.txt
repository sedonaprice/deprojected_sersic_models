.. sersic_profile_mass_VC documentation master file, created by
   sphinx-quickstart on Fri Jun 25 14:05:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========================
``sersic_profile_mass_VC``
==========================


The ``sersic_profile_mass_VC`` package contains code to calculate various profiles
for deprojected, flattened (and also elongated) Sérsic mass distributions.
These calculations follow and extend the derivation of rotation curves for flattened Sérsic profiles
presented by `Noordermeer, 2008, MNRAS, 385, 1359`_.
Full details about the calculations in this
package and in the precomputed tables are given in `Price et al., in prep, 2022`_.


.. _Noordermeer, 2008, MNRAS, 385, 1359: https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract
.. _Price et al., in prep, 2022: LINK_ADS

As these calculations require numerical integration, it is also possible to
reload pre-computed profiles from tables and interpolate as needed.

Tables for a wide range of Sérsic index :math:`n` and intrinsic axis ratio :math:`q_0`
are available for :ref:`download<downloads>`.


.. _quick_start:

Quickstart
==========

.. plot::
    :include-source:

    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import sersic_profile_mass_VC as spm

    # Environment variable containing path to location of pre-computed tables
    table_dir = os.getenv('SERSIC_PROFILE_MASS_VC_DATADIR')

    # Sérsic profile properties
    total_mass = 1.e11
    Reff = 5.0
    n = 1.0
    R = np.arange(0., 30.1, 0.1)

    # Flattening/elongation array (invq = 1/q; q = c/a)
    invq_arr = [1., 2.5, 3.33, 5., 10.]

    # Calculate & plot interpolated circular velocity profiles at r for each invq
    plt.figure(figsize=(4,3.5))
    for invq in invq_arr:
        vc = spm.interpolate_sersic_profile_VC(R=R, total_mass=total_mass, Reff=Reff,
                                           n=n, invq=invq, path=table_dir)
        plt.plot(R, vc, '-', label=r'$q_0$={:0.2f}'.format(1./invq))

    plt.xlabel('Radius [kpc]')
    plt.ylabel('Circular velocity [km/s]')
    plt.legend(title='Intrinsic axis ratio')

    plt.tight_layout()
    plt.show()




.. toctree::
   :hidden:

   self


.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation.rst
   table_downloads.rst
   acknowledgement.rst
   api.rst

.. toctree::
  :maxdepth: 1
  :caption: Tutorials

  sersic_profile_mass_VC_table_interp_example.ipynb
  sersic_profile_mass_VC_profile_example.ipynb
  sersic_profile_mass_VC_plot_example.ipynb



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
