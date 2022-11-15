.. deprojected_sersic_models documentation master file, created by
   sphinx-quickstart on Fri Jun 25 14:05:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============================
``deprojected_sersic_models``
=============================


The ``deprojected_sersic_models`` package contains code to calculate various profiles
for deprojected, flattened (and also elongated) Sérsic mass distributions.
These calculations follow and extend the derivation of rotation curves for flattened Sérsic profiles
presented by `Noordermeer, 2008, MNRAS, 385, 1359`_.
Full details about the calculations in this
package and in the precomputed tables are given in `Price et al., 2022, A&A 665 A159`_.

.. _Noordermeer, 2008, MNRAS, 385, 1359: https://ui.adsabs.harvard.edu/abs/2008MNRAS.385.1359N/abstract
.. _Price et al., 2022, A&A 665 A159: https://ui.adsabs.harvard.edu/abs/2022A%26A...665A.159P/abstract

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
    import deprojected_sersic_models as deproj_sersic

    # Environment variable containing path to location of pre-computed tables
    table_dir = os.getenv('DEPROJECTED_SERSIC_MODELS_DATADIR')

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
        vc = deproj_sersic.interpolate_sersic_profile_VC(R=R, total_mass=total_mass, Reff=Reff,
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
   downloads.rst
   acknowledgement.rst
   api.rst

.. toctree::
  :maxdepth: 1
  :caption: Tutorials

  deprojected_sersic_models_table_interp_example.ipynb
  deprojected_sersic_models_profile_example.ipynb
  deprojected_sersic_models_plot_example.ipynb



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
