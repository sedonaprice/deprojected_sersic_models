.. sersic_profile_mass_VC documentation master file, created by
   sphinx-quickstart on Fri Jun 25 14:05:17 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

======================
sersic_profile_mass_VC
======================


The `sersic_profile_mass_VC` package contains code to calculate various profiles
for deprojected, flattened (and also prolate) Sersic mass distributions.

As these calculations require numerical integration, it is also possible to
reload pre-computed profiles from tables and interpolate as needed.

Tables for a wide range of Sersic index :math:`n` and intrinsic axis ratio :math:`q`
are available for download.


Quickstart
==========

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
    for invq in invq_arr:
        vc = spm.interpolate_sersic_profile_VC(r=r, total_mass=total_mass, Reff=Reff,
                                           n=n, invq=invq, path=table_dir)
        plt.plot(r, vc, '-', label='q={:0.2f}'.format(1./invq))

    plt.xlabel('Radius [kpc]')
    plt.ylabel('Circular velocity [km/s]')
    plt.legend(title='Intrinic axis ratio')

    plt.tight_layout()
    plt.show()


.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation.rst
   table_downloads.rst
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
