.. _install:
.. highlight:: shell

============
Installation
============

.. _install_source:

From Source
-----------

``sersic_profile_mass_VC`` can be installed from source.
Before installing the package, you will need to install python (v3)
and the dependent packages (`numpy`, `scipy`, `matplotlib`, `astropy`, `dill`).

.. tip::
    To handle dependencies (including possible clashes with other packages),
    we recommend using a separate python environment (such as an Anaconda python environment).

    The Anaconda `installation`_ and `environments`_ guides lead through the necessary steps
    to create a new Anaconda python environment.

.. _installation: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html



The package can be installed from the command line as follows:

.. code-block:: console

    $ tar zxvf sersic_profile_mass_VC-N.N.N.tar.gz
    $ cd sersic_profile_mass_VC-N.N.N
    $ python setup.py install

where N.N.N should be replaced with the current version number.
After the installation is complete, you should be able to access the module by running
`import sersic_profile_mass_VC` within python.


.. _clone_repo:

Code Repository
---------------

The most up-to-date version of the code can be obtained
by cloning the repository on GitHub.
From within the target code location directory, run:

 .. code-block:: console

    $ git clone git://github.com/sedonaprice/sersic_profile_mass_VC.git

This installation directory should also be added to your system `$PYTHONPATH` variable.
