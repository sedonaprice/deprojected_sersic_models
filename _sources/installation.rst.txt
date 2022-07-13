.. _install:
.. highlight:: shell

============
Installation
============

.. _install_source:

From Source
-----------

``deprojected_sersic_models`` can be installed from source.
Before installing the package, you will need to install python (v3)
and the dependent packages (`numpy`, `scipy`, `matplotlib`, `astropy`, `dill`).

.. tip::
    To handle dependencies (including possible clashes with other packages),
    we recommend using a separate python environment (such as an Anaconda python environment).

    The Anaconda `installation`_ and `environments`_ guides lead through the necessary steps
    to create a new Anaconda python environment.

.. _installation: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html
.. _environments: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


**Download v1.1**: `tar.gz`_ | `zip`_

.. _tar.gz: https://github.com/sedonaprice/deprojected_sersic_models/archive/refs/tags/v1.1.tar.gz
.. _zip: https://github.com/sedonaprice/deprojected_sersic_models/archive/refs/tags/v1.1.zip

The most recent release version can be downloaded from the `repository`_.

.. _repository: https://github.com/sedonaprice/deprojected_sersic_models/releases

The package can then be installed from the command line as follows:

.. code-block:: console

    $ tar zxvf deprojected_sersic_models-N.N.N.tar.gz
    $ cd deprojected_sersic_models-N.N.N
    $ python setup.py install

where N.N.N should be replaced with the current version number.
After the installation is complete, you should be able to access the module by running
`import deprojected_sersic_models` within python.


.. _clone_repo:

Code Repository
---------------

The most up-to-date version of the code can be obtained
by cloning the repository on GitHub: `https://github.com/sedonaprice/deprojected_sersic_models`_.
From within the target code location directory, run:

.. _https://github.com/sedonaprice/deprojected_sersic_models: https://github.com/sedonaprice/deprojected_sersic_models

 .. code-block:: console

    $ git clone git://github.com/sedonaprice/deprojected_sersic_models.git

This installation directory should also be added to your system `$PYTHONPATH` variable.
