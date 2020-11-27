.. _installation-label:

Installation
============

Try on Binder without local installation
----------------------------------------

The `Binder project <https://mybinder.readthedocs.io>`_ provides pre-configured environment in the cloud. You just need a web browser to access it. Please follow the Binder link on `xESMF's GitHub page <https://github.com/pangeo-data/xESMF>`_.

Install on local machine with Conda
-----------------------------------

xESMF requires Python>=3.6. The major dependencies are xarray and ESMPy. The best way to install them is using Conda_.

First, `install miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_. Then, we recommend creating a new, clean environment:

.. code-block:: bash

    $ conda create -n xesmf_env python=3.7
    $ conda activate xesmf_env

Getting xESMF is as simple as:

.. code-block:: bash

    $ conda install -c conda-forge xesmf

.. warning::

    One some platforms you might get :code:`ImportError: Regrid(filename) requires PIO and does not work if ESMF has not been built with MPI support`. (see `this comment <https://github.com/JiaweiZhuang/xESMF/issues/47#issuecomment-582421822>`_). A quick workaround is to constrain ESMPy version :code:`conda install -c conda-forge xesmf esmpy=8.0.0`.

We also highly recommend those extra packages for full functionality:

.. code-block:: bash

    # to support all features in xESMF
    $ conda install -c conda-forge dask netCDF4

    # optional dependencies for executing all notebook examples
    $ conda install -c conda-forge matplotlib cartopy jupyterlab

Alternatively, you can first install dependencies, and then use ``pip`` to install xESMF:

.. code-block:: bash

    $ conda install -c conda-forge esmpy xarray scipy dask netCDF4
    $ pip install xesmf

Testing your installation
-------------------------

xESMF itself is a lightweight package, but its dependency ESMPy is a quite heavy and sometimes might be installed incorrectly. To validate & debug your installation, you can use pytest to run the test suites:

.. code-block:: bash

    $ pip install pytest
    $ pytest -v --pyargs xesmf  # should all pass

A common cause of error (especially for HPC cluster users) is that pre-installed modules like NetCDF, MPI, and ESMF are incompatible with the conda-installed equivalents. Make sure you have a clean environment when running ``conda install`` (do not ``module load`` other libraries). See `this issue <https://github.com/JiaweiZhuang/xESMF/issues/55#issuecomment-514298498>`_ for more discussions.

Notes for Windows users
-----------------------

The ESMPy conda package is currently only available for Linux and Mac OSX.
Windows users can try the
`Linux subsystem <https://docs.microsoft.com/en-us/windows/wsl/about>`_
or `docker-miniconda <https://hub.docker.com/r/continuumio/miniconda3/>`_ .

Installing scientific software on Windows can often be a pain, and
`Docker <https://www.docker.com>`_ is a pretty good workaround.
It takes some time to learn but worths the effort.
Check out this `tutorial on using Docker with Anaconda
<https://towardsdatascience.com/
how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5>`_.

This problem is being investigated.
See `this other issue <https://github.com/conda-forge/esmpy-feedstock/issues/8>`_.

Install development version from GitHub repo
--------------------------------------------

To get the latest version that is not uploaded to PyPI_ yet::

    $ pip install --upgrade git+https://github.com/pangeo-data/xESMF.git

Developers can track source code change::

    $ git clone https://github.com/pangeo-data/xESMF.git
    $ cd xESMF
    $ pip install -e .

.. _xarray: http://xarray.pydata.org
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _Conda: https://docs.conda.io/
.. _PyPI: https://pypi.python.org/pypi
.. _NESII: https://www.esrl.noaa.gov/gsd/nesii/
