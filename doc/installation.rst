.. _installation-label:

Installation
============

Try on Binder without local installation
----------------------------------------

The `Binder project <https://mybinder.readthedocs.io>`_ provides pre-configured environment in the cloud. You just need a web browser to access it. Please follow the Binder link on `xESMF's GitHub page <https://github.com/JiaweiZhuang/xESMF>`_.

Install on local machine with Conda
-----------------------------------

xESMF requires Python>=3.5. The major dependencies are xarray and ESMPy. The best way to install them is using Conda_.

`Installing miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html>`_, and then install dependencies in a new environment:

.. code-block:: bash

    # recommend creating a new, clean environment
    $ conda create -n xesmf_env python=3.7
    $ conda activate xesmf_env

    # install common dependencies for fully functionality
    $ conda install -c conda-forge esmpy xarray scipy dask netCDF4

    # optional dependencies for executing all notebook examples
    $ conda install -c conda-forge matplotlib cartopy jupyterlab

After dependencies are properly installed, get xesmf:

.. code-block:: bash

    $ pip install xesmf

.. warning::

    The conda channel is not yet actively maintained by the author. ``conda install -c conda-forge xesmf`` can give you out-dated versions. For now, stick to ``pip install`` for xesmf itself.

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
See `this issue <https://github.com/conda-forge/esmpy-feedstock/issues/8>`_.

Install development version from GitHub repo
--------------------------------------------

To get the latest version that is not uploaded to PyPI_ yet::

    $ pip install --upgrade git+https://github.com/JiaweiZhuang/xESMF.git

Developers can track source code change::

    $ git clone https://github.com/JiaweiZhuang/xESMF.git
    $ cd xESMF
    $ pip install -e .

.. _xarray: http://xarray.pydata.org
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _Conda: https://docs.conda.io/
.. _PyPI: https://pypi.python.org/pypi
.. _NESII: https://www.esrl.noaa.gov/gsd/nesii/
