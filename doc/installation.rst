.. _installation-label:

Installation
============

The quickest way
----------------

xESMF requires Python>=3.5. The major dependencies are xarray and ESMPy.
The best way to install them is using Anaconda_.::

    $ conda install xarray
    $ conda install -c conda-forge esmpy

Then install xesmf::

    $ pip install xesmf

Done! You can go to the next tutorial.

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

Install xESMF from GitHub repo
------------------------------

To get the latest version that is not uploaded to PyPI_ yet::

    $ pip install --upgrade git+https://github.com/JiaweiZhuang/xESMF.git

Developers can track source code change::

    $ git clone https://github.com/JiaweiZhuang/xESMF.git
    $ cd xESMF
    $ pip install -e .

.. _xarray: http://xarray.pydata.org
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _Anaconda: https://www.continuum.io/downloads
.. _PyPI: https://pypi.python.org/pypi
.. _NESII: https://www.esrl.noaa.gov/gsd/nesii/
