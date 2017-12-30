Installation
============

The quickest way
----------------

xESMF depends on xarray and ESMPy. The best way to install them is using Anaconda_::

    $ conda install xarray
    $ conda install -c nesii/label/dev-esmf -c conda-forge esmpy

(ESMPy is currently on NESII_'s `own channel <https://anaconda.org/NESII/esmpy>`_.
but also needs to pull dependencies from conda-forge.)

Then install xesmf::

    $ pip install xesmf

Notes for Windows users
-----------------------

The ESMPy conda package is currently only available Linux and Mac OSX.
Windows users can try the Linux subsystem
or `docker-miniconda <https://hub.docker.com/r/continuumio/miniconda3/>`_ .

`Docker tutorial <https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5>`_

Also see
https://github.com/conda-forge/esmpy-feedstock/issues/8

Install xESMF from source
-------------------------

To get the latest version::

    $ pip install --upgrade git+https://github.com/JiaweiZhuang/xESMF.git

To track source code change::

    $ git clone https://github.com/JiaweiZhuang/xESMF.git
    $ cd xESMF
    $ pip install -e .

.. _xarray: http://xarray.pydata.org
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _Anaconda: https://www.continuum.io/downloads
.. _PyPI: https://pypi.python.org/pypi
.. _NESII: https://www.esrl.noaa.gov/gsd/nesii/
