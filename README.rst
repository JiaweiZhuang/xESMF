xESMF: Use ESMF regridding through xarray
===========================================

|Build Status|

xESMF aims to combine ESMF_/ESMPy_'s regridding power and xarray_'s elegance.

Supports all `ESMF regridding algorithms <https://www.earthsystemcog.org/projects/esmf/regridding>`_:

- bilinear
- first-order conservative
- nearest neighbour (either source- or destination-based)
- high-order patch

Supports remapping between arbitrary quadrilateral (i.e. logically rectilinear) grids.

(Irregular meshes like hexagonal grids don't fit very well with xarray's data model.
ESMPy's Mesh Class might help.)

Installation
------------

Building ESMF/ESMPy from the source code is very daunting. Fortunately,
NESII_ provides a pre-compiled, Python3.6-compatible `anaconda package
<https://anaconda.org/NESII/esmpy>`_ for ESMPy::

    $ conda install -c nesii/label/dev-esmf -c conda-forge esmpy==7.1.0.dev38

It is on NESII's own channel but it also needs to pull dependencies from conda-forge.

**This Python3-compatible ESMPy is currently only available on Linux.** Mac or Windows users can
use `docker-miniconda <https://hub.docker.com/r/continuumio/miniconda3/>`_ as a temporary solution.

Installing the rest of packages is straightforward::

    $ conda install xarray
    $ pip install git+https://github.com/JiaweiZhuang/xESMF.git

Get Started
-----------

TODO: add examples and docs

.. _ESMF: https://www.earthsystemcog.org/projects/esmf/
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _xarray: http://xarray.pydata.org
.. _NESII: https://www.esrl.noaa.gov/gsd/nesii/


.. |Build Status| image:: https://api.travis-ci.org/JiaweiZhuang/xESMF.svg
   :target: https://travis-ci.org/JiaweiZhuang/xESMF
   :alt: travis-ci build status
