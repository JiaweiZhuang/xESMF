xESMF: Use ESMF regridding through xarray
===========================================

xESMF aims to combine ESMF_/ESMPy_'s regridding power and xarray_'s elegance.

Supports all `ESMF regridding algorithms <https://www.earthsystemcog.org/projects/esmf/regridding>`_,
including: 

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
NEISS_ provides a pre-compiled, Python3.6-compatible `anaconda package
<https://anaconda.org/NESII/esmpy>`_ for ESMPy::

    $ conda config --add channels conda-forge  
    $ conda install -c nesii/label/dev-esmf esmpy

It is on NESII's own channel but it also needs to pull dependencies from conda-forge.

**This Python3-compatible ESMPy is currently only available on Linux.** Mac or Windows users can
use `docker-miniconda <https://hub.docker.com/r/continuumio/miniconda/>`_ as a temporary solution.

Installing the rest of packages is straightforward::

    $ conda install xarray
    $ pip install git+https://github.com/JiaweiZhuang/xESMF.git 

Get Started
-----------

See `example notebook <illustration_highlevel.ipynb>`_

Design Idea
-----------

ESMF -> ESMPy -> low-level numpy wrapper -> high-level xarray wrapper

Using low-level numpy wrapper is still much easier than using the original ESMPy.
See `this module <xesmf/lowlevel.py>`_. 

The numpy wrapper just simplifies ESMPy's API without adding too many customizations. 
This low level wrapper should be relatively stable. 
Advanced designs can be added to the xarray-wrapper level. 

Issues & Plans
--------------

- Dask intergration. ESMF/ESMPy parallelizes in the horizontal, using MPI for horizontal domain decomposition. 
  With dask, parallelizing over other dimensions (e.g. lev, time) would be a much better option.

- Dump offline regridding weights. 
  Currently the regridding weights is hided in the Fortran level and invisible in the Python level.

- For multi-tiled grids like `Cubed-Sphere <http://acmg.seas.harvard.edu/geos/cubed_sphere.html>`_,
  Conservative regridding will work correcly by just regridding each tile and adding up the results. 
  But other methods don't not correctly handle tile edges.

Additional Links
----------------
- A modern tutorial on ESMPy: https://github.com/nawendt/esmpy-tutorial, 
which is much more accessible than `the official tutorial 
<http://www.earthsystemmodeling.org/esmf_releases/last_built/esmpy_doc/html/examples.html>`_.


.. _ESMF: https://www.earthsystemcog.org/projects/esmf/
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _xarray: http://xarray.pydata.org
.. _NESII: https://www.esrl.noaa.gov/gsd/nesii/
