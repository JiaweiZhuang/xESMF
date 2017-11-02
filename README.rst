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
NESII_ provides a pre-compiled, Python3.6-compatible `anaconda package
<https://anaconda.org/NESII/esmpy>`_ for ESMPy::

    $ conda config --add channels conda-forge  
    $ conda install -c nesii/label/dev-esmf esmpy

It is on NESII's own channel but it also needs to pull dependencies from conda-forge.

**This Python3-compatible ESMPy is currently only available on Linux.** Mac or Windows users can
use `docker-miniconda <https://hub.docker.com/r/continuumio/miniconda3/>`_ as a temporary solution.

Installing the rest of packages is straightforward::

    $ conda install xarray
    $ pip install git+https://github.com/JiaweiZhuang/xESMF.git 

Get Started
-----------

See `example notebook <examples/illustration_highlevel.ipynb>`_

Design Idea
-----------

ESMF -> ESMPy -> low-level numpy wrapper -> high-level xarray wrapper

The numpy wrapper just simplifies ESMPy's API without adding too many customizations. 
This low level wrapper should be relatively stable. 
Advanced designs can be added to the xarray-wrapper level. 

Note for developrs:

- To build the high-level wrapper based on the low-level one,
  see `this notebook for using the low-level wrapper <examples/illustration_lowlevel.ipynb>`_
  and `highlevel.py <xesmf/highlevel.py>`_ for current implementation. 

- To modify the low-level wrapper,
  see this tutorial on ESMPy: https://github.com/nawendt/esmpy-tutorial
  (much more accessible than `ESMPy's official tutorial
  <http://www.earthsystemmodeling.org/esmf_releases/last_built/esmpy_doc/html/examples.html>`_)
  and `lowlevel.py <xesmf/lowlevel.py>`_ for current implementation.

Issues & Plans
--------------

- Dask intergration. ESMF/ESMPy parallelizes in the horizontal, using MPI for horizontal domain decomposition. 
  With dask, parallelizing over other dimensions (e.g. lev, time) would be a much better option.

- Dump offline regridding weights. 
  Currently the regridding weights are hided in the Fortran level and invisible in the Python level.

- For multi-tiled grids like `Cubed-Sphere <https://github.com/JiaweiZhuang/cubedsphere>`_,
  Conservative regridding will work correcly by just regridding each tile and adding up the results. 
  But other methods do not correctly handle tile edges.

- Improve Masking. Currently, np.nan will be mapped to np.nan, but masking created by np.ma will be ignored.

- Improve API design. Current API is just experimental, more like a proof-of-concept. 
  The trickest part is matching xarray's coordinate dimension with ESMPy's expectation.


.. _ESMF: https://www.earthsystemcog.org/projects/esmf/
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _xarray: http://xarray.pydata.org
.. _NESII: https://www.esrl.noaa.gov/gsd/nesii/
