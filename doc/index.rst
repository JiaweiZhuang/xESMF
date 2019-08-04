xESMF: Universal Regridder for Geospatial Data
==============================================

xESMF is a Python package for
`regridding <https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/regridding-overview>`_.
It is

- **Powerful**: It uses ESMF_/ESMPy_ as backend and can regrid between **general curvilinear grids**
  with all `ESMF regridding algorithms <https://www.earthsystemcog.org/projects/esmf/regridding>`_,
  such as **bilinear**, **conservative** and **nearest neighbour**.
- **Easy-to-use**: It abstracts away ESMF's complicated infrastructure
  and provides a simple, high-level API, compatible with xarray_ as well as basic numpy arrays.
- **Fast**: It is faster than ESMPy's original Fortran regridding engine in serial case
  (surprise! here's :doc:`benchmark <./Backend>`), and also supports dask_ for `out-of-core, parallel computation http://xarray.pydata.org/en/stable/dask.html>`_.

.. note::

    xESMF is at early stage of development and will keep improving in the future.
    The major regridding API should be quite stable,
    but minor ultilities could change in the next version.
    If you find any bugs or would like to request any enhancements,
    please `raise an issue on GitHub <https://github.com/JiaweiZhuang/xESMF/issues>`_.

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Overview

   why
   other_tools
   limitations

.. toctree::
   :maxdepth: 1
   :caption: Beginner tutorials

   installation
   Rectilinear_grid
   Curvilinear_grid
   Pure_numpy

.. toctree::
   :maxdepth: 1
   :caption: Intermediate tutorials

   Compare_algorithms
   Reuse_regridder

.. toctree::
   :maxdepth: 1
   :caption: Technical notes

   Backend

.. toctree::
   :maxdepth: 1
   :caption: API

   user_api
   internal_api

.. _ESMF: https://www.earthsystemcog.org/projects/esmf/
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _xarray: http://xarray.pydata.org
.. _dask: https://dask.org/