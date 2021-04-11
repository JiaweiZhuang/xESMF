Why invent a new regridding package
===================================

For scientific correctness
--------------------------

Traditional interpolation routines, such as
`interp2d in Scipy <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.interpolate.interp2d.html>`_
and
`interp2 in MATLAB <https://www.mathworks.com/help/matlab/ref/interp2.html>`_,
assume flat 2D planes and do not consider the spherical geometry of the earth.
They are great for image processing, but will produce incorrect/distorted results for geospatial data.

Also, traditional interpolation algorithms are typically based on piecewise polynomials ("splines").
While being highly accurate in terms of error convergence, they often lack desired physical properties such as
conservation (total mass should be conserved) and monotonicity (air density cannot go negative).

For emerging new grid types
---------------------------

Non-orthogonal grids are becoming popular in numerical models
(`Staniforth and Thuburn 2012 <http://onlinelibrary.wiley.com/doi/10.1002/qj.958/full>`_),
but traditional tools often assume standard lat-lon grids.

xESMF can regrid between general curvilinear (i.e. quadrilateral or "logically rectilinear") grids, like

- The `Cubed-Sphere <http://acmg.seas.harvard.edu/geos/cubed_sphere.html>`_ grid
  in `GFDL-FV3 <https://www.gfdl.noaa.gov/fv3/>`_
- The `Latitude-Longitude-Cap grid <https://www.geosci-model-dev.net/8/3071/2015/>`_
  in `MITgcm <http://mitgcm.org>`_
- The `Lambert Conformal grid <https://en.wikipedia.org/wiki/Lambert_conformal_conic_projection>`_
  in WRF

However, xESMF does not yet support non-quadrilateral grids,
like the hexagonal grid in `MPAS <https://mpas-dev.github.io>`_.
See :ref:`irregular_meshes-label` for more information.

For usability and simplicity
----------------------------

:ref:`Current geospatial regridding tools <other_tools-label>` tend to have non-trivial learning curves.
xESMF tries to be simple and intuitive.
Instead of inventing a new data structure, it relies on well-estabilished standards
(numpy and xarray), so users don't need to learn a bunch of new syntaxes or even a new software stack.

xESMF can track metadata in ``xarray.DataArray`` / ``xarray.Dataset``, and
also work with basic ``numpy.ndarray``.
This means any Python users can use it easily, even if being unfamiliar with xarray.

The choice of Python and Anaconda also makes xESMF :ref:`extremely easy to install <installation-label>`.
