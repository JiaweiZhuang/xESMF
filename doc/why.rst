Why inventing a new regridding package
======================================

For scientific correctness
--------------------------

There exist many interpolation routines, such as
`interp2d in Scipy <https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.interpolate.interp2d.html>`_
and
`interp2 in MATLAB <https://www.mathworks.com/help/matlab/ref/interp2.html>`_.
They are great for image processing, but will lead to incorrect/distorted results for geospatial data,
because they assume flat 2D plane and do not consider spherical geometry of the earth.

Also, traditional interpolation algorithms are typically based on piecewise polynomials ("splines").
While being highly accurate in terms of error convergence, they often lack desired physical properties such as
conservation (total mass should be conserved) and motononicity (air density cannot go negative).

For new grid types
------------------

the Cubed-Sphere grid in GFDL-FV3,
the Lambert Conformal projection in WRF,
the Latitude-Longitude-Cap grid in MITgcm,

:ref:`irregular_meshes-label`

For simplicity and usability
----------------------------

:ref:`other_tools-label`

xESMF can track the metadata in xarray ``DataArray`` and ``DataSet``,
but also accept the basic ``numpy.ndarray``.
Don't learn a new API. Also see "Other regridding tools".
