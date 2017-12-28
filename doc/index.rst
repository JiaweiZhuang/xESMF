xESMF: Universal Regridder for Geospatial Data
==============================================

xESMF aims to combine ESMF/ESMPy's regridding power and xarray's elegance.

xESMF provides you ...
----------------------

- **Multiple algorithms**.
  xESMF is able to use all ESMF regridding methods, including
  bilinear,
  first-order conservative,
  nearest neighbour (either source or destination based)
  and high-order patch recovery.

- **Aribtray quariteral grid support**.
  This means xESMF can deal with
  the Cubed-Sphere grid in GFDL-FV3,
  the Lambert Conformal projection in WRF,
  the Latitude-Longitude-Cap grid in MITgcm,
  etc.
  (Irregular meshes like hexagonal grids don't fit very well with xarray's data model.
  ESMF/ESMPy is able to handle irregular meshes but designing an elegant frontend for that is very challenging.)

- **Transparancy**.
  xESMF can track the metadata in xarray ``DataArray`` and ``DataSet``,
  but also accept the basic ``numpy.ndarray``.
  Don't learn a new API. Also see "Other regridding tools".

- **Speed**.
  xESMF uses ``scipy.sparse`` to apply the regridding operation
  and is 2~3 times faster than the native ESMPy version (link to benchmark).
  xESMF can cache the regridder and can sometimes be orders of magnitude faster
  than 1-step regridders. Also see "Why need two-step regridding".

Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   quick_start
   built_in_example

.. toctree::
   :maxdepth: 1
   :caption: API

   user_api
   internal_api

Why we need a new regridding package
------------------------------------

Scipy and MATLAB have interp2d. They are excellent for image processing
but not suited for geospatial data.
In geoscience we have various grid geometries on a sphere,
while traditional intepolation schemes assume flat 2D data.

We also need many special algorithms,
such as conserving the total amount of mass or flux.

Other regridding tools
----------------------

They are all great tools and helped the author a lot.

Native ESMF/ESMPy
https://www.earthsystemcog.org/projects/esmf/
https://www.earthsystemcog.org/projects/esmpy/

UV-CDAT
https://uvcdat.llnl.gov/documentation/cdms/cdms_4.html

NCL
https://www.ncl.ucar.edu/Applications/regrid.shtml

Iris
http://scitools.org.uk/iris/docs/v1.10.0/userguide/interpolation_and_regridding.html

TempestRemap
https://github.com/ClimateGlobalChange/tempestremap

Current limitations
-------------------
- **Irregular meshes**.

- **Vector regridding**.

- **Parallel regridding**.
