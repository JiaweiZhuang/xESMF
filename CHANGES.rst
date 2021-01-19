What's new
==========

0.5.2 (01-20-2011)
------------------

Bug fixes
~~~~~~~~~

* Restore original behavior for lon/lat discovery, uses cf-xarray if lon/lat not found in dataset (:pull:`64`)
* Solve issue of dimension order in dataset (#53) with (:pull:`66`)

0.5.1 (01-11-2011)
------------------

Documentation
~~~~~~~~~~~~~
* Update installation instructions to mention that PyPi only holds xesmf up to version 0.3.0.

New features
~~~~~~~~~~~~
* Regridded xarray.Dataset now preserves the name and attributes of target coordinates (:pull:`60`)

Bug fixes
~~~~~~~~~
* Fix doc build for API/Regridder (:pull:`61`)


0.5.0 (11-11-2020)
------------------

Breaking changes
~~~~~~~~~~~~~~~~
* Deprecate `esmf_grid` in favor of `Grid.from_xarray`
* Deprecate `esmf_locstream` in favor of `LocStream.from_xarray`
* Installation requires numpy>=1.16 and cf-xarray>=0.3.1

New features
~~~~~~~~~~~~
* Create `ESMF.Mesh` objects from `shapely.polygons` (:pull:`24`). By `Pascal Bourgault <https://github.com/aulemahal>`_
* New class `SpatialAverager` offers user-friendly mechanism to average a 2-D field over a polygon. Includes support to handle interior holes and multi-part geometries. (:pull:`24`) By `Pascal Bourgault <https://github.com/aulemahal>`_
* Automatic detection of coordinates and computation of vertices based on cf-xarray. (:pull:`49`) By `Pascal Bourgault <https://github.com/aulemahal>`_

Bug fixes
~~~~~~~~~
* Fix serialization bug when using dask's distributed scheduler (:pull:`39`).
  By `Pascal Bourgault <https://github.com/aulemahal>`_.

Internal changes
~~~~~~~~~~~~~~~~
* Subclass `ESMF.Mesh` and create `from_polygon` method
* Subclass `ESMF.Grid` and `ESMF.LocStream` and create `from_xarray` methods.
* New `BaseRegridder` class, with support for `Grid`, `LocStream` and `Mesh` objects. Not all regridding methods are supported for `Mesh` objects.
* Refactor `Regridder` to subclass `BaseRegridder`.


0.4.0 (01-10-2020)
------------------
The git repo is now hosted by pangeo-data (https://github.com/pangeo-data/xESMF)

Breaking changes
~~~~~~~~~~~~~~~~
* By default, weights are not written to disk, but instead kept in memory.
* Installation requires ESMPy 8.0.0 and up.

New features
~~~~~~~~~~~~
* The `Regridder` object now takes a `weights` argument accepting a scipy.sparse COO matrix,
  a dictionary, an xarray.Dataset, or a path to a netCDF file created by ESMF. If None, weights
  are computed and can be written to disk using the `to_netcdf` method. This `weights` parameter
  replaces the `filename` and `reuse_weights` arguments, which are preserved for backward compatibility (:pull:`3`).
  By `David Huard <https://github.com/huard>`_ and `Raphael Dussin <https://github.com/raphaeldussin>`_
* Added documentation discussion how to compute weights from a shell using MPI, and reuse from xESMF (:pull:`12`).
  By `Raphael Dussin <https://github.com/raphaeldussin>`_
* Add support for masks in :py:func`esmf_grid`. This avoid NaNs to bleed into the interpolated values.
  When using a mask and the `conservative` regridding method, use a new method called
  `conservative_normed` to properly handle normalization (:pull:`1`).
  By `Raphael Dussin <https://github.com/raphaeldussin>`_


0.3.0 (06-03-2020)
------------------

New features
~~~~~~~~~~~~
* Add support for `ESMF.LocStream` `(#81) <https://github.com/JiaweiZhuang/xESMF/pull/81>`_
  By `Raphael Dussin <https://github.com/raphaeldussin>`_


0.2.2 (07-10-2019)
------------------

New features
~~~~~~~~~~~~
* Add option to allow degenerated grid cells `(#61) <https://github.com/JiaweiZhuang/xESMF/pull/61>`_
  By `Jiawei Zhuang <https://github.com/JiaweiZhuang>`_


0.2.0 (04-08-2019)
------------------

Breaking changes
~~~~~~~~~~~~~~~~
All user-facing APIs in v0.1.x should still work exactly the same. That said, because some internal codes have changed a lot, there might be unexpected edge cases that break current user code. If that happens, you can revert to the previous version by `pip install xesmf==0.1.2` and follow `old docs <https://xesmf.readthedocs.io/en/v0.1.2/>`_.

New features
~~~~~~~~~~~~
* Lazy evaluation on dask arrays (uses :py:func:`xarray.apply_ufunc` and :py:func:`dask.array.map_blocks`)
* Automatic looping over variables in an xarray Dataset
* Add tutorial notebooks on those new features

By `Jiawei Zhuang <https://github.com/JiaweiZhuang>`_


0.1.2 (03-08-2019)
------------------
This release mostly contains internal clean-ups to facilitate future development.

New features
~~~~~~~~~~~~
* Deprecates `regridder.A` in favor of `regridder.weights`
* Speed-up test suites by using coarser grids
* Use parameterized tests when appropriate
* Fix small memory leaks from `ESMF.Grid`
* Properly assert ESMF enums

By `Jiawei Zhuang <https://github.com/JiaweiZhuang>`_


0.1.1 (31-12-2017)
------------------
Initial release.
By `Jiawei Zhuang <https://github.com/JiaweiZhuang>`_
