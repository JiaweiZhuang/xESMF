What's new
==========

0.4.x
-----
The git repo is now hosted by pangeo-data (https://github.com/pangeo-data/xESMF)

Breaking changes
~~~~~~~~~~~~~~~~
* By default, weights are not written to disk, but instead kept in memory.

New features
~~~~~~~~~~~~
* The `Regridder` object now takes a `weights` argument accepting a scipy.sparse COO matrix,
  a dictionary, an xarray.Dataset, or a path to a netCDF file created by ESMF. If None, weights
  are computed and can be written to disk using the `to_netcdf` method. This `weights` parameter
  replaces the `filename` and `reuse_weights` arguments, which are preserved for backward compatibility.








