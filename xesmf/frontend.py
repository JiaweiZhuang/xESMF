'''
Frontend for xESMF, exposed to users.
'''

import numpy as np
import xarray as xr
import os
import warnings

from . backend import (esmf_grid, esmf_locstream, add_corner,
                       esmf_regrid_build, esmf_regrid_finalize)

from . smm import read_weights, apply_weights, add_nans_to_weights

try:
    import dask.array as da
    dask_array_type = (da.Array,)  # for isinstance checks
except ImportError:
    dask_array_type = ()

def as_2d_mesh(lon, lat):

    if (lon.ndim, lat.ndim) == (2, 2):
        assert lon.shape == lat.shape, 'lon and lat should have same shape'
    elif (lon.ndim, lat.ndim) == (1, 1):
        lon, lat = np.meshgrid(lon, lat)
    else:
        raise ValueError('lon and lat should be both 1D or 2D')

    return lon, lat


def ds_to_ESMFgrid(ds, need_bounds=False, periodic=None, append=None):
    '''
    Convert xarray DataSet or dictionary to ESMF.Grid object.

    Parameters
    ----------
    ds : xarray DataSet or dictionary
        Contains variables ``lon``, ``lat``,
        and optionally ``lon_b``, ``lat_b`` if need_bounds=True.

        Shape should be ``(n_lat, n_lon)`` or ``(n_y, n_x)``,
        as normal C or Python ordering. Will be then tranposed to F-ordered.

    need_bounds : bool, optional
        Need cell boundary values?

    periodic : bool, optional
        Periodic in longitude?

    Returns
    -------
    grid : ESMF.Grid object

    '''

    # use np.asarray(dr) instead of dr.values, so it also works for dictionary
    lon = np.asarray(ds['lon'])
    lat = np.asarray(ds['lat'])
    lon, lat = as_2d_mesh(lon, lat)

    if 'mask' in ds:
        mask = np.asarray(ds['mask'])
    else:
        mask = None

    # tranpose the arrays so they become Fortran-ordered
    if mask is not None:
        grid = esmf_grid(lon.T, lat.T, periodic=periodic, mask=mask.T)
    else:
        grid = esmf_grid(lon.T, lat.T, periodic=periodic, mask=None)

    if need_bounds:
        lon_b = np.asarray(ds['lon_b'])
        lat_b = np.asarray(ds['lat_b'])
        lon_b, lat_b = as_2d_mesh(lon_b, lat_b)
        add_corner(grid, lon_b.T, lat_b.T)

    return grid, lon.shape


def ds_to_ESMFlocstream(ds):
    '''
    Convert xarray DataSet or dictionary to ESMF.LocStream object.

    Parameters
    ----------
    ds : xarray DataSet or dictionary
        Contains variables ``lon``, ``lat``.

    Returns
    -------
    locstream : ESMF.LocStream object

    '''

    lon = np.asarray(ds['lon'])
    lat = np.asarray(ds['lat'])

    if len(lon.shape) > 1:
        raise ValueError("lon can only be 1d")
    if len(lat.shape) > 1:
        raise ValueError("lat can only be 1d")

    assert lon.shape == lat.shape

    locstream = esmf_locstream(lon, lat)

    return locstream, (1,) + lon.shape


class Regridder(object):
    def __init__(self, ds_in, ds_out, method, periodic=False,
                 filename=None, reuse_weights=False,
                 extrap_method=None, extrap_dist_exponent=None,
                 extrap_num_src_pnts=None,
                 weights=None, ignore_degenerate=None,
                 locstream_in=False, locstream_out=False):
        """
        Make xESMF regridder

        Parameters
        ----------
        ds_in, ds_out : xarray DataSet, or dictionary
            Contain input and output grid coordinates. Look for variables
            ``lon``, ``lat``, optionally ``lon_b``, ``lat_b`` for
            conservative methods, and ``mask``. Note that for `mask`,
            the ESMF convention is used, where masked values are identified
            by 0, and non-masked values by 1.

            Shape can be 1D (n_lon,) and (n_lat,) for rectilinear grids,
            or 2D (n_y, n_x) for general curvilinear grids.
            Shape of bounds should be (n+1,) or (n_y+1, n_x+1).

            If either dataset includes a 2d mask variable, that will also be
            used to inform the regridding.

        method : str
            Regridding method. Options are

            - 'bilinear'
            - 'conservative', **need grid corner information**
            - 'conservative_normed', **need grid corner information**
            - 'patch'
            - 'nearest_s2d'
            - 'nearest_d2s'

        periodic : bool, optional
            Periodic in longitude? Default to False.
            Only useful for global grids with non-conservative regridding.
            Will be forced to False for conservative regridding.

        filename : str, optional
            Name for the weight file. The default naming scheme is::

                {method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc

            e.g. bilinear_400x600_300x400.nc

        reuse_weights : bool, optional
            Whether to read existing weight file to save computing time.
            False by default (i.e. re-compute, not reuse).

        extrap_method : str, optional
            Extrapolation method. Options are

            - 'inverse_dist'
            - 'nearest_s2d'

        extrap_dist_exponent : float, optional
            The exponent to raise the distance to when calculating weights for the
            extrapolation method. If none are specified, defaults to 2.0

        extrap_num_src_pnts : int, optional
            The number of source points to use for the extrapolation methods
            that use more than one source point. If none are specified, defaults to 8

        weights : None, coo_matrix, dict, str, Dataset, Path,
            Regridding weights, stored as
              - a scipy.sparse COO matrix,
              - a dictionary with keys `row_dst`, `col_src` and `weights`,
              - an xarray Dataset with data variables `col`, `row` and `S`,
              - or a path to a netCDF file created by ESMF.
            If None, compute the weights.

        ignore_degenerate : bool, optional
            If False (default), raise error if grids contain degenerated cells
            (i.e. triangles or lines, instead of quadrilaterals)

        locstream_in: bool, optional
            input is a LocStream (list of locations)

        locstream_out: bool, optional
            output is a LocStream (list of locations)

        Returns
        -------
        regridder : xESMF regridder object

        """

        # record basic switches
        if method in ['conservative', 'conservative_normed']:
            self.need_bounds = True
            periodic = False  # bound shape will not be N+1 for periodic grid
        else:
            self.need_bounds = False

        self.method = method
        self.periodic = periodic
        self.reuse_weights = reuse_weights
        self.extrap_method = extrap_method
        self.extrap_dist_exponent = extrap_dist_exponent
        self.extrap_num_src_pnts = extrap_num_src_pnts
        self.ignore_degenerate = ignore_degenerate
        self.locstream_in = locstream_in
        self.locstream_out = locstream_out

        methods_avail_ls_in = ['nearest_s2d', 'nearest_d2s']
        methods_avail_ls_out = ['bilinear', 'patch'] + methods_avail_ls_in

        if locstream_in and self.method not in methods_avail_ls_in:
            raise ValueError(f'locstream input is only available for method in {methods_avail_ls_in}')
        if locstream_out and self.method not in methods_avail_ls_out:
            raise ValueError(f'locstream output is only available for method in {methods_avail_ls_out}')

        # construct ESMF grid, with some shape checking
        if locstream_in:
            self._grid_in, shape_in = ds_to_ESMFlocstream(ds_in)
        else:
            self._grid_in, shape_in = ds_to_ESMFgrid(ds_in,
                                                     need_bounds=self.need_bounds,
                                                     periodic=periodic
                                                     )
        if locstream_out:
            self._grid_out, shape_out = ds_to_ESMFlocstream(ds_out)
        else:
            self._grid_out, shape_out = ds_to_ESMFgrid(ds_out,
                                                       need_bounds=self.need_bounds
                                                       )

        # record output grid and metadata
        self._lon_out = np.asarray(ds_out['lon'])
        self._lat_out = np.asarray(ds_out['lat'])

        if self._lon_out.ndim == 2:
            try:
                self.lon_dim = self.lat_dim = ds_out['lon'].dims
            except:
                self.lon_dim = self.lat_dim = ('y', 'x')

            self.out_horiz_dims = self.lon_dim

        elif self._lon_out.ndim == 1:
            try:
                self.lon_dim, = ds_out['lon'].dims
                self.lat_dim, = ds_out['lat'].dims
            except:
                self.lon_dim = 'lon'
                self.lat_dim = 'lat'

            self.out_horiz_dims = (self.lat_dim, self.lon_dim)

        # record grid shape information
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.n_in = shape_in[0] * shape_in[1]
        self.n_out = shape_out[0] * shape_out[1]

        # some logic about reusing weights with either filename or weights args
        if reuse_weights and (filename is None) and (weights is None):
            raise ValueError("to reuse weights, you need to provide either filename or weights")

        if not reuse_weights and weights is None:
            weights = self._compute_weights()  # Dictionary of weights
        else:
            weights = filename if filename is not None else weights

        assert weights is not None

        # Convert weights, whatever their format, to a sparse coo matrix
        self.weights = read_weights(weights, self.n_in, self.n_out)

        # replace zeros by NaN in mask
        if 'mask' in ds_out:
            self.weights = add_nans_to_weights(self.weights)

        # follows legacy logic of writing weights if filename is provided
        if filename is not None and not reuse_weights:
            self.to_netcdf(filename=filename)

        # set default weights filename if none given
        self.filename = self._get_default_filename() if filename is None else filename

    @property
    def A(self):
        message = (
            "regridder.A is deprecated and will be removed in future versions. "
            "Use regridder.weights instead."
        )

        warnings.warn(message, DeprecationWarning)
        # DeprecationWarning seems to be ignored by certain Python environments
        # Also print to make sure users notice this.
        print(message)
        return self.weights

    def _get_default_filename(self):
        # e.g. bilinear_400x600_300x400.nc
        filename = ('{0}_{1}x{2}_{3}x{4}'.format(self.method,
                    self.shape_in[0], self.shape_in[1],
                    self.shape_out[0], self.shape_out[1],)
                    )
        if self.periodic:
            filename += '_peri.nc'
        else:
            filename += '.nc'

        return filename

    def _compute_weights(self):
        regrid = esmf_regrid_build(self._grid_in, self._grid_out, self.method,
                                   extrap_method = self.extrap_method,
                                   extrap_dist_exponent = self.extrap_dist_exponent,
                                   extrap_num_src_pnts = self.extrap_num_src_pnts,
                                   ignore_degenerate=self.ignore_degenerate)

        w = regrid.get_weights_dict(deep_copy=True)
        esmf_regrid_finalize(regrid)  # only need weights, not regrid object
        return w

    def __repr__(self):
        info = ('xESMF Regridder \n'
                'Regridding algorithm:       {} \n'
                'Weight filename:            {} \n'
                'Reuse pre-computed weights? {} \n'
                'Input grid shape:           {} \n'
                'Output grid shape:          {} \n'
                'Output grid dimension name: {} \n'
                'Periodic in longitude?      {}'
                .format(self.method,
                        self.filename,
                        self.reuse_weights,
                        self.shape_in,
                        self.shape_out,
                        self.out_horiz_dims,
                        self.periodic)
                )

        return info

    def __call__(self, indata, keep_attrs=False):
        """
        Apply regridding to input data.

        Parameters
        ----------
        indata : numpy array, dask array, xarray DataArray or Dataset.
            The rightmost two dimensions must be the same as ``ds_in``.
            Can have arbitrary additional dimensions.

            Examples of valid shapes

            - (n_lat, n_lon), if ``ds_in`` has shape (n_lat, n_lon)
            - (n_time, n_lev, n_y, n_x), if ``ds_in`` has shape (Ny, n_x)

            Transpose your input data if the horizontal dimensions are not
            the rightmost two dimensions.

        keep_attrs : bool, optional
            Keep attributes for xarray DataArrays or Datasets.
            Defaults to False.

        Returns
        -------
        outdata : Data type is the same as input data type.
            On the same horizontal grid as ``ds_out``,
            with extra dims in ``dr_in``.

            Assuming ``ds_out`` has the shape of (n_y_out, n_x_out),
            examples of returning shapes are

            - (n_y_out, n_x_out), if ``dr_in`` is 2D
            - (n_time, n_lev, n_y_out, n_x_out), if ``dr_in`` has shape
              (n_time, n_lev, n_y, n_x)

        """

        if isinstance(indata, np.ndarray):
            return self.regrid_numpy(indata)
        elif isinstance(indata, dask_array_type):
            return self.regrid_dask(indata)
        elif isinstance(indata, xr.DataArray):
            return self.regrid_dataarray(indata, keep_attrs=keep_attrs)
        elif isinstance(indata, xr.Dataset):
            return self.regrid_dataset(indata, keep_attrs=keep_attrs)
        else:
            raise TypeError(
                "input must be numpy array, dask array, "
                "xarray DataArray or Dataset!")

    def regrid_numpy(self, indata):
        """See __call__()."""

        if self.locstream_in:
            indata = np.expand_dims(indata, axis=-2)

        outdata = apply_weights(self.weights, indata,
                                self.shape_in, self.shape_out)
        return outdata

    def regrid_dask(self, indata):
        """See __call__()."""

        extra_chunk_shape = indata.chunksize[0:-2]

        output_chunk_shape = extra_chunk_shape + self.shape_out

        outdata = da.map_blocks(
            self.regrid_numpy,
            indata,
            dtype=float,
            chunks=output_chunk_shape
        )

        return outdata

    def regrid_dataarray(self, dr_in, keep_attrs=False):
        """See __call__()."""

        # example: ('lat', 'lon') or ('y', 'x')
        if self.locstream_in:
            input_horiz_dims = dr_in.dims[-1:]
        else:
            input_horiz_dims = dr_in.dims[-2:]

        # apply_ufunc needs a different name for output_core_dims
        # example: ('lat', 'lon') -> ('lat_new', 'lon_new')
        # https://github.com/pydata/xarray/issues/1931#issuecomment-367417542
        if self.locstream_out:
            temp_horiz_dims = ['dummy', 'locations']
        else:
            temp_horiz_dims = [s + '_new' for s in input_horiz_dims]

        if self.locstream_in and not self.locstream_out:
            temp_horiz_dims = ['dummy_new'] + temp_horiz_dims


        dr_out = xr.apply_ufunc(
            self.regrid_numpy, dr_in,
            input_core_dims=[input_horiz_dims],
            output_core_dims=[temp_horiz_dims],
            dask='parallelized',
            output_dtypes=[float],
            output_sizes={temp_horiz_dims[0]: self.shape_out[0],
                          temp_horiz_dims[1]: self.shape_out[1]
                          },
            keep_attrs=keep_attrs
        )

        if not self.locstream_out:
            # rename dimension name to match output grid
            dr_out = dr_out.rename(
                {temp_horiz_dims[0]: self.out_horiz_dims[0],
                 temp_horiz_dims[1]: self.out_horiz_dims[1]
                }
            )

        # append output horizontal coordinate values
        # extra coordinates are automatically tracked by apply_ufunc
        if self.locstream_out:
            dr_out.coords['lon'] = xr.DataArray(self._lon_out, dims=('locations',))
            dr_out.coords['lat'] = xr.DataArray(self._lat_out, dims=('locations',))
        else:
            dr_out.coords['lon'] = xr.DataArray(self._lon_out, dims=self.lon_dim)
            dr_out.coords['lat'] = xr.DataArray(self._lat_out, dims=self.lat_dim)

        dr_out.attrs['regrid_method'] = self.method

        if self.locstream_out:
            dr_out = dr_out.squeeze(dim='dummy')

        return dr_out

    def regrid_dataset(self, ds_in, keep_attrs=False):
        """See __call__()."""

        # most logic is the same as regrid_dataarray()
        # the major caution is that some data variables might not contain
        # the correct horizontal dimension names.

        # get the first data variable to infer input_core_dims
        name, dr_in = next(iter(ds_in.items()))

        if self.locstream_in:
            input_horiz_dims = dr_in.dims[-1:]
        else:
            input_horiz_dims = dr_in.dims[-2:]

        if self.locstream_out:
            temp_horiz_dims = ['dummy', 'locations']
        else:
            temp_horiz_dims = [s + '_new' for s in input_horiz_dims]

        if self.locstream_in and not self.locstream_out:
            temp_horiz_dims = ['dummy_new'] + temp_horiz_dims

        # help user debugging invalid horizontal dimensions
        print('using dimensions {} from data variable {} '
              'as the horizontal dimensions for this dataset.'
              .format(input_horiz_dims, name)
              )

        ds_out = xr.apply_ufunc(
            self.regrid_numpy, ds_in,
            input_core_dims=[input_horiz_dims],
            output_core_dims=[temp_horiz_dims],
            dask='parallelized',
            output_dtypes=[float],
            output_sizes={temp_horiz_dims[0]: self.shape_out[0],
                          temp_horiz_dims[1]: self.shape_out[1]
                          },
            keep_attrs=keep_attrs
        )

        if not self.locstream_out:
            # rename dimension name to match output grid
            ds_out = ds_out.rename(
                {temp_horiz_dims[0]: self.out_horiz_dims[0],
                 temp_horiz_dims[1]: self.out_horiz_dims[1]
                }
            )

        # append output horizontal coordinate values
        # extra coordinates are automatically tracked by apply_ufunc
        if self.locstream_out:
            ds_out.coords['lon'] = xr.DataArray(self._lon_out, dims=('locations',))
            ds_out.coords['lat'] = xr.DataArray(self._lat_out, dims=('locations',))
        else:
            ds_out.coords['lon'] = xr.DataArray(self._lon_out, dims=self.lon_dim)
            ds_out.coords['lat'] = xr.DataArray(self._lat_out, dims=self.lat_dim)

        ds_out.attrs['regrid_method'] = self.method

        if self.locstream_out:
            ds_out = ds_out.squeeze(dim='dummy')

        return ds_out

    def to_netcdf(self, filename=None):
        '''Save weights to disk as a netCDF file.'''
        if filename is None:
            filename = self.filename
        w = self.weights
        dim = "n_s"
        ds = xr.Dataset({"S": (dim, w.data), "col": (dim, w.col + 1), "row": (dim, w.row + 1)})
        ds.to_netcdf(filename)
        return filename

