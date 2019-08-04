'''
Frontend for xESMF, exposed to users.
'''

import numpy as np
import xarray as xr
import os
import warnings

from . backend import (esmf_grid, add_corner,
                       esmf_regrid_build, esmf_regrid_finalize)

from . smm import read_weights, apply_weights


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

    # tranpose the arrays so they become Fortran-ordered
    grid = esmf_grid(lon.T, lat.T, periodic=periodic)

    if need_bounds:
        lon_b = np.asarray(ds['lon_b'])
        lat_b = np.asarray(ds['lat_b'])
        lon_b, lat_b = as_2d_mesh(lon_b, lat_b)
        add_corner(grid, lon_b.T, lat_b.T)

    return grid, lon.shape


class Regridder(object):
    def __init__(self, ds_in, ds_out, method, periodic=False,
                 filename=None, reuse_weights=False):
        """
        Make xESMF regridder

        Parameters
        ----------
        ds_in, ds_out : xarray DataSet, or dictionary
            Contain input and output grid coordinates. Look for variables
            ``lon``, ``lat``, and optionally ``lon_b``, ``lat_b`` for
            conservative method.

            Shape can be 1D (n_lon,) and (n_lat,) for rectilinear grids,
            or 2D (n_y, n_x) for general curvilinear grids.
            Shape of bounds should be (n+1,) or (n_y+1, n_x+1).

        method : str
            Regridding method. Options are

            - 'bilinear'
            - 'conservative', **need grid corner information**
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

        Returns
        -------
        regridder : xESMF regridder object

        """

        # record basic switches
        if method == 'conservative':
            self.need_bounds = True
            periodic = False  # bound shape will not be N+1 for periodic grid
        else:
            self.need_bounds = False

        self.method = method
        self.periodic = periodic
        self.reuse_weights = reuse_weights

        # construct ESMF grid, with some shape checking
        self._grid_in, shape_in = ds_to_ESMFgrid(ds_in,
                                                 need_bounds=self.need_bounds,
                                                 periodic=periodic
                                                 )
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

            self.horiz_dims = self.lon_dim

        elif self._lon_out.ndim == 1:
            try:
                self.lon_dim, = ds_out['lon'].dims
                self.lat_dim, = ds_out['lat'].dims
            except:
                self.lon_dim = 'lon'
                self.lat_dim = 'lat'

            self.horiz_dims = (self.lat_dim, self.lon_dim)

        # record grid shape information
        self.shape_in = shape_in
        self.shape_out = shape_out
        self.n_in = shape_in[0] * shape_in[1]
        self.n_out = shape_out[0] * shape_out[1]

        if filename is None:
            self.filename = self._get_default_filename()
        else:
            self.filename = filename

        # get weight matrix
        self._write_weight_file()
        self.weights = read_weights(self.filename, self.n_in, self.n_out)

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

    def get_A(self):
        warnings.warn(
            "regridder.A is deprecated and will be removed in future versions. "
            "Use regridder.weights instead.", DeprecationWarning
            )
        print("Do not use A!")
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

    def _write_weight_file(self):

        if os.path.exists(self.filename):
            if self.reuse_weights:
                print('Reuse existing file: {}'.format(self.filename))
                return  # do not compute it again, just read it
            else:
                print('Overwrite existing file: {} \n'.format(self.filename),
                      'You can set reuse_weights=True to save computing time.')
                os.remove(self.filename)
        else:
            print('Create weight file: {}'.format(self.filename))

        regrid = esmf_regrid_build(self._grid_in, self._grid_out, self.method,
                                   filename=self.filename)
        esmf_regrid_finalize(regrid)  # only need weights, not regrid object

    def clean_weight_file(self):
        """
        Remove the offline weight file on disk.

        To save the time on re-computing weights, you can just keep the file,
        and set "reuse_weights=True" when initializing the regridder next time.
        """
        if os.path.exists(self.filename):
            print("Remove file {}".format(self.filename))
            os.remove(self.filename)
        else:
            print("File {} is already removed.".format(self.filename))

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
                        self.horiz_dims,
                        self.periodic)
                )

        return info

    def __call__(self, a):
        """
        Shortcut for ``regrid_numpy()`` and ``regrid_dataarray()``.

        Parameters
        ----------
        a : xarray DataArray or numpy array

        Returns
        -------
        xarray DataArray or numpy array
            Regridding results. Type depends on input.
        """
        # TODO: DataSet support

        if isinstance(a, np.ndarray):
            regrid_func = self.regrid_numpy
        elif isinstance(a, xr.DataArray):
            regrid_func = self.regrid_dataarray
        else:
            raise TypeError("input must be numpy array or xarray DataArray!")

        return regrid_func(a)

    def regrid_numpy(self, indata):
        """
        Regrid pure numpy array. Shape requirement is the same as
        ``regrid_dataarray()``

        Parameters
        ----------
        indata : numpy array

        Returns
        -------
        outdata : numpy array

        """

        # check shape
        shape_horiz = indata.shape[-2:]  # the rightmost two dimensions
        assert shape_horiz == self.shape_in, (
             'The horizontal shape of input data is {}, different from that of'
             'the regridder {}!'.format(shape_horiz, self.shape_in)
             )

        outdata = apply_weights(self.weights, indata,
                                self.shape_out[0], self.shape_out[1])
        return outdata

    def regrid_dataarray(self, dr_in):
        """
        Regrid xarray DataArray, track metadata.

        Parameters
        ----------
        dr_in : xarray DataArray
            The rightmost two dimensions must be the same as ``ds_in``.
            Can have arbitrary additional dimensions.

            Examples of valid shapes

            - (Nlat, Nlon), if ``ds_in`` has shape (Nlat, Nlon)
            - (N2, N1, Ny, Nx), if ``ds_in`` has shape (Ny, Nx)

        Returns
        -------
        dr_out : xarray DataArray
            On the same horizontal grid as ``ds_out``,
            with extra dims in ``dr_in``.

            Assuming ``ds_out`` has the shape of (Ny_out, Nx_out),
            examples of returning shapes are

            - (Ny_out, Nx_out), if ``dr_in`` is 2D
            - (N2, N1, Ny_out, Nx_out), if ``dr_in`` has shape
              (N2, N1, Ny, Nx)

        """

        # apply regridding to pure numpy array
        outdata = self.regrid_numpy(dr_in.values)

        # track metadata
        varname = dr_in.name
        extra_dims = dr_in.dims[0:-2]

        dr_out = xr.DataArray(outdata,
                              dims=extra_dims+self.horiz_dims,
                              name=varname)

        dr_out.coords['lon'] = xr.DataArray(self._lon_out, dims=self.lon_dim)
        dr_out.coords['lat'] = xr.DataArray(self._lat_out, dims=self.lat_dim)

        # append extra dimension coordinate value
        for dim in extra_dims:
            dr_out.coords[dim] = dr_in.coords[dim]

        dr_out.attrs['regrid_method'] = self.method

        return dr_out

    def regrid_dataset(self, ds_in):
        raise NotImplementedError("Only support regrid_dataarray() for now.")
