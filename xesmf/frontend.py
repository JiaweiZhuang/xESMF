'''
Frontend for xESMF, exposed to users.
'''

import numpy as np
import xarray as xr
import os

from . backend import esmf_regrid_build, esmf_regrid_finalize
from . util import ds_to_ESMFgrid
from . smm import read_weights, apply_weights


def _check_center_dim(ds_in, ds_out):
    """
    make sure the cell center dimension agree
    """
    assert ds_in['lon'].ndim == 2, "dimension of ds_in['lon'] should be 2"

    dim_name = ds_in['lon'].dims
    assert ds_in['lat'].dims == dim_name, (
           "ds_in['lat'] should have the same dimension name as ds_in['lon']")
    assert ds_out['lon'].dims == ds_out['lat'].dims == dim_name, (
           "ds_out['lon'] and ds_out['lat'] should have the same "
           "dimension name as ds_in['lon']")

    return dim_name


def _check_bound_dim(ds_in, ds_out):
    """
    make sure the cell bound dimension agree
    """

    # existence
    assert ('lon_b' in ds_in) and ('lat_b' in ds_in), (
           "conservative method need 'lon_b' and 'lat_b' in ds_in")
    assert ('lon_b' in ds_out) and ('lat_b' in ds_out), (
           "conservative method need 'lon_b' and 'lat_b' in ds_out")

    # rank
    assert ds_in['lon_b'].ndim == 2, "dimension of ds_in['lon_b'] should be 2"

    # naming
    dim_b_name = ds_in['lon_b'].dims
    assert ds_in['lat_b'].dims == dim_b_name, (
           'lat_b should have the same dimension name as lon_b')
    assert ds_out['lon_b'].dims == ds_out['lat_b'].dims == dim_b_name, (
           "ds_out['lon_b'] and ds_out['lat_b'] should have the same"
           "dimension name as ds_in['lon_b']")

    return dim_b_name


def _check_bound_shape(ds):
    """
    make sure bound has shape N+1 compared to center
    """
    s = ds['lon'].shape
    s_b = ds['lon_b'].shape

    assert (s_b[0] == s[0]+1) and (s_b[1] == s[1]+1), (
           "ds['lon'] should be one box larger than ds['lon_b']")


class Regridder(object):
    def __init__(self, ds_in, ds_out, method,
                 filename=None, reuse_weights=False):
        """
        Make xESMF regridder

        Parameters
        ----------
        ds_in, ds_out : xarray DataSet
            Contain input and output grid coordinates. Look for variables
            'lon', 'lat', and optionally 'lon_b', 'lat_b' for conservative
            method.

        method : str, optional
            Regridding method. Options are
            - 'bilinear'
            - 'conservative', need grid corner information
            - 'patch'
            - 'nearest_s2d'
            - 'nearest_d2s'

        filename : bool, optional
            Name for the weight file. The default naming scheme is
            {method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc,
            e.g. bilinear_400x600_300x400.nc

        reuse_weights : bool, optional
            Whether to read existing weight file to save computing time.
            False by default (i.e. re-compute, not reuse).

        Returns
        -------
        regridder : xESMF regridder object

        """

        # TODO: also accept 1D coodinate array for rectilinear grids
        self.dim_name = _check_center_dim(ds_in, ds_out)

        # get grid shape information
        # Use (Ny, Nx) instead of (Nlat, Nlon),
        # because ds can be general curvilinear grids
        # For rectilinear grids, (Ny, Nx) == (Nlat, Nlon)
        self.Ny_in, self.Nx_in = ds_in['lon'].shape
        self.Ny_out, self.Nx_out = ds_out['lon'].shape
        self.N_in = ds_in['lon'].size
        self.N_out = ds_out['lon'].size

        # only copy coordinate values do not copy data
        self.coords_in = ds_in.coords.to_dataset().copy()
        self.coords_out = ds_out.coords.to_dataset().copy()

        if method == 'conservative':
            self.dim_b_name = _check_bound_dim(ds_in, ds_out)
            _check_bound_shape(ds_in)
            _check_bound_shape(ds_out)
        else:
            self.dim_b_name = None

        self.method = method
        self.reuse_weights = reuse_weights

        if filename is None:
            # e.g. bilinear_400x600_300x400.nc
            filename = ('{0}_{1}x{2}_{3}x{4}.nc'.format(method,
                        self.Ny_in, self.Nx_in,
                        self.Ny_out, self.Nx_out)
                        )
        self.filename = filename

        self.write_weights(ds_in, ds_out)
        self.A = read_weights(self.filename, self.N_in, self.N_out)

    def __str__(self):
        info = ('xESMF Regridder \n'
                'Regridding algorithm:       {} \n'
                'Weight filename:            {} \n'
                'Reuse pre-computed weights? {} \n'
                'Input grid shape:           {} \n'
                'Output grid shape:          {} \n'
                'Grid dimension name:        {} \n'
                .format(self.method,
                        self.filename,
                        self.reuse_weights,
                        (self.Ny_in, self.Nx_in),
                        (self.Ny_out, self.Nx_out),
                        self.dim_name
                        )
                )

        if self.method == 'conservative':
            info += 'Boundary dimension name:    {} \n'.format(self.dim_b_name)

        return info

    def __repr__(self):
        return self.__str__()

    def __call__(self, a):
        """
        Shortcut for regrid_numpy() and regrid_dataarray()
        """
        # TODO: DataSet support

        if isinstance(a, np.ndarray):
            regrid_func = self.regrid_numpy
        elif isinstance(a, xr.DataArray):
            regrid_func = self.regrid_dataarray
        else:
            raise TypeError("input must be numpy array or xarray DataArray!")

        return regrid_func(a)

    def write_weights(self, ds_in, ds_out):
        """
        Write offline weight file, which will be read in at the next step.
        """

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

        grid_in = ds_to_ESMFgrid(ds_in)
        grid_out = ds_to_ESMFgrid(ds_out)
        regrid = esmf_regrid_build(grid_in, grid_out, self.method,
                                   filename=self.filename)

        # we only need the weight file, not the regrid object
        esmf_regrid_finalize(regrid)

    def regrid_numpy(self, indata):
        """
        Regrid pure numpy array

        """
        outdata = apply_weights(self.A, indata, self.Ny_out, self.Nx_out)
        return outdata

    def regrid_dataarray(self, dr_in):
        """
        Regrid xarray DataArray, track metadata.

        Parameters
        ----------
        dr_in : xarray DataArray
            The rightmost two dimensions must be the same as ds_in.
            Can have arbitrary additional dimensions.

            Examples of valid dimensions:
            - (Nlat, Nlon), if ds_in has shape (Nlat, Nlon)
            - (N2, N1, Ny, Nx), if ds_in has shape (Ny, Nx)

        Returns
        -------
        dr_out : xarray DataArray
            On the same horizontal grid as ds_out, with extra dims in dr_in.

            Examples of returning dimensions,
            assuming ds_out has the shape of (Ny_out, Nx_out):
            - (Ny_out, Nx_out), if dr_in is 2D
            - (N2, N1, Ny_out, Nx_out), if dr_in has shape (N2, N1, Ny, Nx)
        """

        # check dimension naming
        dims_in = dr_in.dims
        dims_extra = dims_in[0:-2]
        dims_horiz = dims_in[-2:]

        assert dims_horiz == self.dim_name, (
                'The horizontal two dimensions of dr_in are {}, different from'
                ' that of the regridder {}!'.format(dims_horiz, self.dim_name)
                )

        # check shape
        shape_horiz = dr_in.shape[-2:]
        assert shape_horiz == (self.Ny_in, self.Nx_in), (
             'The horizontal shape of dr_in are {}, different from that of '
             'the regridder {}!'.format(shape_horiz, (self.Ny_in, self.Nx_in))
             )

        # apply regridding to pure numpy array
        outdata = self.regrid_numpy(dr_in.values)

        # use a tempory DataSet to get horizontal grid information
        varname = dr_in.name
        ds_out_temp = self.coords_out.copy()
        ds_out_temp[varname] = (dims_in, outdata)  # same dim name as dr_in
        dr_out = ds_out_temp[varname]

        # recover coordinate values for extra dimensions
        for dim in dims_extra:
            dr_out.coords[dim] = dr_in.coords[dim]

        # TODO: add Attributes like regridding method

        return dr_out

    def regrid_dataset(self, ds_in):
        raise NotImplementedError("Only support regrid_dataarray() for now.")
