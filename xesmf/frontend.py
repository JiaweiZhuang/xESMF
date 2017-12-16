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

        # Use (Ny, Nx) instead of (Nlat, Nlon),
        # because ds can be general curvilinear grids
        # For rectilinear grids, (Ny, Nx) == (Nlat, Nlon)
        self.dim_name = _check_center_dim(ds_in, ds_out)

        self.Ny_in, self.Nx_in = ds_in['lon'].shape
        self.Ny_out, self.Nx_out = ds_out['lon'].shape
        self.N_in = ds_in['lon'].size
        self.N_out = ds_out['lon'].size

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

    def __call__(self, dr_in):
        return self.apply_weights(dr_in)

    def write_weights(self, ds_in, ds_out):

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

    def apply_weights(self, dr_in):
        indata = dr_in.values
        outdata = apply_weights(self.A, indata, self.Ny_out, self.Nx_out)

        # TODO: append metadata

        return outdata
