import os
import numpy as np
import ESMF
import xesmf as xe
from xesmf.backend import (warn_f_contiguous, warn_lat_range,
                           esmf_grid, add_corner,
                           esmf_regrid_build, esmf_regrid_apply,
                           esmf_regrid_finalize)
from xesmf.smm import read_weights, apply_weights

from numpy.testing import assert_equal, assert_almost_equal
import pytest

# We use pure numpy arrays to test backend
# xarray DataSet is only used at the very beginning as a quick way to make data
coord_names = ['lon', 'lat', 'lon_b', 'lat_b']

ds_in = xe.util.grid_global(20, 12)
lon_in, lat_in, lon_b_in, lat_b_in = [ds_in[name].values
                                      for name in coord_names]

ds_out = xe.util.grid_global(15, 9)
lon_out, lat_out, lon_b_out, lat_b_out = [ds_out[name].values
                                          for name in coord_names]

# shortcut to test a single grid
lon, lat, lon_b, lat_b = [lon_in, lat_in, lon_b_in, lat_b_in]

# input test data
ds_in['data'] = xe.data.wave_smooth(ds_in['lon'], ds_in['lat'])
data_in = ds_in['data'].values

# reference output data, calculated analytically
ds_out['data_ref'] = xe.data.wave_smooth(ds_out['lon'], ds_out['lat'])
data_ref = ds_out['data_ref'].values

# 4D data to test broadcasting, increasing linearly with time and lev
ds_in.coords['time'] = np.arange(1, 11)
ds_in.coords['lev'] = np.arange(1, 51)
ds_in['data4D'] = ds_in['time'] * ds_in['lev'] * ds_in['data']
data4D_in = ds_in['data4D'].values


def test_warn_f_on_array():
    a = np.zeros([2, 2], order='C')
    with pytest.warns(UserWarning):
        warn_f_contiguous(a)


def test_warn_f_on_grid():
    # should throw a warning if not passing transpose
    with pytest.warns(UserWarning):
        esmf_grid(lon, lat)


def test_warn_lat_range():
    # latitude goes to -100 (invalid value)
    ds_temp = xe.util.grid_2d(-180, 180, 10, -100, 90, 5)
    with pytest.warns(UserWarning):
        warn_lat_range(ds_temp['lat'].values)
    with pytest.warns(UserWarning):
        warn_lat_range(ds_temp['lat_b'].values)


def test_esmf_grid_with_corner():

    # only center coordinate, no corners
    # remember to pass transpose (F-ordered) to backend
    grid = esmf_grid(lon.T, lat.T)

    # make sure coordinate values agree
    assert_equal(grid.coords[0][0], lon.T)
    assert_equal(grid.coords[0][1], lat.T)

    # make sure meta data agree
    assert not grid.has_corners  # no corner yet!
    assert grid.staggerloc == [True, False, False, False]
    assert grid.coord_sys is ESMF.CoordSys.SPH_DEG
    assert grid.rank == 2
    assert_equal(grid.size[0], lon.T.shape)
    assert_equal(grid.upper_bounds[0], lon.T.shape)
    assert_equal(grid.lower_bounds[0], np.array([0, 0]))

    # now add corner information
    add_corner(grid, lon_b.T, lat_b.T)

    # coordinate values
    assert_equal(grid.coords[3][0], lon_b.T)
    assert_equal(grid.coords[3][1], lat_b.T)

    # metadata
    assert grid.has_corners  # should have corner now
    assert grid.staggerloc == [True, False, False, True]
    assert_equal(grid.size[3], lon_b.T.shape)
    assert_equal(grid.upper_bounds[3], lon_b.T.shape)
    assert_equal(grid.lower_bounds[3], np.array([0, 0]))


def test_esmf_build_bilinear():

    grid_in = esmf_grid(lon_in.T, lat_in.T)
    grid_out = esmf_grid(lon_out.T, lat_out.T)

    regrid = esmf_regrid_build(grid_in, grid_out, 'bilinear')
    assert regrid.unmapped_action is ESMF.UnmappedAction.IGNORE
    assert regrid.regrid_method is ESMF.RegridMethod.BILINEAR

    # they should share the same memory
    regrid.srcfield.grid is grid_in
    regrid.dstfield.grid is grid_out

    esmf_regrid_finalize(regrid)


def test_regrid():

    # use conservative regridding as an example,
    # since it is the most well-tested studied one in papers

    # TODO: possible to break this long test into smaller tests?
    # not easy due to strong dependencies.

    grid_in = esmf_grid(lon_in.T, lat_in.T)
    grid_out = esmf_grid(lon_out.T, lat_out.T)

    # no corner info yet, should not be able to use conservative
    with pytest.raises(ValueError):
        esmf_regrid_build(grid_in, grid_out, 'conservative')

    # now add corners
    add_corner(grid_in, lon_b_in.T, lat_b_in.T)
    add_corner(grid_out, lon_b_out.T, lat_b_out.T)

    # also write to file for scipy regridding
    filename = 'test_weights.nc'
    if os.path.exists(filename):
        os.remove(filename)
    regrid = esmf_regrid_build(grid_in, grid_out, 'conservative',
                               filename=filename)
    assert regrid.regrid_method is ESMF.RegridMethod.CONSERVE

    # apply regridding using ESMPy's native method
    data_out_esmpy = esmf_regrid_apply(regrid, data_in.T).T

    rel_err = (data_out_esmpy - data_ref)/data_ref  # relative error
    assert np.max(np.abs(rel_err)) < 0.05

    # apply regridding using scipy
    A = read_weights(filename, lon_in.size, lon_out.size)
    Nlat_out, Nlon_out = lon_out.shape
    data_out_scipy = apply_weights(A, data_in, Nlat_out, Nlon_out)

    # must be exactly the same as esmpy's result!
    # TODO: this fails once but I cannot replicate it.
    # Maybe assert_equal is too strict for scipy vs esmpy comparision
    assert_equal(data_out_scipy, data_out_esmpy)

    # finally, test broadcasting with scipy
    # TODO: need to test broadcasting with ESMPy backend?
    # We only use Scipy in frontend, and ESMPy is just for backend benchmark
    # However, it is useful to compare performance and show scipy is 3x faster
    data4D_out = apply_weights(A, data4D_in, Nlat_out, Nlon_out)

    # data over broadcasting dimensions should agree
    assert_almost_equal(data4D_in.mean(axis=(2, 3)),
                        data4D_out.mean(axis=(2, 3)),
                        decimal=10)

    # clean-up
    esmf_regrid_finalize(regrid)
    os.remove(filename)


def test_regrid_periodic_wrong():

    # not using periodic grid
    grid_in = esmf_grid(lon_in.T, lat_in.T)
    grid_out = esmf_grid(lon_out.T, lat_out.T)

    assert grid_in.num_peri_dims == 0
    assert grid_in.periodic_dim is None

    regrid = esmf_regrid_build(grid_in, grid_out, 'bilinear')
    data_out_esmpy = esmf_regrid_apply(regrid, data_in.T).T

    rel_err = (data_out_esmpy - data_ref)/data_ref  # relative error
    assert np.max(np.abs(rel_err)) == 1.0  # some data will be missing

    # clean-up
    esmf_regrid_finalize(regrid)


def test_regrid_periodic_correct():

    # only need to specific periodic for input grid
    grid_in = esmf_grid(lon_in.T, lat_in.T, periodic=True)
    grid_out = esmf_grid(lon_out.T, lat_out.T)

    assert grid_in.num_peri_dims == 1
    assert grid_in.periodic_dim == 0  # the first axis, longitude

    regrid = esmf_regrid_build(grid_in, grid_out, 'bilinear')
    data_out_esmpy = esmf_regrid_apply(regrid, data_in.T).T

    rel_err = (data_out_esmpy - data_ref)/data_ref  # relative error
    assert np.max(np.abs(rel_err)) < 0.065
    # clean-up
    esmf_regrid_finalize(regrid)
