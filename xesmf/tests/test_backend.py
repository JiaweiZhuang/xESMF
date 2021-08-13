import os

import ESMF
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_almost_equal, assert_equal

import xesmf as xe
from xesmf.backend import (
    Grid,
    LocStream,
    add_corner,
    esmf_regrid_apply,
    esmf_regrid_build,
    esmf_regrid_finalize,
    warn_f_contiguous,
    warn_lat_range,
)
from xesmf.smm import apply_weights, read_weights

# We use pure numpy arrays to test backend
# xarray DataSet is only used at the very beginning as a quick way to make data
coord_names = ['lon', 'lat', 'lon_b', 'lat_b']

ds_in = xe.util.grid_global(20, 12)
lon_in, lat_in, lon_b_in, lat_b_in = [ds_in[name].values for name in coord_names]

ds_out = xe.util.grid_global(15, 9)
lon_out, lat_out, lon_b_out, lat_b_out = [ds_out[name].values for name in coord_names]

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
        Grid.from_xarray(lon, lat)


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
    grid = Grid.from_xarray(lon.T, lat.T)

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

    grid_in = Grid.from_xarray(lon_in.T, lat_in.T)
    grid_out = Grid.from_xarray(lon_out.T, lat_out.T)

    regrid = esmf_regrid_build(grid_in, grid_out, 'bilinear')
    assert regrid.unmapped_action is ESMF.UnmappedAction.IGNORE
    assert regrid.regrid_method is ESMF.RegridMethod.BILINEAR

    # they should share the same memory
    regrid.srcfield.grid is grid_in
    regrid.dstfield.grid is grid_out

    esmf_regrid_finalize(regrid)


def test_esmf_extrapolation():

    grid_in = Grid.from_xarray(lon_in.T, lat_in.T)
    grid_out = Grid.from_xarray(lon_out.T, lat_out.T)

    regrid = esmf_regrid_build(grid_in, grid_out, 'bilinear')
    data_out_esmpy = esmf_regrid_apply(regrid, data_in.T).T
    # without extrapolation, the first and last lines/columns = 0
    assert data_out_esmpy[0, 0] == 0

    regrid = esmf_regrid_build(
        grid_in,
        grid_out,
        'bilinear',
        extrap_method='inverse_dist',
        extrap_num_src_pnts=3,
        extrap_dist_exponent=1,
    )
    data_out_esmpy = esmf_regrid_apply(regrid, data_in.T).T
    # the 3 closest points in data_in are 2.010, 2.005, and 1.992. The result should be roughly equal to 2.0
    assert np.round(data_out_esmpy[0, 0], 1) == 2.0


def test_regrid():

    # use conservative regridding as an example,
    # since it is the most well-tested studied one in papers

    # TODO: possible to break this long test into smaller tests?
    # not easy due to strong dependencies.

    grid_in = Grid.from_xarray(lon_in.T, lat_in.T)
    grid_out = Grid.from_xarray(lon_out.T, lat_out.T)

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
    regrid = esmf_regrid_build(grid_in, grid_out, 'conservative', filename=filename)
    assert regrid.regrid_method is ESMF.RegridMethod.CONSERVE

    # apply regridding using ESMPy's native method
    data_out_esmpy = esmf_regrid_apply(regrid, data_in.T).T

    rel_err = (data_out_esmpy - data_ref) / data_ref  # relative error
    assert np.max(np.abs(rel_err)) < 0.05

    # apply regridding using scipy
    weights = read_weights(filename, lon_in.size, lon_out.size).data
    shape_in = lon_in.shape
    shape_out = lon_out.shape
    data_out_scipy = apply_weights(weights, data_in, shape_in, shape_out)

    # must be almost exactly the same as esmpy's result!
    assert_almost_equal(data_out_scipy, data_out_esmpy)

    # finally, test broadcasting with scipy
    # TODO: need to test broadcasting with ESMPy backend?
    # We only use Scipy in frontend, and ESMPy is just for backend benchmark
    # However, it is useful to compare performance and show scipy is 3x faster
    data4D_out = apply_weights(weights, data4D_in, shape_in, shape_out)

    # data over broadcasting dimensions should agree
    assert_almost_equal(data4D_in.mean(axis=(2, 3)), data4D_out.mean(axis=(2, 3)), decimal=10)

    # clean-up
    esmf_regrid_finalize(regrid)
    os.remove(filename)


def test_regrid_periodic_wrong():

    # not using periodic grid
    grid_in = Grid.from_xarray(lon_in.T, lat_in.T)
    grid_out = Grid.from_xarray(lon_out.T, lat_out.T)

    assert grid_in.num_peri_dims == 0
    assert grid_in.periodic_dim is None

    regrid = esmf_regrid_build(grid_in, grid_out, 'bilinear')
    data_out_esmpy = esmf_regrid_apply(regrid, data_in.T).T

    rel_err = (data_out_esmpy - data_ref) / data_ref  # relative error
    assert np.max(np.abs(rel_err)) == 1.0  # some data will be missing

    # clean-up
    esmf_regrid_finalize(regrid)


def test_regrid_periodic_correct():

    # only need to specific periodic for input grid
    grid_in = Grid.from_xarray(lon_in.T, lat_in.T, periodic=True)
    grid_out = Grid.from_xarray(lon_out.T, lat_out.T)

    assert grid_in.num_peri_dims == 1
    assert grid_in.periodic_dim == 0  # the first axis, longitude

    regrid = esmf_regrid_build(grid_in, grid_out, 'bilinear')
    data_out_esmpy = esmf_regrid_apply(regrid, data_in.T).T

    rel_err = (data_out_esmpy - data_ref) / data_ref  # relative error
    assert np.max(np.abs(rel_err)) < 0.065
    # clean-up
    esmf_regrid_finalize(regrid)


def test_esmf_locstream():
    lon = np.arange(5)
    lat = np.arange(5)

    ls = LocStream.from_xarray(lon, lat)
    assert isinstance(ls, ESMF.LocStream)

    lon2d, lat2d = np.meshgrid(lon, lat)
    with pytest.raises(ValueError):
        ls = LocStream.from_xarray(lon2d, lat2d)
    with pytest.raises(ValueError):
        ls = LocStream.from_xarray(lon, lat2d)
    with pytest.raises(ValueError):
        ls = LocStream.from_xarray(lon2d, lat)

    grid_in = Grid.from_xarray(lon_in.T, lat_in.T, periodic=True)
    esmf_regrid_build(grid_in, ls, 'bilinear')
    esmf_regrid_build(ls, grid_in, 'nearest_s2d')


def test_read_weights(tmp_path):
    fn = tmp_path / 'weights.nc'

    grid_in = Grid.from_xarray(lon_in.T, lat_in.T)
    grid_out = Grid.from_xarray(lon_out.T, lat_out.T)

    regrid_memory = esmf_regrid_build(grid_in, grid_out, method='bilinear')
    esmf_regrid_build(grid_in, grid_out, method='bilinear', filename=str(fn))

    w = regrid_memory.get_weights_dict(deep_copy=True)
    sm = read_weights(w, lon_in.size, lon_out.size)

    # Test Path and string to netCDF file against weights dictionary
    np.testing.assert_array_equal(
        read_weights(fn, lon_in.size, lon_out.size).data.todense(), sm.data.todense()
    )
    np.testing.assert_array_equal(
        read_weights(str(fn), lon_in.size, lon_out.size).data.todense(), sm.data.todense()
    )

    # Test xr.Dataset
    np.testing.assert_array_equal(
        read_weights(xr.open_dataset(fn), lon_in.size, lon_out.size).data.todense(),
        sm.data.todense(),
    )

    # Test COO matrix
    np.testing.assert_array_equal(
        read_weights(sm, lon_in.size, lon_out.size).data.todense(), sm.data.todense()
    )

    # Test failures
    with pytest.raises(IOError):
        read_weights(tmp_path / 'wrong_file.nc', lon_in.size, lon_out.size)

    with pytest.raises(ValueError):
        read_weights({}, lon_in.size, lon_out.size)

    with pytest.raises(ValueError):
        ds = xr.open_dataset(fn)
        read_weights(ds.drop_vars('col'), lon_in.size, lon_out.size)


def test_deprecated():
    from xesmf.backend import esmf_grid, esmf_locstream

    lon = np.arange(5)
    lat = np.arange(5)
    gr = Grid.from_xarray(lon_in.T, lat_in.T)
    ls = LocStream.from_xarray(lon, lat)

    with pytest.warns(DeprecationWarning):
        np.testing.assert_allclose(gr.coords[0], esmf_grid(lon_in.T, lat_in.T).coords[0])

    with pytest.warns(DeprecationWarning):
        out = dict(esmf_locstream(lon, lat).items())
        for key, val in ls.items():
            np.testing.assert_array_equal(val, out[key])
