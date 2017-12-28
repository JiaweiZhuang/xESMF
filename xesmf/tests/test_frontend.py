import numpy as np
import xarray as xr
import xesmf as xe

from numpy.testing import assert_equal, assert_almost_equal
import pytest

# same test data as test_backend.py, but here we can use xarray DataSet
ds_in = xe.util.grid_global(5, 4)
ds_out = xe.util.grid_global(2, 2)

ds_in['data'] = xe.data.wave_smooth(ds_in['lon'], ds_in['lat'])
ds_out['data_ref'] = xe.data.wave_smooth(ds_out['lon'], ds_out['lat'])

# 4D data to test broadcasting, increasing linearly with time and lev
ds_in.coords['time'] = np.arange(1, 11)
ds_in.coords['lev'] = np.arange(1, 51)
ds_in['data4D'] = ds_in['time'] * ds_in['lev'] * ds_in['data']


def test_build_regridder():
    # 'patch' is too slow to test
    for method in ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s']:
        regridder = xe.Regridder(ds_in, ds_out, method)
        regridder.clean_weight_file()


def test_conservative_without_bounds():
    with pytest.raises(KeyError):
        xe.Regridder(ds_in.drop('lon_b'), ds_out, 'conservative')


def test_build_regridder_from_dict():
    lon_in = ds_in['lon'].values
    lat_in = ds_in['lat'].values
    lon_out = ds_out['lon'].values
    lat_out = ds_out['lat'].values
    regridder = xe.Regridder({'lon': lon_in, 'lat': lat_in},
                             {'lon': lon_out, 'lat': lat_out},
                             'bilinear')
    regridder.clean_weight_file()


def test_regrid():
    regridder = xe.Regridder(ds_in, ds_out, 'conservative')

    outdata = regridder(ds_in['data'].values)  # pure numpy array
    dr_out = regridder(ds_in['data'])  # xarray DataArray

    # DataArray and numpy array should lead to the same result
    assert_equal(outdata, dr_out.values)

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - dr_out)/ds_out['data_ref']
    assert np.max(np.abs(rel_err)) == pytest.approx(0.03126, abs=1e-5)

    # check metadata
    assert_equal(dr_out['lat'].values, ds_out['lat'].values)
    assert_equal(dr_out['lon'].values, ds_out['lon'].values)

    # test broadcasting
    dr_out_4D = regridder(ds_in['data4D'])

    # data over broadcasting dimensions should agree
    assert_almost_equal(ds_in['data4D'].values.mean(axis=(2, 3)),
                        dr_out_4D.values.mean(axis=(2, 3)),
                        decimal=10)

    # check metadata
    xr.testing.assert_identical(dr_out_4D['time'], ds_in['time'])
    xr.testing.assert_identical(dr_out_4D['lev'], ds_in['lev'])
