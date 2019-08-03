import os
import numpy as np
import xarray as xr
import xesmf as xe
from xesmf.frontend import as_2d_mesh

from numpy.testing import assert_equal, assert_almost_equal
import pytest

# same test data as test_backend.py, but here we can use xarray DataSet
ds_in = xe.util.grid_global(20, 12)
ds_out = xe.util.grid_global(15, 9)

ds_in['data'] = xe.data.wave_smooth(ds_in['lon'], ds_in['lat'])
ds_out['data_ref'] = xe.data.wave_smooth(ds_out['lon'], ds_out['lat'])

# 4D data to test broadcasting, increasing linearly with time and lev
ds_in.coords['time'] = np.arange(1, 11)
ds_in.coords['lev'] = np.arange(1, 51)
ds_in['data4D'] = ds_in['time'] * ds_in['lev'] * ds_in['data']


def test_as_2d_mesh():
    # 2D grid should not change
    lon2d = ds_in['lon'].values
    lat2d = ds_in['lat'].values
    assert_equal((lon2d, lat2d), as_2d_mesh(lon2d, lat2d))

    # 1D grid should become 2D
    lon1d = lon2d[0, :]
    lat1d = lat2d[:, 0]
    assert_equal((lon2d, lat2d), as_2d_mesh(lon1d, lat1d))

    # mix of 1D and 2D should fail
    with pytest.raises(ValueError):
        as_2d_mesh(lon1d, lat2d)


# 'patch' is too slow to test
methods_list = ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s']

@pytest.mark.parametrize('method', methods_list)
def test_build_regridder(method):
    regridder = xe.Regridder(ds_in, ds_out, method)

    # check screen output
    assert repr(regridder) == str(regridder)
    assert 'xESMF Regridder' in str(regridder)
    assert method in str(regridder)

    regridder.clean_weight_file()


def test_existing_weights():
    # the first run
    method = 'bilinear'
    regridder = xe.Regridder(ds_in, ds_out, method)

    # make sure we can reuse weights
    assert os.path.exists(regridder.filename)
    regridder_reuse = xe.Regridder(ds_in, ds_out, method,
                                   reuse_weights=True)
    assert regridder_reuse.A.shape == regridder.A.shape

    # or can also overwrite it
    xe.Regridder(ds_in, ds_out, method)

    # clean-up
    regridder.clean_weight_file()
    assert not os.path.exists(regridder.filename)


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
    assert np.max(np.abs(rel_err)) < 0.05

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

    # clean-up
    regridder.clean_weight_file()


def test_regrid_periodic_wrong():
    # not using periodic option
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear')

    dr_out = regridder(ds_in['data'])  # xarray DataArray

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - dr_out)/ds_out['data_ref']
    assert np.max(np.abs(rel_err)) == 1.0  # some data will be missing

    # clean-up
    regridder.clean_weight_file()


def test_regrid_periodic_correct():
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear', periodic=True)

    dr_out = regridder(ds_in['data'])

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - dr_out)/ds_out['data_ref']
    assert np.max(np.abs(rel_err)) < 0.065

    # clean-up
    regridder.clean_weight_file()


def ds_2d_to_1d(ds):
    ds_temp = ds.reset_coords()
    ds_1d = xr.merge([ds_temp['lon'][0, :], ds_temp['lat'][:, 0]])
    ds_1d.coords['lon'] = ds_1d['lon']
    ds_1d.coords['lat'] = ds_1d['lat']
    return ds_1d


def test_regrid_with_1d_grid():
    ds_in_1d = ds_2d_to_1d(ds_in)
    ds_out_1d = ds_2d_to_1d(ds_out)

    regridder = xe.Regridder(ds_in_1d, ds_out_1d, 'bilinear', periodic=True)

    dr_out = regridder(ds_in['data'])

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - dr_out)/ds_out['data_ref']
    assert np.max(np.abs(rel_err)) < 0.065

    # metadata should be 1D
    assert_equal(dr_out['lon'].values, ds_out_1d['lon'].values)
    assert_equal(dr_out['lat'].values, ds_out_1d['lat'].values)

    # clean-up
    regridder.clean_weight_file()
