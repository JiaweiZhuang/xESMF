import os
import numpy as np
import xarray as xr
import xesmf as xe
from xesmf.frontend import as_2d_mesh, default_var_names as dvn

from numpy.testing import assert_equal, assert_almost_equal
import pytest


# same test data as test_backend.py, but here we can use xarray DataSet
ds_in = xe.util.grid_global(20, 12)
ds_out = xe.util.grid_global(15, 9)

horiz_shape_in = ds_in['lon'].shape
horiz_shape_out = ds_out['lon'].shape

ds_in['data'] = xe.data.wave_smooth(ds_in['lon'], ds_in['lat'])
ds_out['data_ref'] = xe.data.wave_smooth(ds_out['lon'], ds_out['lat'])

# 4D data to test broadcasting, increasing linearly with time and lev
ds_in.coords['time'] = np.arange(7) + 1
ds_in.coords['lev'] = np.arange(11) + 1
ds_in['data4D'] = ds_in['time'] * ds_in['lev'] * ds_in['data']
ds_out['data4D_ref'] = ds_in['time'] * ds_in['lev'] * ds_out['data_ref']

# use non-divisible chunk size to catch edge cases
ds_in_chunked = ds_in.chunk({'time': 3, 'lev': 2})

ds_locs = xr.Dataset()
ds_locs['lat'] = xr.DataArray(data=[-20, -10, 0, 10],
                              dims=('locations',),
                              attrs={"long_name": "latitude", "units": "degrees_north"})
ds_locs['lon'] = xr.DataArray(data=[0, 5, 10, 15],
                              dims=('locations',),
                              attrs={"long_name": "longitude", "units": "degrees_east"})


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

@pytest.mark.parametrize("locstream_in,locstream_out,method", [
                         (False, False, 'conservative'),
                         (False, False, 'bilinear'),
                         (False, True, 'bilinear'),
                         (False, False, 'nearest_s2d'),
                         (False, True, 'nearest_s2d'),
                         (True, False, 'nearest_s2d'),
                         (True, True, 'nearest_s2d'),
                         (False, False, 'nearest_d2s'),
                         (False, True, 'nearest_d2s'),
                         (True, False, 'nearest_d2s'),
                         (True, True, 'nearest_d2s')
                         ])
def test_build_regridder(method, locstream_in, locstream_out):
    din = ds_locs if locstream_in else ds_in
    dout = ds_locs if locstream_out else ds_out

    regridder = xe.Regridder(din, dout, method,
                             locstream_in=locstream_in,
                             locstream_out=locstream_out)

    # check screen output
    assert repr(regridder) == str(regridder)
    assert 'xESMF Regridder' in str(regridder)
    assert method in str(regridder)

    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_existing_weights(vn):
    # the first run
    method = 'bilinear'
    regridder = xe.Regridder(ds_in, ds_out, method, var_names=vn)

    # make sure we can reuse weights
    assert os.path.exists(regridder.filename)
    regridder_reuse = xe.Regridder(ds_in, ds_out, method,
                                   reuse_weights=True, var_names=vn)
    assert regridder_reuse.A.shape == regridder.A.shape

    # or can also overwrite it
    xe.Regridder(ds_in, ds_out, method)

    # clean-up
    regridder.clean_weight_file()
    assert not os.path.exists(regridder.filename)


def test_conservative_without_bounds():
    with pytest.raises(KeyError):
        xe.Regridder(ds_in.drop_vars('lon_b'), ds_out, 'conservative', var_names=dvn)


def test_build_regridder_from_dict():
    lon_in = ds_in['lon'].values
    lat_in = ds_in['lat'].values
    lon_out = ds_out['lon'].values
    lat_out = ds_out['lat'].values
    regridder = xe.Regridder({'lon': lon_in, 'lat': lat_in},
                             {'lon': lon_out, 'lat': lat_out},
                             'bilinear')
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_periodic_wrong(vn):
    # not using periodic option
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear', var_names=vn)

    dr_out = regridder(ds_in['data'])  # xarray DataArray

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - dr_out)/ds_out['data_ref']
    assert np.max(np.abs(rel_err)) == 1.0  # some data will be missing

    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_periodic_correct(vn):
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear', periodic=True, var_names=vn)

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


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_with_1d_grid(vn):
    ds_in_1d = ds_2d_to_1d(ds_in)
    ds_out_1d = ds_2d_to_1d(ds_out)

    regridder = xe.Regridder(ds_in_1d, ds_out_1d, 'bilinear', periodic=True, var_names=vn)

    dr_out = regridder(ds_in['data'])

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - dr_out)/ds_out['data_ref']
    assert np.max(np.abs(rel_err)) < 0.065

    # metadata should be 1D
    assert_equal(dr_out['lon'].values, ds_out_1d['lon'].values)
    assert_equal(dr_out['lat'].values, ds_out_1d['lat'].values)

    # clean-up
    regridder.clean_weight_file()


# TODO: consolidate (regrid method, input data types) combination
# using pytest fixtures and parameterization

@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dataarray(vn):
    # xarray.DataArray containing in-memory numpy array

    regridder = xe.Regridder(ds_in, ds_out, 'conservative', var_names=vn)

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


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dataarray_to_locstream(vn):
    # xarray.DataArray containing in-memory numpy array

    regridder = xe.Regridder(ds_in, ds_locs, 'bilinear', locstream_out=True, var_names=vn)

    outdata = regridder(ds_in['data'].values)  # pure numpy array
    dr_out = regridder(ds_in['data'])  # xarray DataArray

    # DataArray and numpy array should lead to the same result
    assert_equal(outdata.squeeze(), dr_out.values)

    # clean-up
    regridder.clean_weight_file()

    with pytest.raises(ValueError):
        regridder = xe.Regridder(ds_in, ds_locs, 'conservative', locstream_out=True)


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dataarray_from_locstream(vn):
    # xarray.DataArray containing in-memory numpy array

    regridder = xe.Regridder(ds_locs, ds_in, 'nearest_s2d', locstream_in=True, var_names=vn)

    outdata = regridder(ds_locs['lat'].values)  # pure numpy array
    dr_out = regridder(ds_locs['lat'])  # xarray DataArray

    # DataArray and numpy array should lead to the same result
    assert_equal(outdata, dr_out.values)

    # clean-up
    regridder.clean_weight_file()

    with pytest.raises(ValueError):
        regridder = xe.Regridder(ds_locs, ds_in, 'bilinear', locstream_in=True)
    with pytest.raises(ValueError):
        regridder = xe.Regridder(ds_locs, ds_in, 'patch', locstream_in=True)
    with pytest.raises(ValueError):
        regridder = xe.Regridder(ds_locs, ds_in, 'conservative', locstream_in=True)


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dask(vn):
    # chunked dask array (no xarray metadata)

    regridder = xe.Regridder(ds_in, ds_out, 'conservative', var_names=vn)

    indata = ds_in_chunked['data4D'].data
    outdata = regridder(indata)

    # lazy dask arrays have incorrect shape attribute due to last chunk
    assert outdata.compute().shape == indata.shape[:-2] + horiz_shape_out
    assert outdata.chunksize == indata.chunksize[:-2] + horiz_shape_out

    outdata_ref = ds_out['data4D_ref'].values
    rel_err = (outdata.compute() - outdata_ref) / outdata_ref
    assert np.max(np.abs(rel_err)) < 0.05

    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dask_to_locstream(vn):
    # chunked dask array (no xarray metadata)

    regridder = xe.Regridder(ds_in, ds_locs, 'bilinear', locstream_out=True, var_names=vn)

    indata = ds_in_chunked['data4D'].data
    outdata = regridder(indata)

    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dask_from_locstream(vn):
    # chunked dask array (no xarray metadata)

    regridder = xe.Regridder(ds_locs, ds_in, 'nearest_s2d', locstream_in=True, var_names=vn)

    outdata = regridder(ds_locs['lat'].data)

    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dataarray_dask(vn):
    # xarray.DataArray containing chunked dask array

    regridder = xe.Regridder(ds_in, ds_out, 'conservative', var_names=vn)

    dr_in = ds_in_chunked['data4D']
    dr_out = regridder(dr_in)

    assert dr_out.data.shape == dr_in.data.shape[:-2] + horiz_shape_out
    assert dr_out.data.chunksize == dr_in.data.chunksize[:-2] + horiz_shape_out

    # data over broadcasting dimensions should agree
    assert_almost_equal(dr_in.values.mean(axis=(2, 3)),
                        dr_out.values.mean(axis=(2, 3)),
                        decimal=10)

    # check metadata
    xr.testing.assert_identical(dr_out['time'], dr_in['time'])
    xr.testing.assert_identical(dr_out['lev'], dr_in['lev'])
    assert_equal(dr_out['lat'].values, ds_out['lat'].values)
    assert_equal(dr_out['lon'].values, ds_out['lon'].values)

    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dataarray_dask_to_locstream(vn):
    # xarray.DataArray containing chunked dask array

    regridder = xe.Regridder(ds_in, ds_locs, 'bilinear', locstream_out=True, var_names=vn)

    dr_in = ds_in_chunked['data4D']
    dr_out = regridder(dr_in)

    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dataarray_dask_from_locstream(vn):
    # xarray.DataArray containing chunked dask array

    regridder = xe.Regridder(ds_locs, ds_in, 'nearest_s2d', locstream_in=True, var_names=vn)

    outdata = regridder(ds_locs['lat'])

    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dataset(vn):
    # xarray.Dataset containing in-memory numpy array

    regridder = xe.Regridder(ds_in, ds_out, 'conservative', var_names=vn)

    # `ds_out` already refers to output grid object
    # TODO: use more consistent variable namings across tests
    ds_result = regridder(ds_in)

    # output should contain all data variables
    assert set(ds_result.data_vars.keys()) == set(ds_in.data_vars.keys())

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - ds_result['data'])/ds_out['data_ref']
    assert np.max(np.abs(rel_err)) < 0.05

    # data over broadcasting dimensions should agree
    assert_almost_equal(ds_in['data4D'].values.mean(axis=(2, 3)),
                        ds_result['data4D'].values.mean(axis=(2, 3)),
                        decimal=10)

    # check metadata
    xr.testing.assert_identical(ds_result['time'], ds_in['time'])
    xr.testing.assert_identical(ds_result['lev'], ds_in['lev'])
    assert_equal(ds_result['lat'].values, ds_out['lat'].values)
    assert_equal(ds_result['lon'].values, ds_out['lon'].values)

    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dataset_to_locstream(vn):
    # xarray.Dataset containing in-memory numpy array

    regridder = xe.Regridder(ds_in, ds_locs, 'bilinear', locstream_out=True, var_names=vn)
    ds_result = regridder(ds_in)
    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_regrid_dataset_from_locstream(vn):
    # xarray.Dataset containing in-memory numpy array

    regridder = xe.Regridder(ds_locs, ds_in, 'nearest_s2d', locstream_in=True, var_names=vn)
    outdata = regridder(ds_locs)
    # clean-up
    regridder.clean_weight_file()


@pytest.mark.parametrize("vn", [dvn, None])
def test_ds_to_ESMFlocstream(vn):
    import ESMF
    from xesmf.frontend import ds_to_ESMFlocstream

    locstream, shape = ds_to_ESMFlocstream(ds_locs, var_names=vn)
    assert isinstance(locstream, ESMF.LocStream)
    assert shape == (1, 4,)
    with pytest.raises(ValueError):
        locstream, shape = ds_to_ESMFlocstream(ds_in, var_names=vn)

    ds_bogus = ds_in.copy()
    ds_bogus['lon'] = ds_locs['lon']
    with pytest.raises(ValueError):
        locstream, shape = ds_to_ESMFlocstream(ds_bogus, var_names=vn)
