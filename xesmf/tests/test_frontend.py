import os
import warnings

import cf_xarray as cfxr
import dask
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from shapely.geometry import MultiPolygon, Polygon

import xesmf as xe
from xesmf.frontend import as_2d_mesh

dask_schedulers = ['threaded_scheduler', 'processes_scheduler', 'distributed_scheduler']


# same test data as test_backend.py, but here we can use xarray DataSet
ds_in = xe.util.grid_global(20, 12)
ds_in.lat.attrs['standard_name'] = 'latitude'
ds_in.lon.attrs['standard_name'] = 'longitude'
ds_out = xe.util.grid_global(15, 9)

horiz_shape_in = ds_in['lon'].shape
horiz_shape_out = ds_out['lon'].shape

ds_in['data'] = xe.data.wave_smooth(ds_in['lon'], ds_in['lat'])
ds_out['data_ref'] = xe.data.wave_smooth(ds_out['lon'], ds_out['lat'])

# 4D data to test broadcasting, increasing linearly with time and lev
ds_in.coords['time'] = np.arange(7) + 1
ds_in.coords['lev'] = np.arange(11) + 1
ds_in['data4D'] = ds_in['time'] * ds_in['lev'] * ds_in['data']
ds_in['data4D_f4'] = ds_in['data4D'].astype('f4')
ds_out['data4D_ref'] = ds_in['time'] * ds_in['lev'] * ds_out['data_ref']

# use non-divisible chunk size to catch edge cases
ds_in_chunked = ds_in.chunk({'time': 3, 'lev': 2})

ds_locs = xr.Dataset()
ds_locs['lat'] = xr.DataArray(data=[-20, -10, 0, 10], dims=('locations',))
ds_locs['lon'] = xr.DataArray(data=[0, 5, 10, 15], dims=('locations',))


# For polygon handling and spatial average
ds_savg = xr.Dataset(
    coords={
        'lat': (('lat',), [0.5, 1.5]),
        'lon': (('lon',), [0.5, 1.5, 2.5]),
        'lat_b': (('lat_b',), [0, 1, 2]),
        'lon_b': (('lon_b',), [0, 1, 2, 3]),
    },
    data_vars={'abc': (('lon', 'lat'), [[1.0, 2.0], [3.0, 4.0], [2.0, 4.0]])},
)
polys = [
    Polygon([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5]]),  # Simple triangle polygon
    MultiPolygon(
        [
            Polygon([[0.25, 1.25], [0.25, 1.75], [0.75, 1.75], [0.75, 1.25]]),
            Polygon([[1.25, 1.25], [1.25, 1.75], [1.75, 1.75], [1.75, 1.25]]),
        ]
    ),  # Multipolygon on 2 and 4
    Polygon(
        [[0, 0], [0, 1], [2, 1], [2, 0]],
        holes=[[[0.5, 0.25], [0.5, 0.75], [1.0, 0.75], [1.0, 0.25]]],
    ),  # Simple polygon covering 1 and 3 with hole over 1
    Polygon([[1, 1], [1, 3], [3, 3], [3, 1]]),  # Polygon partially outside, covering a part of 4
    Polygon([[3, 3], [3, 4], [4, 4], [4, 3]]),  # Polygon totally outside
    Polygon(
        [
            [0, 0],
            [0.5, 0.5],
            [0, 1],
            [0.5, 1.5],
            [0, 2],
            [2, 2],
            [1.5, 1.5],
            [2, 1],
            [1.5, 0.5],
            [2, 0],
        ]
    ),  # Long multifaceted polygon
    [
        Polygon([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5]]),
        MultiPolygon(
            [
                Polygon([[0.25, 1.25], [0.25, 1.75], [0.75, 1.75], [0.75, 1.25]]),
                Polygon([[1, 1], [1, 2], [2, 2], [2, 1]]),
            ]
        ),
    ],  # Combination of Polygon and MultiPolygon with two different areas
]
exps_polys = [1.75, 3, 2.1429, 4, 0, 2.5, [1.75, 3.6]]


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


@pytest.mark.parametrize(
    'locstream_in,locstream_out,method,unmapped_to_nan',
    [
        (False, False, 'conservative', False),
        (False, False, 'bilinear', False),
        (False, True, 'bilinear', False),
        (False, False, 'nearest_s2d', False),
        (False, True, 'nearest_s2d', False),
        (True, False, 'nearest_s2d', False),
        (True, True, 'nearest_s2d', False),
        (False, False, 'nearest_d2s', False),
        (False, True, 'nearest_d2s', False),
        (True, False, 'nearest_d2s', False),
        (True, True, 'nearest_d2s', False),
        (False, False, 'conservative', True),
        (False, False, 'bilinear', True),
        (False, True, 'bilinear', True),
        (False, False, 'nearest_s2d', True),
        (False, True, 'nearest_s2d', True),
        (True, False, 'nearest_s2d', True),
        (True, True, 'nearest_s2d', True),
        (False, False, 'nearest_d2s', True),
        (False, True, 'nearest_d2s', True),
        (True, False, 'nearest_d2s', True),
        (True, True, 'nearest_d2s', True),
    ],
)
def test_build_regridder(method, locstream_in, locstream_out, unmapped_to_nan):
    din = ds_locs if locstream_in else ds_in
    dout = ds_locs if locstream_out else ds_out

    regridder = xe.Regridder(
        din, dout, method, locstream_in=locstream_in, locstream_out=locstream_out
    )

    # check screen output
    assert repr(regridder) == str(regridder)
    assert 'xESMF Regridder' in str(regridder)
    assert method in str(regridder)


def test_existing_weights():
    # the first run
    method = 'bilinear'
    regridder = xe.Regridder(ds_in, ds_out, method)
    fn = regridder.to_netcdf()

    # make sure we can reuse weights
    assert os.path.exists(fn)
    regridder_reuse = xe.Regridder(ds_in, ds_out, method, weights=fn)
    assert regridder_reuse.weights.shape == regridder.weights.shape

    # this should also work with reuse_weights=True
    regridder_reuse = xe.Regridder(ds_in, ds_out, method, reuse_weights=True, weights=fn)
    assert regridder_reuse.weights.shape == regridder.weights.shape

    # or can also overwrite it
    xe.Regridder(ds_in, ds_out, method)

    # check legacy args still work
    regridder = xe.Regridder(ds_in, ds_out, method, filename='wgts.nc')
    regridder_reuse = xe.Regridder(ds_in, ds_out, method, reuse_weights=True, filename='wgts.nc')
    assert regridder_reuse.weights.shape == regridder.weights.shape

    # check fails on non-existent file
    with pytest.raises(OSError):
        regridder_reuse = xe.Regridder(
            ds_in, ds_out, method, reuse_weights=True, filename='fakewgts.nc'
        )

    # check fails if no weights are provided
    with pytest.raises(ValueError):
        regridder_reuse = xe.Regridder(ds_in, ds_out, method, reuse_weights=True)


def test_to_netcdf(tmp_path):
    from xesmf.backend import Grid, esmf_regrid_build

    # Let the frontend write the weights to disk
    xfn = tmp_path / 'ESMF_weights.nc'
    method = 'bilinear'
    regridder = xe.Regridder(ds_in, ds_out, method, unmapped_to_nan=False)
    regridder.to_netcdf(filename=xfn)

    grid_in = Grid.from_xarray(ds_in['lon'].values.T, ds_in['lat'].values.T)
    grid_out = Grid.from_xarray(ds_out['lon'].values.T, ds_out['lat'].values.T)

    # Let the ESMPy backend write the weights to disk
    efn = tmp_path / 'weights.nc'
    esmf_regrid_build(grid_in, grid_out, method=method, filename=str(efn))

    x = xr.open_dataset(xfn)
    e = xr.open_dataset(efn)
    xr.testing.assert_identical(x, e)


def test_to_netcdf_nans(tmp_path):
    from xesmf.backend import Grid, esmf_regrid_build

    # Let the frontend write the weights to disk
    xfn = tmp_path / 'ESMF_weights_nans.nc'
    method = 'bilinear'
    regridder = xe.Regridder(ds_in, ds_out, method, unmapped_to_nan=True)
    regridder.to_netcdf(filename=xfn)

    grid_in = Grid.from_xarray(ds_in['lon'].values.T, ds_in['lat'].values.T)
    grid_out = Grid.from_xarray(ds_out['lon'].values.T, ds_out['lat'].values.T)

    # Let the ESMPy backend write the weights to disk
    efn = tmp_path / 'weights_nans.nc'
    esmf_regrid_build(grid_in, grid_out, method=method, filename=str(efn))

    x = xr.open_dataset(xfn)
    e = xr.open_dataset(efn)

    # Reformat to sparse COO matrix
    smat = xe.smm.read_weights(e, np.prod(ds_in['lon'].shape), np.prod(ds_out['lon'].shape))
    # Add NaNs to weights
    smat = xe.smm.add_nans_to_weights(smat)
    # Updating the dataset
    e_nans = xr.Dataset(
        {
            'S': (['n_s'], smat.data.data),
            'row': (['n_s'], smat.data.coords[0, :] + 1),
            'col': (['n_s'], smat.data.coords[1, :] + 1),
        }
    )
    # Comparison
    xr.testing.assert_identical(x, e_nans)


def test_conservative_without_bounds():
    with pytest.raises(KeyError):
        xe.Regridder(ds_in.drop_vars('lon_b'), ds_out, 'conservative')


def test_build_regridder_from_dict():
    lon_in = ds_in['lon'].values
    lat_in = ds_in['lat'].values
    lon_out = ds_out['lon'].values
    lat_out = ds_out['lat'].values
    _ = xe.Regridder({'lon': lon_in, 'lat': lat_in}, {'lon': lon_out, 'lat': lat_out}, 'bilinear')


def test_regrid_periodic_wrong():
    # not using periodic option
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear', unmapped_to_nan=False)

    dr_out = regridder(ds_in['data'])  # xarray DataArray

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - dr_out) / ds_out['data_ref']
    assert np.max(np.abs(rel_err)) == 1.0  # some data will be missing


def test_regrid_periodic_correct():
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear', periodic=True)

    dr_out = regridder(ds_in['data'])

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - dr_out) / ds_out['data_ref']
    assert np.max(np.abs(rel_err)) < 0.065


def ds_2d_to_1d(ds):
    ds_temp = ds.reset_coords()
    ds_1d = xr.merge([ds_temp['lon'][0, :], ds_temp['lat'][:, 0]])
    ds_1d.coords['lon'] = ds_1d['lon']
    ds_1d.coords['lat'] = ds_1d['lat']
    return ds_1d


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize(
    'data_in',
    [
        pytest.param(np.array(ds_in['data']), id='np.ndarray'),
        pytest.param(xr.DataArray(ds_in['data']), id='xr.DataArray input'),
        pytest.param(xr.Dataset(ds_in[['data']]), id='xr.Dataset input'),
        pytest.param(ds_in['data'].chunk(), id='da.Array input'),
    ],
)
def test_regridded_respects_input_dtype(dtype, data_in):
    """Tests regridded output has same dtype as input"""
    data_in = data_in.astype(dtype)
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear')  # Make this a fixture?
    out = regridder(data_in)

    if 'data' in data_in:
        # When data_in is xr.Dataset, a mapping...
        assert out['data'].dtype == data_in['data'].dtype
    else:
        assert out.dtype == data_in.dtype


def test_regrid_with_1d_grid():
    ds_in_1d = ds_2d_to_1d(ds_in)
    ds_out_1d = ds_2d_to_1d(ds_out)

    regridder = xe.Regridder(ds_in_1d, ds_out_1d, 'bilinear', periodic=True)

    dr_out = regridder(ds_in['data'])

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - dr_out) / ds_out['data_ref']
    assert np.max(np.abs(rel_err)) < 0.065

    # metadata should be 1D
    assert_equal(dr_out['lon'].values, ds_out_1d['lon'].values)
    assert_equal(dr_out['lat'].values, ds_out_1d['lat'].values)


def test_regrid_with_1d_grid_infer_bounds():
    ds_in_1d = ds_2d_to_1d(ds_in).rename(x='lon', y='lat')
    ds_out_1d = ds_2d_to_1d(ds_out).rename(x='lon', y='lat')

    regridder = xe.Regridder(ds_in_1d, ds_out_1d, 'conservative', periodic=True)

    dr_out = regridder(ds_in['data'])

    # compare with provided-bounds solution
    dr_exp = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)(ds_in['data'])

    assert_allclose(dr_out, dr_exp)


def test_regrid_cfbounds():
    # Test regridding when bounds are given in cf format with a custom "bounds" name.
    ds = ds_in.copy().drop_vars(['lat_b', 'lon_b'])
    ds['lon_bounds'] = cfxr.vertices_to_bounds(ds_in.lon_b, ('bnds', 'y', 'x'))
    ds['lat_bounds'] = cfxr.vertices_to_bounds(ds_in.lat_b, ('bnds', 'y', 'x'))
    ds.lat.attrs['bounds'] = 'lat_bounds'
    ds.lon.attrs['bounds'] = 'lon_bounds'

    regridder = xe.Regridder(ds, ds_out, 'conservative', periodic=True)
    dr_out = regridder(ds['data'])
    # compare with provided-bounds solution
    dr_exp = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)(ds_in['data'])
    assert_allclose(dr_out, dr_exp)


# TODO: consolidate (regrid method, input data types) combination
# using pytest fixtures and parameterization


@pytest.mark.parametrize('use_cfxr', [True, False])
def test_regrid_dataarray(use_cfxr):
    # xarray.DataArray containing in-memory numpy array
    if use_cfxr:
        ds_in2 = ds_in.rename(lat='Latitude', lon='Longitude')
        ds_out2 = ds_out.rename(lat='Latitude', lon='Longitude')
    else:
        ds_in2 = ds_in
        ds_out2 = ds_out

    regridder = xe.Regridder(ds_in2, ds_out2, 'conservative')

    outdata = regridder(ds_in2['data'].values)  # pure numpy array
    dr_out = regridder(ds_in2['data'])  # xarray DataArray

    # DataArray and numpy array should lead to the same result
    assert_equal(outdata, dr_out.values)

    # compare with analytical solution
    rel_err = (ds_out2['data_ref'] - dr_out) / ds_out2['data_ref']
    assert np.max(np.abs(rel_err)) < 0.05

    # check metadata
    lat_name = 'Latitude' if use_cfxr else 'lat'
    lon_name = 'Longitude' if use_cfxr else 'lon'
    xr.testing.assert_identical(dr_out[lat_name], ds_out2[lat_name])
    xr.testing.assert_identical(dr_out[lon_name], ds_out2[lon_name])

    # test broadcasting
    dr_out_4D = regridder(ds_in2['data4D'])

    # data over broadcasting dimensions should agree
    assert_almost_equal(
        ds_in2['data4D'].values.mean(axis=(2, 3)),
        dr_out_4D.values.mean(axis=(2, 3)),
        decimal=10,
    )

    # check metadata
    xr.testing.assert_identical(dr_out_4D['time'], ds_in2['time'])
    xr.testing.assert_identical(dr_out_4D['lev'], ds_in2['lev'])

    # test transposed
    dr_out_4D_t = regridder(ds_in2['data4D'].transpose(..., 'time', 'lev'))
    xr.testing.assert_identical(dr_out_4D, dr_out_4D_t)

    # test renamed dim
    if not use_cfxr:
        dr_out_rn = regridder(ds_in2.rename(y='why')['data'])
        xr.testing.assert_identical(dr_out, dr_out_rn)


@pytest.mark.parametrize('use_dask', [True, False])
def test_regrid_dataarray_endianess(use_dask):
    # xarray.DataArray containing in-memory numpy array
    regridder = xe.Regridder(ds_in, ds_out, 'conservative')

    exp = regridder(ds_in['data'])  # Normal (little-endian)
    # with pytest.warns(UserWarning, match='Input array has a dtype not supported'):

    if use_dask:
        indata = ds_in.data.astype('>f8').chunk()
    else:
        indata = ds_in.data.astype('>f8')

    out = regridder(indata)  # big endian

    # Results should be the same
    assert_equal(exp.values, out.values)
    assert out.dtype == '>f8'


def test_regrid_dataarray_to_locstream():
    # xarray.DataArray containing in-memory numpy array

    regridder = xe.Regridder(ds_in, ds_locs, 'bilinear', locstream_out=True)

    outdata = regridder(ds_in['data'].values)  # pure numpy array
    dr_out = regridder(ds_in['data'])  # xarray DataArray
    dr_out_t = regridder(ds_in['data'].transpose())  # Transpose to test name matching

    # DataArray and numpy array should lead to the same result
    assert_equal(outdata.squeeze(), dr_out.values)
    assert_equal(outdata.squeeze(), dr_out_t.values)

    with pytest.raises(ValueError):
        regridder = xe.Regridder(ds_in, ds_locs, 'conservative', locstream_out=True)


def test_regrid_dataarray_from_locstream():
    # xarray.DataArray containing in-memory numpy array

    regridder = xe.Regridder(ds_locs, ds_in, 'nearest_s2d', locstream_in=True)

    outdata = regridder(ds_locs['lat'].values)  # pure numpy array
    dr_out = regridder(ds_locs['lat'])  # xarray DataArray
    # New dim and transpose to test name-matching
    dr_out_2D = regridder(ds_locs['lat'].expand_dims(other=[1]).transpose('locations', 'other'))

    # DataArray and numpy array should lead to the same result
    assert_equal(outdata, dr_out.values)
    assert_equal(outdata, dr_out_2D.sel(other=1).values)

    with pytest.raises(ValueError):
        regridder = xe.Regridder(ds_locs, ds_in, 'bilinear', locstream_in=True)
    with pytest.raises(ValueError):
        regridder = xe.Regridder(ds_locs, ds_in, 'patch', locstream_in=True)
    with pytest.raises(ValueError):
        regridder = xe.Regridder(ds_locs, ds_in, 'conservative', locstream_in=True)


@pytest.mark.parametrize('scheduler', dask_schedulers)
def test_regrid_dask(request, scheduler):
    # chunked dask array (no xarray metadata)
    scheduler = request.getfixturevalue(scheduler)
    regridder = xe.Regridder(ds_in, ds_out, 'conservative')

    indata = ds_in_chunked['data4D'].data
    # Use ridiculous small chunk size value to be sure it _isn't_ impacting computation.
    with dask.config.set({'array.chunk-size': '1MiB'}):
        outdata = regridder(indata)

    assert dask.is_dask_collection(outdata)

    # lazy dask arrays have incorrect shape attribute due to last chunk
    assert outdata.shape == indata.shape[:-2] + horiz_shape_out
    assert outdata.chunksize == indata.chunksize[:-2] + horiz_shape_out

    # Check that the number of tasks hasn't exploded.
    n_task_in = len(indata.__dask_graph__().keys())
    n_task_out = len(outdata.__dask_graph__().keys())
    assert (n_task_out / n_task_in) < 3

    outdata_ref = ds_out['data4D_ref'].values
    rel_err = (outdata.compute() - outdata_ref) / outdata_ref
    assert np.max(np.abs(rel_err)) < 0.05


@pytest.mark.parametrize('scheduler', dask_schedulers)
def test_regrid_dask_to_locstream(request, scheduler):
    # chunked dask array (no xarray metadata)

    scheduler = request.getfixturevalue(scheduler)
    regridder = xe.Regridder(ds_in, ds_locs, 'bilinear', locstream_out=True)

    indata = ds_in_chunked['data4D'].data
    outdata = regridder(indata)
    assert dask.is_dask_collection(outdata)


@pytest.mark.parametrize('scheduler', dask_schedulers)
def test_regrid_dask_from_locstream(request, scheduler):
    # chunked dask array (no xarray metadata)

    scheduler = request.getfixturevalue(scheduler)
    regridder = xe.Regridder(ds_locs, ds_in, 'nearest_s2d', locstream_in=True)

    outdata = regridder(ds_locs.chunk()['lat'].data)
    assert dask.is_dask_collection(outdata)


@pytest.mark.parametrize('scheduler', dask_schedulers)
def test_regrid_dataarray_dask(request, scheduler):
    # xarray.DataArray containing chunked dask array
    scheduler = request.getfixturevalue(scheduler)
    regridder = xe.Regridder(ds_in, ds_out, 'conservative')

    dr_in = ds_in_chunked['data4D']
    dr_out = regridder(dr_in)
    assert dask.is_dask_collection(dr_out)

    assert dr_out.data.shape == dr_in.data.shape[:-2] + horiz_shape_out
    assert dr_out.data.chunksize == dr_in.data.chunksize[:-2] + horiz_shape_out

    # data over broadcasting dimensions should agree
    assert_almost_equal(dr_in.values.mean(axis=(2, 3)), dr_out.values.mean(axis=(2, 3)), decimal=10)

    # check metadata
    xr.testing.assert_identical(dr_out['time'], dr_in['time'])
    xr.testing.assert_identical(dr_out['lev'], dr_in['lev'])
    assert_equal(dr_out['lat'].values, ds_out['lat'].values)
    assert_equal(dr_out['lon'].values, ds_out['lon'].values)


@pytest.mark.parametrize('scheduler', dask_schedulers)
def test_regrid_dataarray_dask_to_locstream(request, scheduler):
    # xarray.DataArray containing chunked dask array
    scheduler = request.getfixturevalue(scheduler)
    regridder = xe.Regridder(ds_in, ds_locs, 'bilinear', locstream_out=True)

    dr_in = ds_in_chunked['data4D']
    dr_out = regridder(dr_in)
    assert dask.is_dask_collection(dr_out)


@pytest.mark.parametrize('scheduler', dask_schedulers)
def test_regrid_dataarray_dask_from_locstream(request, scheduler):
    # xarray.DataArray containing chunked dask array

    scheduler = request.getfixturevalue(scheduler)
    regridder = xe.Regridder(ds_locs, ds_in, 'nearest_s2d', locstream_in=True)

    outdata = regridder(ds_locs.chunk()['lat'])
    assert dask.is_dask_collection(outdata)


def test_regrid_dataset():
    # xarray.Dataset containing in-memory numpy array

    regridder = xe.Regridder(ds_in, ds_out, 'conservative')

    # `ds_out` already refers to output grid object
    # TODO: use more consistent variable namings across tests
    ds_result = regridder(ds_in)

    # output should contain all data variables
    assert set(ds_result.data_vars.keys()) == set(ds_in.data_vars.keys())

    # compare with analytical solution
    rel_err = (ds_out['data_ref'] - ds_result['data']) / ds_out['data_ref']
    assert np.max(np.abs(rel_err)) < 0.05

    # data over broadcasting dimensions should agree
    assert_almost_equal(
        ds_in['data4D'].values.mean(axis=(2, 3)),
        ds_result['data4D'].values.mean(axis=(2, 3)),
        decimal=10,
    )

    assert ds_result['data4D'].dtype == np.dtype('f8')
    assert ds_result['data4D_f4'].dtype == np.dtype('f4')

    # check metadata
    xr.testing.assert_identical(ds_result['time'], ds_in['time'])
    xr.testing.assert_identical(ds_result['lev'], ds_in['lev'])
    assert_equal(ds_result['lat'].values, ds_out['lat'].values)
    assert_equal(ds_result['lon'].values, ds_out['lon'].values)

    # Allow (but skip) other non spatial variables
    ds_result2 = regridder(ds_in.assign(nonspatial=ds_in.x * ds_in.time))
    xr.testing.assert_identical(ds_result2, ds_result)


@pytest.mark.parametrize('scheduler', dask_schedulers)
def test_regrid_dataset_dask(request, scheduler):
    scheduler = request.getfixturevalue(scheduler)
    # xarray.Dataset containing dask array
    regridder = xe.Regridder(ds_in, ds_out, 'conservative')

    # `ds_out` already refers to output grid object
    ds_result = regridder(ds_in.chunk())

    # output should contain all data variables
    assert set(ds_result.data_vars.keys()) == set(ds_in.data_vars.keys())
    assert dask.is_dask_collection(ds_result)
    assert ds_result.data.dtype == ds_in.data.dtype

    ds_in_f4 = ds_in.copy()
    ds_in_f4['data'] = ds_in_f4.data.astype('float32')
    ds_in_f4['data4D'] = ds_in_f4.data4D.astype('float32')
    ds_result = regridder(ds_in_f4.chunk())
    assert ds_result.data.dtype == 'float32'


def test_regrid_dataset_to_locstream():
    # xarray.Dataset containing in-memory numpy array

    regridder = xe.Regridder(ds_in, ds_locs, 'bilinear', locstream_out=True)
    regridder(ds_in)


def test_build_regridder_with_masks():
    dsi = ds_in.copy()
    dsi['mask'] = xr.DataArray(np.random.randint(2, size=ds_in['data'].shape), dims=('y', 'x'))
    # 'patch' is too slow to test
    for method in [
        'bilinear',
        'conservative',
        'conservative_normed',
        'nearest_s2d',
        'nearest_d2s',
    ]:
        regridder = xe.Regridder(dsi, ds_out, method)

        # check screen output
        assert repr(regridder) == str(regridder)
        assert 'xESMF Regridder' in str(regridder)
        assert method in str(regridder)


def test_regrid_dataset_from_locstream():
    # xarray.Dataset containing in-memory numpy array

    regridder = xe.Regridder(ds_locs, ds_in, 'nearest_s2d', locstream_in=True)
    regridder(ds_locs)


def test_ds_to_ESMFlocstream():
    import ESMF

    from xesmf.frontend import ds_to_ESMFlocstream

    locstream, shape, names = ds_to_ESMFlocstream(ds_locs)
    assert isinstance(locstream, ESMF.LocStream)
    assert shape == (
        1,
        4,
    )
    assert names == ('locations',)
    with pytest.raises(ValueError):
        locstream, shape, names = ds_to_ESMFlocstream(ds_in)
    ds_bogus = ds_in.copy()
    ds_bogus['lon'] = ds_locs['lon']
    with pytest.raises(ValueError):
        locstream, shape, names = ds_to_ESMFlocstream(ds_bogus)


@pytest.mark.parametrize('poly,exp', list(zip(polys, exps_polys)))
def test_spatial_averager(poly, exp):
    if isinstance(poly, (Polygon, MultiPolygon)):
        poly = [poly]
    savg = xe.SpatialAverager(ds_savg, poly, geom_dim_name='my_geom')
    out = savg(ds_savg.abc)
    assert_allclose(out, exp, rtol=1e-3)

    assert 'my_geom' in out.dims


def test_compare_weights_from_poly_and_grid():
    """Confirm that the weights are identical when they are computed from a grid->grid and grid->poly."""

    # Global grid
    ds = xe.util.grid_global(20, 12, cf=True)

    # A single destination tile
    tile = xe.util.cf_grid_2d(-40, -80, -40, 0, 80, 80)
    ds['a'] = xr.DataArray(
        np.ones((ds.lon.size, ds.lat.size)),
        coords={'lat': ds.lat, 'lon': ds.lon},
        dims=('lon', 'lat'),
    )

    # Create polygon from tile corners
    x1, x2 = tile.lon_bounds
    y1, y2 = tile.lat_bounds
    poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

    # Regrid using two identical destination grids (in theory)
    rgrid = xe.Regridder(ds, tile, method='conservative')
    rpoly = xe.SpatialAverager(ds, [poly])

    # Normally, weights should be identical, but this fails
    np.testing.assert_array_almost_equal(rgrid.weights.data.todense(), rpoly.weights.data.todense())

    # Visualize the weights
    wg = np.reshape(rgrid.weights.data.todense(), ds.a.T.shape)
    wp = np.reshape(rpoly.weights.data.todense(), ds.a.T.shape)

    ds['wg'] = (('lat', 'lon'), wg)
    ds['wp'] = (('lat', 'lon'), wp)

    # Figure of weights in two cases
    # from matplotlib import pyplot as plt
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ds.wg.plot(ax=ax1); ds.wp.plot(ax=ax2)
    # ax1.set_title("Regridder weights")
    # ax2.set_title("SpatialAverager weights")
    # plt.show()

    # Check that source area affects weights
    i = ds.indexes['lon'].get_loc(-55, method='nearest')
    j1 = ds.indexes['lat'].get_loc(12, method='nearest')
    j2 = ds.indexes['lat'].get_loc(72, method='nearest')
    assert ds.wg.isel(lon=i, lat=j1) > ds.wg.isel(lon=i, lat=j2)
    assert ds.wp.isel(lon=i, lat=j1).data > ds.wp.isel(lon=i, lat=j2).data  # Fails


def test_polys_to_ESMFmesh():
    import ESMF

    from xesmf.frontend import polys_to_ESMFmesh

    # No overlap but multi + holes
    with warnings.catch_warnings(record=True) as rec:
        mesh, shape = polys_to_ESMFmesh([polys[1], polys[2], polys[4]])

    assert isinstance(mesh, ESMF.Mesh)
    assert shape == (1, 4)
    assert len(rec) == 1
    assert 'Some passed polygons have holes' in rec[0].message.args[0]


@pytest.mark.parametrize(
    'method, skipna, na_thres, nvalid',
    [
        ('bilinear', False, 1.0, 380),
        ('bilinear', True, 1.0, 395),
        ('bilinear', True, 0.0, 380),
        ('bilinear', True, 0.5, 388),
        ('bilinear', True, 1.0, 395),
        ('conservative', False, 1.0, 385),
        ('conservative', True, 1.0, 394),
        ('conservative', True, 0.0, 385),
        ('conservative', True, 0.5, 388),
        ('conservative', True, 1.0, 394),
    ],
)
def test_skipna(method, skipna, na_thres, nvalid):
    dai = ds_in['data4D'].copy()
    dai[0, 0, 4:6, 4:6] = np.nan
    rg = xe.Regridder(ds_in, ds_out, method)
    dao = rg(dai, skipna=skipna, na_thres=na_thres)
    assert int(dao[0, 0, 1:-1, 1:-1].notnull().sum()) == nvalid


def test_non_cf_latlon():
    ds_in_noncf = ds_in.copy()
    ds_in_noncf.lon.attrs = {}
    ds_in_noncf.lat.attrs = {}
    # Test non-CF lat/lon extraction for both DataArray and Dataset
    xe.Regridder(ds_in_noncf['data'], ds_out, 'bilinear')
    xe.Regridder(ds_in_noncf, ds_out, 'bilinear')


@pytest.mark.parametrize(
    'var_renamer,dim_out',
    [
        ({}, 'locations'),
        ({'lon': {'locations': 'foo'}, 'lat': {'locations': 'foo'}}, 'foo'),
        ({'lon': {'locations': 'foo'}, 'lat': {'locations': 'bar'}}, 'locations'),
    ],
)
def test_locstream_dim_name(var_renamer, dim_out):

    ds_locs_renamed = ds_locs.copy()
    for var, renamer in var_renamer.items():
        ds_locs_renamed[var] = ds_locs_renamed[var].rename(renamer)

    regridder = xe.Regridder(ds_in, ds_locs_renamed, 'bilinear', locstream_out=True)
    expected = {'lev', 'time', 'x_b', 'y_b', dim_out}
    actual = set(regridder(ds_in).dims)
    assert expected == actual


def test_spatial_averager_mask():
    poly = Polygon([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]])
    ds = ds_savg.copy(deep=True)
    ds.abc[1, 1] = np.nan
    savg = xe.SpatialAverager(ds, [poly], geom_dim_name='my_geom')

    # Without mask, we expect NaNs to propagate
    out = savg(ds.abc)
    assert out.isnull()

    # With masking, the NaN should be ignored.
    mask = ds.abc.notnull()
    dsm = ds.assign(
        mask=mask.T
    )  # TODO: open an issue about the fact that this fails without a transpose.
    savg = xe.SpatialAverager(dsm, [poly], geom_dim_name='my_geom')
    out = savg(dsm.abc)
    assert_allclose(out, 2, rtol=1e-3)
