import os
import warnings

import cf_xarray  # noqa
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
    data_vars={'abc': (('lon', 'lat'), [[1, 2], [3, 4], [2, 4]])},
)
polys = [
    Polygon([[0.5, 0.5], [0.5, 1.5], [1.5, 0.5]]),  # Simple triangle polygon
    MultiPolygon(
        [
            Polygon([[0.25, 1.25], [0.25, 1.75], [0.75, 1.75], [0.75, 1.25]]),
            Polygon([[1.25, 1.25], [1.25, 1.75], [1.75, 1.75], [1.75, 1.25]]),
        ]
    ),  # Multipolygon on 3 and 4
    Polygon(
        [[0, 0], [0, 1], [2, 1], [2, 0]],
        holes=[[[0.5, 0.25], [0.5, 0.75], [1.0, 0.75], [1.0, 0.25]]],
    ),  # Simple polygon covering 1 and 2 with hole spanning on both
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
]
exps_polys = [1.75, 3, 2.1429, 4, 0, 2.5]


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
    'locstream_in,locstream_out,method',
    [
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
        (True, True, 'nearest_d2s'),
    ],
)
def test_build_regridder(method, locstream_in, locstream_out):
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
    assert regridder_reuse.A.shape == regridder.A.shape

    # this should also work with reuse_weights=True
    regridder_reuse = xe.Regridder(ds_in, ds_out, method, reuse_weights=True, weights=fn)
    assert regridder_reuse.A.shape == regridder.A.shape

    # or can also overwrite it
    xe.Regridder(ds_in, ds_out, method)

    # check legacy args still work
    regridder = xe.Regridder(ds_in, ds_out, method, filename='wgts.nc')
    regridder_reuse = xe.Regridder(ds_in, ds_out, method, reuse_weights=True, filename='wgts.nc')
    assert regridder_reuse.A.shape == regridder.A.shape

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
    regridder = xe.Regridder(ds_in, ds_out, method)
    regridder.to_netcdf(filename=xfn)

    grid_in = Grid.from_xarray(ds_in['lon'].values.T, ds_in['lat'].values.T)
    grid_out = Grid.from_xarray(ds_out['lon'].values.T, ds_out['lat'].values.T)

    # Let the ESMPy backend write the weights to disk
    efn = tmp_path / 'weights.nc'
    esmf_regrid_build(grid_in, grid_out, method=method, filename=str(efn))

    x = xr.open_dataset(xfn)
    e = xr.open_dataset(efn)
    xr.testing.assert_identical(x, e)


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
    regridder = xe.Regridder(ds_in, ds_out, 'bilinear')

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
    outdata = regridder(indata)

    assert dask.is_dask_collection(outdata)

    # lazy dask arrays have incorrect shape attribute due to last chunk
    assert outdata.compute().shape == indata.shape[:-2] + horiz_shape_out
    assert outdata.chunksize == indata.chunksize[:-2] + horiz_shape_out

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

    # check metadata
    xr.testing.assert_identical(ds_result['time'], ds_in['time'])
    xr.testing.assert_identical(ds_result['lev'], ds_in['lev'])
    assert_equal(ds_result['lat'].values, ds_out['lat'].values)
    assert_equal(ds_result['lon'].values, ds_out['lon'].values)


def test_regrid_dataset_to_locstream():
    # xarray.Dataset containing in-memory numpy array

    regridder = xe.Regridder(ds_in, ds_locs, 'bilinear', locstream_out=True)
    regridder(ds_in)


def test_build_regridder_with_masks():
    ds_in['mask'] = xr.DataArray(np.random.randint(2, size=ds_in['data'].shape), dims=('y', 'x'))
    print(ds_in)
    # 'patch' is too slow to test
    for method in [
        'bilinear',
        'conservative',
        'conservative_normed',
        'nearest_s2d',
        'nearest_d2s',
    ]:
        regridder = xe.Regridder(ds_in, ds_out, method)

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
    savg = xe.SpatialAverager(ds_savg, [poly], geom_dim_name='my_geom')
    out = savg(ds_savg.abc)
    assert_allclose(out, exp, rtol=1e-3)

    assert 'my_geom' in out.dims


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
