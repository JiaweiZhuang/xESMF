import numpy as np
import xesmf as xe
import xarray as xr
import pytest

from . test_frontend import ds_in


def test_grid_global():
    ds = xe.util.grid_global(1.5, 1.5)
    refshape = (120, 240)
    refshape_b = (121, 241)

    assert ds['lon'].values.shape == refshape
    assert ds['lat'].values.shape == refshape
    assert ds['lon_b'].values.shape == refshape_b
    assert ds['lat_b'].values.shape == refshape_b


def test_grid_global_bad_resolution():
    with pytest.warns(UserWarning):
        xe.util.grid_global(1.5, 1.23)

    with pytest.warns(UserWarning):
        xe.util.grid_global(1.23, 1.5)


def test_cf_bnds_1d():
    xc, xb = xe.util._grid_1d(0, 2, 1)
    cfb = xe.util._cf_bnds_1d(xb)
    np.testing.assert_array_equal(cfb, [[0, 1], [1, 2]])
    new_xb = xe.util.cf_to_esm_bnds_1d(cfb)
    np.testing.assert_array_equal(new_xb, xb)


def test_cf_bnds_2d():
    ds = xe.util.grid_2d(0, 2, 1, 10, 11, 1)
    eb = xe.util.cf_to_esm_bnds_2d(ds['lat_bnds'])
    np.testing.assert_array_equal(eb, ds['lat_b'])


def test_cf_lon_lat():
    # Dataset
    lon, lat = xe.util.cf_lon_lat(ds_in)
    xr.testing.assert_equal(lon, ds_in["lon"])
    xr.testing.assert_equal(lat, ds_in["lat"])

    # DataArray
    lon, lat = xe.util.cf_lon_lat(ds_in["data"])
    xr.testing.assert_equal(lon, ds_in["lon"])
    xr.testing.assert_equal(lat, ds_in["lat"])

    # Dict
    lon, lat = xe.util.cf_lon_lat(dict(ds_in["data"].coords))
    xr.testing.assert_equal(lon, ds_in["lon"])
    xr.testing.assert_equal(lat, ds_in["lat"])

    # Missing attributes
    ds_miss = ds_in.copy()
    ds_miss.lon.attrs["long_name"] = ""
    with pytest.raises(ValueError):
        xe.util.cf_lon_lat(ds_miss)

    ds_miss = ds_in.copy()
    ds_miss.lat.attrs["long_name"] = ""
    with pytest.raises(ValueError):
        xe.util.cf_lon_lat(ds_miss)


def test_cf_bnds():
    ds_miss = ds_in.copy()
    ds_miss.lat.attrs.pop("bounds")
    with pytest.raises(ValueError):
        xe.util.cf_bnds(ds_miss, ds_miss["lat"])

    ds_miss = ds_in.copy()
    ds_miss.lat.attrs["bounds"] = "kabong"
    with pytest.raises(ValueError):
        xe.util.cf_bnds(ds_miss, ds_miss["lat"])
