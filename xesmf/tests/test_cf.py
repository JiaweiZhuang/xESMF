import pytest
import numpy as np
import xarray as xr
import xesmf.cf as xecf


coord_base = xr.DataArray(np.arange(2), dims='d')
da = xr.DataArray(np.ones(2), dims='d', name='temp')


def test_cf_get_lon_name_warns():
    with pytest.warns(UserWarning):
        xecf.get_lon_name(da.to_dataset())


@pytest.mark.parametrize("name,coord_specs", [
    ('lon', coord_base),
    ('longitude', coord_base),
    ('x', ('d', coord_base, {'units': 'degrees_east'})),
    ('x', ('d', coord_base, {'standard_name': 'longitude'})),
    ('xu', ('d', coord_base, {'standard_name': 'longitude_at_U_location'})),
    ('x', ('d', coord_base, {'axis': 'X'})),
    ('x', ('d', coord_base, {'units': 'degrees_east', 'axis': 'Y'})),
    ])
def test_cf_get_lon_name(name, coord_specs):
    assert xecf.get_lon_name(da.assign_coords(**{name: coord_specs})) == name


def test_cf_get_lat_name_warns():
    with pytest.warns(UserWarning):
        xecf.get_lat_name(da.to_dataset())


@pytest.mark.parametrize("name,coord_specs", [
    ('lat', coord_base),
    ('latitude', coord_base),
    ('y', ('d', coord_base, {'units': 'degrees_north'})),
    ('y', ('d', coord_base, {'standard_name': 'latitude'})),
    ('yu', ('d', coord_base, {'standard_name': 'latitude_at_V_location'})),
    ('y', ('d', coord_base, {'axis': 'Y'})),
    ('y', ('d', coord_base, {'units': 'degrees_north', 'axis': 'X'})),
    ])
def test_cf_get_lat_name(name, coord_specs):
    assert xecf.get_lat_name(da.assign_coords(**{name: coord_specs})) == name


def test_cf_decode_cf():

    yy, xx = np.mgrid[:3, :4].astype('d')
    yyb, xxb = np.mgrid[:4, :5] - .5

    xx = xr.DataArray(xx, dims=['ny', 'nx'], attrs={'units': "degrees_east"})
    xxb = xr.DataArray(xxb, dims=['nyb', 'nxb'])
    yy = xr.DataArray(yy, dims=['ny', 'nx'])
    yyb = xr.DataArray(yyb, dims=['nyb', 'nxb'])

    ds = xr.Dataset({
        'xx': xx,
        'xx_b': xxb,
        'latitude': yy,
        'latitude_bounds': yyb})
    ds_decoded = xecf.decode_cf(ds)
    assert sorted(list(ds_decoded)) == ['lat', 'lat_b', 'lon', 'lon_b']
