"""
Decode longitude and latitude of an `xarray.Dataset` or a `xarray.DataArray`
according to CF conventions.
"""

import warnings

CF_SPECS = {
    'lon': {
        'name': ['lon', 'longitude'],
        'units': ['degrees_east', 'degree_east', 'degree_E', 'degrees_E',
                  'degreeE', 'degreesE'],
        'standard_name': ['longitude'],
        'axis': 'X'
        },
    'lat': {
        'name': ['lat', 'latitude'],
        'units': ['degrees_north', 'degree_north', 'degree_N', 'degrees_N',
                  'degreeN', 'degreesN'],
        'standard_name': ['latitude'],
        'axis': 'Y'
        }
    }



def get_coord_name_from_specs(ds, specs):
    '''
    Get the name of a `xarray.DataArray` according to search specifications

    Parameters
    ----------
    ds: xarray.DataArray or xarray.Dataset
    specs: dict

    Returns
    -------
    str: Name of the xarray.DataArray
    '''
    # Targets
    ds_names = []
    if hasattr(ds, 'coords'):
        ds_names.extend(list(ds.coords))
    if hasattr(ds, 'variables'):
        ds_names.extend(list(ds.variables))

    # Search in names first
    if 'name' in specs:
        for specs_name in specs['name']:
            if specs_name in ds_names:
                return specs_name

    # Search in units attributes
    if 'units' in specs:
        for ds_name in ds_names:
            if hasattr(ds[ds_name], 'units'):
                for specs_units in specs['units']:
                    if ds[ds_name].units == specs_units:
                        return ds_name

    # Search in standard_name attributes
    if 'standard_name' in specs:
        for ds_name in ds_names:
            if hasattr(ds[ds_name], 'standard_name'):
                for specs_sn in specs['standard_name']:
                    if (ds[ds_name].standard_name == specs_sn or
                            ds[ds_name].standard_name.startswith(
                                specs_sn+'_at_')):
                        return ds_name

    # Search in axis attributes
    if 'axis' in specs:
        for ds_name in ds_names:
            if (hasattr(ds[ds_name], 'axis') and
                    ds[ds_name].axis.lower() == specs['axis'].lower()):
                return ds_name


def get_lon_name(ds):
    '''
    Get the longitude name in a `xarray.Dataset` or a `xarray.DataArray`

    Parameters
    ----------
    ds: xarray.DataArray or xarray.Dataset

    Returns
    -------
    str: Name of the xarray.DataArray
    '''
    lon_name = get_coord_name_from_specs(ds, CF_SPECS['lon'])
    if lon_name is not None:
        return lon_name
    warnings.warn('longitude not found in dataset')


def get_lat_name(ds):
    '''
    Get the latitude name in a `xarray.Dataset` or a `xarray.DataArray`

    Parameters
    ----------
    ds: xarray.DataArray or xarray.Dataset

    Returns
    -------
    str: Name of the xarray.DataArray
    '''
    lat_name = get_coord_name_from_specs(ds, CF_SPECS['lat'])
    if lat_name is not None:
        return lat_name
    warnings.warn('latitude not found in dataset')


def decode_cf(ds):
    '''
    Search for longitude and latitude and rename them

    Parameters
    ----------
    ds: xarray.DataArray or xarray.Dataset

    Returns
    -------
    ds: xarray.DataArray or xarray.Dataset
    '''
    # Longitude
    lon_name = get_lon_name(ds)
    if lon_name is not None:
        ds = ds.rename({lon_name: 'lon'})
        for suffix in ('_b', '_bounds'):
            if lon_name+suffix in ds:
                ds = ds.rename({lon_name+suffix: 'lon_b'})

    # Latitude
    lat_name = get_lat_name(ds)
    if lat_name is not None:
        ds = ds.rename({lat_name: 'lat'})
        for suffix in ('_b', '_bounds'):
            if lat_name+suffix in ds:
                ds = ds.rename({lat_name+suffix: 'lat_b'})

    return ds
