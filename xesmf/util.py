import xesmf as xe
import numpy as np
import xarray as xr
import warnings

LON = 'lon'
LAT = 'lat'


def _grid_1d(start_b, end_b, step):
    '''
    1D grid centers and bounds

    Parameters
    ----------
    start_b, end_b : float
        start/end position. Bounds, not centers.

    step: float
        step size, i.e. grid resolution

    Returns
    -------
    centers : 1D numpy array

    bounds : 1D numpy array, with one more element than centers

    '''
    bounds = np.arange(start_b, end_b+step, step)
    centers = (bounds[:-1] + bounds[1:])/2

    return centers, bounds


def grid_2d(lon0_b, lon1_b, d_lon,
            lat0_b, lat1_b, d_lat):
    '''
    2D rectilinear grid centers and bounds

    Parameters
    ----------
    lon0_b, lon1_b : float
        Longitude bounds

    d_lon : float
        Longitude step size, i.e. grid resolution

    lat0_b, lat1_b : float
        Latitude bounds

    d_lat : float
        Latitude step size, i.e. grid resolution

    Returns
    -------
    ds : xarray DataSet with coordinate values

    '''

    lon_1d, lon_b_1d = _grid_1d(lon0_b, lon1_b, d_lon)
    lat_1d, lat_b_1d = _grid_1d(lat0_b, lat1_b, d_lat)

    lon, lat = np.meshgrid(lon_1d, lat_1d)
    lon_b, lat_b = np.meshgrid(lon_b_1d, lat_b_1d)

    ds = xr.Dataset(coords={'lon': (['y', 'x'], lon),
                            'lat': (['y', 'x'], lat),
                            'lon_b': (['y_b', 'x_b'], lon_b),
                            'lat_b': (['y_b', 'x_b'], lat_b)
                            }
                    )

    return ds


def grid_global(d_lon, d_lat):
    '''
    Global 2D rectilinear grid centers and bounds

    Parameters
    ----------
    d_lon : float
        Longitude step size, i.e. grid resolution

    d_lat : float
        Latitude step size, i.e. grid resolution

    Returns
    -------
    ds : xarray DataSet with coordinate values

    '''

    if not np.isclose(360/d_lon, 360//d_lon):
        warnings.warn('360 cannot be divided by d_lon = {}, '
                      'might not cover the globe uniformally'.format(d_lon))

    if not np.isclose(180/d_lat, 180//d_lat):
        warnings.warn('180 cannot be divided by d_lat = {}, '
                      'might not cover the globe uniformally'.format(d_lat))

    return grid_2d(-180, 180, d_lon, -90, 90, d_lat)


def _regrid_it(da, d_lon, d_lat, **kwargs):
    '''
    Global 2D rectilinear grid centers and bounds

    Parameters
    ----------
    da : xarray DataArray
        Contain input and output grid coordinates. Look for variables
        ``lon``, ``lat``, and optionally ``lon_b``, ``lat_b`` for
        conservative method.

        Shape can be 1D (Nlon,) and (Nlat,) for rectilinear grids,
        or 2D (Ny, Nx) for general curvilinear grids.
        Shape of bounds should be (N+1,) or (Ny+1, Nx+1).

    d_lon : float
        Longitude step size, i.e. grid resolution

    d_lat : float
        Latitude step size, i.e. grid resolution

    Returns
    -------
    da : xarray DataArray with coordinate values

    '''
    try:
        grid_out = {LON: np.arange(da[LON].min(), da[LON].max() + d_lon, d_lon),
                    LAT: np.arange(da[LAT].min(), da[LAT].max() + d_lat, d_lat)}
        regridder = xe.Regridder(da, grid_out, **kwargs)
        return regridder(da)
    except KeyError:
        warnings.warn('Skipping {0} because it has no '
                      'lon / lat coordinate'.format(da.name))
        return da


def regrid_it(ds, d_lon=1, d_lat=None, method='bilinear',
              periodic=False, filename=None, reuse_weights=True):
    '''
    Quick regridding

    Parameters
    ----------
    ds : xarray DataSet
        Contain input and output grid coordinates. Look for variables
        ``lon``, ``lat``, and optionally ``lon_b``, ``lat_b`` for
        conservative method.

        Shape can be 1D (Nlon,) and (Nlat,) for rectilinear grids,
        or 2D (Ny, Nx) for general curvilinear grids.
        Shape of bounds should be (N+1,) or (Ny+1, Nx+1).

    d_lon : float, optional
        Longitude step size, i.e. grid resolution; if not provided,
        will equal 1

    d_lat : float, optional
        Latitude step size, i.e. grid resolution; if not provided,
        will equal d_lon

    method : str
        Regridding method. Options are

        - 'bilinear'
        - 'conservative', **need grid corner information**
        - 'patch'
        - 'nearest_s2d'
        - 'nearest_d2s'

    periodic : bool, optional
        Periodic in longitude? Default to False.
        Only useful for global grids with non-conservative regridding.
        Will be forced to False for conservative regridding.

    filename : str, optional
        Name for the weight file. The default naming scheme is::

            {method}_{Ny_in}x{Nx_in}_{Ny_out}x{Nx_out}.nc

        e.g. bilinear_400x600_300x400.nc

    reuse_weights : bool, optional
        Whether to read existing weight file to save computing time.
        False by default (i.e. re-compute, not reuse).

    Returns
    -------
    ds : xarray DataSet with coordinate values or DataArray

    '''
    if d_lat is None:
        d_lat = d_lon

    kwargs = {
        'd_lon': d_lon,
        'd_lat': d_lat,
        'method': method,
        'periodic':periodic,
        'filename': filename,
        'reuse_weights': reuse_weights,
    }

    if isinstance(ds, xr.Dataset):
        ds = xr.merge(_regrid_it(ds[var], **kwargs)
                      for var in ds.data_vars)
    else:
        ds = _regrid_it(ds, **kwargs)

    return ds
