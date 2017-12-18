import numpy as np
import xarray as xr
import warnings

from . backend import esmf_grid, add_corner


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
        Logitude step size, i.e. grid resolution

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
        Logitude step size, i.e. grid resolution

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


def ds_to_ESMFgrid(ds):
    '''
    Convert xarray DataSet to ESMF.Grid object.

    Parameters
    ----------
    ds : xarray DataSet
        Contains variables ``lon``, ``lat``,
        and optionally ``lon_b``, ``lat_b``.

        Shape should be ``(Nlat, Nlon)`` or ``(Ny, Nx)``,
        as normal C or Python ordering. Will be then tranposed to F-ordered.

    Returns
    -------
    grid : ESMF.Grid object

    '''

    # tranpose the arrays so they become Fortran-ordered
    lon, lat = ds['lon'].values, ds['lat'].values
    grid = esmf_grid(lon.T, lat.T)

    # cell bounds are optional
    try:
        lon_b, lat_b = ds['lon_b'].values, ds['lat_b'].values
        add_corner(grid, lon_b.T, lat_b.T)
    except:
        pass

    return grid
