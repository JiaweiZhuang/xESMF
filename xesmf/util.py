import numpy as np
import xarray as xr
import warnings


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


def _cf_bnds_1d(bounds):
    """Return a CF-compliant coordinate boundaries.

    Parameters
    ----------
    bounds: array (n+1)
      Array with ordered left and right boundaries.

    Returns
    -------
    array (n, 2)
      Left [:,0] and right [:1] boundaries.
    """
    return np.vstack((bounds[:-1], bounds[1:])).T


def _cf_bnds_2d(bounds):
    """Return a CF-compliant coordinate boundaries.

    Parameters
    ----------
    bounds: array (n+1, m+1)
      Array with ordered left and right boundaries.

    Returns
    -------
    array (n, m, 4)
      Counter
    """
    return np.dstack((bounds[:-1, :-1], bounds[1:, :-1], bounds[1:, 1:], bounds[:-1, 1:]))


def cf_to_esm_bnds_1d(vertices):
    """Convert 2D CF-compliant boundaries to 1D ESM boundaries."""
    v = vertices
    return np.concatenate((v[:, 0], v[-1:, 1]))


def cf_to_esm_bnds_2d(vertices):
    """Convert 3D CF-compliant boundaries to 2D ESM boundaries."""
    v = vertices
    tl = v[:, :, 0]
    bl = v[-1:, :, 1]
    tr = v[:, -1:, 3]
    br = v[-1:, -1:, 2]
    return np.block([[tl, tr],
                     [bl, br]])


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

    ds = xr.Dataset(coords={'lon': (['y', 'x'], lon, {"long_name": "longitude", "bounds": "lon_bnds"}),
                            'lat': (['y', 'x'], lat, {"long_name": "latitude", "bounds": "lat_bnds"}),
                            'lon_b': (['y_b', 'x_b'], lon_b),
                            'lat_b': (['y_b', 'x_b'], lat_b),
                            'lon_bnds': (['y', 'x', 'nv'], _cf_bnds_2d(lon_b)),
                            'lat_bnds': (['y', 'x', 'nv'], _cf_bnds_2d(lat_b))
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
