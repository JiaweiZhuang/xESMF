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
    """Return CF-compliant coordinate bounds.

    Parameters
    ----------
    bounds: array (n+1)
      Array with ordered left and right boundaries.

    Returns
    -------
    array (n, 2)
      Left [:,0] and right [:1] bounds.
    """
    return np.vstack((bounds[:-1], bounds[1:])).T


def _cf_bnds_2d(bounds):
    """Return CF-compliant coordinate bounds.

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
    """Convert 2D CF-compliant bounds to 1D ESM boundaries."""
    v = vertices
    return np.concatenate((v[:, 0], v[-1:, 1]))


# TODO: This is probably not true in general, and depends on the convention used, starting point, clockwise, etc.
def cf_to_esm_bnds_2d(vertices):
    """Convert 3D CF-compliant bounds to 2D ESM boundaries."""
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


def is_longitude(var):
    """Return True if variable is a longitude.

    A variable is considered a longitude if its `long_name` attribute is 'longitude'.

    Parameters
    ----------
    var: xr.Variable
      Coordinate variable.

    Return
    ------
    bool
      True if variable is a longitude.

    Example
    -------
    >>> # Identify the longitude coordinate in a dataset.
    >>> lon = next(filter(is_longitude, ds.coords))
    """
    return var.attrs.get("long_name") == "longitude"


def is_latitude(var):
    """Return True if variable is a latitude.

    A variable is considered a latitude if its `long_name` attribute is 'latitude'.

    Parameters
    ----------
    var: xr.Variable
      Coordinate variable.

    Return
    ------
    bool
      True if variable is a latitude.
    """
    return var.attrs.get("long_name") == "latitude"


def cf_lon_lat(ds):
    """Return longitude and latitude.

    Identify the longitude and latitude coordinates based on the value of the `long_name` attribute.

    Parameters
    ----------
    ds: xr.Dataset
      Dataset storing coordinate information.

    Return
    ------
    xr.DataArray, xr.DataArray
      Longitude and latitude coordinates.
    """
    if isinstance(ds, xr.Dataset):
        x = list(ds.coords.values()) + list(ds.data_vars.values())
    elif isinstance(ds, xr.DataArray):
        x = ds.coords.values()
    else:  # dict
        x = ds.values()

    lon = list(filter(is_longitude, x))
    lat = list(filter(is_latitude, x))

    msg = "No {0} coordinate found." \
          "Identify the longitude variable by setting its `long_name` attribute to `{0}`."

    if not lon:
        raise ValueError(msg.format("longitude"))

    if not lat:
        raise ValueError(msg.format("latitude"))

    return lon[0], lat[0]


def cf_bnds(ds, coord):
    """Return the coordinate boundaries for a given coordinate.

    Identify the boundaries based on the value of the `bounds` attribute of the coordinate.
    If coordinate has no `bounds` attribute, raise an error.

    Parameters
    ----------
    ds: xr.Dataset
      Dataset storing coordinate information.
    coord: xr.DataArray
      Coordinate whose boundaries should be returned.

    Return
    ------
    xr.DataArray
      The coordinate boundaries.
    """
    key = coord.attrs.get("bounds")
    if key is None:
        raise ValueError(f"No bounds found for {coord.name}."
                         "Identify the longitude bounds by setting the `bounds` attribute of the longitude variable "
                         "to an existing variable name.")
    if key not in ds:
        raise ValueError(f"The variable {key} identified as the bounds for {coord.name} is not in this dataset.")

    return ds[key]


def cf_lon_lat_bnds(ds):
    """Return longitude and latitude boundaries.

    Parameters
    ----------
    ds: xr.Dataset
      Dataset storing coordinate information.

    Return
    ------
    xr.DataArray, xr.DataArray
      Longitude and latitude boundaries.
    """
    lon, lat = cf_lon_lat(ds)
    return cf_bnds(ds ,lon), cf_bnds(ds, lat)


def get_lon_lat(ds, var_names=None):
    """Return longitude and latitude coordinates.

    Parameters
    ----------
    ds: xr.Dataset
      Dataset storing coordinate information.
    var_names: dict
      Dictionary with keys `lon` and `lat` identifying the name of the variables storing longitude and latitude
      coordinates respectively.

    Return
    ------
    np.ndarray, np.ndarray
      Longitude and latitude arrays.
    """
    if var_names is None:
        out = cf_lon_lat(ds)
    else:
        out = ds[var_names["lon"]], ds[var_names["lat"]]

    return map(np.asarray, out)


def get_lon_lat_bnds(ds, var_names=None):
    """Return longitude and latitude coordinate boundaries.

    Parameters
    ----------
    ds: xr.Dataset
      Dataset storing coordinate information.
    var_names: dict
      Dictionary with keys `lon_b` and `lat_b` identifying the name of the variables storing longitude and latitude
      coordinate boundaries respectively.

    Return
    ------
    np.ndarray, np.ndarray
      Longitude and latitude boundary arrays.
    """
    if var_names is None:
        lon_bnds, lat_bnds = cf_lon_lat_bnds(ds)
        if lon_bnds.ndim == 2:
            out = map(cf_to_esm_bnds_1d, [lon_bnds, lat_bnds])
        elif lon_bnds.ndim == 3:
            out = map(cf_to_esm_bnds_2d, [lon_bnds, lat_bnds])
    else:
        out = ds[var_names["lon_b"]], ds[var_names["lat_b"]]

    return map(np.asarray, out)
