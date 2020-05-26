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
    """Convert 2D CF-compliant bounds to 1D ESM boundaries.

    Parameters
    ----------
    vertices: array (n, 2)
      Contiguous left and right cell boundaries.

    Returns
    -------
    array (n+1)
      Corners coordinate.
    """
    v = vertices
    return np.concatenate((v[:, 0], v[-1:, 1]))


# TODO: This is probably not true in general, and depends on the convention used, starting point, clockwise, etc.
def cf_to_esm_bnds_2d(vertices):
    """Convert 3D CF-compliant bounds to 2D ESM boundaries.

    Parameters
    ----------
    vertices: array (n, m, 4)
      Cell corners coordinate starting from the upper left corner and going in the anti-clockwise direction.

        0 --- 3
        |  c  |
        1 --- 2

    Returns
    -------
    array (n+1, m+1)
      Corners coordinate.
    """
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

    ds = xr.Dataset(coords={'lon': (['y', 'x'], lon,
                                    {"standard_name": "longitude", "bounds": "lon_bnds",  "units":"degrees_east"}),
                            'lat': (['y', 'x'], lat,
                                    {"standard_name": "latitude", "bounds": "lat_bnds", "units": "degrees_north"}),
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

    A variable is considered a longitude if
    - its `units` are 'degrees_east' or a synonym, or
    - its `standard_name` is 'longitude'.

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
    return has_units_of_longitude(var) or has_standard_name_of_longitude(var)


def is_latitude(var):
    """Return True if variable is a latitude.

    A variable is considered a latitude if
    - its `units` are 'degrees_north' or a synonym, or
    - its `standard_name` is 'latitude'.

    Parameters
    ----------
    var: xr.Variable
      Coordinate variable.

    Return
    ------
    bool
      True if variable is a latitude.
    """
    return has_units_of_latitude(var) or has_standard_name_of_latitude(var)


def has_standard_name_of_longitude(var):
    """Whether variable has longitude as is `standard_name` attribute.

    Parameters
    ----------
    var: xr.Variable, xr.DataArray,
      Coordinate.

    Returns
    -------
    bool
      True if variable's `standard_name` attribute starts with longitude.
    """
    key = "standard_name"
    value = var.attrs.get(key, "")
    return value == "longitude" or value.startswith("longitude_at_")


def has_standard_name_of_latitude(var):
    """Whether variable has latitude as is `standard_name` attribute.

    Parameters
    ----------
    var: xr.Variable, xr.DataArray,
      Coordinate.

    Returns
    -------
    bool
      True if variable's `standard_name` attribute starts with latitude.
    """
    key = "standard_name"
    value = var.attrs.get(key, "")
    return value == "latitude" or value.startswith("latitude_at_")


def has_units_of_longitude(var):
    """Whether variable has units describing longitude coordinates.

    Recognized units of longitude are:
      - degrees_east
      - degree_east
      - degree_E
      - degrees_E
      - degreeE
      - degreesE

    Parameters
    ----------
    var: xr.Variable, xr.DataArray,
      Coordinate.

    Returns
    -------
    bool
      True if variable's `units` attribute has longitude coordinates.
    """
    return var.attrs.get("units") in ("degrees_east", "degree_east", "degree_E", "degrees_E", "degreeE", "degreesE")


def has_units_of_latitude(var):
    """Whether variable has units describing latitude coordinates.

    Recognized units of longitude are:
      - degrees_north
      - degree_north
      - degree_N
      - degrees_N
      - degreeN
      - degreesN

    Parameters
    ----------
    var: xr.Variable, xr.DataArray,
      Coordinate.

    Returns
    -------
    bool
      True if variable's `units` attribute has latitude coordinates.
    """
    return var.attrs.get("units") in ("degrees_north", "degree_north", "degree_N", "degrees_N", "degreeN", "degreesN")


def cf_lon_lat(ds):
    """Return longitude and latitude.

    Identify the longitude and latitude coordinates based on the value of the `units` or `standard_name` attributes.

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
    """Return longitude and latitude boundary variables.

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
    return cf_bnds(ds, lon), cf_bnds(ds, lat)


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
      Longitude and latitude arrays with shape (n, ), (m, ) for cartesian (rectangular) coordinates and (n, m), (n, m)
      otherwise.
    """
    if var_names is None or "lon" not in var_names or "lat" not in var_names:
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
      Longitude and latitude boundary arrays with shape (n+1), (m+1) for cartesian (rectangular) coordinates,
      and (n+1, m+1), (n+1, m+1) otherwise.
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
