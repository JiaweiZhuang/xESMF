import warnings

import numpy as np
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon


def _grid_1d(start_b, end_b, step):
    """
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
    """

    bounds = np.arange(start_b, end_b + step, step)
    centers = (bounds[:-1] + bounds[1:]) / 2

    return centers, bounds


def grid_2d(lon0_b, lon1_b, d_lon, lat0_b, lat1_b, d_lat):
    """
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

    """

    lon_1d, lon_b_1d = _grid_1d(lon0_b, lon1_b, d_lon)
    lat_1d, lat_b_1d = _grid_1d(lat0_b, lat1_b, d_lat)

    lon, lat = np.meshgrid(lon_1d, lat_1d)
    lon_b, lat_b = np.meshgrid(lon_b_1d, lat_b_1d)

    ds = xr.Dataset(
        coords={
            'lon': (['y', 'x'], lon, {'standard_name': 'longitude'}),
            'lat': (['y', 'x'], lat, {'standard_name': 'latitude'}),
            'lon_b': (['y_b', 'x_b'], lon_b),
            'lat_b': (['y_b', 'x_b'], lat_b),
        }
    )

    return ds


def grid_global(d_lon, d_lat):
    """
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

    """

    if not np.isclose(360 / d_lon, 360 // d_lon):
        warnings.warn(
            '360 cannot be divided by d_lon = {}, '
            'might not cover the globe uniformally'.format(d_lon)
        )

    if not np.isclose(180 / d_lat, 180 // d_lat):
        warnings.warn(
            '180 cannot be divided by d_lat = {}, '
            'might not cover the globe uniformally'.format(d_lat)
        )

    return grid_2d(-180, 180, d_lon, -90, 90, d_lat)


def _flatten_poly_list(polys):
    """Iterator flattening MultiPolygons."""
    for i, poly in enumerate(polys):
        if isinstance(poly, MultiPolygon):
            for sub_poly in poly:
                yield (i, sub_poly)
        else:
            yield (i, poly)


def split_polygons_and_holes(polys):
    """Split the exterior boundaries and the holes for a list of polygons.

    If MultiPolygons are encountered in the list, they are flattened out
    in their constituents.

    Parameters
    ----------
    polys : Sequence of shapely Polygons or MultiPolygons

    Returns
    -------
    exteriors : list of Polygons
        The polygons without any holes
    holes : list of Polygons
        Holes of the polygons as polygons
    i_ext : list of integers
       The index in `polys` of each polygon in `exteriors`.
    i_hol : list of integers
       The index in `polys` of the owner of each hole in `holes`.
    """
    exteriors = []
    holes = []
    i_ext = []
    i_hol = []
    for (i, poly) in _flatten_poly_list(polys):
        exteriors.append(Polygon(poly.exterior))
        i_ext.append(i)
        holes.extend(map(Polygon, poly.interiors))
        i_hol.extend([i] * len(poly.interiors))

    return exteriors, holes, i_ext, i_hol
