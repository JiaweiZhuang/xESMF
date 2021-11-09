"""
Standard test data for regridding benchmark.
"""

import numpy as np


def wave_smooth(lon, lat):
    r"""
    Spherical harmonic with low frequency.

    Parameters
    ----------
    lon, lat : 2D numpy array or xarray DataArray
         Longitute/Latitude of cell centers

    Returns
    -------
    f : 2D numpy array or xarray DataArray depending on input
        2D wave field

    Notes
    -------
    Equation from [1]_ [2]_:

    .. math:: Y_2^2 = 2 + \cos^2(\\theta) \cos(2 \phi)

    References
    ----------
    .. [1] Jones, P. W. (1999). First-and second-order conservative remapping
       schemes for grids in spherical coordinates. Monthly Weather Review,
       127(9), 2204-2210.

    .. [2] Ullrich, P. A., Lauritzen, P. H., & Jablonowski, C. (2009).
       Geometrically exact conservative remapping (GECoRe): regular
       latitude–longitude and cubed-sphere grids. Monthly Weather Review,
       137(6), 1721-1741.
    """
    # degree to radius, make a copy
    lat = lat / 180.0 * np.pi
    lon = lon / 180.0 * np.pi

    f = 2 + np.cos(lat) ** 2 * np.cos(2 * lon)
    return f
