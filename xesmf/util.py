import numpy as np
import xarray as xr

def _grid_1d(start_b, end_b, step):

    bounds = np.arange(start_b, end_b+step, step)
    centers = (bounds[:-1] + bounds[1:])/2

    return centers, bounds

def grid_2d(lon0_b, lon1_b, d_lon,
            lat0_b, lat1_b, d_lat):
    
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
    
    return grid_2d(-180, 180, d_lon, -90, 90, d_lat)