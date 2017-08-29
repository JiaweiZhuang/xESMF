'''
This module interfaces ESMPy and xarray, based on low-level wrappers.

Major challenges are:
- ESMPy requires additional dimensions to be on the rightmost side, 
for example [Nlat, Nlon, Ntime, Nlev]. However, data from NetCDF file
mostly has the shape of [Ntime, Nlev, Nlat, Nlon]. To ensure input and 
output data have consistent shapes, xarray's transpose() method is heavily used,
but it might not be very memory-efficient.

- dask integration (not available yet). ESMF/ESMPy parallelizes in the horizontal,
using MPI for horizontal domain decomposition. With dask, parallelizing over
other dimensions (e.g. lev, time) would be a much better option.

- Error checking. Because ESMPy's error message is hard to interpret,
there's a danger for debugging if we build too many levels of wrappers.

Jiawei Zhuang (08/28/2017)
'''

import xarray as xr
from . lowlevel import ESMF_grid, ESMF_regrid


def _ds_to_ESMFgrid(ds, verbose=False):
    '''
    Convert xarray DataSet to ESMF Grid object.
    
    Parameters
    ----------
    ds: xarray DataSet
        Contains coordinate information. 
        Look for variables 'lon', 'lat', 'lon_b', 'lat_b'.
        Shape must be 2D, as ESMF_grid() expects.
        
    verbose: bool, optional
        Print diagnostics for debugging.
    
    Returns
    -------   
    grid: ESMF Grid class
    
    '''
    
    lon, lat = ds['lon'].values, ds['lat'].values
    if [lon.ndim, lat.ndim] != [2, 2]:
        raise ValueError('coordinate variables must be 2D (Nlat, Nlon)')
    
    # cell bounds are optional
    try:
        lon_b, lat_b = ds['lon_b'].values, ds['lat_b'].values
    except:
        lon_b, lat_b = None, None
    
    grid = ESMF_grid(lon, lat, lon_b=lon_b, lat_b=lat_b, verbose=verbose)
    
    return grid

def regrid(ds_in, ds_out, dr_in, method='bilinear', verbose=False):
    '''
    Regrid xarray DataArray using ESMF regridding.
    
    Parameters
    ----------
    ds_in, ds_out: xarray DataSet
        Contain input and output grid information.
        Look for variables 'lon', 'lat', 'lon_b', 'lat_b'.
        Shape must be 2D, as ESMF_grid() expects.
        
    dr_in: xarray DataArray
        Can have arbitrary additional dimensions.
        
    method: method: str, optional
        Regridding method. Options are
        - 'bilinear'
        - 'conservative', need grid corner information
        - 'patch'
        - 'nearest_s2d'
        - 'nearest_d2s'
        See ESMF_regrid_build()
        
    verbose: bool, optional
        Print diagnostics for debugging.
    
    Returns
    -------   
    dr_out: xarray DataArray
        On the same horizontal grid as ds_out with extra dimensions in dr_in.
    
    '''
    
    # get input dimensions
    dims_grid_in = list(ds_in['lon'].dims)
    dims_dr_in = list(dr_in.dims)
    
    if verbose:
        print('input grid dimensions:', dims_grid_in)
        print('input data dimensions:', dims_dr_in)
    
    # figure out additional dimensions (e.g. time, level)
    dims_ESMF_extra = list(dr_in.dims)
    dims_ESMF_extra.remove(dims_grid_in[0])
    dims_ESMF_extra.remove(dims_grid_in[1])
    
    if verbose:
        print('additional dimensions:', dims_ESMF_extra)
    
    # move all additional dimensions to the right 
    dims_to_ESMF = dims_grid_in + dims_ESMF_extra
    dr_in_to_ESMF = dr_in.transpose(*dims_to_ESMF)
    
    if verbose:
        print('To match ESMPy API, rearrange input data dimensions to:',
              dr_in_to_ESMF.dims)
    
    # get raw numpy array
    indata = dr_in_to_ESMF.values
    
    # Call low-level functions to perform regridding
    sourcegrid = _ds_to_ESMFgrid(ds_in, verbose=verbose)
    destgrid = _ds_to_ESMFgrid(ds_out, verbose=verbose)
    outdata = ESMF_regrid(sourcegrid, destgrid, indata, method=method)
    sourcegrid.destroy();destgrid.destroy(); # avoid memory leak
    
    # now the output data is available as a numpy array
    # need to construct a full DataArray by adding metadata.
    
    # get output dimension information
    dims_grid_out = list(ds_out['lon'].dims)
    dims_from_ESMF = dims_grid_out + dims_ESMF_extra
    
    # pass regridding results to the tempory DataSet to get grid information
    ds_out_temp = ds_out.copy()
    varname = dr_in.name
    ds_out_temp[varname] = (dims_from_ESMF, outdata)
    dr_out = ds_out_temp[varname]
    
    if verbose:
        print('output dimensions from ESMPy:', dr_out.dims)
    
    # move additional dimensions to the left.
    dims_return =  dims_ESMF_extra + dims_grid_out
    dr_out = dr_out.transpose(*dims_return)
    
    if verbose:
        print('rearrange output dimensions to:', dr_out.dims)
    
    # add coordinate values for additional dimensions
    for extra_dim in dims_ESMF_extra:
        dr_out.coords[extra_dim] = dr_in.coords[extra_dim]
    
    return dr_out
