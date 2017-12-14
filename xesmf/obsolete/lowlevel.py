'''
This module wraps ESMPy's complicated API and allows users to create
ESMF Grid and Regrid objects only using basic numpy arrays.

ESMPy's original API interacts with the underlying ESMF Fortran routines, 
so simple operations like grid generation require many lines of codes, mostly
about pointer handling.

General idea:
- Use simple, functional programming. Because ESMF/ESMPy Classes are complicated 
enough, building new Classes upon them will make debugging very difficult.

- ESMPy is hard to debug, because the program often dies in the Fortran level.
It would be useful to add some error checking in this wrapper level.

- In this low-level wrapper, don't use modules more advanced than numpy. Advanced 
modules like xarray should interface with this low-level wrapper, not ESMPy itself.

Jiawei Zhuang (08/28/2017)
'''

import numpy as np
import ESMF


def ESMF_grid(lon, lat, lon_b=None, lat_b=None, verbose=True):
    '''
    Create an ESMF grid object, which can be an input for ESMF_regrid()
    
    Parameters
    ----------
    lon, lat : numpy array of shape (Nlat, Nlon)
        Longitute/Latitude of cell centers

    lon_b, lat_b : optional, numpy array of shape (Nlat+1, Nlon+1)
        Longitute/Latitude of cell boundaries (corners)
        Corner information is needed by conservative regridding.

    verbose: bool, optional
        Print diagnostics for debugging.
        

    Returns
    -------   
    grid: ESMF Grid class

    '''
    
    # ESMPy documentation claims that if staggerloc and coord_sys are None,
    # they will be set to default values (CENTER and SPH_DEG).
    # However, they actually need to be set explicitly, 
    # otherwise grid._coord_sys and grid._staggerloc will still be None.
    grid = ESMF.Grid(np.array(lon.shape), 
                     staggerloc = ESMF.StaggerLoc.CENTER,
                     coord_sys = ESMF.CoordSys.SPH_DEG)
    
    # The grid object points to the underlying Fortran arrays in the original ESMF.
    # To modify lat/lon coordinates, need to get pointers to them (shown as numpy arrays) 
    lon_pointer = grid.get_coords(coord_dim=0, 
                                  staggerloc=ESMF.StaggerLoc.CENTER)
    lat_pointer = grid.get_coords(coord_dim=1, 
                                  staggerloc=ESMF.StaggerLoc.CENTER)
    
    # Use [...] to avoid overwritting the object. Only change array values. 
    lon_pointer[...] = lon
    lat_pointer[...] = lat
    
    # Corner information is only for conservative regridding,
    # not for other methods like bilinear or nearest neighbour.
    if (lon_b is None) or (lat_b is None):
        if verbose:
            print('Not passing cell corners. Conservative regridding is NOT available.')
        
    else:
        if verbose:
            print('Passing cell corners. Conservative regridding is available.')
        
        grid.add_coords(staggerloc=ESMF.StaggerLoc.CORNER)
        lon_b_pointer = grid.get_coords(coord_dim=0, 
                                        staggerloc=ESMF.StaggerLoc.CORNER)
        lat_b_pointer = grid.get_coords(coord_dim=1, 
                                        staggerloc=ESMF.StaggerLoc.CORNER)
        lon_b_pointer[...] = lon_b
        lat_b_pointer[...] = lat_b

    return grid


def ESMF_regrid_build(sourcegrid, destgrid, method='bilinear', extra_dims=None):
    '''
    Create an ESMF regrid object, containing regridding weights.


    Parameters
    ----------
    sourcegrid, destgrid: ESMF Grid class
        Source and destination grids.
        Users should create them by ESMF_grid() instead of ESMPy's original API.
    
    method: str, optional
        Regridding method. Options are
        - 'bilinear'
        - 'conservative', need grid corner information
        - 'patch'
        - 'nearest_s2d'
        - 'nearest_d2s'
        
    extra_dims: list of integers, optional
        Extra dimensions (e.g. time or levels) in the data field
        
        Extra dimensions will be stacked to fastest-changing dimensions, 
        i.e. following Fortran-like instead of C-like conventions.
        For example, if extra_dims=[N1, N2], then the data field dimension 
        will be [Nlat, Nlon, N1, N2], which is different from commonly-seen
        shape [Ntime, Nlev, Nlat, Nlon] in numpy/xarray


    Returns
    -------   
    grid: ESMF Grid class

    '''
    
    # use shorter, clearer names for options in ESMF.RegridMethod 
    method_dict ={'bilinear':ESMF.RegridMethod.BILINEAR,
                  'conservative':ESMF.RegridMethod.CONSERVE,
                  'patch':ESMF.RegridMethod.PATCH,
                  'nearest_s2d':ESMF.RegridMethod.NEAREST_STOD,
                  'nearest_d2s':ESMF.RegridMethod.NEAREST_DTOS
                 }
    try:
        RegridMethod = method_dict[method]
    except:
        raise ValueError('method should be chosen from {0}'.format(method_dict.keys()) )
        
    # conservative regridding needs cell corner information
    if method == 'conservative':
        if not sourcegrid.has_corners:
            raise ValueError('source grid has no corner information. '
                             'cannot use conservative regridding.')
        if not destgrid.has_corners:
            raise ValueError('destination grid has no corner information. '
                             'cannot use conservative regridding.')
            
    # ESMF.Regrid requires Field (Grid+data) as input, not just Grid.
    # Extra dimensions are specified when constructing the Field objects,
    # not when constructing the Regrid object later on.
    sourcefield = ESMF.Field(sourcegrid, ndbounds=extra_dims)
    destfield = ESMF.Field(destgrid, ndbounds=extra_dims)

    # Calculate regridding weights.
    # Must set unmapped_action to IGNORE, otherwise the function will fail,
    # if the destination grid is larger than the source grid.
    regrid = ESMF.Regrid(sourcefield, destfield, 
                         regrid_method = RegridMethod,
                         unmapped_action = ESMF.UnmappedAction.IGNORE)
    
    return regrid


def ESMF_regrid_apply(regrid, indata):
    '''
    Apply existing regridding weights to the data field.
    

    Parameters
    ----------
        regrid: ESMF Regrid class
            Contains the mapping of the source grid to the destination grid.
            Users should create them by ESMF_regrid_build() instead of ESMPy's original API.

        indata: numpy array of shape [Nlat, Nlon, N1, N2, ...]
            Extra dimensions [N1, N2, ...] are specified in ESMF_regrid_build()

    Returns
    -------   
    outdata: numpy array of shape [Nlat_out, Nlon_out, N1, N2, ...]
    
    '''
    
    # Get the pointers to source and destination fields.
    # Because the regrid object points to its underlying field&grid, 
    # we can just pass regrid from ESMF_regrid_build() to ESMF_regrid_apply(),
    # without having to pass all the field&grid objects.
    sourcefield = regrid.srcfield
    destfield = regrid.dstfield
    
    # pass numpy array to the underlying Fortran array
    sourcefield.data[...] = indata

    # apply regridding weights
    destfield = regrid(sourcefield, destfield)
    
    # avoid sharing the same memory with the ESMF regrid object
    outdata = destfield.data.copy()
    
    return outdata


def ESMF_regrid_finalize(regrid, verbose=True):
    '''
    Free the underlying Fortran array to avoid memory leak
    
    Parameters
    ----------
    regrid: ESMF Regrid class
        
    verbose: bool, optional
        Print finalized state. Should be all True.
        
    '''

    regrid.srcfield.destroy()
    regrid.dstfield.destroy()
    regrid.destroy()
    
    # double check
    if verbose:
        print('finalized state:', regrid.finalized,
              regrid.srcfield.finalized, regrid.dstfield.finalized)


def ESMF_regrid(sourcegrid, destgrid, indata, method='bilinear'):
    '''
    A wrapper that builds, applys and finalizes regridding weights at once.
    
    To regrid many variables, it will be more efficient to break this function 
    into 3 separate steps, i.e. 
    - Call ESMF_regrid_build() only once to calculate regridding weights.
    - Call ESMF_regrid_apply() multiple times on many data fields. 
      Applying weights should be much faster than calculating them.
    - Call ESMF_regrid_finalize() at last to avoid memory leaks.

    Parameters
    ----------
    Same as ESMF_regrid_build() and ESMF_regrid_apply()

    Returns
    -------   
    Same as ESMF_regrid_apply()
    
    '''
    
    # calculate regridding weights
    # the first two dimensions of indata are lon&lat, other dimensions are extra.
    regrid = ESMF_regrid_build(sourcegrid, destgrid, method=method, extra_dims=indata.shape[2:])
    
    # apply regridding weights
    outdata = ESMF_regrid_apply(regrid, indata)
    
    # free underlying Fortran memory to avoid memory leaks
    ESMF_regrid_finalize(regrid, verbose=False)
    
    return outdata
