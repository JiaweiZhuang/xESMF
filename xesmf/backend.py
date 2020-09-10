'''
Backend for xESMF. This module wraps ESMPy's complicated API and can create
ESMF Grid and Regrid objects only using basic numpy arrays.

General idea:

1) Only use pure numpy array in this low-level backend. xarray should only be
used in higher-level APIs which interface with this low-level backend.

2) Use simple, procedural programming here. Because ESMPy Classes are
complicated enough, building new Classes will make debugging very difficult.

3) Add some basic error checking in this wrapper level.
ESMPy is hard to debug because the program often dies in the Fortran level.
So it would be helpful to catch some common mistakes in Python level.
'''

import numpy as np
import ESMF
from shapely.geometry import MultiPolygon
import warnings
import os


def warn_f_contiguous(a):
    '''
    Give a warning if input array if not Fortran-ordered.

    ESMPy expects Fortran-ordered array. Passing C-ordered array will slow down
    performance due to memory rearrangement.

    Parameters
    ----------
    a : numpy array
    '''
    if not a.flags['F_CONTIGUOUS']:
        warnings.warn("Input array is not F_CONTIGUOUS. "
                      "Will affect performance.")


def warn_lat_range(lat):
    '''
    Give a warning if latitude is outside of [-90, 90]

    Longitute, on the other hand, can be in any range,
    since the it the transform is done in (x, y, z) space.

    Parameters
    ----------
    lat : numpy array
    '''
    if (lat.max() > 90.0) or (lat.min() < -90.0):
        warnings.warn("Latitude is outside of [-90, 90]")


def esmf_grid(lon, lat, periodic=False, mask=None):
    '''
    Create an ESMF.Grid object, for constructing ESMF.Field and ESMF.Regrid.

    Parameters
    ----------
    lon, lat : 2D numpy array
         Longitute/Latitude of cell centers.

         Recommend Fortran-ordering to match ESMPy internal.

         Shape should be ``(Nlon, Nlat)`` for rectilinear grid,
         or ``(Nx, Ny)`` for general quadrilateral grid.

    periodic : bool, optional
        Periodic in longitude? Default to False.
        Only useful for source grid.

    mask : 2D numpy array, optional
        Grid mask. According to the ESMF convention, masked cells
        are set to 0 and unmasked cells to 1.

        Shape should be ``(Nlon, Nlat)`` for rectilinear grid,
        or ``(Nx, Ny)`` for general quadrilateral grid.

    Returns
    -------
    grid : ESMF.Grid object
    '''

    # ESMPy expects Fortran-ordered array.
    # Passing C-ordered array will slow down performance.
    for a in [lon, lat]:
        warn_f_contiguous(a)

    warn_lat_range(lat)

    # ESMF.Grid can actually take 3D array (lon, lat, radius),
    # but regridding only works for 2D array
    assert lon.ndim == 2, "Input grid must be 2D array"
    assert lon.shape == lat.shape, "lon and lat must have same shape"

    staggerloc = ESMF.StaggerLoc.CENTER  # actually just integer 0

    if periodic:
        num_peri_dims = 1
    else:
        num_peri_dims = None

    # ESMPy documentation claims that if staggerloc and coord_sys are None,
    # they will be set to default values (CENTER and SPH_DEG).
    # However, they actually need to be set explicitly,
    # otherwise grid._coord_sys and grid._staggerloc will still be None.
    grid = ESMF.Grid(np.array(lon.shape), staggerloc=staggerloc,
                     coord_sys=ESMF.CoordSys.SPH_DEG,
                     num_peri_dims=num_peri_dims)

    # The grid object points to the underlying Fortran arrays in ESMF.
    # To modify lat/lon coordinates, need to get pointers to them
    lon_pointer = grid.get_coords(coord_dim=0, staggerloc=staggerloc)
    lat_pointer = grid.get_coords(coord_dim=1, staggerloc=staggerloc)

    # Use [...] to avoid overwritting the object. Only change array values.
    lon_pointer[...] = lon
    lat_pointer[...] = lat

    # Follows SCRIP convention where 1 is unmasked and 0 is masked.
    # See https://github.com/NCPP/ocgis/blob/61d88c60e9070215f28c1317221c2e074f8fb145/src/ocgis/regrid/base.py#L391-L404
    if mask is not None:
        # remove fractional values
        mask = np.where(mask == 0, 0, 1)
        # convert array type to integer (ESMF compat)
        grid_mask = mask.astype(np.int32)
        if not (grid_mask.shape == lon.shape):
            raise ValueError(
                "mask must have the same shape as the latitude/longitude"
                "coordinates, got: mask.shape = %s, lon.shape = %s" %
                (mask.shape, lon.shape))
        grid.add_item(ESMF.GridItem.MASK, staggerloc=ESMF.StaggerLoc.CENTER,
                      from_file=False)
        grid.mask[0][:] = grid_mask

    return grid


def esmf_locstream(lon, lat):
    '''
    Create an ESMF.LocStream object, for contrusting ESMF.Field and ESMF.Regrid

    Parameters
    ----------
    lon, lat : 1D numpy array
         Longitute/Latitude of cell centers.

    Returns
    -------
    locstream : ESMF.LocStream object
    '''

    if len(lon.shape) > 1:
        raise ValueError("lon can only be 1d")
    if len(lat.shape) > 1:
        raise ValueError("lat can only be 1d")

    assert lon.shape == lat.shape

    location_count = len(lon)

    locstream = ESMF.LocStream(location_count,
                               coord_sys=ESMF.CoordSys.SPH_DEG)

    locstream["ESMF:Lon"] = lon.astype(np.dtype('f8'))
    locstream["ESMF:Lat"] = lat.astype(np.dtype('f8'))

    return locstream


def add_corner(grid, lon_b, lat_b):
    '''
    Add corner information to ESMF.Grid for conservative regridding.

    Not needed for other methods like bilinear or nearest neighbour.

    Parameters
    ----------
    grid : ESMF.Grid object
        Generated by ``esmf_grid()``. Will be modified in-place.

    lon_b, lat_b : 2D numpy array
        Longitute/Latitude of cell corner
        Recommend Fortran-ordering to match ESMPy internal.
        Shape should be ``(Nlon+1, Nlat+1)``, or ``(Nx+1, Ny+1)``
    '''

    # codes here are almost the same as esmf_grid(),
    # except for the "staggerloc" keyword
    staggerloc = ESMF.StaggerLoc.CORNER  # actually just integer 3

    for a in [lon_b, lat_b]:
        warn_f_contiguous(a)

    warn_lat_range(lat_b)

    assert lon_b.ndim == 2, "Input grid must be 2D array"
    assert lon_b.shape == lat_b.shape, "lon_b and lat_b must have same shape"
    assert np.array_equal(lon_b.shape, grid.max_index+1), (
           "lon_b should be size (Nx+1, Ny+1)")
    assert (grid.num_peri_dims == 0) and (grid.periodic_dim is None), (
           "Cannot add corner for periodic grid")

    grid.add_coords(staggerloc=staggerloc)

    lon_b_pointer = grid.get_coords(coord_dim=0, staggerloc=staggerloc)
    lat_b_pointer = grid.get_coords(coord_dim=1, staggerloc=staggerloc)

    lon_b_pointer[...] = lon_b
    lat_b_pointer[...] = lat_b


def esmf_mesh(polys):
    """
    Create an ESMF.Mesh object from a list of polygons.

    All exterior ring points are added to the mesh as nodes and each polygon
    is added as an element, with the polygon centroid as the element's coordinates.

    Parameters
    ----------
    polys : sequence of shapely Polygon
        There must be no overlap between any of the polygons, holes are ignored.

    Returns
    -------
    mesh : ESMF.Mesh
        A mesh where each polygon is represented as an Element.
    """
    node_coords = []
    element_types = []
    element_conn = []
    element_coords = []
    for poly in polys:
        ring = poly.exterior
        element_coords.append(poly.centroid.xy)
        element_types.append(len(ring.coords) - 1)
        for coord in (ring.coords[:-1] if ring.is_ccw else ring.coords[:0:-1]):
            if coord not in node_coords:
                node_coords.append(coord)
            element_conn.append(node_coords.index(coord))

    mesh = ESMF.Mesh(2, 2, coord_sys=ESMF.CoordSys.SPH_DEG)
    node_num = len(node_coords)
    mesh.add_nodes(node_num, np.arange(node_num) + 1, np.array(node_coords).ravel(), np.zeros(node_num))
    elem_num = len(element_types)
    mesh.add_elements(elem_num, np.arange(elem_num) + 1, np.array(element_types), np.array(element_conn), element_coords=np.array(element_coords).ravel() + 1)
    return mesh


def esmf_regrid_build(sourcegrid, destgrid, method,
                      filename=None, extra_dims=None, ignore_degenerate=None):
    '''
    Create an ESMF.Regrid object, containing regridding weights.

    Parameters
    ----------
    sourcegrid, destgrid : ESMF.Grid or ESMF.Mesh object
        Source and destination grids.

        Should create them by ``esmf_grid()``
        (with optionally ``add_corner()``),
        instead of ESMPy's original API.

    method : str
        Regridding method. Options are

        - 'bilinear'
        - 'conservative', **need grid corner information**
        - 'conservative_normed', **need grid corner information**
        - 'patch'
        - 'nearest_s2d'
        - 'nearest_d2s'

    filename : str, optional
        Offline weight file. **Require ESMPy 7.1.0.dev38 or newer.**
        With the weights available, we can use Scipy's sparse matrix
        multiplication to apply weights, which is faster and more Pythonic
        than ESMPy's online regridding. If None, weights are stored in
        memory only.

    extra_dims : a list of integers, optional
        Extra dimensions (e.g. time or levels) in the data field

        This does NOT affect offline weight file, only affects online regrid.

        Extra dimensions will be stacked to the fastest-changing dimensions,
        i.e. following Fortran-like instead of C-like conventions.
        For example, if extra_dims=[Nlev, Ntime], then the data field dimension
        will be [Nlon, Nlat, Nlev, Ntime]

    ignore_degenerate : bool, optional
        If False (default), raise error if grids contain degenerated cells
        (i.e. triangles or lines, instead of quadrilaterals)

    Returns
    -------
    grid : ESMF.Grid object

    '''

    # use shorter, clearer names for options in ESMF.RegridMethod
    method_dict = {'bilinear': ESMF.RegridMethod.BILINEAR,
                   'conservative': ESMF.RegridMethod.CONSERVE,
                   'conservative_normed': ESMF.RegridMethod.CONSERVE,
                   'patch': ESMF.RegridMethod.PATCH,
                   'nearest_s2d': ESMF.RegridMethod.NEAREST_STOD,
                   'nearest_d2s': ESMF.RegridMethod.NEAREST_DTOS
                   }
    try:
        esmf_regrid_method = method_dict[method]
    except:
        raise ValueError('method should be chosen from '
                         '{}'.format(list(method_dict.keys())))

    # conservative regridding needs cell corner information
    if method in ['conservative', 'conservative_normed']:
        if not sourcegrid.has_corners:
            raise ValueError('source grid has no corner information. '
                             'cannot use conservative regridding.')
        if not isinstance(destgrid, ESMF.Mesh) and not destgrid.has_corners:
            raise ValueError('destination grid has no corner information. '
                             'cannot use conservative regridding.')

    # ESMF.Regrid requires Field (Grid+data) as input, not just Grid.
    # Extra dimensions are specified when constructing the Field objects,
    # not when constructing the Regrid object later on.
    sourcefield = ESMF.Field(sourcegrid, ndbounds=extra_dims)
    if isinstance(destgrid, ESMF.Mesh):
        destfield = ESMF.Field(destgrid, meshloc=ESMF.MeshLoc.ELEMENT, ndbounds=extra_dims)
    else:
        destfield = ESMF.Field(destgrid, ndbounds=extra_dims)

    # ESMF bug? when using locstream objects, options src_mask_values
    # and dst_mask_values produce runtime errors
    allow_masked_values = True
    if isinstance(sourcefield.grid, ESMF.api.locstream.LocStream):
        allow_masked_values = False
    if isinstance(destfield.grid, ESMF.api.locstream.LocStream):
        allow_masked_values = False

    # ESMPy will throw an incomprehensive error if the weight file
    # already exists. Better to catch it here!
    if filename is not None:
        assert not os.path.exists(filename), (
            'Weight file already exists! Please remove it or use a new name.')

    # re-normalize conservative regridding results
    # https://github.com/JiaweiZhuang/xESMF/issues/17
    if method == 'conservative_normed':
        norm_type = ESMF.NormType.FRACAREA
    else:
        norm_type = ESMF.NormType.DSTAREA

    # Calculate regridding weights.
    # Must set unmapped_action to IGNORE, otherwise the function will fail,
    # if the destination grid is larger than the source grid.
    kwargs=dict(filename=filename,
                regrid_method=esmf_regrid_method,
                unmapped_action=ESMF.UnmappedAction.IGNORE,
                ignore_degenerate=ignore_degenerate,
                norm_type=norm_type,
                factors=filename is None)
    if allow_masked_values:
        kwargs.update(dict(src_mask_values=[0], dst_mask_values=[0]))

    regrid = ESMF.Regrid(sourcefield, destfield, **kwargs)

    return regrid


def esmf_regrid_apply(regrid, indata):
    '''
    Apply existing regridding weights to the data field,
    using ESMPy's built-in functionality.

    xESMF use Scipy to apply weights instead of this.
    This is only for benchmarking Scipy's result and performance.

    Parameters
    ----------
    regrid : ESMF.Regrid object
        Contains the mapping from the source grid to the destination grid.

        Users should create them by esmf_regrid_build(),
        instead of ESMPy's original API.

    indata : numpy array of shape ``(Nlon, Nlat, N1, N2, ...)``
        Extra dimensions ``(N1, N2, ...)`` are specified in
        ``esmf_regrid_build()``.

        Recommend Fortran-ordering to match ESMPy internal.

    Returns
    -------
    outdata : numpy array of shape ``(Nlon_out, Nlat_out, N1, N2, ...)``

    '''

    # Passing C-ordered input data will be terribly slow,
    # since indata is often quite large and re-ordering memory is expensive.
    warn_f_contiguous(indata)

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

    return destfield.data


def esmf_regrid_finalize(regrid):
    '''
    Free the underlying Fortran array to avoid memory leak.

    After calling ``destroy()`` on regrid or its fields, we cannot use the
    regrid method anymore, but the input and output data still exist.

    Parameters
    ----------
    regrid : ESMF.Regrid object

    '''

    regrid.destroy()
    regrid.srcfield.destroy()
    regrid.dstfield.destroy()
    regrid.srcfield.grid.destroy()
    regrid.dstfield.grid.destroy()

    # double check
    assert regrid.finalized
    assert regrid.srcfield.finalized
    assert regrid.dstfield.finalized
    assert regrid.srcfield.grid.finalized
    assert regrid.dstfield.grid.finalized
