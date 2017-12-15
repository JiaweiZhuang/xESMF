import os
import numpy as np
import ESMF
import xesmf as xe
from xesmf.backend import (warn_f_contiguous, esmf_grid, add_corner,
                           esmf_regrid_build, esmf_regrid_finalize)
from xesmf.smm import read_weights, apply_weights

from numpy.testing import assert_equal
import pytest

# We use pure numpy arrays to test backend
coord_names = ['lon', 'lat', 'lon_b', 'lat_b']

ds_in = xe.util.grid_global(2, 2)
lon_in, lat_in, lon_b_in, lat_b_in = [ds_in[name].values
                                      for name in coord_names]

ds_out = xe.util.grid_global(5, 4)
lon_out, lat_out, lon_b_out, lat_b_out = [ds_out[name].values
                                          for name in coord_names]

# shortcut to test a single grid
lon, lat, lon_b, lat_b = [lon_in, lat_in, lon_b_in, lat_b_in]


def test_flag():
    # some shortcuts for ESMF flags
    assert ESMF.StaggerLoc.CENTER == 0
    assert ESMF.StaggerLoc.CORNER == 3

    assert ESMF.CoordSys.CART == 0
    assert ESMF.CoordSys.SPH_DEG == 1  # only use this!
    assert ESMF.CoordSys.SPH_RAD == 2

    assert ESMF.UnmappedAction.ERROR == 0
    assert ESMF.UnmappedAction.IGNORE == 1  # only use this!

    assert ESMF.RegridMethod.BILINEAR == 0
    assert ESMF.RegridMethod.PATCH == 1
    assert ESMF.RegridMethod.CONSERVE == 2
    assert ESMF.RegridMethod.NEAREST_STOD == 3
    assert ESMF.RegridMethod.NEAREST_DTOS == 4


def test_warn_f_on_array():
    a = np.zeros([2, 2], order='C')
    with pytest.warns(UserWarning):
        warn_f_contiguous(a)


def test_warn_f_on_grid():
    # should throw a warning if not passing transpose
    with pytest.warns(UserWarning):
        esmf_grid(lon, lat)


def test_esmf_grid_with_corner():

    # only center coordinate, no corners
    # remember to pass transpose (F-ordered) to backend
    grid = esmf_grid(lon.T, lat.T)

    # make sure coordinate values agree
    assert_equal(grid.coords[0][0], lon.T)
    assert_equal(grid.coords[0][1], lat.T)

    # make sure meta data agree
    assert not grid.has_corners  # no corner yet!
    assert grid.staggerloc == [True, False, False, False]
    assert grid.coord_sys == 1
    assert grid.rank == 2
    assert_equal(grid.size[0], lon.T.shape)
    assert_equal(grid.upper_bounds[0], lon.T.shape)
    assert_equal(grid.lower_bounds[0], np.array([0, 0]))

    # now add corner information
    add_corner(grid, lon_b.T, lat_b.T)

    # coordinate values
    assert_equal(grid.coords[3][0], lon_b.T)
    assert_equal(grid.coords[3][1], lat_b.T)

    # metadata
    assert grid.has_corners  # should have corner now
    assert grid.staggerloc == [True, False, False, True]
    assert_equal(grid.size[3], lon_b.T.shape)
    assert_equal(grid.upper_bounds[3], lon_b.T.shape)
    assert_equal(grid.lower_bounds[3], np.array([0, 0]))


def test_esmf_regrid_build():
    grid_in = esmf_grid(lon_in.T, lat_in.T)
    grid_out = esmf_grid(lon_out.T, lat_out.T)

    # first test bilinear regridding
    regrid = esmf_regrid_build(grid_in, grid_out, 'bilinear')
    assert regrid.regrid_method == 0

    # they should share the same memory
    regrid.srcfield.grid is grid_in
    regrid.dstfield.grid is grid_out

    # then test conservative regridding
    # no corner info yet, should not be able to use conservative
    with pytest.raises(ValueError):
        esmf_regrid_build(grid_in, grid_out, 'conservative')

    add_corner(grid_in, lon_b_in.T, lat_b_in.T)
    add_corner(grid_out, lon_b_out.T, lat_b_out.T)
    regrid = esmf_regrid_build(grid_in, grid_out, 'conservative')

    esmf_regrid_finalize(regrid)


def test_esmf_regrid_apply():
    pass


def test_write_weight_file():
    pass


def test_read_weights():
    pass


def test_apply_weights():
    pass
