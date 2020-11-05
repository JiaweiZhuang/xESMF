import pytest

import xesmf as xe


def test_grid_global():
    ds = xe.util.grid_global(1.5, 1.5)
    refshape = (120, 240)
    refshape_b = (121, 241)

    assert ds['lon'].values.shape == refshape
    assert ds['lat'].values.shape == refshape
    assert ds['lon_b'].values.shape == refshape_b
    assert ds['lat_b'].values.shape == refshape_b


def test_grid_global_bad_resolution():
    with pytest.warns(UserWarning):
        xe.util.grid_global(1.5, 1.23)

    with pytest.warns(UserWarning):
        xe.util.grid_global(1.23, 1.5)
