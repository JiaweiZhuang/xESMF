import numpy as np
import sparse as sps
import xarray as xr

import xesmf as xe


def test_add_nans_to_weights():
    """testing adding Nans to empty rows in sparse matrix"""
    # create input sparse matrix with one empty row (j=2)
    coords = np.array([[0, 3, 1, 0], [0, 3, 1, 2]])
    data = np.array([4.0, 5.0, 7.0, 9.0])
    Matin = sps.COO(coords, data, shape=(4, 4))

    # this is what is expected to come out (Nan added at i=0, j=2)
    coords = np.array([[0, 3, 1, 0, 2], [0, 3, 1, 2, 0]])
    data = np.array([4.0, 5.0, 7.0, 9.0, np.nan])
    expected = sps.COO(coords, data, shape=(4, 4))

    Matout = xe.smm.add_nans_to_weights(xr.DataArray(Matin, dims=('in', 'out')))
    assert np.allclose(expected.todense(), Matout.data.todense(), equal_nan=True)

    # Matrix without empty rows should return the same
    coords = np.array([[0, 3, 1, 0, 2], [0, 3, 1, 2, 1]])
    data = np.array([4.0, 5.0, 7.0, 9.0, 10.0])
    Matin = sps.COO(coords, data, shape=(4, 4))

    Matout = xe.smm.add_nans_to_weights(xr.DataArray(Matin, dims=('in', 'out')))
    assert np.allclose(Matin.todense(), Matout.data.todense())
