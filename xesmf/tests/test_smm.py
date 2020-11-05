import numpy as np
import scipy.sparse as sps

import xesmf as xe


def test_add_nans_to_weights():
    """ testing adding Nans to empty rows in sparse matrix """
    # create input sparse matrix with one empty row (j=2)
    row = np.array([0, 3, 1, 0])
    col = np.array([0, 3, 1, 2])
    data = np.array([4.0, 5.0, 7.0, 9.0])
    Matin = sps.coo_matrix((data, (row, col)), shape=(4, 4))

    # this is what is expected to come out (Nan added at i=0, j=2)
    row = np.array([0, 3, 1, 0, 2])
    col = np.array([0, 3, 1, 2, 0])
    data = np.array([4.0, 5.0, 7.0, 9.0, np.nan])
    expected = sps.coo_matrix((data, (row, col)), shape=(4, 4))

    Matout = xe.smm.add_nans_to_weights(Matin)
    assert np.allclose(expected.toarray(), Matout.toarray(), equal_nan=True)

    # Matrix without empty rows should return the same
    row = np.array([0, 3, 1, 0, 2])
    col = np.array([0, 3, 1, 2, 1])
    data = np.array([4.0, 5.0, 7.0, 9.0, 10.0])
    Matin = sps.coo_matrix((data, (row, col)), shape=(4, 4))

    Matout = xe.smm.add_nans_to_weights(Matin)
    assert np.allclose(Matin.toarray(), Matout.toarray())
