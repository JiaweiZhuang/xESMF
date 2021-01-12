import os
import warnings

import cf_xarray  # noqa
import dask
import numpy as np
import pytest
import xarray as xr
import pydap
import xesmf


def test_MOM6_to_1x1():
    """ check regridding of MOM6 to regular lon/lat """

    dataurl = 'http://35.188.34.63:8080/thredds/dodsC/OM4p5/'
    MOM6 = xr.open_dataset(f'{dataurl}/ocean_monthly_z.200301-200712.nc4',
                           chunks={'time':1, 'z_l': 1},
                           drop_variables=['average_DT', 'average_T1', 'average_T2'],
                           engine='pydap')

    grid_1x1 = xr.Dataset()
    grid_1x1['lon'] = xr.DataArray(data=0.5 + np.arange(360), dims=('x'))
    grid_1x1['lat'] = xr.DataArray(data=0.5 -90 + np.arange(180), dims=('y'))

    regrid_to_1x1 = xesmf.Regridder(MOM6.rename({'geolon': 'lon',
                                                 'geolat': 'lat'}),
                                                  grid_1x1, 'bilinear', periodic=True)

    thetao_1x1 = regrid_to_1x1(MOM6['thetao'])
    assert np.allclose(thetao_1x1.isel(time=0, z_l=0, x=200, y=100).values, 27.15691922)
