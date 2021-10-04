import cftime
import numpy as np
import pytest
import xarray as xr

import xesmf

mom6like = xr.Dataset(
    data_vars=dict(
        tos=(['time', 'yh', 'xh'], np.random.rand(2, 180, 360)),
    ),
    coords=dict(
        xq=xr.DataArray(
            np.arange(-300, 60 + 1),
            dims=['xq'],
            attrs={
                'long_name': 'q point nominal longitude',
                'units': 'degrees_east',
                'cartesian_axis': 'X',
            },
        ),
        yq=xr.DataArray(
            np.arange(-90, 90 + 1),
            dims=['yq'],
            attrs={
                'long_name': 'q point nominal latitude',
                'units': 'degrees_north',
                'cartesian_axis': 'Y',
            },
        ),
        xh=xr.DataArray(
            0.5 + np.arange(-300, 60),
            dims=['xh'],
            attrs={
                'long_name': 'h point nominal longitude',
                'units': 'degrees_east',
                'cartesian_axis': 'X',
            },
        ),
        yh=xr.DataArray(
            0.5 + np.arange(-90, 90),
            dims=['yh'],
            attrs={
                'long_name': 'h point nominal latitude',
                'units': 'degrees_north',
                'cartesian_axis': 'Y',
            },
        ),
        time=xr.DataArray(
            [
                cftime.DatetimeNoLeap(2007, 1, 16, 12, 0, 0, 0),
                cftime.DatetimeNoLeap(2007, 2, 15, 0, 0, 0, 0),
            ],
            dims=['time'],
        ),
        reference_time=cftime.DatetimeNoLeap(1901, 1, 1, 0, 0, 0, 0),
    ),
    attrs=dict(description='Synthetic MOM6 data'),
)


def test_mom6like_to_5x5():
    """regression test for MOM6 grid"""

    grid_5x5 = xr.Dataset()
    grid_5x5['lon'] = xr.DataArray(data=0.5 + np.arange(0, 360, 5), dims=('x'))
    grid_5x5['lat'] = xr.DataArray(data=0.5 - 90 + np.arange(0, 180, 5), dims=('y'))

    # multiple definition for lon/lat results in failure to determine
    # which coordinate set to use.
    with pytest.raises(ValueError):
        regrid_to_5x5 = xesmf.Regridder(mom6like, grid_5x5, 'bilinear', periodic=True)

    regrid_to_5x5 = xesmf.Regridder(
        mom6like.rename({'xh': 'lon', 'yh': 'lat'}), grid_5x5, 'bilinear', periodic=True
    )

    tos_regridded = regrid_to_5x5(mom6like['tos'])
    assert tos_regridded.shape == ((2, 36, 72))


@pytest.mark.testcases
def test_mom6_to_1x1():
    """check regridding of MOM6 to regular lon/lat"""

    dataurl = 'http://35.188.34.63:8080/thredds/dodsC/OM4p5/'
    mom6 = xr.open_dataset(
        f'{dataurl}/ocean_monthly_z.200301-200712.nc4',
        chunks={'time': 1, 'z_l': 1},
        drop_variables=['average_DT', 'average_T1', 'average_T2'],
        decode_times=False,
        engine='pydap',
    )

    grid_1x1 = xr.Dataset()
    grid_1x1['lon'] = xr.DataArray(data=0.5 + np.arange(360), dims=('x'))
    grid_1x1['lat'] = xr.DataArray(data=0.5 - 90 + np.arange(180), dims=('y'))

    regrid_to_1x1 = xesmf.Regridder(
        mom6.rename({'geolon': 'lon', 'geolat': 'lat'}), grid_1x1, 'bilinear', periodic=True
    )

    thetao_1x1 = regrid_to_1x1(mom6['thetao'])
    assert np.allclose(thetao_1x1.isel(time=0, z_l=0, x=200, y=100).values, 27.15691922)
