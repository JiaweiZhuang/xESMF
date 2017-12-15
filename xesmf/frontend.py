'''
Frontend for xESMF, exposed to users.
'''

import numpy as np
import xarray as xr
import os

from . backend import esmf_regrid_build, esmf_regrid_finalize
from . util import ds_to_ESMFgrid
from . smm import read_weights, apply_weights


class Regridder(object):
    def __init__(self, ds_in, ds_out, method,
                 filename=None, clobber=False, reuse_weights=False):

        self.method = method

        self.Nlat_in, self.Nlon_in = ds_in['lon'].shape
        self.Nlat_out, self.Nlon_out = ds_out['lon'].shape
        self.N_in = ds_in['lon'].size
        self.N_out = ds_out['lon'].size

        self.clobber = clobber
        self.reuse_weights = reuse_weights

        if filename is None:
            # e.g. bilinear_600x400_400x300.nc
            filename = ('{0}_{1}x{2}_{3}x{4}.nc'.format(method,
                        self.Nlat_in, self.Nlon_in,
                        self.Nlat_out, self.Nlon_out)
                        )
        self.filename = filename

        self.write_weights(ds_in, ds_out)
        self.A = read_weights(self.filename, self.N_in, self.N_out)

    def __str__(self):
        return ('xESMF Regridder \n'
                'Regridding algorithm:       {} \n'
                '(Nlat_in, Nlon_in):         {} \n'
                '(Nlat_out, Nlon_out):       {} \n'
                'Weight filename:            {} \n'
                .format(self.method,
                        (self.Nlat_in, self.Nlon_in),
                        (self.Nlat_out, self.Nlon_out),
                        self.filename)
                )

    def __repr__(self):
        return self.__str__()

    def __call__(self, dr_in):
        return self.apply_weights(dr_in)

    def write_weights(self, ds_in, ds_out):

        if os.path.exists(self.filename):
            if self.clobber:
                print('overwrite existing file: {}'.format(self.filename))
                os.remove(self.filename)
            elif self.reuse_weights:
                print('reuse existing file: {}'.format(self.filename))
                return
            else:
                raise ValueError('Weight file {} already exists! Please:\n'
                                 '(1) set clobber=True to overwrite it,\n'
                                 'or (2) set reuse_weights=True to reuse it,\n'
                                 'or (3) set filename="your_custom_name.nc"\n'
                                 .format(self.filename)
                                 )

        grid_in = ds_to_ESMFgrid(ds_in)
        grid_out = ds_to_ESMFgrid(ds_out)
        regrid = esmf_regrid_build(grid_in, grid_out, self.method,
                                   filename=self.filename)

        # we only need the weight file, not the regrid object
        esmf_regrid_finalize(regrid)

    def apply_weights(self, dr_in):
        indata = dr_in.values
        outdata = apply_weights(self.A, indata, self.Nlon_out, self.Nlat_out)

        # TODO: append metadata

        return outdata
