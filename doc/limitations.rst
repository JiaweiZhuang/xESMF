Current limitations
===================

.. _irregular_meshes-label:

Irregular meshes
----------------

Irregular meshes like hexagonal grids don't fit very well with xarray's data model.
ESMF/ESMPy is able to handle irregular meshes but designing an elegant frontend for that is very challenging.

Vector regridding
-----------------

Can rotate the vector first.

Parallel regridding
-------------------

In future plan. See GitHub issue.

https://github.com/JiaweiZhuang/xESMF/issues/3
