Current limitations
===================

.. _irregular_meshes-label:

Irregular meshes
----------------

xESMF only supports quadrilateral grids and doesn't support weirder grids
like triangular or hexagonal meshes.

ESMPy is actually able to deal with general irregular meshes
(`example <http://www.earthsystemmodeling.org/esmf_releases/
last_built/esmpy_doc/html/examples.html#create-a-5-element-mesh>`_),
but designing an elegant front-end for that is very challenging.
Plain 2D arrays cannot describe irregular meshes.
There needs to be additional information for connectivitity, as suggested by
`UGRID Conventions <http://ugrid-conventions.github.io/ugrid-conventions/>`_.

xarray's data model, although powerful, can only describe quadrilateral grids
(including multi-tile quadrilateral grids like the cubed-sphere).
If there is an elegant data model in Python for irregular meshes,
interfacing that with ESMPy should not be super difficult.

Parallel regridding
-------------------

xESMF currently only runs in serial.
Parallel options are being investgated.
See `issue#3 <https://github.com/JiaweiZhuang/xESMF/issues/3>`_.

Vector regridding
-----------------

Like almost all regridding packages, xESMF assumes scalar fields.
The most common way to remap winds is to rotate/re-decompose the
wind components (U and V) to the new direction,
and then regrid each component individually using a scalar regridding function.

Exact conservation of vector properities (like divergence and vorticity)
is beyond the scope of almost all regridding packages.
Using bilinear algorithm on each component should lead to OK results in most cases.
