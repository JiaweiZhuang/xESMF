.. _other_tools-label:

Other geospatial regridding tools
=================================

Here is a brief overview of other regridding tools that the author is aware of
(for geospatial data on the sphere, excluding traditional image resizing functions).
They are all great tools and have helped the author a lot in both scientific research
and xESMF development. Check them out if xESMF cannot suit your needs.

- `ESMF <https://www.earthsystemcog.org/projects/esmf/>`_ (*Fortran package*).
  Although its name "Earth System Modeling Framework" doesn't indicate a regridding
  functionality, it actually contains the most powerful regridding engine in the world.
  It is widely used in Earth System Models (ESMs) as both the software infrastructure
  and the regridder for transforming data between the atmospheric, ocean, and land components.
  It can deal with general irregular meshes, in either 2D or 3D.

  ESMF is a huge beast, containing
  `one million lines of source code <https://www.earthsystemcog.org/projects/esmf/sloc_annual>`_.
  Even just compiling it requires some effort.
  It is more for building ESMs than for data analysis.

- `ESMPy <https://www.earthsystemcog.org/projects/esmpy/>`_ (*Python interface to ESMF*).
  ESMPy provides a much simpler way to use ESMF's regridding functionality.
  The greatest thing is, it is pre-compiled as a
  `conda package <https://anaconda.org/NESII/esmpy>`_,
  so you can install it with one-click and don't have to go through
  the daunting compiling process on your own.
  However, ESMPy is a complicated Python API that controls a huge beast hidden behind.
  It is not as intuitive as native Python packages,
  and a simple regridding task would require more than 10 lines of arcane code.

  If you want to involve in xESMF development you need to know ESMPy.
  Check out this nice
  `tutorial <https://github.com/nawendt/esmpy-tutorial>`_ before going to the
  `official doc <http://www.earthsystemmodeling.org/esmf_releases/last_built/esmpy_doc/html/index.html>`_.

- `TempestRemap <https://github.com/ClimateGlobalChange/tempestremap>`_
  (*C++ package*). A pretty mordern and powerful package,
  supporting arbitrary-order conservative remapping.
  It also can generate cubed-sphere grids on the fly
  and can be modified to support many cubed-sphere grid variations.
  (`example <https://github.com/JiaweiZhuang/Tempest_for_GCHP>`_, only if you can read C++)

- `SCRIP <http://oceans11.lanl.gov/trac/SCRIP>`_ (*Fortran package*).
  A very old pacakge, once popular but **no longer maintained** (long live SCRIP).
  You should not use it now, but should know that it exists.
  Newer regridding packages often follow its standards --
  you will see "SCRIP format" here and there, for example in ESMF or TempestRemap.

- `Regridder in NCL <https://www.ncl.ucar.edu/Applications/regrid.shtml>`_
  (*NCAR Command Language*).
  Has bilinear and conservative algorithms for rectilinear grids,
  and also supports some specialized curvilinear grids.
  There is also a `ESMF wrapper <https://www.ncl.ucar.edu/Applications/ESMF.shtml>`_
  that works for more grid types.

Regridders in other tools (often wraps ESMF or SCRIP):

- `Regridder in NCO <http://nco.sourceforge.net/nco.html#Regridding>`_
  (*command line tool*)
- `Regridder in Iris <http://scitools.org.uk/iris/docs/v1.10.0/userguide/interpolation_and_regridding.html>`_
  (*Python package*)
- `Regridder in UV-CDAT <https://uvcdat.llnl.gov/documentation/cdms/cdms_4.html>`_
  (*Python package*)
