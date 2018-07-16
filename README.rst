xESMF: Universal Regridder for Geospatial Data
==============================================

|pypi| |Build Status| |codecov| |docs| |license| |DOI|

xESMF is a Python package for
`regridding <https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/regridding-overview>`_.
It is

- **Powerful**: It uses ESMF_/ESMPy_ as backend and can regrid between **general curvilinear grids**
  with all `ESMF regridding algorithms <https://www.earthsystemcog.org/projects/esmf/regridding>`_,
  such as **bilinear**, **conservative** and **nearest neighbour**.
- **Easy-to-use**: It abstracts away ESMF's complicated infrastructure
  and provides a simple, high-level API, compatible with xarray_ as well as basic numpy arrays.
- **Fast**: It is faster than ESMPy's original Fortran regridding engine in serial case,
  and parallel capability will be added in the next version.

Please see `online documentation <http://xesmf.readthedocs.io/en/latest/>`_.


.. _ESMF: https://www.earthsystemcog.org/projects/esmf/
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _xarray: http://xarray.pydata.org

.. |pypi| image:: https://badge.fury.io/py/xesmf.svg
   :target: https://badge.fury.io/py/xesmf
   :alt: pypi package

.. |Build Status| image:: https://api.travis-ci.org/JiaweiZhuang/xESMF.svg
   :target: https://travis-ci.org/JiaweiZhuang/xESMF
   :alt: travis-ci build status

.. |codecov| image:: https://codecov.io/gh/JiaweiZhuang/xESMF/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/JiaweiZhuang/xESMF
   :alt: code coverage

.. |docs| image:: https://readthedocs.org/projects/xesmf/badge/?version=latest
   :target: http://xesmf.readthedocs.io/en/latest/?badge=latest
   :alt: documentation status

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/JiaweiZhuang/xESMF/blob/master/LICENSE
   :alt: license

.. |DOI| image:: https://zenodo.org/badge/101709596.svg
   :target: https://zenodo.org/badge/latestdoi/101709596
   :alt: DOI
