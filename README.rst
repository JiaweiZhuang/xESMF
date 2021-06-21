xESMF: Universal Regridder for Geospatial Data
==============================================

|Binder| |conda| |Build Status| |codecov| |docs| |license| |DOI|

xESMF is a Python package for
`regridding <https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/regridding-overview>`_.
It is

- **Powerful**: It uses ESMF_/ESMPy_ as backend and can regrid between **general curvilinear grids**
  with all `ESMF regridding algorithms <https://www.earthsystemcog.org/projects/esmf/regridding>`_,
  such as **bilinear**, **conservative** and **nearest neighbour**.
- **Easy-to-use**: It abstracts away ESMF's complicated infrastructure
  and provides a simple, high-level API, compatible with xarray_ as well as basic numpy arrays.
- **Fast**: It is faster than ESMPy's original Fortran regridding engine in serial case, and also supports dask_ for `out-of-core, parallel computation <http://xarray.pydata.org/en/stable/dask.html>`_.

Please see `online documentation <http://pangeo-xesmf.readthedocs.io/en/latest/>`_, or `play with example notebooks on Binder <https://mybinder.org/v2/gh/pangeo-data/xESMF/master?filepath=doc%2Fnotebooks>`_.

For new users, I also recommend reading `How to ask for help <https://pangeo-xesmf.readthedocs.io/en/latest/#how-to-ask-for-help>`_ and `How to support xESMF <https://pangeo-xesmf.readthedocs.io/en/latest/#how-to-support-xesmf>`_.

.. _ESMF: https://www.earthsystemcog.org/projects/esmf/
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _xarray: http://xarray.pydata.org
.. _dask: https://dask.org/

.. |conda| image:: https://img.shields.io/conda/dn/conda-forge/xesmf.svg
   :target: https://anaconda.org/conda-forge/xesmf

.. |Build Status| image:: https://img.shields.io/github/workflow/status/pangeo-data/xESMF/CI?logo=github
   :target: https://github.com/pangeo-data/xESMF/actions
   :alt: github-ci build status

.. |codecov| image:: https://codecov.io/gh/pangeo-data/xESMF/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/pangeo-data/xESMF
   :alt: code coverage

.. |docs| image:: https://readthedocs.org/projects/pangeo-xesmf/badge/?version=latest
   :target: http://pangeo-xesmf.readthedocs.io/en/latest/?badge=latest
   :alt: documentation status

.. |license| image:: https://img.shields.io/badge/License-MIT-blue.svg
   :target: https://github.com/pangeo-data/xESMF/blob/master/LICENSE
   :alt: license

.. |DOI| image:: https://zenodo.org/badge/281126933.svg
   :target: https://zenodo.org/badge/latestdoi/281126933
   :alt: DOI

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/pangeo-data/xESMF/master?filepath=doc%2Fnotebooks
   :alt: binder
