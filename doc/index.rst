xESMF: Universal Regridder for Geospatial Data
==============================================

xESMF is a Python package for
`regridding <https://climatedataguide.ucar.edu/climate-data-tools-and-analysis/regridding-overview>`_.
It is

- **Powerful**: It uses ESMF_/ESMPy_ as backend and can regrid between **general curvilinear grids**
  with all `ESMF regridding algorithms <https://www.earthsystemcog.org/projects/esmf/regridding>`_,
  such as **bilinear**, **conservative** and **nearest neighbour**.
- **Easy-to-use**: It abstracts away ESMF's complicated infrastructure
  and provides a simple, high-level API, compatible with xarray_ as well as basic numpy arrays.
- **Fast**: It is :doc:`faster than <./notebooks/Backend>` ESMPy's original Fortran regridding engine in serial case, and also supports dask_ for `out-of-core, parallel computation <http://xarray.pydata.org/en/stable/dask.html>`_.


.. _ESMF: https://www.earthsystemcog.org/projects/esmf/
.. _ESMPy: https://www.earthsystemcog.org/projects/esmpy/
.. _xarray: http://xarray.pydata.org
.. _dask: https://dask.org/


Contents
--------

.. toctree::
   :maxdepth: 1
   :caption: Overview

   why
   other_tools
   limitations

.. toctree::
   :maxdepth: 1
   :caption: Beginner tutorials

   installation
   notebooks/Rectilinear_grid
   notebooks/Curvilinear_grid
   notebooks/Pure_numpy
   notebooks/Dataset
   notebooks/Spatial_Averaging

.. toctree::
   :maxdepth: 1
   :caption: Intermediate tutorials

   notebooks/Dask
   notebooks/Compare_algorithms
   notebooks/Reuse_regridder
   notebooks/Using_LocStream
   notebooks/Masking
   large_problems_on_HPC

.. toctree::
   :maxdepth: 1
   :caption: Technical notes

   changes
   notebooks/Backend

.. toctree::
   :maxdepth: 1
   :caption: API

   user_api
   internal_api


How to ask for help
-------------------

The `GitHub issue tracker <https://github.com/JiaweiZhuang/xESMF/issues>`_ is the primary place for bug reports. If you hit any issues, I recommend the following steps:

- First, `search for existing issues <https://help.github.com/en/articles/searching-issues-and-pull-requests>`_. Other people are likely to hit the same problem and probably have already found the solution.

- For a new bug, please `craft a minimal bug report <https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_ with reproducible code. Use synthetic data or `upload <https://help.github.com/en/articles/file-attachments-on-issues-and-pull-requests>`_ a small sample of input data (~1 MB) so I can quickly reproducible your error.

- For platform-dependent problems (such as kernel dying and installation error), please also show how to reproduce your system environment, otherwise I have no way to diagnose the issue. The best approach is probably finding an `official Docker image <https://docs.docker.com/docker-hub/official_images/>`_ that is closest to your OS (such as `Ubuntu <https://hub.docker.com/_/ubuntu/>`_ or `CentOS <https://hub.docker.com/_/centos/>`_), and build your Python environment starting with such image, to see whether the error still exists. Alternatively you can select from public cloud images, such as `Amazon Machine Images <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/AMIs.html>`_ or `Google Cloud Images <https://cloud.google.com/compute/docs/images>`_. If the error only happens on your institution's HPC cluster, please contact the system administrator for help.

For general "how-to" questions that are not bugs, you can also post on `StackOverflow <https://stackoverflow.com/>`_ (ref: `xarray questions <https://stackoverflow.com/questions/tagged/python-xarray>`_) and send me the link. For small questions also feel free to @ me `on Twitter <https://twitter.com/Jiawei_Zhuang_>`_.


**The "Don'ts"**:

- Do not describe your problem in a private email, as this would require me to reply similar emails many times. `Keep all discussions in public places <https://matthewrocklin.com/blog/2019/02/28/slack-github>`_ like GitHub or StackOverflow.
- Do not only show the error/problem without providing the steps to reproduce it.
- Do not take screenshots of your code, as they are not copy-pastable.


How to support xESMF
--------------------

xESMF is so far my personal unfunded project; most development happens during my (very limited) free time at graduate school. Your support in any form will be appreciated.

The easy ways (takes several seconds):

- `Give a star <https://help.github.com/en/articles/saving-repositories-with-stars>`_ to its `GitHub repository <https://github.com/JiaweiZhuang/xESMF>`_.
- Share it via social media like Twitter; introduce it to your friends/advisors/students.

More advanced ways:

- Cite xESMF in your scientific publications. Currently the best way is to cite the DOI: https://doi.org/10.5281/zenodo.1134365.
- If you'd like to contribute code, see this `preliminary contributor guide <https://github.com/JiaweiZhuang/xESMF/issues/28>`_. Also see `Contributing to xarray <http://xarray.pydata.org/en/stable/contributing.html>`_ for more backgrounds.
