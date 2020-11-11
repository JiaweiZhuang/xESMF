import os

from setuptools import find_packages, setup

DISTNAME = 'xesmf'
DESCRIPTION = 'Universal Regridder for Geospatial Data'
AUTHOR = 'Jiawei Zhuang'
AUTHOR_EMAIL = 'jiaweizhuang@g.harvard.edu'
URL = 'https://github.com/pangeo-data/xESMF'
LICENSE = 'MIT'
PYTHON_REQUIRES = '>=3.6'
USE_SCM_VERSION = {'version_scheme': 'post-release', 'local_scheme': 'dirty-tag'}
SETUP_REQUIRES = ['setuptools_scm', 'setuptools>=30.3.0']

# https://github.com/rtfd/readthedocs.org/issues/5512#issuecomment-475024373
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['esmpy>=8.0.0', 'xarray', 'numpy', 'scipy']

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Topic :: Scientific/Engineering',
]


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name=DISTNAME,
    license=LICENSE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    long_description=readme(),
    long_description_content_type='text/x-rst',
    python_requires=PYTHON_REQUIRES,
    install_requires=INSTALL_REQUIRES,
    url=URL,
    packages=find_packages(),
    use_scm_version=USE_SCM_VERSION,
    setup_requires=SETUP_REQUIRES,
)
