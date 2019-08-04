from setuptools import setup, find_packages
import os

VERSION = '0.1.2'
DISTNAME = 'xesmf'
DESCRIPTION = "Universal Regridder for Geospatial Data"
AUTHOR = 'Jiawei Zhuang'
AUTHOR_EMAIL = 'jiaweizhuang@g.harvard.edu'
URL = 'https://github.com/JiaweiZhuang/xESMF'
LICENSE = 'MIT'
PYTHON_REQUIRES = '>=3.5'

# https://github.com/rtfd/readthedocs.org/issues/5512#issuecomment-475024373
on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    INSTALL_REQUIRES = []
else:
    INSTALL_REQUIRES = ['esmpy', 'xarray', 'numpy', 'scipy']

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Topic :: Scientific/Engineering',
]


def readme():
    with open('README.rst') as f:
        return f.read()


setup(name=DISTNAME,
      version=VERSION,
      license=LICENSE,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      classifiers=CLASSIFIERS,
      description=DESCRIPTION,
      long_description=readme(),
      python_requires=PYTHON_REQUIRES,
      install_requires=INSTALL_REQUIRES,
      url=URL,
      packages=find_packages())
