# flake8: noqa

from . import data, util
from .frontend import Regridder, SpatialAverager

try:
    from ._version import __version__
except ImportError:
    __version__ = 'unknown'
