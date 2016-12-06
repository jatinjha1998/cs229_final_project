"""The trading module

Contains code to download stock data, read in csv, run benchmarks
and trade stocks."""

from os.path import join, dirname, abspath

__all__ = ['MODULE_LOC', \
           'PROJECT_LOC', \
           'DATA_LOC']

MODULE_LOC = dirname(abspath(__file__))
PROJECT_LOC =  abspath(join(MODULE_LOC, '..', '..'))
DATA_LOC = join(PROJECT_LOC, 'data')

from .stock_history import *
from .portfolio import *
from .benchmarks import *
from .rl import *
from .nn import *

