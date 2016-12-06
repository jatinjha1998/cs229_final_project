''' Stock History sub-module

Reads in stock history data. Assumes data only has two columns:
    Date, Close
'''

__all__ = ['download_stock_histories', \
           'read_stock_history', \
           'load_stock_pairs', \
           'STOCK_DATA_DIR', \
           'get_lo_beta_stock_symbols', \
           'get_hi_beta_stock_symbols', \
           'get_stock_betas', \
           'LO_BETA_DIR', \
           'HI_BETA_DIR', \
           'StockPair', \
           'get_stock_pairs']

import glob
import itertools
from os.path import join, splitext, basename

import numpy as np
import pandas as pd
import pandas_datareader.data as datareader
from sklearn.model_selection import train_test_split

from . import DATA_LOC

STOCK_DATA_DIR = join(DATA_LOC, 'stocks')
LO_BETA_DIR = join(STOCK_DATA_DIR, 'low_beta')
HI_BETA_DIR = join(STOCK_DATA_DIR, 'high_beta')

DEFAULT_START_DATE = np.datetime64('2015-01-01')
DEFAULT_END_DATE = np.datetime64('2016-01-01')

#holds a low-beta and high-beta stock
class StockPair:
    def __init__(self, lo, hi, hist_lo, hist_hi):
        # stock ticker symbols (strings)
        self.lo = lo
        self.hi = hi
        # stock history (pandas data arrays)
        self.hist_lo = hist_lo
        self.hist_hi = hist_hi

def get_stock_pairs(train_size=0.8):
    '''Reads in csv's of stock data and returns arrays of StockPair objects.

    Returns all possible combinations of low- and high-beta stocks, with
    `training_size` specifying the split into the training and test vectors,
    respectively.
    For more info, see `sklearn.model_selection.train_test_split`'''

    lo_beta_files = get_lo_beta_stock_symbols()
    hi_beta_files = get_hi_beta_stock_symbols()
    stock_pairs = []

    for (L, H) in itertools.product(lo_beta_files, hi_beta_files):
        # the actual histories
        hist_lo = read_stock_history(join(LO_BETA_DIR, L + '.csv'))
        hist_hi = read_stock_history(join(HI_BETA_DIR, H + '.csv'))

        stock_pairs.append(StockPair(L, H, hist_lo, hist_hi))

    return train_test_split(stock_pairs, train_size=train_size)

def get_lo_beta_stock_symbols():
    '''List of ticker symbols for low-beta stocks'''
    return [line.strip() for line in
            open(join(LO_BETA_DIR, 'stock_symbols.txt'))]

def get_hi_beta_stock_symbols():
    '''List of ticker symbols for high-beta stocks'''
    return [line.strip() for line in
            open(join(HI_BETA_DIR, 'stock_symbols.txt'))]

def get_stock_betas():
    betas = dict()
    for line in open(join(STOCK_DATA_DIR, 'betas.csv')):
        sym, beta = line.split(',')
        betas[sym] = float(beta)

    return betas

def download_stock_histories(path, stock,
        start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE,
        source='google'):
    ''' Uses ``pandas_datareader`` to query the source for closing stock prices.

    Drops all columns except for closing prices with the date as the index.
    Saves the resulting pandas.Series(es) ``<stock>.csv``.
    Will remove any nans from data, so if stock has no closing prices
    for a period in the time frame, those indices will not show up.
    Notice: it saves the *Series*, so there will be no header information

    Args:
    path : string
        path to save output to.
    stock : string or list of strings
        stock(s) to download.
    start_date : numpy.datetime64 = datetime64('2015-01-01')
    end_date : numpy.datetime64 = datetime64('2016-01-01')
        start and end times to query over
    source : string or None = 'google'
        'google' or 'yahoo'
    '''

    if end_date <= start_date:
        raise ValueError('Start date is not before end date')

    stock_data = datareader.DataReader(stock,
            source, start_date, end_date)
    if isinstance(stock_data, pd.Panel):
        stock_data = stock_data.transpose(2, 1, 0)
        for i in stock_data:
            d = stock_data[i]['Close'].dropna()
            d.to_csv(join(path, '{:s}.csv'.format(i)))
    elif isinstance(stock_data, pd.DataFrame):
        stock_data['Close'].dropna().to_csv(join(path, '{:s}.csv'.format(stock)))

read_stock_history = pd.Series.from_csv

def load_stock_pairs(path: str):
    '''Loads trading.StockPairs of the stock pairs (comma seperated) in path'''
    # load the test set stock pairs from the saved file
    stock_pairs = []

    with open(path, 'r') as f:
        for line in f:
            (L, H) = line.strip().split(',')
            hist_lo = read_stock_history(join(LO_BETA_DIR,
                        L.strip() + '.csv'))
            hist_hi = read_stock_history(join(HI_BETA_DIR,
                        H.strip() + '.csv'))

            stock_pairs.append(StockPair(L, H, hist_lo, hist_hi))

    return stock_pairs

#def read_stock_history(f, **kwargs):
#    '''Reads the csv ``f`` using ``pandas.Series.from_csv`` and passes on ``**kwargs``'''
#    return pandas.Series.from_csv(path=f, **kwargs).astype(numpy.float64)

