""" Stock History sub-module

Reads in stock history data. Assumes data only has two columns
    Date, Close
"""

__all__ = ['download_stock_histories', \
           'read_stock_history']


from os.path import join
import numpy
import pandas
import pandas_datareader.data

default_start_date = numpy.datetime64('2015-01-01'),
default_end_date = numpy.datetime64('2016-01-01'),

def download_stock_histories(path, stock, 
        start_date=default_start_date, end_date=default_end_date,
        source='google'):
    """ Uses ``pandas_datareader`` to query the source for closing stock prices.

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
    """

    if end_date <= start_date:
        raise ValueError('Start date is not before end date')

    stock_data = pandas_datareader.data.DataReader(stock, 
            source, start_date, end_date)
    if isinstance(stock_data, pandas.Panel):
        stock_data = stock_data.transpose(2, 1, 0)
        for i in stock_data:
            d = stock_data[i]['Close'].dropna()
            d.to_csv(join(path, '{:s}.csv'.format(i)))
    elif isinstance(stock_data, pandas.DataFrame):
        stock_data['Close'].dropna().to_csv(join(path, '{:s}.csv'.format(stock)))

read_stock_history = pandas.Series.from_csv

#def read_stock_history(f, **kwargs):
#    """Reads the csv ``f`` using ``pandas.Series.from_csv`` and passes on ``**kwargs``"""
#    return pandas.Series.from_csv(path=f, **kwargs).astype(numpy.float64)

