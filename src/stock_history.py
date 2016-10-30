""" Stock History Module

Reads in stock history data. Assumes data only has two columns
    Date, Close
"""

from os.path import join
import numpy
import pandas
import pandas_datareader.data

default_start_date = numpy.datetime64('2015-01-01'),
default_end_date = numpy.datetime64('2016-01-01'),

def download_stock_histories(path, stock, 
        start_date=default_start_date, end_date=default_end_date,
        source='google'):
    """ Uses pandas datareader to query the source for stock data.

    Drops all columns except for Close. Date should be the index.
    Saves to 'stock.csv'. 

    Args:
    path (string):
        path to save output to.
    stock (string or list of strings):
        stock(s) to download.
    start_date (numpy.datetime64):
    end_date (numpy.datetime64):
        start and end times to query over
    source (string or None):
        'google' or 'yahoo'
    """

    if end_date <= start_date:
        raise ValueError('Start date is not before end date')

    stock_data = pandas_datareader.data.DataReader(stock, 
            source, start_date, end_date)
    if isinstance(stock_data, pandas.Panel):
        stock_data = stock_data.transpose(2, 1, 0)
        for i in stock_data:
            d = stock_data[i]['Close']
            d.to_csv(join(path, '{:s}.csv'.format(i)))
    elif isinstance(stock_data, pandas.DataFrame):
        d['Close'].to_csv(join(path, '{:s}.csv'.format(stock)))

def read_stock_history(f, **kwargs):
    """ Sets default values for pandas.read_csv and passes on **kwargs """
    return pandas.read_csv(f, index_col='Date', names=['Date', 'Close'], 
        header=0, parse_dates=[0], **kwargs).astype(numpy.float64)

