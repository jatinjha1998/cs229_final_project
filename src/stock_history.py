""" Stock History Module

Reads in stock history data.
Assumes data is in the Google Finance format (google.com/finance) with columns:
    Date, Open, High, Low, Close, Volume
"""

import pandas
import numpy

def read_stock_history(f, **kwargs):
    """ Sets default values for pandas.read_csv """
    return pandas.read_csv(f, index_col='Date', 
        names=['Date','Open','High','Low','Close','Volume'],
        header=0, parse_dates=[0], 
        usecols=['Date', 'Close'], **kwargs).astype(numpy.float64)

