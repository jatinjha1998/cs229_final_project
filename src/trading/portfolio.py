""" Portfolio sub-module

A portfolio is a pandas.DataFrame with a time index and the following columns:
    ca : cost of A's stocks
    na : number of A's stocks
    cb : cost of b's stocks
    nb : number of B's stocks
    cash : cash assets
    total : total worth of portfolio
"""

__all__ = ['make_portfolio', \
           'allocate_stocks']

import numpy
import pandas

def make_portfolio(ca=0, na=0, cb=0, nb=0, cash=0, total=0, index=None):
    """Makes a portfolia dataframe"""
    return pandas.DataFrame(data={'ca': ca, 'na': na, 
                                  'cb': cb, 'nb': nb, 'cash': cash,
                                  'total': total}, index=index)

def allocate_stocks(total_amount=1E6, 
                    stock_a=20, stock_b=16, 
                    target_weights=(0.5, 0.5)):
    """Return number of stocks to buy reach target_weights

    Args:
    stock_a : float
    stock_b : float
        Cost of each stock
    initial_value : float = 1,000,000
        Initial cash value of the portfolio
    target_weights : (float, float) = (0.5, 0.5)
        Percentage of stocks to aim to allocate into each stock

    Return:
    (na, nb) : (int, int)
        Number of each stock to buy
    """
    na = int(numpy.floor(target_weights[0] * total_amount / stock_a))
    ca = na * stock_a
    cb = total_amount - ca
    nb = int(cb // stock_b)

    return (na, nb)

