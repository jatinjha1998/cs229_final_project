""" Benchmarks sub-module

Functions to return portfolios using benchmarks
"""

__all__ = ['do_nothing_benchmark']

import numpy
import pandas
from .portfolio import *

def _benchmark_validate_args(**kwargs):
    if not all(kwargs['stock_a'].index == kwargs['stock_b'].index):
        raise ValueError('stocks are not over same time period')

    if kwargs['initial_value'] <= 0:
        raise ValueError('initial value must be positive')

    if abs(sum(kwargs['target_weights']) - 1) > 1E-15:
        raise ValueError('target weights must sum to one')

def do_nothing_benchmark(stock_a, stock_b, initial_value=1e6,
                         target_weights=(0.5, 0.5), trans_cost=0.0):

    """Simulates stock perfomance if left alone

    Returns portfolio performance if left alone after over time period

    Args:
    stock_a : pandas.DataFrame
    stock_b : pandas.DataFrame
        Dataframes with (ideally) pd.datetime64 indices and closing
        prices for stocks
    initial_value : float = 1,000,000
        Initial cash value of the portfolio
    target_weights : (float, float) = (0.5, 0.5)
        Percentage of stocks to aim to allocate into each stock
    trans_cost : float = 0
        transaction cost of buying or selling a stock

    Return:
    portfolio : pandas.DataFrame
        A DataFrame over the same index with net portfolio worth
    """
    _benchmark_validate_args(**locals())

    # buy initial assest
    (na, nb) = allocate_stocks(initial_value, 
                               stock_a.iloc[0] + trans_cost, 
                               stock_b.iloc[0] + trans_cost, 
                               target_weights)

    cash = initial_value - na * stock_a.iloc[0] - \
            nb * stock_b.iloc[0] - 2 * trans_cost
    return make_portfolio(cash, na, nb, na * stock_a + nb * stock_b)

