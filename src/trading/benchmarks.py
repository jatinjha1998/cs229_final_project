""" Benchmarks sub-module

Functions to return portfolios using benchmarks
"""

__all__ = ['do_nothing_benchmark',
           'rebalance_benchmark']

import numpy
import pandas
from .portfolio import *

def _benchmark_validate_args(**kwargs):
    # ignore unequal lengths since we are dropping all nans
    #if not all(kwargs['stock_a'].index == kwargs['stock_b'].index):
    #    raise ValueError('stocks are not over same time period')

    if kwargs['initial_value'] <= 0:
        raise ValueError('initial value must be positive')

    if abs(sum(kwargs['target_weights']) - 1) > 1E-15:
        raise ValueError('target weights must sum to one')

def do_nothing_benchmark(stock_a, stock_b, initial_value=1e6,
                         target_weights=(0.5, 0.5), trans_cost=0.01):

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
    (s1, s2, cash) = allocate_stocks(initial_value,
           stock_a.iloc[0],
           stock_b.iloc[0],
           trans_cost=trans_cost,
           target_weights=target_weights)

    return make_portfolio(stock_a, s1.num,
                          stock_b, s2.num,
                          cash,
                          s1.num * stock_a + s2.num * stock_b + cash)

def rebalance_benchmark(stock_a, stock_b, initial_value=1e6,
                   target_weights=(0.5, 0.5), trans_cost=0.01,
                   rebalance_period=5):
    """Simulates stock performance if rebalanced every so often

    Returns portfolio performance that is rebalanced to maintain the original
    portfolio ratio

    Args:
    stock_a : pandas.DataFrame
    stock_b : pandas.DataFrame
        Dataframes with (ideally) pd.datetime64 indices and closing
        prices for stocks
    initial_value : float = 1,000,000
        Initial cash value of the portfolio
    target_weights : (float, float) = (0.5, 0.5)
        Percentage of stocks to be allocated into each stock
    trans_cost : float = 0
        transaction cost of buying or selling a stock
    rebalance_period : integer = 5
        Length in (stock) days of rebalancing period, starts on the first day
        of each period. ``1`` means rebalance ever day

    Return:
    portfolio : portfolio (pandas.DataFrame)
        A portfolio DataFrame over the same index
    """
    _benchmark_validate_args(**locals())

    if (rebalance_period <= 0 or rebalance_period >= len(stock_a)):
        raise ValueError('rebalance_period must be in (0, time period)')

    rebalance_period = int(rebalance_period)
    portfolio = make_portfolio(cost_lo=stock_a, cost_hi=stock_b)

    s1 = StockHolding()
    s2 = StockHolding()
    cash = initial_value

    for k, g in portfolio.groupby(numpy.arange(len(portfolio)) //
                                  rebalance_period):
        index = g.index
        start = index[0]
        end = index[-1]

        if (s1.total == 0) and (s2.total == 0):
            (s1, s2, cash) = allocate_stocks(initial_value,
                       stock_a.loc[start],
                       stock_b.loc[start],
                       trans_cost = trans_cost,
                       target_weights = target_weights)
        else:
            s1.cost = stock_a.loc[start]
            s2.cost = stock_b.loc[start]

            total = s1 + s2 + cash
            (lrg_stk, sml_stk) = (s1, s2) if \
                ((s1.total / total) > target_weights[0]) else (s2, s1)

            # difference in actual vs desired
            percent_trade = abs(lrg_stk.total / total - target_weights[0])
            (lrg_stk, sml_stk, cash) = trade_stocks(percent_trade,
                lrg_stk, sml_stk, cash, trans_cost)

        portfolio.loc[index, ('num_lo', 'num_hi')] = (s1.num, s2.num)
        portfolio.loc[index, 'cash'] = cash
        portfolio.loc[index, 'total'] = s1.num * stock_a[index] + \
                                        s2.num * stock_b[index] + cash

    return portfolio

