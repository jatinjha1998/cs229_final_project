""" Benchmarks sub-module

Functions to return portfolios using benchmarks
"""

__all__ = ['do_nothing_benchmark',
           'rebalance_benchmark',
           'minmax_benchmark']

import numpy
import pandas
import math
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

def minmax_benchmark(stock_a, stock_b, initial_value=1e6,
                 target_weights=(0.5, 0.5), trans_cost=0.01, opt="max"):
    """Simulates stock performance of worst-performing algorithm

    Args:
    stock_a : pandas.DataFrame
    stock_b : pandas.DataFrame
        Dataframes with (ideally) pd.datetime64 indices and closing
        prices for stocks
    initial_value : float = 1,000,000
        Initial cash value of the portfolio
    trans_cost : float = 0
        transaction cost of buying or selling a stock

    Return:
    portfolio : portfolio (pandas.DataFrame)
        A portfolio DataFrame over the same index
    """

    # ========================================
    # Initializing variables
    # ========================================

    actions = [-0.25, -0.1, -0.05, 0, 0.05, 0.1, 0.25]

    # Validating arguments
    _benchmark_validate_args(**locals())
    opt = False if opt.lower() == "min" else True
    # Initializing portfolio
    portfolio = make_portfolio(cost_lo=stock_a, cost_hi=stock_b)
    holding_A = StockHolding()
    holding_B = StockHolding()
    cash = initial_value
    # Initializing row iterators
    iter_A = stock_a.iteritems()
    iter_B = stock_b.iteritems()

    # Setting current values
    (cur_date_A, holding_A.cost) = iter_A.next()
    (cur_date_B, holding_B.cost) = iter_B.next()

    # Checking dates
    while cur_date_A != cur_date_B:
        # Debug message
        print "WARN: Mismatched dates found"
        # Compare days, advance 'slower' date
        if cur_date_A < cur_date_B:
            (cur_date_A, holding_A.cost) = iter_A.next()
        else:
            (cur_date_B, holding_B.cost) = iter_B.next()

    # Initializing holdings
    (holding_A, holding_B, cash) = allocate_stocks(initial_value,
               holding_A.cost, holding_B.cost,
               trans_cost = trans_cost,
               target_weights = target_weights)

    # ========================================
    # Iterating across histories
    # ========================================

    try:
        while True:
            # Write to output

            date  = cur_date_A
            total = holding_A.total + holding_B.total + cash
            total = round(total, 2)
            portfolio.loc[date, ('num_lo', 'num_hi')] = (holding_A.num, holding_B.num)
            portfolio.loc[date, 'cash']  = cash
            portfolio.loc[date, 'total'] = total

            # Getting next stocks
            (nxt_date_A, nxt_val_A) = iter_A.next()
            (nxt_date_B, nxt_val_B) = iter_B.next()

            # Checking dates
            while nxt_date_A != nxt_date_B:
                # Debug message
                print "WARN: Mismatched dates found"
                # Compare days, advance 'slower' date
                if nxt_date_A < nxt_date_B:
                    (nxt_date_A, nxt_val_A) = iter_A.next()
                else:
                    (nxt_date_B, nxt_val_B) = iter_B.next()

            # Comparing growth options
            minmax   = -float("Inf") if opt else float("Inf")
            minmax_A = 0
            minmax_B = 0
            minmax_C = 0
            for action in actions:
                diff  = total * action
                if diff < 0:
                    # If negative action, then sell B and buy A
                    diff = abs(diff)
                    num_sold = min(holding_B.num, int(math.floor(diff / holding_B.cost)))
                    val_sold = num_sold * holding_B.cost + cash
                    num_buy  = int(math.floor(val_sold / holding_A.cost))
                    num_A = holding_A.num + num_buy
                    num_B = holding_B.num - num_sold
                else:
                    # If positive action, then sell A and buy B
                    num_sold = min(holding_A.num, int(math.floor(diff / holding_A.cost)))
                    val_sold = num_sold * holding_A.cost + cash
                    num_buy  = int(math.floor(val_sold / holding_B.cost))
                    num_B = holding_B.num + num_buy
                    num_A = holding_A.num - num_sold
                new_cash = total - (num_A * holding_A.cost + num_B * holding_B.cost)
                new_total = num_A * nxt_val_A + num_B * nxt_val_B + new_cash
                if (opt and (minmax < new_total)) \
                or ((not opt) and (minmax > new_total)) :
                    # If 'maximum' and new total larger than max, or
                    # If 'minimum' and new total less than min
                    minmax_A = num_A
                    minmax_B = num_B
                    minmax_C = new_cash
                    minmax   = new_total
            holding_A.num = minmax_A
            holding_B.num = minmax_B
            cash          = minmax_C
            # Updating values
            (cur_date_A, holding_A.cost) = (nxt_date_A, nxt_val_A)
            (cur_date_B, holding_B.cost) = (nxt_date_B, nxt_val_B)

    except StopIteration:
        return portfolio
