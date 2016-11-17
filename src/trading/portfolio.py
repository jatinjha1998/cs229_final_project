""" Portfolio sub-module

A portfolio is a pandas.DataFrame with a time index and the following columns:
    cost_lo : cost of the low-volatility stock
    num_lo : number of the low-volatility stocks
    cost_hi : cost of the high-volatility stock
    num_hi : number of the high-volatility stocks
    cash : cash assets
    total : total worth of portfolio
"""

__all__ = ['make_portfolio', \
           'allocate_stocks',\
           'trade_stocks',\
           'StockHolding']

import numpy
import pandas

class StockHolding:
    def __init__(self, cost=0, num=0, name=''):
        self.cost = cost
        self.num = num
        self.name = name

    def __getattr__(self, attr):
        if attr == 'total':
            return self.cost * self.num

    def __eq__(self, other):
        return self.total == other.total

    def __gt__(self, other):
         return self.total > other.total

    def __add__(self, other):
        return self.total + other.total

    def __repr__(self):
        name_str =  '' if (self.name == '')  else self.name + ': '
        return '{:s}{:d} at {:.2f}$ ({:.2f}$)'.format(name_str,
               int(self.num), self.cost, self.total)

def make_portfolio(cost_lo=0, num_lo=0, cost_hi=0, num_hi=0,
        cash=0, total=0, index=None):
    """Makes a portfolia dataframe"""
    return pandas.DataFrame(data={'cost_lo': cost_lo, 'num_lo': num_lo,
                                  'cost_hi': cost_hi, 'num_hi': num_hi,
                                  'cash': cash, 'total': total}, index=index)



def allocate_stocks(total_amount=1E6,
                    cost_a=16, cost_b=16, trans_cost=0.5,
                    target_weights=(0.5, 0.5)):
    """Return number of stocks to buy reach target_weights

    Args:
    total_amount: float = 1,000,000
        Initial cash value of the portfolio
    cost_a : float
    cost_b : float
        Cost of each stock
    trans_cost : float
        Transaction cost
    target_weights : (float, float) = (0.5, 0.5)
        Percentage of stocks to aim to allocate into each stock

    Return:
    (s1, s2) : (StockHolding, StockHolding)
    cash: float
    """
    na = numpy.floor(target_weights[0] * total_amount /
            (cost_a + trans_cost))
    s1 = StockHolding(cost_a, na)
    cash = total_amount - s1.total - na * trans_cost

    nb = numpy.floor(cash // (cost_b + trans_cost))
    s2 = StockHolding(cost_b, nb)
    cash -= (s2.total + nb * trans_cost)
    return (s1, s2, cash)

def trade_stocks(percent_trade: float, s1: StockHolding,  s2: StockHolding,
                 cash: float, trans_cost: float=0.16, total=0):
    """Sell off percent_trade * total of the s1 to buy s2"""
    if not (0 < percent_trade < 1):
        raise ValueError('percent_trade ({}) must be in (0, 1)'.format(percent_trade))

    if total == 0:
        total = s1 + s2 + cash

    # sell off from the larger stock first
    delta_s1 = numpy.floor(percent_trade * total / s1.cost)

    # take into account transaction cost
    cash += delta_s1 * (s1.cost - trans_cost)
    s1.num -= delta_s1

    # buy from smaller stock
    delta_s2 = numpy.floor(cash / (s2.cost + trans_cost))
    cash -= delta_s2 * (s2.cost + trans_cost)
    s2.num += delta_s2

    return (s1, s2, cash)

