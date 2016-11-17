""" Model sub-module

Deals with specifying the problem, environment, actions, and model state
"""

__all__ = ['Price', \
           'Shares', \
           'Action', \
           'actions', \
           'State', \
           'SharpeRatio']

import numpy as np
import pandas as pd

from .portfolio import *

Price = np.float64
Shares = np.uint

Action = np.float64
# percent of total to sell of the lo-beta stock and buy of hi-beta
actions = np.array([-0.25, -0.10, -0.05, 0, 0.05, 0.10, 0.25], dtype=Action)

class State:
    """Defines the agent's state for Q-learning"""
    @staticmethod
    def num_states():
        return 8

    def __init__(self, cost_lo_prev: Price, cost_lo: Price, num_lo: Shares,
                cost_hi_prev: Price, cost_hi: Price, num_hi: Shares,
                cash: Shares, trans_cost: Price=0,
                name_lo: str='', name_hi: str =''):
        self.cost_lo_prev = cost_lo_prev
        self.lo = StockHolding(cost_lo, num_lo, name_lo)

        self.cost_hi_prev = cost_hi_prev
        self.hi = StockHolding(cost_hi, num_hi, name_hi)

        self.cash = cash
        self.trans_cost = trans_cost

    def __getattr__(self, attr):
        if attr == 'total':
            return self.lo + self.hi + self.cash
        elif attr == 'state':
            return (self.cost_lo_prev, self.lo.cost, self.lo.num,
                self.cost_hi_prev, self.hi.cost, self.hi.num,
                self.cash, self.total)

    def update(self, cost_lo_new, cost_hi_new):
        """Update stock costs with a new one and returns the old total"""
        old_total = self.total

        self.cost_lo_prev, self.lo.cost = self.lo.cost, cost_lo_new
        self.cost_hi_prev, self.hi.cost = self.hi.cost, cost_hi_new

        return old_total

    def execute_trade(self, action: Action):
        """Sell off lo to buy hi (if action > 0, vice-versa if < 0) Returns old total"""
        old_total = self.total

        if (not action in actions) or (action == 0):
            return

        (buy_stk, sell_stk) = (self.hi, self.lo) \
            if (action > 0) \
            else (self.lo, self.hi)

        percent_trade = np.minimum(sell_stk.total / self.total, abs(action))

        (buy_stk, sell_stk, self.cash) = trade_stocks(percent_trade,
                sell_stk, buy_stk, cash=self.cash, trans_cost=self.trans_cost)

        return old_total


class SharpeRatio:
    """Online (incremental) estimation of Sharpe's Ratio

    Uses sample std dev (sₙ), not σₙ
    Taken from:http://math.stackexchange.com/a/103025
    """

    def __init__(self, returns: np.array=np.empty(0)):
        if returns.size == 0:
            self.mean = 0
            self.std = 0
        else:
            self.mean = np.mean(returns)
            self.std = np.std(returns)
        self.n = returns.size

    def update(self, rt: np.float64):
        old_mean, old_std = self.mean, self.std
        self.n += 1
        n = self.n

        if n == 1:
            self.mean = rt
            self.std = 0
        else:
            self.mean = ( old_mean * (n-1) + rt ) / n
            self.std = np.sqrt( (n-2)/(n-1)*old_std**2 + (rt - old_mean)**2 / (n) )

        return old_mean, old_std

