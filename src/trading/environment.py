""" Model sub-module

Deals with specifying the problem, environment, actions, and model state
"""

__all__ = ['Price', \
           'Shares', \
           'Action', \
           'actions', \
           'State']

import numpy as np
import pandas as pd

from .portfolio import *

Price = np.float64
Shares = np.uint

Action = np.float64
# percent of total to sell of the lo-beta stock and buy of hi-beta
actions = np.array([-0.25, -0.10, -0.05, 0, 0.05, 0.10, 0.25], dtype=Action)

class State:
    def __init__(self, cost_lo_prev: Price, cost_lo: Price, num_lo: Shares,
                cost_hi_prev: Price, cost_hi: Price, num_hi: Shares,
                cash: Shares, name_lo: str='', name_hi: str =''):
        self.cost_lo_prev = cost_lo_prev
        self.lo = StockHolding(cost_lo, num_lo, name_lo)

        self.cost_hi_prev = cost_hi_prev
        self.hi = StockHolding(cost_hi, num_hi, name_hi)

        self.cash = cash

    def __getattr__(self, attr):
        if attr == 'total':
            return self.lo + self.hi + self.cash
        elif attr == 'state':
            return (self.cost_lo_prev, self.lo.cost, self.lo.num,
                self.cost_hi_prev, self.hi.cost, self.hi.num,
                self.cash, self.total)

    def update(self, cost_lo_new, cost_hi_new):
        """Update stock costs with a new one"""
        self.cost_lo_prev, self.lo.cost = self.lo.cost, cost_lo_new
        self.cost_hi_prev, self.hi.cost = self.hi.cost, cost_hi_new

    def execute_trade(self, action: Action):
        """Sell off lo to buy hi (if action > 0, vice-versa if < 0)"""
        if (not action in actions) or (action == 0):
            return

        (buy_stk, sell_stk) = (self.hi, self.lo) \
            if (action > 0) \
            else (self.lo, self.hi)

        percent_trade = np.minimum(sell_stk.total / self.total, abs(action))

        (buy_stk, sell_stk, self.cash) = trade_stocks(percent_trade,
                sell_stk, buy_stk, cash=self.cash, trans_cost=trans_cost)

