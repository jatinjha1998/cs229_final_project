""" Model sub-module

Deals with specifying the problem, environment, actions, and model state
"""

__all__ = ['Price', \
           'Shares', \
           'Action', \
           'actions', \
           'State', \
           'last_return_reward', \
           'sharpe_ratio_reward']

import numpy as np
import pandas as pd

from .portfolio import *
from .stock_history import *

Price = np.float64
Shares = np.uint

Action = np.float64
# percent of total to sell of the lo-beta stock and buy of hi-beta
actions = np.array([-0.25, -0.10, -0.05, 0, 0.05, 0.10, 0.25], dtype=Action)

class State:
    """Defines the agent's state for Q-learning

    Includes the two stock histories, number of stocks owned, current reward,
    cash, etc."""

    @staticmethod
    def num_states():
        return 8

    def __init__(self, stocks: StockPair, cash: Price=1e6,
            target_weights=(0.5, 0.5), trans_cost: Price=0.01):
        """Initializes state by buying stocks to reach target weights (lo, hi)"""
        self.trans_cost = trans_cost

        # keep the stock holding objects just for API ease
        self.lo, self.hi, self.cash = allocate_stocks(cash,
                stocks.hist_lo[0], stocks.hist_hi[0],
                trans_cost=trans_cost, target_weights=target_weights,
                symbol_a=stocks.lo, symbol_b=stocks.hi)

        self.portfolio = make_portfolio(cost_lo=stocks.hist_lo,
                cost_hi=stocks.hist_hi)

        # assume both stocks have the same length
        self.MAX_T = len(self.portfolio)
        # current location in stock histories
        self.t = 0
        self.step()

    def __getattr__(self, attr):
        if attr == 'total':
            return self.lo + self.hi + self.cash
        elif attr == 'state':
            return (self.portfolio.ix[self.t-1, 'cost_lo'],
                self.lo.cost, self.lo.num,
                self.portfolio.ix[self.t-1, 'cost_hi'],
                self.hi.cost, self.hi.num,
                self.cash, self.total)

    def step(self):
        """Update state one time step forward"""
        if self.t == self.MAX_T:
            raise StopIteration

        old_total = self.total
        self.portfolio.ix[self.t, ['num_lo', 'num_hi', 'cash', 'total']] = \
            [self.lo.num, self.hi.num, self.cash, self.total]

        self.t += 1
        self.lo.cost = self.portfolio.ix[self.t, 'cost_lo']
        self.hi.cost = self.portfolio.ix[self.t, 'cost_hi']

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

# NOTE: returns don't take into account the current day (m.t) because no
#  action has been executed on that day. the current return (r_t) for the
#  action done on day == m.t will only be available after m.t is performed with
#  m.step()
def last_return_reward(m: State):
    """Uses last return as the reward"""

    return (m.portfolio.ix[m.t-1, 'total'] - m.portfolio.ix[m.t-2, 'total']) \
        if (m.t > 1) else 0


def sharpe_ratio_reward(m: State):
    """Sharpe Ratio of a portfolio up to (and including) index t, zero based

    Uses sample std dev (sₙ), not σₙ
    """
    if m.t < 2:
        return 0

    returns = m.portfolio.ix[0:m.t, 'total'].values
    returns = returns[1:m.t] - returns[0:(m.t-1)]
    μ = np.mean(returns)
    # unbiased (meh) estimator of std dev
    s = np.std(returns, ddof=1)
    return μ/s if (s != 0) else 0

