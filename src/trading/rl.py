''' RL sub-module

Deals with specifying the problem, environment, actions, and model state
'''

__all__ = ['Price', \
           'Shares', \
           'Action', \
           'actions', \
           'State', \
           'create_penalized_returns_reward', \
           'sharpe_ratio_reward', \
           'choose_actions']

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
    '''Defines the agent's state for Q-learning

    Includes the two stock histories, number of stocks owned, current reward,
    cash, etc.
    '''

    def num_states(self):
        return 4 + self.d * 2

    def __init__(self, stocks: StockPair, cash: Price=1e6,
            target_weights=(0.5, 0.5), d: int=2,
            trans_cost: Price=0.01, **extras):
        '''Initializes state by buying stocks to reach target weights (lo, hi)

        d is the number of days (including the current) to input to model
        '''
        self.trans_cost = trans_cost

        if (d < 1):
            raise ValueError('d must be greater than 0')

        # life starts on the last day of d
        t = max(d - 2, 0)
        # keep the stock holding objects just for API ease
        self.lo, self.hi, self.cash = allocate_stocks(cash,
                stocks.hist_lo[d], stocks.hist_hi[d],
                trans_cost=trans_cost, target_weights=target_weights,
                symbol_a=stocks.lo, symbol_b=stocks.hi)

        self.portfolio = make_portfolio(cost_lo=stocks.hist_lo,
                cost_hi=stocks.hist_hi)

        #  pdrevious days will be NaNs . . .
        self.portfolio.ix[0:(t+1), ['num_lo', 'num_hi', 'cash', 'total']] =\
            np.nan

        # assume both stocks have the same length
        self.MAX_T = len(self.portfolio)
        # current location in stock histories
        self.d = d
        self.t = t
        self.step()

    def __getattr__(self, attr):
        if attr == 'total':
            return self.lo + self.hi + self.cash
        elif attr == 'state':
            return (self.lo.num,
                *self.portfolio.ix[(self.t-self.d+1):\
                        (self.t+1), 'cost_lo'].tolist(),
                self.hi.num,
                *self.portfolio.ix[(self.t-self.d+1):\
                        (self.t+1), 'cost_hi'].tolist(),
                self.cash, self.total)

    def step(self):
        '''Update state one time step forward'''
        old_total = self.total
        self.portfolio.ix[self.t, ['num_lo', 'num_hi', 'cash', 'total']] = \
            [self.lo.num, self.hi.num, self.cash, self.total]

        self.t += 1

        if self.t >= (self.MAX_T - 1):
            raise StopIteration

        self.lo.cost = self.portfolio.ix[self.t, 'cost_lo']
        self.hi.cost = self.portfolio.ix[self.t, 'cost_hi']

        return old_total

    def execute_trade(self, action: Action):
        '''Sell off lo to buy hi (if action > 0, vice-versa if < 0)
        Returns old total'''
        old_total = self.total

        (buy_stk, sell_stk) = (self.hi, self.lo) \
            if (action > 0) \
            else (self.lo, self.hi)

        # can't sell more than you have
        percent_trade = np.minimum(sell_stk.total / self.total, abs(action))

        if (action in actions) and (np.abs(percent_trade) > 1E-4):
            # gotta deal with zeros . . .
            (buy_stk, sell_stk, self.cash) = trade_stocks(percent_trade,
                sell_stk, buy_stk, cash=self.cash, trans_cost=self.trans_cost)

        return old_total

# NOTE: returns don't take into account the current day (m.t) because no
#  action has been executed on that day. the current return (r_t) for the
#  action done on day == m.t will only be available after m.t is performed with
#  m.step()
def create_penalized_returns_reward(l: np.float=2):
    '''returns a function that calculates the most recent reward minus the
    l * std of the reward for the entire period'''

    def penalized_reward(m):
        if isinstance(m, pd.DataFrame):
            t = len(m)
            return (m.ix[-1, 'total'] - m.ix[-2, 'total'] - \
                l * np.std(m.total, ddof=1)) \
                if (t > 1) else 0
        else:
            return (m.portfolio.ix[m.t-1, 'total'] - \
                m.portfolio.ix[m.t-2, 'total'] - \
                l * np.std(m.portfolio.total, ddof=1)) \
                if (m.t > 1) else 0

    return penalized_reward

def sharpe_ratio_reward(m):
    '''Sharpe Ratio of a portfolio up to (and including) index t, zero based

    Uses sample std dev (sₙ), not σₙ
    '''
    # we get the values because else the date index would make subtraction
    #  across the same values
    if isinstance(m, pd.DataFrame):
        returns = m.ix[:, 'total'].dropna().values
    else:
        # have to drop nans because may be later initialization days
        returns = m.portfolio.ix[:m.t, 'total'].dropna().values

    t = len(returns)
    if t < 3:
        # no returns yet if just one day old
        # two days old, one return, sharpe ratio undefined . . .
        return 0

    returns = returns[1:t] - returns[:(t-1)]
    mu = np.mean(returns)
    # unbiased (meh) estimator of std dev
    s = np.std(returns, ddof=1)
    return mu/s if (s != 0) else 0

def choose_actions(qvalues: np.array, eps=0.15):
    '''Picks the action with the largest value w.p. (1-eps), otherwise random

    Args:
    qvalues: np.array
        2d array of Q values for each action (axis 1, each column)
        per training element (state) across axis 0 (each row))
    eps: float
        Probability of choosing a random action
    '''
    chosen_actions = np.argmax(qvalues, axis=1)
    # choose whethar to take random at random, randomly (uniform)
    take_random = np.random.rand(qvalues.shape[0]) < eps
    # generate random epsilon-greedy actions, randomly (also uniform)
    chosen_actions[take_random] = np.random.randint(actions.size,
           size=sum(take_random))

    return chosen_actions

