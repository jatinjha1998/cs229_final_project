""" Model sub-module

Deals with specifying the problem, environment, actions, and model state
"""

__all__ = ['Price', \
           'Shares', \
           'Action', \
           'actions', \
           'State', \
           'last_return_reward', \
           'sharpe_ratio_reward', \
           'choose_actions', \
           'create_model', \
           'copy_model', \
           'track_model']

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

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

    def __init__(self, stocks, cash=1e6,
            target_weights=(0.5, 0.5), trans_cost=0.01):
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
        old_total = self.total
        self.portfolio.ix[self.t, ['num_lo', 'num_hi', 'cash', 'total']] = \
            [self.lo.num, self.hi.num, self.cash, self.total]

        self.t += 1

        if self.t == self.MAX_T:
            raise StopIteration

        self.lo.cost = self.portfolio.ix[self.t, 'cost_lo']
        self.hi.cost = self.portfolio.ix[self.t, 'cost_hi']

        return old_total

    def execute_trade(self, action):
        """Sell off lo to buy hi (if action > 0, vice-versa if < 0) Returns old total"""
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
def last_return_reward(m):
    """Uses last return as the reward"""
    if isinstance(m, pd.DataFrame):
        return m.ix[-1, 'total'] - m.ix[-2, 'total']
    else:
        return (m.portfolio.ix[m.t-1, 'total'] - \
            m.portfolio.ix[m.t-2, 'total']) \
            if (m.t > 1) else 0


def sharpe_ratio_reward(m):
    """Sharpe Ratio of a portfolio up to (and including) index t, zero based

    Uses sample std dev (s_n), not sigma_n
    """
    if isinstance(m, pd.DataFrame):
        returns = m.ix[:, 'total'].values
        t = len(m)
    else:
        returns = m.portfolio.ix[0:m.t, 'total'].values
        t = m.t

    if t < 3:
        # no returns yet if just one day old
        # two days old, one return, sharpe ratio undefined . . .
        return 0

    returns = returns[1:t] - returns[0:(t-1)]
    mu = np.mean(returns)
    # unbiased (meh) estimator of std dev
    s = np.std(returns, ddof=1)
    return miu/s if (s != 0) else 0

def choose_actions(qvalues, epsilon=0.15):
    """Picks the action with the largest value w.p. (1-epsilon), otherwise random

    Args:
    qvalues: np.array
        2d array of Q values for each action (axis 1, each column)
        per training element (state) across axis 0 (each row))
    epsilon: float
        Probability of choosing a random action
    """
    chosen_actions = np.argmax(qvalues, axis=1)
    # choose whethar to take random at random, randomly (uniform)
    take_random = np.random.rand(qvalues.shape[0]) < epsilon
    # generate random epsilon-greedy actions, randomly (also uniform)
    chosen_actions[take_random] = np.random.randint(actions.size,
           size=sum(take_random))

    return chosen_actions

def create_model(n, k, H, non_linearity, init, optimizer):
    model = Sequential([
        Dense(input_dim=n, output_dim=H, init=init),
        Activation(non_linearity),
        Dense(output_dim=H, init=init), # H 1
        Activation(non_linearity),
        Dense(output_dim=H, init=init), # H 2
        Activation(non_linearity),
        Dense(output_dim=H, init=init), # H 3
        Activation(non_linearity),
        Dense(output_dim=k)])

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer, metrics=['mean_squared_error'])

    return model

def copy_model(target, model):
    target.set_weights(model.get_weights())

def track_model(target, model, tau):
    model_weights = model.get_weights()
    target_weights = target.get_weights()
    new_weights = [tau * m + (1-tau)*t for (m, t) in \
        zip(model_weights, target_weights)]
    target.set_weights(new_weights)

