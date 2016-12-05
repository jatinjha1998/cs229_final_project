""" Neural network submodule

Deals with the actual training
"""

__all__ = ['create_model', \
           'copy_model', \
           'track_model']

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation

from .portfolio import *
from .stock_history import *
from .rl import *

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

def track_model(target, model, τ):
    model_weights = model.get_weights()
    target_weights = target.get_weights()
    new_weights = [τ * m + (1-τ)*t for (m, t) in \
        zip(model_weights, target_weights)]
    target.set_weights(new_weights)

#my_init = 'glorot_normal'
_scale = 1E-4
def my_init(shape, name=None):
    return initializations.normal(shape, scale=_scale, name=name)

#α = 0.0001
#opt = SGD(lr=α, decay=1e-5, momentum=0.95, nesterov=True)

def train_model(states, actions, D: Int=6, γ: np.float=0.99,
    ϵ: np.float=0.15, H = 100, non_lin = 'relu', opt=Adam(),
    τ :np.flaot=0.001, debug=False, debug_every=2500):
    """Takes a list of states and trains a neural net

    states:
        a list of Trading.states. They should all have the same d value
    actions:
        the actions the nn can take (output of the Q function)
    d:
        size of experience replay
    γ:
        discount factor
    ϵ:
        ϵ-greedy parameter
    H:
        hidden layer size
    non_lin:
        activation function
    init:
        layer initialization
    opt:
        SGD optimizer
    τ:
        target-model drift

    debug, debug_ever:
        whether to output debugging information, and how ofter to do so
    """

    # training record
    train_record = pd.DataFrame(columns=('reward', 'loss'))
    i = 0

    # number of inputs and outputs
    n = states[0].num_states()
    k = actions.size

    model = create_model(n=n, k=k, H=H, non_linearity=non_lin,
        init=init, optimizer=opt)
    target = create_model(n=n, k=k, H=H, non_linearity=non_lin,
        init=init, optimizer=opt)
    # start off with exact same initialization for target nn
    copy_model(target, model)

    # start training
    while True:
        if available_states == []:
            # nothing left :(
            break
        # fill the experience replay
        elif len(available_states) < D:
            # getting close to the end
            exp_rep = np.random.permutation(available_states)
        else:
            exp_rep = np.random.choice(available_states, size=D, replace=False)

        # the actual size of the experience replay
        d = len(exp_rep)

        # actual state values of each portfolio
        states = np.array([st.state for st in exp_rep])

        qvalues = model.predict(states)

        # max_a w/ ϵ
        chosen_actions = trd.choose_actions(qvalues, ϵ)

        for (st, a) in zip(exp_rep, trd.actions[chosen_actions]):
            # execute the action
            st.execute_trade(a)

            # step forward to the next day
            try:
                st.step()
            except StopIteration:
                # reached end of data; no more stepping for this one
                available_states.remove(st)

        states_prime = np.array([st.state for st in exp_rep])
        rewards = np.array([reward(st) for st in  exp_rep])

        # max_a' Q(s', a')
        # use target network
        qvalues_prime = np.max(target.predict(states_prime), axis=1)

        # the values we want (to minimize the MSE of)
        qvalues[np.arange(0,d), chosen_actions] = rewards + γ * qvalues_prime

        # train the network
        loss = model.train_on_batch(states, qvalues)
        loss = np.asscalar(loss[-1])

        # allow the target to drift behind
        track_model(target, model, τ)

        if np.isnan(loss) or (np.infty in qvalues) or (np.infty in qvalues_prime):
            # we hit the rails . . .
            break

        # append new value
        # not very efficient, but this probably not the slowest step
        train_record.loc[i,:] = [np.mean(rewards), loss]

        if (debug) and (i % debug_every == 0):
            print('iter:  {:7d}\tloss:  {:<16g}'.format(i, loss))

        i += 1

    return (model, train_record)
