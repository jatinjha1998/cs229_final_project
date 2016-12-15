''' Neural network training submodule

Deals with the actual training
'''

__all__ = ['create_model', \
           'copy_model', \
           'track_model', \
           'train_model', \
           'train_mult_models', \
           'my_init',\
           'MODEL_LOC', \
           'Q_PORT_DIR', \
           'NOTHING_DIR', \
           'REBAL_DIR', \
           'MAX_DIR', \
           'MIN_DIR']

from os.path import join
from os import mkdir
import pickle
import numpy as np
import pandas as pd
from keras import initializations
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, Activation

from .stock_history import *
from .rl import *
from . import DATA_LOC

MODEL_LOC = join(DATA_LOC, 'model')
Q_PORT_DIR = 'q'
NOTHING_DIR = 'nothing'
REBAL_DIR = 'rebal'
MAX_DIR = 'max'
MIN_DIR = 'min'

def create_model(n, k, H, non_linearity, init):
    model = Sequential([
        Dense(input_dim=n, output_dim=H, init=init),
        Activation(non_linearity),
        Dense(output_dim=H, init=init), # H 1
        Activation(non_linearity),
        Dense(output_dim=H, init=init), # H 2
        Activation(non_linearity),
        Dense(output_dim=H, init=init), # H 3
        Activation(non_linearity),
        Dense(output_dim=H, init=init), # H 4
        Activation(non_linearity),
        Dense(output_dim=k)])

    return model

def copy_model(target, model):
    target.set_weights(model.get_weights())

def track_model(target, model, tau):
    model_weights = model.get_weights()
    target_weights = target.get_weights()
    new_weights = [tau * m + (1-tau)*t for (m, t) in \
        zip(model_weights, target_weights)]
    target.set_weights(new_weights)

my_init = 'glorot_normal'
_scale = 1E-4
#def my_init(shape, name=None):
#    return initializations.normal(shape, scale=_scale, name=name)

#alpha = 0.0001
#opt = SGD(lr=alpha, decay=1e-5, momentum=0.95, nesterov=True)

def train_model(states, actions, D: np.int64=6, gamma: np.float64=0.99,
    eps: np.float64=0.15, H: np.int64=100, non_lin='relu', optimizer=Adam(),
    reward=sharpe_ratio_reward, tau :np.float=0.001, init=my_init,
    debug=False, debug_every: np.int64=2500, **additional):
    '''Takes a list of states and trains a neural net

    states:
        a list of Trading.states. They should all have the same d value
    actions:
        the actions the nn can take (output of the Q function)
    d:
        size of experience replay
    gamma:
        discount factor
    eps:
        eps-greedy parameter
    H:
        hidden layer size
    non_lin:
        activation function
    init:
        layer initialization
    optimizer:
        SGD optimizer, type keras.Optimizer
    tau:
        target-model drift
    reward:
        function that takes in a Trading.State and returns a reward

    debug, debug_ever:
        whether to output debugging information, and how ofter to do so

    returns
        model: the model
        train_record: record of reward and mse loss
        theta: a dict of all the thetas used in the model
    '''

    #thetas
    theta = {'reward': reward,
         'D': D,
         'gamma': gamma,
         'eps': eps,
         'H': H,
         'non_lin': non_lin,
         'tau': tau}

    # training record
    train_record = pd.DataFrame(columns=('reward', 'loss'))
    i = 0

    # number of inputs and outputs
    n = states[0].num_states()
    k = actions.size

    # list to delete from, keep all the portfolio states in portfolio_states
    #  generates a (shallow) copy rather than copy the list's reference
    available_states = states[:]

    model = create_model(n=n, k=k, H=H, non_linearity=non_lin,
        init=init)
    target = create_model(n=n, k=k, H=H, non_linearity=non_lin,
        init=init)

    model.compile(loss='mean_squared_error',
                  optimizer=optimizer, metrics=['mean_squared_error'])
    target.compile(loss='mean_squared_error',
                  optimizer=optimizer, metrics=['mean_squared_error'])

    # start off with exact same initialization for target nn
    copy_model(target, model)

    # start training
    while True:
        if available_states == []:
            # nothing left :(
            print('finished training')
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
        states = np.asarray([st.state for st in exp_rep])

        qvalues = model.predict(states)

        # max_a w/ eps
        chosen_actions = choose_actions(qvalues, eps)

        for (st, a) in zip(exp_rep, actions[chosen_actions]):
            # execute the action
            st.execute_trade(a)

            # step forward to the next day
            try:
                st.step()
            except StopIteration:
                # reached end of data; no more stepping for this one
                available_states.remove(st)

        states_prime = np.asarray([st.state for st in exp_rep])
        rewards = np.array([reward(st) for st in  exp_rep])

        # max_a' Q(s', a')
        # use target network
        qvalues_prime = np.max(target.predict(states_prime), axis=1)

        # the values we want (to minimize the MSE of)
        qvalues[np.arange(0,d), chosen_actions] = rewards + gamma * qvalues_prime

        # train the network
        loss = model.train_on_batch(states, qvalues)
        loss = np.asscalar(loss[-1])

        # allow the target to drift behind
        track_model(target, model, tau)

        if np.isnan(loss) or (np.infty in qvalues) or (np.infty in qvalues_prime):
            # we hit the rails . . .
            break

        # append new value
        # not very efficient, but this probably not the slowest step
        train_record.loc[i,:] = [np.mean(rewards), loss]

        if (debug) and (i % debug_every == 0):
            print('iter:  {:7d}\tloss:  {:<16g}'.format(i, loss))

        i += 1

    return (model, train_record, theta)

def train_mult_models(thetas, state_init, train_size=0.8,
        debug=False, debug_every=5000):
    '''Unpacks a list of thetas into train_model and wraps folder gen

    Wraps the train_model function to generate the model folder and test files
    and other admin associated with the model

    thetas: list of dictionaries
        unpacks each dictionary as key_word arguments to train_model
    state_init: dictionary to pass to State initialization
    '''
    train_data, test_data = get_stock_pairs(train_size)

    for (i, theta) in enumerate(thetas):
        assert 'reward_name' in theta

        if debug:
            print('model: {:6d} of {:6d}'.format(i+1, len(thetas)))

        # regenerate models each time because they may have different 'd's
        portfolio_states = [State(p, **state_init, **theta) for p in train_data]

        (model, train_record, mdl_theta) = train_model(portfolio_states,
                actions, **theta, debug=debug, debug_every=debug_every)

        # update theta
        theta.update(state_init)
        theta.update(mdl_theta)

        MODEL_DIR = '{:d}days_{:s}'\
            .format(theta['d'], theta['reward_name'])
        MODEL_DIR = join(MODEL_LOC, MODEL_DIR)

        # make all the necessary folders
        try:
            mkdir(MODEL_DIR)
            mkdir(join(MODEL_DIR, Q_PORT_DIR))
            mkdir(join(MODEL_DIR, NOTHING_DIR))
            mkdir(join(MODEL_DIR, REBAL_DIR))
        except Exception:
            pass

        # save everything
        model.save(join(MODEL_DIR, 'model.h5'))

        # except the actual reward function
        #  because you can't save functions
        del theta['reward']
        pickle.dump(theta, open(join(MODEL_DIR, 'theta.p'), 'wb' ))

        # save test portfolios
        with open(join(MODEL_DIR, 'test.csv'), 'w') as f:
            for p in test_data:
                f.write('{:s},{:s}\n'.format(p.lo, p.hi))

