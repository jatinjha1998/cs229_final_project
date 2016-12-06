''' Neural network testing and comparison submodule

Generates test runs on portfolios and benchmarks and pretty plots
'''

__all__ = ['test_models', \
          'portfolio_metrics',\
          'cust_plt']

from os.path import join
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from keras.models import load_model

from .stock_history import load_stock_pairs
from .portfolio import portfolio_returns
from .benchmarks import *
from .rl import *
from .nn import *

def portfolio_metrics(portfolio, reward):
    rts = portfolio_returns(portfolio)
    return (sharpe_ratio_reward(portfolio),
            np.mean(rts), np.std(rts, ddof=1),
            reward(portfolio))

def cust_plt(ax, data, label='', **kwargs):
    ax.plot_date(data.index, data, ls='solid', marker='', label=label, **kwargs)

def test_models(model_dirs: list, rebalance_period: np.int=30, bins: int=75):
    '''Takes a list of strings of model names (relative to Trading.MODEL_LOC

    rebalance_period:
        rebalancing parameter for rebalance benchmark
    bins:
        bins for histogram
    '''
    for (i, model_dir) in enumerate(model_dirs):
        print('running model {:s} ({:d} of {:d})'.format(model_dir,
                i+1, len(model_dirs)))

        model_dir = join(MODEL_LOC, model_dir)

        # all the model and portfolio parameters
        theta = pickle.load(open(join(model_dir, 'theta.p'), 'rb' ))

        reward_name = theta['reward_name']

        if reward_name.startswith('pen_return'):
            (_, l) = theta['reward_name'].split('=')
            reward = create_penalized_returns_reward(float(l.replace('_', '.')))
        elif reward_name.startswith('sharpe_reward'):
            reward = sharpe_ratio_reward
        else:
            raise Exception('invalid reward name')

        stock_pairs = load_stock_pairs(join(model_dir, 'test.csv'))

        # load the test states from the stock pairs
        test_states = [State(p, **theta) for p in stock_pairs]
        available_test_states = test_states[:]

        model = load_model(join(model_dir, 'model.h5'))

        # run the model on each test portfolio
        while True:
            if available_test_states == []:
                break

            states = np.array([st.state for st in available_test_states])

            qvalues = model.predict(states)
            chosen_actions = np.argmax(qvalues, axis=1)

            for (st, a) in zip(available_test_states, actions[chosen_actions]):
                # execute the action
                st.execute_trade(a)

                try:
                    st.step()
                except StopIteration:
                    available_test_states.remove(st)

        # save off the test runs
        for st in test_states:
            file_name = '{:s}_{:s}.csv'.format(st.lo.symbol, st.hi.symbol)
            st.portfolio.to_csv(join(model_dir, Q_PORT_DIR, file_name))

        # run the benchmarks
        do_nothing_portfolios = []
        rebalance_portfolios = []

        for st in stock_pairs:
            nothing = do_nothing_benchmark(st.hist_lo, st.hist_hi,
                            initial_value=theta['cash'],
                            trans_cost=theta['trans_cost'])
            rebal = rebalance_benchmark(st.hist_lo, st.hist_hi,
                            rebalance_period=rebalance_period,
                            initial_value=theta['cash'],
                            trans_cost=theta['trans_cost'])

            do_nothing_portfolios.append(nothing)
            rebalance_portfolios.append(rebal)

            nothing.to_csv(join(model_dir, NOTHING_DIR,
                        '{:s}_{:s}.csv'.format(st.lo, st.hi)))
            rebal.to_csv(join(model_dir, REBAL_DIR,
                        '{:s}_{:s}.csv'.format(st.lo, st.hi)))

        # generate pretty pictures
        fmt = '%.0e'

        matplotlib.rc('font', size=12)
        # turn off interactive mode and don't plot them
        plt.ioff()
        lbl_str = '{:s} ({:.5f})'

        assert len(test_states) == len(do_nothing_portfolios)
        assert len(do_nothing_portfolios) == len(rebalance_portfolios)

        # make plots of them
        for i in range(0, len(test_states)):
            st = test_states[i]
            nothing = do_nothing_portfolios[i]
            rebal = rebalance_portfolios[i]

            # returns for each
            port_returns = portfolio_returns(st.portfolio)
            rebal_returns = portfolio_returns(rebal)
            nothing_returns = portfolio_returns(nothing)

            lo_name = st.lo.symbol
            hi_name = st.hi.symbol

            f = plt.figure(figsize=(15, 7))

            # portfolio over time
            ax1 = plt.subplot2grid((3, 2), (0,0), rowspan=3)
            cust_plt(ax1, nothing.total,
                     lbl_str.format('Do Nothing', reward(nothing)))
            cust_plt(ax1, rebal.total,
                     lbl_str.format('Rebalance {:d}'.format(rebalance_period),
                         reward(rebal)))
            cust_plt(ax1, st.portfolio.total,
                     lbl_str.format('Q Learning', reward(st.portfolio)))

            ax1.axhline(y=theta['cash'], color='black')
            ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
            ax1.set_ylim(bottom=0)
            plt.xticks(rotation=40)
            ax1.set_title('{:s} vs {:s}'.format(lo_name, hi_name))

            # histogram of returns
            ax2 = plt.subplot2grid((3, 2), (0,1))
            ax2.hist(nothing_returns, bins, label='Do Nothing')
            ax2.legend(loc='upper left')
            ax2.xaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))

            ax3 = plt.subplot2grid((3, 2), (1,1), sharex=ax2)
            ax3.hist(rebal_returns, bins, label='Rebalance')
            ax3.legend(loc='upper left')
            ax3.xaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))

            ax4 = plt.subplot2grid((3, 2), (2,1), sharex=ax3)
            ax4.hist(port_returns, bins, label='Q Learning')
            ax4.legend(loc='upper left')
            ax4.xaxis.set_major_formatter(mtick.FormatStrFormatter(fmt))

            # save the picture
            f.savefig(join(model_dir, '{:s}_{:s}_cmp.png'.format(lo_name, hi_name)))
            plt.close(f)

        # evaluate the portfolio performances
        q_perf = []
        onothing_perf = []
        rebal_perf = []

        for st in test_states:
            q_perf.append(portfolio_metrics(st.portfolio, reward))

        for p in do_nothing_portfolios:
            nothing_perf.append(portfolio_metrics(p, reward))

        for p in rebalance_portfolios:
            rebal_perf.append(portfolio_metrics(p, reward))

        q_perf = np.mean(np.asarray(q_perf), axis=0)
        nothing_perf = np.mean(np.asarray(nothing_perf), axis=0)
        rebal_perf = np.mean(np.asarray(rebal_perf), axis=0)

        np.savetxt(join(MODEL_DIR, 'nothing_perf.csv'), nothing_perf)
        np.savetxt(join(MODEL_DIR, 'rebal_perf.csv'), rebal_perf)
        np.savetxt(join(MODEL_DIR, 'q_perf.csv'), q_perf)

