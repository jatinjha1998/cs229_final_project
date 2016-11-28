# ========================================
# Importing modules
# ========================================
from itertools import product
import glob
import sys
from os.path import join, splitext, basename
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import trading as trd

# If Python version is 3+, reload module
if int(sys.version[0]) >= 3:
    import importlib
    importlib.reload(trd.portfolio)
    importlib.reload(trd.benchmarks)
    importlib.reload(trd)

# ========================================
# Initializing variables
# ========================================

# Filepaths
lo_path    = trd.LO_BETA_DIR + "/"
hi_path    = trd.HI_BETA_DIR + "/"
out_path   = "./../out/"
bench_path = trd.STOCK_DATA_DIR + "/../"
bench_nth_name = "bench-nth/"
bench_rbl_name = "bench-rbl/"
bench_max_name = "bench-max/"
bench_min_name = "bench-min/"

# Variables
output_num    = 10
initial_value = 1E6
trans_cost    = 0.001
rebalance_amt = 30

# ========================================
# Plotting benchmarks
# ========================================

# Plot over datea
def cust_plt(ax, data, label='', **kwargs):
    ax.plot_date(data.index, data, ls='solid', marker='', label=label, **kwargs)

# Choosing 10 random pairs of low- / high-beta stocks
data, _ = trd.get_stock_pairs(output_num)

# Initializing figure
matplotlib.rc('font', size=16)
f = plt.figure(figsize=(25, 80))

# Iterating across data
for i, pair in enumerate(data):
    # Getting stock pair data
    lo_name = pair.lo
    hi_name = pair.hi
    lo_hist = pair.hist_lo
    hi_hist = pair.hist_hi
    
    # Plotting original stock histories
    ax1 = plt.subplot(output_num, 2, 2*i + 1)
    cust_plt(ax1, lo_hist, lo_name)
    cust_plt(ax1, hi_hist, hi_name)
    # Formatting
    ax1.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    plt.xticks(rotation=40)
    
    # 'Nothing' benchmark data
    bench_nth = trd.do_nothing_benchmark(lo_hist, hi_hist, trans_cost=trans_cost, initial_value=initial_value)
    # 'Rebalance' benchmark data
    bench_rbl = trd.rebalance_benchmark(lo_hist, hi_hist, rebalance_period=rebalance_amt, initial_value=initial_value, trans_cost=trans_cost)
    # 'Maximum' benchmark data
    bench_max = trd.minmax_benchmark(lo_hist, hi_hist, initial_value=initial_value, trans_cost=trans_cost, opt="max")
    # 'Minimum' benchmark data
    bench_min = trd.minmax_benchmark(lo_hist, hi_hist, initial_value=initial_value, trans_cost=trans_cost, opt="min")
    
    # Plotting outputs
    ax2 = plt.subplot(output_num, 2, 2*i + 2)
    cust_plt(ax2, bench_nth.total, 'do nothing')
    cust_plt(ax2, bench_rbl.total, 'rebalance (' + str(rebalance_amt) + ')')
    # TODO: ERROR
    cust_plt(ax2, bench_max.total, 'maximum')
    cust_plt(ax2, bench_min.total, 'minimum')
    # Formatting
    ax2.axhline(y=initial_value, color='black')
    ax2.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    ax2.set_yscale('log')
    plt.xticks(rotation=40)
    
    # Saving outputs
    path_nth = bench_path + bench_nth_name + lo_name + "_" + hi_name + ".csv"
    path_rbl = bench_path + bench_rbl_name + str(rebalance_amt) + "_" + lo_name + "_" + hi_name + ".csv"
    path_max = bench_path + bench_max_name + lo_name + "_" + hi_name + ".csv"
    path_min = bench_path + bench_min_name + lo_name + "_" + hi_name + ".csv"
    bench_nth.to_csv(path_nth)
    bench_rbl.to_csv(path_rbl)
    bench_max.to_csv(path_max)
    bench_min.to_csv(path_min)
    
# Saving figure
fig_path = out_path + "benchmarks.png"
f.savefig(fig_path, transparent=True)