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
out_path   = "./../out/"

# Variables
output_num    = 10
initial_value = 1E6
trans_cost    = 0.001

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
    
    # Generating benchmark data
    do_nothing = trd.do_nothing_benchmark(lo_hist, hi_hist, trans_cost=trans_cost, initial_value=initial_value)
    rebal = trd.rebalance_benchmark(lo_hist, hi_hist, rebalance_period=30, initial_value=initial_value, trans_cost=trans_cost)
       
    # Plotting benchmark outputs
    ax2 = plt.subplot(output_num, 2, 2*i + 2)
    cust_plt(ax2, do_nothing.total, 'do nothing')
    cust_plt(ax2, rebal.total, 'rebalance (30)')
    # Formatting
    ax2.axhline(y=initial_value, color='black')
    ax2.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
    ax2.set_ylim(bottom=0)
    plt.xticks(rotation=40)
    
# Saving figure
fig_path = out_path + "benchmarks.png"
f.savefig(fig_path, transparent=True)