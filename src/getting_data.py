# ========================================
# Importing modules
# ========================================

import sys
from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import trading as trd

# If Python version is 3+, reload module
if int(sys.version[0]) >= 3:
    import importlib
    importlib.reload(trd.stock_history)
    importlib.reload(trd)

# ========================================
# Initializing variables
# ========================================

# Filepaths
stock_path = trd.STOCK_DATA_DIR + "/"
out_path   = "./../out/"
beta_name  = "beta_values.csv"

# Variables
start_date = '2001-07-01'
end_date   = '2016-07-02'

# ========================================
# Loading stock data
# ========================================

# Getting stock symbols
low_beta = trd.get_lo_beta_stock_symbols()
high_beta = trd.get_hi_beta_stock_symbols()
# Downloading stock histories
start_date = np.datetime64(start_date)
end_date = np.datetime64(end_date)
trd.download_stock_histories(trd.LO_BETA_DIR, low_beta, start_date)
trd.download_stock_histories(trd.HI_BETA_DIR, high_beta, start_date)

# Opening beta file
betas = {}
beta_path = stock_path + beta_name
beta_file = ""
try:
    beta_file = open(beta_path, 'rb')
except IOError:
    print "ERROR: Can't find beta values"
    exit(1)
# Populating beta values
for row in beta_file.readlines():
    (symbol,beta) = row.split(",")
    betas[symbol] = float(beta)

# ========================================
# Plotting stock histories
# ========================================

# Initializing figure
f = plt.figure(figsize=(14, 5))

# Plotting low-beta stocks
ax1 = plt.subplot(1, 2, 1)
for s in low_beta: 
    # Reading stock histories
    c = trd.read_stock_history(join(trd.LO_BETA_DIR, s + '.csv'))
    # Plotting data
    ax1.plot_date(c.index, c, label='{:s} ({:.2f})'.format(s, betas[s]), ls='solid', marker='')
# Formatting
plt.xticks(rotation=40)
ax1.set_title('Low Beta Stocks')
ax1.legend(loc='upper left')

# Plotting high-beta stocks
ax2 = plt.subplot(1, 2, 2, sharey=ax1)
for s in high_beta: 
    # Reading stock histories
    c = trd.read_stock_history(join(trd.HI_BETA_DIR, s + '.csv'))
    # Plotting data
    ax2.plot_date(c.index, c, label='{:s} ({:.2f})'.format(s, betas[s]), ls='solid', marker='')
# Formatting
plt.xticks(rotation=40)
ax2.set_title('High Beta Stocks')
ax2.legend(loc='upper right')

# Saving figure
fig_path = out_path + "stock_history.png"
f.savefig(fig_path,  transparent=True)
