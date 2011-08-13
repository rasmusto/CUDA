"""
make a scatter plot with varying color and size arguments
"""
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook

# load a numpy record array from yahoo csv data with fields date,
# open, close, volume, adj_close from the mpl-data/example directory.
# The record array stores python datetime.date as an object array in
# the date column

fig = plt.figure()
ax = fig.add_subplot(111)

xx  =  [87,  12,  98,  50,  78,  23,  79,  54,  54,  21,  90,  66,  15,  14,  23,  30,  58,  58,  78,  77]
yy  =  [1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  16,  17,  18,  19,  20]

ax.scatter(xx, yy, marker='o', c='g', alpha=0.75)

#ticks = arange(-0.06, 0.061, 0.02)
#xticks(ticks)
#yticks(ticks)

#ax.set_xlabel(r'$\Delta_i$', fontsize=20)
#ax.set_ylabel(r'$\Delta_{i+1}$', fontsize=20)

ax.set_xlabel('xlabel', fontsize=20)
ax.set_ylabel('ylabel', fontsize=20)

ax.set_title('Sample Scatterplot')
ax.grid(True)

plt.show()
