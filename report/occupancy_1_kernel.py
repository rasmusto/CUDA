#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

colors = ['r', 'g', 'b', 'c', 'm', 'k']

fig = plt.figure()
ax = fig.add_subplot(111)

block_size = [10, 14, 16, 20]

occupancy = [0.75,0.875,1,0.8125]

names = [  'Occupancy' ]

ax.plot(block_size, occupancy,    marker='o', c=colors[0], alpha=0.75)

ax.set_xlabel('Block Size', fontsize=20)
ax.set_ylabel('Occupancy', fontsize=20)
ax.set_title('Occupancy vs Block Size (1 Kernel)')
ax.grid(True)

plt.legend( (names), loc=0, borderaxespad=0. )
#####################################

plt.show()
