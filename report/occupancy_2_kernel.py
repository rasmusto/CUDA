#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

colors = ['r', 'g', 'b', 'c', 'm', 'k']

fig = plt.figure()
ax = fig.add_subplot(111)

block_size = [10, 14, 16, 20]

jacobi_occupancy = [0.75,0.875,1,0.8125]
reduction_occupancy = [0.125,0.21,0.25,0.40625]

total_occupancy = [0,0,0,0]
total_occupancy[0] = jacobi_occupancy[0] + reduction_occupancy[0]
total_occupancy[1] = jacobi_occupancy[1] + reduction_occupancy[1]
total_occupancy[2] = jacobi_occupancy[2] + reduction_occupancy[2]
total_occupancy[3] = jacobi_occupancy[3] + reduction_occupancy[3]

names = [   'jacobi_occupancy',
            'reduction_occupancy',
            'total_occupancy']

ax.plot(block_size, jacobi_occupancy,    marker='o', c=colors[0], alpha=0.75)
ax.plot(block_size, reduction_occupancy, marker='o', c=colors[1], alpha=0.75)
ax.plot(block_size, total_occupancy,     marker='o', c=colors[2], alpha=0.75)

ax.set_xlabel('Block Size', fontsize=20)
ax.set_ylabel('Occupancy', fontsize=20)
ax.set_title('Occupancy vs Block Size (2 Kernel)')
ax.grid(True)

plt.legend( (names), loc=0, borderaxespad=0. )
#####################################

plt.show()
