#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

colors = ['r', 'g', 'b', 'c', 'm', 'k']

fig = plt.figure()
ax = fig.add_subplot(111)

matrix_size = [82, 130, 514, 802, 1026, 2050, 4098]

orij5_reduction_mem_throughput = [0.0394654, 0.0697831,0.67654,1.08691,1.30126,1.7109,1.8495]
orij6_reduction_mem_throughput = [0.0391974,0.0697185,0.658554,1.08694,1.28469,1.70396,1.8477]


names = [   'orij5_reduction_mem_throughput',
            'orij6_reduction_mem_throughput']

ax.plot(matrix_size, orij5_reduction_mem_throughput,      marker='o', c=colors[0], alpha=0.75)
ax.plot(matrix_size, orij6_reduction_mem_throughput,      marker='o', c=colors[1], alpha=0.75)

ax.set_xlabel('Matrix Size', fontsize=20)
ax.set_ylabel('Memory Throughput', fontsize=20)
ax.set_title('Memory throughput vs Array Size (Reduction Kernel)')
ax.grid(True)

plt.legend( (names), loc=0, borderaxespad=0. )
#####################################

plt.show()
