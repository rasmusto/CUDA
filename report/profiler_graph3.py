#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

colors = ['r', 'g', 'b', 'c', 'm', 'k']

fig = plt.figure()
ax = fig.add_subplot(111)

matrix_size = [82, 130, 514, 802, 1026, 2050, 4098]

orij5_jacobi_mem_throughput = [41.7332,47.3507,65.9734,76.5066,68.0308,67.5396,66.9734]
orij6_jacobi_mem_throughput = [12.4006,18.1979,31.0801,32.2656,32.6264,33.1604,33.2914]
onekj6_jacobi_mem_throughput =  [9.33343,14.9829,28.3336,30.1745,30.4039,30.9456,31.1163]

names = [   'orij5_jacobi_mem_throughput',
            'orij6_jacobi_mem_throughput',
            'onekj6_jacobi_mem_throughput']

ax.plot(matrix_size, orij5_jacobi_mem_throughput,      marker='o', c=colors[0], alpha=0.75)
ax.plot(matrix_size, orij6_jacobi_mem_throughput,      marker='o', c=colors[1], alpha=0.75)
ax.plot(matrix_size, onekj6_jacobi_mem_throughput,     marker='o', c=colors[2], alpha=0.75)

ax.set_xlabel('Matrix Size', fontsize=20)
ax.set_ylabel('Memory Throughput', fontsize=20)
ax.set_title('Memory throughput vs Array Size (Jacobi Kernel)')
ax.grid(True)

plt.legend( (names), loc=0, borderaxespad=0. )
#####################################

plt.show()
