#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

colors = ['r', 'g', 'b', 'c', 'm', 'k']

fig = plt.figure()
ax = fig.add_subplot(111)

matrix_size = [82, 130, 514, 802, 1026, 2050, 4098]

orij5_reduction_instr_throughput = [0.604332, 0.593645, 0.526618, 0.473386, 0.441889, 0.392522, 0.373636]

orij6_reduction_instr_throughput = [0.600017, 0.593135, 0.544139, 0.473397, 0.451708, 0.396061, 0.374665]

names = [   'orij5_reduction_instr_throughput',
            'orij6_reduction_instr_throughput']


ax.plot(matrix_size, orij5_reduction_instr_throughput,   marker='o', c=colors[0], alpha=0.75)
ax.plot(matrix_size, orij6_reduction_instr_throughput,   marker='o', c=colors[1], alpha=0.75)

ax.set_xlabel('Matrix Size', fontsize=20)
ax.set_ylabel('Instruction throughput', fontsize=20)
ax.set_title('Instruction throughput vs Array Size')
ax.grid(True)

plt.legend( (names), loc=0, borderaxespad=0. )
#####################################

plt.show()
