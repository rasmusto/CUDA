#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

colors = ['r', 'g', 'b', 'c', 'm', 'k']

fig = plt.figure()
ax = fig.add_subplot(111)

matrix_size = [82, 130, 514, 802, 1026, 2050, 4098]

orij5_jacobi_instr_throughput = [0.376713, 0.498613, 0.504114, 0.574964, 0.508802, 0.504233, 0.499359]
orij5_reduction_instr_throughput = [0.604332, 0.593645, 0.526618, 0.473386, 0.441889, 0.392522, 0.373636]

orij6_jacobi_instr_throughput = [0.544438, 0.92697, 1.1464, 1.16945, 1.17699, 1.1939, 1.19694]
orij6_reduction_instr_throughput = [0.600017, 0.593135, 0.544139, 0.473397, 0.451708, 0.396061, 0.374665]

onekj6_jacobi_instr_throughput = [0.666048, 0.947911, 1.12194, 1.16243, 1.16335, 1.17805, 1.18196]

names = [   'orij5_jacobi_instr_throughput',
            'orij5_reduction_instr_throughput',
            'orij6_jacobi_instr_throughput',
            'orij6_reduction_instr_throughput',
            '1kj6_jacobi_instr_throughput']


ax.plot(matrix_size, orij5_jacobi_instr_throughput,      marker='o', c=colors[0], alpha=0.75)
ax.plot(matrix_size, orij5_reduction_instr_throughput,   marker='o', c=colors[1], alpha=0.75)
ax.plot(matrix_size, orij6_jacobi_instr_throughput,      marker='o', c=colors[2], alpha=0.75)
ax.plot(matrix_size, orij6_reduction_instr_throughput,   marker='o', c=colors[3], alpha=0.75)
ax.plot(matrix_size, onekj6_jacobi_instr_throughput,     marker='o', c=colors[4], alpha=0.75)

ax.set_xlabel('Matrix Size', fontsize=20)
ax.set_ylabel('Instructino throughput', fontsize=20)
ax.set_title('Instruction throughput vs Array Size')
ax.grid(True)

plt.legend( (names), loc=0, borderaxespad=0. )
#####################################

plt.show()
