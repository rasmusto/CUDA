#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

colors = ['r', 'g', 'b', 'c', 'm', 'k']

fig = plt.figure()
ax = fig.add_subplot(111)


matrix_size_j5 = [82, 274, 322, 562, 658, 802, 850, 946, 1042, 2050]
j5_speedup = [0.600301614, 19.25699033, 29.01406739, 62.42538161, 69.26964871, 73.65725077, 75.38498251, 77.59565301, 74.4005454, 62.67714628]

matrix_size_j6 = [82, 258, 322, 642, 1026, 2050, 4098]
j6_speedup = [0.605673844, 21.01464555, 37.64777971, 110.2630308, 134.5351917, 128.8959751, 116.2030411]

names = [ 'J5 Speedup', 'J6 Speedup' ]

ax.plot(matrix_size_j5, j5_speedup,    marker='o', c=colors[0], alpha=0.75)
ax.plot(matrix_size_j6, j6_speedup,    marker='o', c=colors[1], alpha=0.75)

ax.set_xlabel('Matrix Size', fontsize=20)
ax.set_ylabel('Speedup', fontsize=20)
ax.set_title('Speedup vs Matrix Size')
ax.grid(True)

plt.legend( (names), loc=0, borderaxespad=0. )
#####################################

plt.show()
