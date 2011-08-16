#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

colors = ['r', 'g', 'b', 'c', 'm', 'k']

#j6mod + 1kmod

files = ['data/j6mod_10.csv',
'data/j6mod_14.csv',
'data/j6mod_16.csv',
'data/j6mod_20.csv']

############### LOOP ##############
fig = plt.figure()
ax = fig.add_subplot(111)

i = 0
for file in files:
    print "parsing ", file
    reader = csv.DictReader(open(file, 'rb'), delimiter='\t', quotechar='"')

    matrix_size = []
    grid_size = []
    iterations = []
    residual = []
    gpu_time = []
    total_time = []
    memory_throughput = []
    instruction_throughput = []
    gflops = []
    occupancy = []

    for row in reader:
        matrix_size.append(float(row['Matrix Size']))
        grid_size.append(float(row['Grid Size']))
        iterations.append(float(row['No of iterations']))
        residual.append(float(row['Residual']))
        gpu_time.append(float(row['GPU Time Taken']))
        total_time.append(float(row['Total Time']))
        #memory_throughput.append(float(row['Memory Throughput']))
        #instruction_throughput.append(float(row['Instruction Throughput']))
        gflops.append(float(row['GFLOPS']))
        #occupancy.append(float(row['Occupancy']))

    ax.scatter(matrix_size, gpu_time, marker='o', c=colors[i%6], alpha=0.75)
    i += 1

ax.set_xlabel('Matrix Size', fontsize=20)
ax.set_ylabel('GPU time taken (s)', fontsize=20)
ax.set_title('GPU time taken vs Array Size')
ax.grid(True)

plt.legend( (files), loc=2, borderaxespad=0. )
#####################################

plt.show()
