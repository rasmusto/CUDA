#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import csv

files = ['data/1kmod_b10.csv',
'data/1kmod_b14.csv',
'data/1kmod_b16.csv',
'data/1kmod_b20.csv',
'data/j5_1k_b16.csv',
'data/j5_original_b16.csv',
'data/j6_1k_b16.csv',
'data/j6_original_b16.csv',
'data/j6mod_b10.csv',
'data/j6mod_b14.csv',
'data/j6mod_b16.csv',
'data/j6mod_b20.csv']

fig = plt.figure()
ax = fig.add_subplot(111)

mpl.rcParams['axes.color_cycle'] = ['r', 'g', 'b', 'c', 'y', 'm', 'k']

for file in files:
    print "parsing ", file
    reader = csv.DictReader(open(file, 'rb'), delimiter='\t', quotechar='"')
#workers = [ageName(row[0], row[1]) for row in reader]

    matrix_size = []
    grid_size = []
    iterations = []
    residual = []
    gpu_time = []
    total_time = []
    memory_throughput = []
    instruction_throughput = []

    for row in reader:
        matrix_size.append(float(row['Matrix Size']))
        grid_size.append(float(row['Grid Size']))
        iterations.append(float(row['No of iterations']))
        residual.append(float(row['Residual']))
        gpu_time.append(float(row['GPU Time Taken']))
        total_time.append(float(row['Total Time']))
        #memory_throughput.append(float(row['Memory Throughput']))
        #instruction_throughput.append(float(row['Instruction Throughput']))

#print matrix_size
#print gpu_time

#print len(matrix_size)
#print len(gpu_time)


    ax.scatter(matrix_size, gpu_time, marker='o', alpha=0.75)

ax.set_xlabel('Matrix Size', fontsize=20)
ax.set_ylabel('Grid Size', fontsize=20)

ax.set_title('1 Kernel: Block size = 10')
ax.grid(True)

plt.show()
