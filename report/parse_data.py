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

colors = ['r', 'g', 'b', 'c', 'm', 'k']

############### LOOP 1 ##############
fig1 = plt.figure(figsize=(10,10))
ax1 = fig1.add_subplot(111)

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

    for row in reader:
        matrix_size.append(float(row['Matrix Size']))
        grid_size.append(float(row['Grid Size']))
        iterations.append(float(row['No of iterations']))
        residual.append(float(row['Residual']))
        gpu_time.append(float(row['GPU Time Taken']))
        total_time.append(float(row['Total Time']))
        #memory_throughput.append(float(row['Memory Throughput']))
        #instruction_throughput.append(float(row['Instruction Throughput']))
    ax1.scatter(matrix_size, gpu_time, marker='o', c=colors[i%6], alpha=0.75)
    i += 1

ax1.set_xlabel('Matrix Size', fontsize=20)
ax1.set_ylabel('Grid Size', fontsize=20)

ax1.set_title('1 Kernel: Block size = 10')
ax1.grid(True)
plt.legend( (files), loc=2, borderaxespad=0. )
#####################################

plt.show()
