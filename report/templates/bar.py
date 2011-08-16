#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt

N = 4
data1 = (20, 35, 30, 35)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, data1, width/2, color='#4671d5')

data2 = (25, 32, 34, 20)
rects2 = ax.bar(ind+width/2, data2, width/2, color='#ffaa00')

data3 = (25, 32, 34, 20)
rects3 = ax.bar(ind+width, data3, width/2, color='#6a48d7')

# add some
ax.set_ylabel('Time (s)')
ax.set_title('Sample Graph')
ax.set_xticks(ind+ 3 * width / 4)
ax.set_xticklabels( ('18x18', '50x50', '82x82', '258x258') )

ax.legend( (rects1[0], rects2[0], rects3[0]), ('Block Size 16x16', 'Block Size 10x10', 'Block Size 20x20' ) )

def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 0.95*height, '%d'%int(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

plt.show()
