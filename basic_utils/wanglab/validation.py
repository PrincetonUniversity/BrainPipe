# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:44:04 2018

@author: Zahra
"""
import re
import matplotlib.pyplot as plt

test = ''
with open('cnn_train_20181009_1528952.out', 'r') as searchfile:
    for line in searchfile:
        if 'TEST:' in line:
            test += line
    searchfile.close()

n = re.compile("(?<='soma_label':\s)(\d+.\d+)")
loss = n.findall(test)
loss = [float(xx) for xx in loss if str(xx)]

#plot
plt.figure()
plt.plot(loss[1:], 'r')
plt.xlabel('# of iterations in thousands')
plt.ylabel('loss value')
plt.title('3D U-net validation curve for H129')          
plt.savefig('val.pdf', dpi = 300, transparent = True)
plt.close()