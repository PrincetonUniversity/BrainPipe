# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 16:44:04 2018

@author: Zahra
"""
import re
import matplotlib.pyplot as plt

#read text file output from PyTorchUtils
test = ''
with open('cnn_train_20181009_1528952.out', 'r') as searchfile:
    for line in searchfile:
        if 'TEST:' in line: #finds all lines with test
            test += line
    searchfile.close()

n = re.compile("(?<='soma_label':\s)(\d+.\d+)") #finds loss values in test lines
loss = n.findall(test)
loss = [round(float(xx), 5) for xx in loss if str(xx)] 

#plot
plt.rcParams.update({'font.size': 8})
plt.figure()
plt.plot(loss, 'r')
plt.ylim(0, 0.02)
plt.xlabel('# of iterations in thousands')
plt.ylabel('loss value')
plt.title('3D U-net validation curve for H129')          
plt.savefig('val_zoom.pdf', dpi = 300)
plt.close()