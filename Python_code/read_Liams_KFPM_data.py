# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 13:52:23 2018

@author: a6q
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt



#%%

filename = r'C:\Users\a6q\exp_data\wanyi_KPFM_map_data\CPD_vspec_70.txt'
data = pd.read_table(filename, header=None).values

data = np.transpose(data)

#to flip horizontally
data2 = np.flip(data, axis=1)

for i in range(len(data2[0])):
    
    offset = np.average(data2[:10, i])

    data2[:,i] =  data2[:,i] - offset

    plt.plot(data2[:,i])
    
plt.show()


new_filename = filename.split('.')[0] + '_processed.txt'

np.savetxt(new_filename, data2, delimiter='\t',
           header='', footer='', comments='')