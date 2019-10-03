# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 18:06:53 2019

@author: a6q
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:19:19 2018

@author: a6q
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as poly


def config_plot(xlabel='x', ylabel='y', size=12,
               setlimits=False, limits=[0,1,0,1]):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    #set axis limits
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))


#%% USER INPUTS

filedir = r'C:\Users\a6q\Desktop\AI-controlled experiment\good_data'
filename = r'2019-08-01_15-40__iv_pp.csv'

#%% load in data file
data = pd.read_csv(filedir + '\\' + filename).dropna()
#%%
# extract bias values and drop bias column
bias = data[data.columns[0]].values
data = data.drop(data.filter(regex='bias').columns, axis=1)

#%% loop over each set of iv curves

traces = np.empty((len(bias), len(data.columns)))
traces_abs = np.empty((len(bias), len(data.columns)))
max_current = np.empty(len(data.columns))
min_current = np.empty(len(data.columns))

colors = cm.rainbow(np.linspace(0, 1, len(data.columns)))

# loop over each et of IV curves
for i in range(len(data.columns)):
    
    # format current array
    current = data[data.columns[i]].values*1e6
    current -= current[np.argmin(np.abs(bias))]
    traces[:, i] = current
    traces_abs[:, i] = np.abs(current)
    
    # save data to arrays
    max_current[i] = current.max()
    min_current[i] = current.min()
    
    plt.plot(bias, current, c=colors[i])
plt.show()

plt.plot(max_current)
plt.title('max current')
plt.show()
    
plt.plot(min_current)
plt.title('min current')
plt.show()







