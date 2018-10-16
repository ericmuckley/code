import glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.signal as filt
import time
import datetime
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)



#%%

#read in data file
data = pd.read_table('exp_data//2018-06-08pedotpss_iv_labeled.txt')

#voltages
v = data.v

data = data.drop('v', axis=1)


#%%

for rh in data:
    plt.plot(v, data[rh]*1e9)
label_axes('Voltage (V)', 'Current (nA)')
plt.show()

#%%


pmin_ind = []
nmin_ind = []



for rh in data:
    
    pmin_ind.append(np.argmin(data[rh].iloc[180:300]))

    nmin_ind.append(np.argmin(data[rh].iloc[150:210]))
    
  
rh = [int(i) for i in list(data)]   
    
pmin = v[pmin_ind]  
nmin = v[nmin_ind]
    
plt.plot(rh, pmin)
plt.plot(rh, nmin)
plt.show()

    
    

#%%


saveip = []


#loop over rh
for rh in data:

    col = data[rh]
    
    
    #loop over points
    for i in range(1, len(data)-1):
        
        if col.iloc[i] < col.iloc[i+1] and col.iloc[i] < col.iloc[i-1]:
            
            saveip.append((float(rh), i))
    



