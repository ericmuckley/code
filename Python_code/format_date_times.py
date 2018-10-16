# -*- coding: utf-8 -*-

import sys, csv, glob, os, time, numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from matplotlib import rcParams
labelsize = 18 #make size of axis tick labels larger
plt.rcParams['xtick.labelsize'] = labelsize 
plt.rcParams['ytick.labelsize'] = labelsize



def find_times(filename='2018-05-04pedotpss', pressure_col_name='p_abs'):
    
    '''Read file which contains timestamps and changing pressures. The 
    function retuns a dataframe with times and corresponding pressures.
    '''
    
    data = pd.read_table(str(filename))
    pressure_col_name = str(pressure_col_name)
    
    p_raw = np.array(data[pressure_col_name])
    
    p_indices = np.array([])
    time_table = []
    
    for i in range(len(data)-1):
        #get indices of times to keep
        if p_raw[i] != p_raw[i+1]:
            p_indices = np.append(p_indices, i).astype(int)
    
            time_table.append([data['date_time'].iloc[i],
                               data[pressure_col_name].iloc[i]])
                
    time_table = pd.DataFrame(time_table, columns=['time', 'pressure'])
    
    return time_table



#%%
    
time_table = find_times('exp_data\\2018-06-15_rh_chamber',
                        pressure_col_name='p_abs')



