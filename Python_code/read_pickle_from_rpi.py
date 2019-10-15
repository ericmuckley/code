# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:39:05 2018

@author: a6q
"""

import sys, glob, os, numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import figure


import time
import datetime
from scipy.optimize import curve_fit

import scipy.signal as filt
from scipy.signal import savgol_filter
from scipy.stats import linregress
import scipy.interpolate as inter
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.interpolate import griddata

import numpy.polynomial.polynomial as poly


def label_axes(xlabel='x', ylabel='y', size=16):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)




    
def get_time_table(filename, pressure_col_name='p_abs'): #'RH stpnt'
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
                               data[str(pressure_col_name)].iloc[i]])
    
    #append last pressure step
    time_table.append([data['date_time'].iloc[-1],
                               data[str(pressure_col_name)].iloc[-1]])
    
    time_table = pd.DataFrame(time_table, columns=['time', 'pressure'])
    
    #add column for formatted timestamps
    ts = [datetime.datetime.strptime(step,'%Y-%m-%d %H:%M:%S.%f'
                                     ) for step in time_table['time']]
    time_table['ts'] = ts
    
    return time_table
     


def vec_stretch(vecx0, vecy0=None, vec_len=100, vec_scale='lin'):
    '''Stretches or compresses x and y values to a new length
    by interpolating using a 3-degree spline fit.
    For only stretching one array, leave vecy0 == None.'''

    #check whether original x scale is linear or log
    if vec_scale == 'lin': s = np.linspace
    if vec_scale == 'log': s = np.geomspace
    
    #create new x values
    vecx0 = np.array(vecx0)
    vecx = s(vecx0[0], vecx0[-1], vec_len)
    
    #if only resizing one array
    if np.all(np.array(vecy0)) == None:
        return vecx
    
    #if resizing two arrays
    if np.all(np.array(vecy0)) != None:        
        #calculate parameters of degree-3 spline fit to original data
        spline_params = splrep(vecx0, vecy0)
        
        #calculate spline at new x values
        vecy = splev(vecx, spline_params)
        
        return vecx, vecy



def get_peaks(vec, n=3):
    '''get indicies and heights of peaks in vector. n parameter specifies
        how many points on each side of the peak should be strictly
        increasing/decreasing in order for it to be consiidered a peak.'''
   
    peak_indices = []
    peak_vals = []
    
    for i in range(n, len(vec)-n):
        
        #differences between points at rising and falling edges of peak
        rising_diff = np.diff(vec[i-n:i+1])
        falling_diff = np.diff(vec[i:i+n+1])
        
        #check if rising edge increases and falling edge decreases
        if np.all(rising_diff>0) and np.all(falling_diff<0):
            peak_indices.append(i)
            peak_vals.append(vec[i])

    peak_indices = np.array(peak_indices).astype(int)
    peak_vals = np.array(peak_vals).astype(float)
    

    return peak_indices, peak_vals


def normalize_vec(vec):
    #normalize intensity of a vector from 0 to 1
    vec2 = np.copy(vec)
    vec2 -= np.min(vec2)
    vec2 /= np.max(vec2)
    return vec2    
    


def remove_minimums(vec0, num_of_mins=8):
    #remove minimums from a vector. this helps clean up the
    #random low outlier points measured by the SARK-110.
    #removes "num_of_mins" number of points
    vec = np.copy(vec0)
    for i in range(num_of_mins):
        #position of minimum
        min_pos = np.argmin(vec)
        #check if minimum is first or last point in vector
        if 3 <= min_pos <= len(vec0)-3:
            vec[min_pos] = np.median(vec[min_pos-2:min_pos+2])
        if min_pos < 3:
            vec[min_pos] = np.median(vec[1:6])
        if min_pos > len(vec0)-3:
            vec[min_pos] = np.median(vec[-7:-1])
    return vec



#%% load pickled dictionary of matrices

filename = 'C:\\Users\\a6q\\exp_data\\saved_readings_copy.pkl'
data0 = pd.read_pickle(filename).iloc[::100, :]

print('done importing')
#%%


#change string datatypes to numeric
data0[['time', 'temp', 'press', 'rh']] = data0[[
        'time', 'temp', 'press', 'rh']].apply(pd.to_numeric)

#data0['ts'] = datetime.datetime.fromtimestamp(int(('45.6').split('.')[0]))


#convert time to elapsed hours
#data0['time'] = np.subtract(data0['time'],  np.min(data0['time']))/3600/24



plt.plot(data0['time'], data0['temp'])
label_axes('Time (days)', 'Temperature (C)')
plt.show()


plt.plot(data0['time'], data0['rh'])
label_axes('Time', 'RH (%)')
plt.show()

plt.plot(data0['time'], data0['press'])
label_axes('Time', 'Pressure (Pa)')
plt.show()


data0['temp'] = normalize_vec(data0['temp'])
data0['rh'] = normalize_vec(data0['rh'])
data0['press'] = normalize_vec(data0['press'])


plt.plot(data0['time'], data0['temp'], c='r', label='temp.')
plt.plot(data0['time'], data0['rh'], c='g', label='press.')
plt.plot(data0['time'], data0['press'], c='b', label='rh')
label_axes('Time (days)', 'Signal')
plt.legend(fontsize=14)
plt.show()












