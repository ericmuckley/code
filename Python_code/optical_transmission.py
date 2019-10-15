# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:28:32 2018

@author: a6q
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
import datetime
from scipy.optimize import curve_fit
import scipy.signal as filt
from scipy.signal import savgol_filter
import scipy.interpolate as inter




def config_plot(xlabel='x', ylabel='y', size=16,
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


def get_time_table(filename, pressure_col_name='rh_setpoint'): #'RH stpnt'
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
    
            time_table.append([data['date/time'].iloc[i],
                               data[str(pressure_col_name)].iloc[i]])
    
    #append last pressure step
    time_table.append([data['date/time'].iloc[-1],
                               data[str(pressure_col_name)].iloc[-1]])
    
    time_table = pd.DataFrame(time_table, columns=['time', 'pressure'])
    
    #add column for formatted timestamps
    ts = [datetime.datetime.strptime(step,'%Y-%m-%d %H:%M:%S.%f'
                                     ) for step in time_table['time']]
    time_table['ts'] = ts
    time_table['elapsed_min'] = np.array(
            time_table['ts']-time_table['ts'].iloc[0]).astype(float)/1e9/60
      
    return time_table

#%% import data
datafile = pd.read_csv('C:\\Users\\a6q\\exp_data\\2018-11-29_PSS_optical_clean.csv')
rhfile = 'C:\\Users\\a6q\\exp_data\\2018-11-29_rh'
time_table = get_time_table(rhfile)

#%%
wl0 = np.array(datafile['wavelength_nm'])
wl = np.linspace(wl0[0], wl0[-1], len(wl0))


all_spec = np.empty((len(wl), 0))
max_wl = np.array([])
max_int = np.array([])
#loop over each spectrum
for i, col in enumerate(datafile.columns[1:]):
    print('spectrum %i/%i' %(i+1, len(datafile.columns[1:])))
    spec0 = np.array(datafile[col])
    spec0 = spec0/1e3
    
    spline = savgol_filter(spec0, 501, 1)#, mode='nearest')
    #fit_spline = inter.UnivariateSpline(wl0, spec0, k=2, s=1.5e3)
    #spline = fit_spline(wl)
    
    
    plt.scatter(wl0, spec0, s=3, c='k', alpha=0.3, label='data')
    plt.plot(wl, spline, lw=3, c='r', label='spline')
    config_plot('Wavelength (nm)', 'Intensity (arb. units)')
    plt.legend()
    plt.title(col, fontsize=16)
    plt.show()
    
    all_spec = np.column_stack((all_spec, spline))
    max_wl = np.append(max_wl, wl[np.argmax(spline)])
    max_int= np.append(max_int, np.max(spline))
 
all_spec_diff = np.copy(all_spec)
for i in range(len(all_spec[0])):
    all_spec_diff[:,i] -=  all_spec[:,0]
    plt.plot(wl, all_spec_diff[:,i], label=str(i))
    plt.legend()
plt.show()

for i in range(len(all_spec[0])):
    plt.plot(wl, all_spec[:,i], label=str(i))
    plt.legend()
plt.show()







#%% format for ML array
rh_list = time_table['pressure'].iloc[:-1]

rh_spline = np.array([2,5,10,15,20,25,30,35,40,45,50,
                      55,60,65,70,75,80,85,90,95])

fit_spline = inter.UnivariateSpline(rh_list[:-1], max_int[:-1], k=2, s=1.5e3)
max_int_spline = fit_spline(rh_spline)

fit_spline = inter.UnivariateSpline(rh_list[:-1], max_wl[:-1], k=2, s=1.5e3)
max_wl_spline = fit_spline(rh_spline)


plt.scatter(rh_list, max_int)
plt.plot(rh_spline, max_int_spline)
config_plot('RH', 'Max intensity')   
plt.show()

plt.scatter(rh_list, max_wl)
plt.plot(rh_spline, max_wl_spline)
config_plot('RH', 'Wavelength at max intensity (nm)')
plt.show()