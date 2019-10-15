# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:19:19 2018

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
from scipy import interpolate
from scipy.interpolate import griddata

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
    
    return time_table
  








def get_good_files(time_table, data_folder, band_num=1):
    '''Collect the data files from "datafolder" which were created
    just before the pressure changes occure in the "time_table"
    time/pressure table. The number of files saved for each pressure
    step is equal to "band_num".'''
    
    #sort data files by time modified
    data_folder.sort(key=os.path.getmtime)
    
    # get timestamp for each data file
    data_time = [datetime.datetime.strptime(time.ctime(os.path.getmtime(
            file)),'%a %b  %d %H:%M:%S %Y') for file in data_folder]
    
    #make list of good files which were measured just pressure changes
    good_files = []   

    #loop over each pressure
    for i in range(len(time_table)):
        
        #loop over each data file in folder
        for j in range(band_num, len(data_folder)): 
    
            #check if data file was created before timestep changed
            if data_time[j] < time_table['ts'].iloc[i]:
                
                #if single band
                if band_num == 1:
                    good_file0 = data_folder[j] 
                    
                #if multiple bands, get files just before pressure change
                if band_num > 1:
                    good_file0 = data_folder[j-band_num:j]
            
        good_files.append(good_file0) 

    #reshape to accomodate multiple bands
    good_files = np.reshape(good_files, (band_num*len(good_files)))
    #print(np.shape(good_files))
    return  good_files





def match_files_to_pressures(time_table, data_folder):
    # label a list of files by the RH level which was present during their
    # creation.
    
    # sort data files by time modified
    data_folder.sort(key=os.path.getmtime)
    # get timestamp for each data file
    data_time = [datetime.datetime.strptime(time.ctime(os.path.getmtime(
                file)),'%a %b  %d %H:%M:%S %Y') for file in data_folder]
    
    #create empty array to hold pressures for each file
    file_pressures = np.zeros(len(data_folder))
    
    #loop over each data file in folder
    for j in range(len(data_folder)): 
        #loop over each pressure
        for i in range(1,len(time_table)-1):
        
            #check if data file was created before timestep changed
            if data_time[j] < time_table['ts'].iloc[i] and data_time[j] > time_table['ts'].iloc[i-1]:
                file_pressures[j] = time_table['pressure'].iloc[i]
            elif data_time[j] < time_table['ts'].iloc[0]:
                file_pressures[j] = time_table['pressure'].iloc[0]
            elif data_time[j] > time_table['ts'].iloc[-2]:
                file_pressures[j] = time_table['pressure'].iloc[-1]    
    return file_pressures
    






def create_video(image_folder, video_name, fps=8, reverse=False):
    #create video out of images saved in a folder
    import cv2
    import os
    
    images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]
    if reverse: images = images[::-1]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()




#%% USER INPUTS
data_folder = glob.glob('C:\\Users\\a6q\\exp_data\\2019-04-18_P3HT_CV_RH/*')
data_folder.sort(key=os.path.getmtime)

good_files = data_folder





sweep_rates0 = list(pd.read_table(good_files[0]))[1:]
sweep_rates = ['2500 mV/s', '250 mV/s', '125 mV/s', '50 mV/s']

#create empty dictionary to hold data
bias0 = np.round(pd.read_table(good_files[0])['V'], 3)

#remove redundant starting and ending data so we have a full complete loop
start_i = np.argmax(bias0)
end_i = np.argmax(bias0[np.arange(len(bias0))!=np.argmax(bias0)]) + 1
bias = bias0[start_i:end_i]

dic = dict.fromkeys(sweep_rates0, bias)

dic['areas'] = np.zeros((len(good_files), len(sweep_rates)))




#%% loop pover each CV file and plot 

#loop over each file
for i, file in enumerate(good_files):
    print('file %i/%i' %(i+1, len(good_files)+1))
    data0 = pd.read_table(file)
    #bias = data0['V']
    
    #loop over each sweep
    for j, col in enumerate(list(data0)[1:]):
        current = np.array(data0[col])*1e6
        current -= current[0]
        current = current[start_i:end_i]
        current[-1] = current[0]

        plt.plot(bias, current, lw=0.3,
                 marker='o',
                 markersize=0.6,
                 label=sweep_rates[j])
        
        
        
        low_current = current[start_i:start_i*3]
        high_current = current[start_i*3:start_i*5]
        
        low_area = np.trapz(low_current, x=bias[start_i:start_i*3])
        high_area = np.trapz(high_current, x=bias[start_i*3:start_i*5])

        tot_area = high_area - low_area       
        #tot_area = np.trapz(current, x=bias)
        
        
        
        
        #print(tot_area)
        dic['areas'][i,j] = tot_area
        
        dic[col] = np.column_stack((dic[col], current))
    
        plt.fill_between(bias, current, 0, alpha=0.09)
    
    #config_plot('Bias (V)', 'Current (nA)',
    #           setlimits=True, limits=[-2.1, 2.1, -1, 1])

    plt.legend(fontsize=8.3, loc='upper left', ncol=3,
               handletextpad=0.1).get_frame().set_linewidth(0.0)
    plt.gcf().set_size_inches(4,3)
    
    plt.axhline(y=0, color='k', alpha=0.2, linewidth=0.5)
    plt.axvline(x=0, color='k', alpha=0.2, linewidth=0.5)
    
    save_pic_filename = 'exp_data\\save_CV_plots_2019-01-28_pss\\fig'+str(i).zfill(3)+'.jpg'
    #plt.savefig(save_pic_filename, format='jpg', dpi=250)
    plt.show()
        
    
    
#%% plot area under curves

for i in range(len(sweep_rates)):
    plt.plot(np.arange(len(dic['areas'])), dic['areas'][:,i], lw=0.5,
             marker='o', markersize=1.5, label=sweep_rates[i])  
config_plot('RH (%)', 'Area (uA V)')
plt.legend(fontsize=12)
plt.show()


#%% plot each sweep rate over time, and maximum current over time

max_currents = np.zeros((len(good_files), len(sweep_rates0)))

# get all sweeps at a given sweeprate
for rate_i, rate in enumerate(sweep_rates0):
    for col in range(1, len(dic[rate][0])):
        
        bias = dic[rate][:, 0]
        current = dic[rate][:, col]
        
        max_current0 = np.amax(current)
        max_currents[col-1, rate_i] = max_current0
        
        
        plt.plot(bias, current)
    plt.title(rate)
    config_plot('Bias', 'Current')
    plt.show()
      

# plot all max currents
for col in range(len(max_currents[0])):
    plt.plot(np.arange(len(max_currents))+1, max_currents[:, col],
             label=sweep_rates[col])
plt.legend()
plt.xlabel('Time')
plt.ylabel('max current')
plt.show()





#%%
make_video = False
if make_video:
    create_video('exp_data\\save_CV_plots_2019-01-28_pss\\',
                 'C:\\Users\\a6q\\Desktop\\PSS_CV_fast.avi', fps=16)



