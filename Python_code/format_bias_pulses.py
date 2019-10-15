# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 09:26:54 2018

@author: a6q
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.interpolate import griddata
import time
from scipy.optimize import curve_fit

import scipy.interpolate as inter


#define general functions
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

                good_file0 = data_folder[j] 

        good_files.append(good_file0) 

    #reshape to accomodate multiple bands
    good_files = np.reshape(good_files, (band_num*len(good_files)))
    #print(np.shape(good_files))
    return  good_files


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


def single_exp(t, A1, tau1, y0):
    return A1 * np.exp(-(t) / tau1) + y0


#%% USER INPUTS
    
data_folder = glob.glob('C:\\Users\\a6q\\exp_data\\2019-02-26_chip2dev3_bs_rh/*')
rh_file = 'C:\\Users\\a6q\\exp_data\\2019-02-26_chip2dev3'


#%% loop over each file

data_folder.sort(key=os.path.getmtime) #sort data files by time modified
time_table = get_time_table(rh_file)
good_files = data_folder# get_good_files(time_table, data_folder)
step_dur = 1
text_ycord = 0

save_data_array = np.zeros((2038, len(good_files)*2))
save_currents = np.zeros((2038, len(good_files)))


#%% find indicies of voltage changes
ref_data = pd.read_table(good_files[0])
ref_time = ref_data.minutes

fit_params = np.zeros((len(good_files), 3))

edge_indicies = np.zeros((len(good_files), 8))

save_splines = np.empty((len(ref_time), 0))

for i, file in enumerate(good_files):

    
    print('%i / %i' %(i+1, len(data_folder)+1))
    data0 = pd.read_table(file)
    #change units to nA and remove minimum offset
    data0.iloc[:,2] = (data0.iloc[:,2] - data0.iloc[0,2])*1e9 
    #save time
    save_data_array[:len(data0), i*2] = data0.iloc[:,0]
    
    fit_splines = False
    if fit_splines == True:
        #fit current to spline
        spline_params = inter.UnivariateSpline(data0.minutes, data0.iloc[:,2], s=1e-2)
        spline = spline_params(ref_time)
        #correct for spline overshoots at bias edges
        bias = np.round(ref_data['V'], decimals=1)
        for j in range(1, len(bias)):
            if bias[j] != bias[j-1]:
                spline[j-4:j+1] = spline[j-10:j-5]
        save_splines = np.column_stack((save_splines, spline))
        plt.plot(ref_time, spline)
  
    #save current
    save_data_array[:len(data0), i*2+1] = data0.iloc[:,2]
    save_currents[:len(data0), i] = data0.iloc[:,2]
    
    
    '''
    #fit exponential decays
    fit_limits = [235, 425]
    guess = [-10, 0.1, 0]
    popt, _ = curve_fit(single_exp, data0.iloc[fit_limits[0]:fit_limits[1], 0],
                        data0.iloc[fit_limits[0]:fit_limits[1], 2],
                                # bounds=(0, np.inf),
                                p0=guess, ftol=1e-14, xtol=1e-14,)
    print(popt)   
    fit_params[i,:] = popt                             
    fit = single_exp(data0.iloc[fit_limits[0]:fit_limits[1], 0], *popt)
    plt.plot(data0.iloc[fit_limits[0]:fit_limits[1], 0], fit, c='r',linewidth=3)
    
    
    
    '''
    plt.plot(data0.iloc[:,0], data0.iloc[:,2], c='k', alpha=0.5)
    config_plot('Time (min)', 'Current (nA)',
                )#setlimits=True)#, limits=[0, 8, -25, 25])
    
    
    plt.show()


'''
    popt, _ = curve_fit(single_exp, data0.iloc[:,2]
                                freq0,
                                g_raw,
                                # bounds=(0, np.inf),
                                      p0=guess)#, ftol=1e-14, xtol=1e-14,)
    g_fit = multiBvD_reY(freq0, *popt)
'''
'''
    fig, ax = plt.subplots()
    ax.plot(data0['minutes'], data0.iloc[:,2], c='k', alpha=0.8, linewidth=0.5)
    plt.title('%i%% RH' %time_table['pressure'].iloc[i], fontsize=16)
    config_plot('Time (min)', 'Current (nA)',
                setlimits=True, limits=[0, 8.5, -25, 25])
    
    #add colored boxes behind data
    ax.axvspan(step_dur, 2*step_dur, facecolor='b', alpha=.13)
    ax.axvspan(3*step_dur, 4*step_dur, facecolor='b', alpha=.08)
    ax.axvspan(5*step_dur, 6*step_dur, facecolor='r', alpha=.08)
    ax.axvspan(7*step_dur, 8*step_dur, facecolor='r', alpha=.13)
    #add text labels inside colored boxes
    plt.text(step_dur+0.15, text_ycord+18, '-0.4 V',
             rotation='vertical', fontsize=16, color = 'b')
    plt.text(step_dur*3+0.15, text_ycord+18, '-0.2 V',
             rotation='vertical', fontsize=16, color = 'b')
    plt.text(step_dur*5+0.15, text_ycord-8, '+0.2 V',
             rotation='vertical', fontsize=16, color = 'r')
    plt.text(step_dur*7+0.15, text_ycord-8, '+0.4 V',
             rotation='vertical', fontsize=16, color = 'r')
    
    plt.gcf().set_size_inches(4,3)
    save_pic_filename = 'exp_data\\save_biaspulse_plots\\'+str(i).zfill(3)+'.jpg'
    #plt.savefig(save_pic_filename, format='jpg', dpi=250)
    plt.show()

'''

#%%
make_video = False
if make_video:
    create_video('exp_data\\save_biaspulse_plots\\',
                 'C:\\Users\\a6q\\Desktop\\PSS_biaspulses.avi', fps=7)


#%%
'''
step_dur = 0.75
text_ycord = 7

for i in range(len(data[0])):
    print('%i/%i' %(i+1, len(data[0])+1))
    
    
    fig, ax = plt.subplots()
      
    #change units to nA and remove minimum offset
    data[:,i] = (data[:,i] - data[0,i]) *1e9 
    
    
    ax.plot(time, data[:,i], c='k', alpha=0.8)
    
    
    
    config_plot('Time (min)', 'Current (nA)',
                setlimits=True, limits=[0, 10, -40, 40])
    
    
    
    plt.title('%i%% RH' %i, fontsize=18)
    
    #add colored boxes behind data
    ax.axvspan(step_dur, 2*step_dur, facecolor='b', alpha=.13)
    ax.axvspan(3*step_dur, 4*step_dur, facecolor='b', alpha=.08)
    ax.axvspan(5*step_dur, 6*step_dur, facecolor='r', alpha=.08)
    ax.axvspan(7*step_dur, 8*step_dur, facecolor='r', alpha=.13)
    
    #add text labels inside colored boxes
    plt.text(step_dur+0.15, text_ycord-14, '-0.25 V',
             rotation='vertical', fontsize=16, color = 'b')
    plt.text(step_dur*3+0.15, text_ycord-14, '-0.5 V',
             rotation='vertical', fontsize=16, color = 'b')
    plt.text(step_dur*5+0.15, text_ycord, '+0.25 V',
             rotation='vertical', fontsize=16, color = 'r')
    plt.text(step_dur*7+0.15, text_ycord, '+0.5 V',
             rotation='vertical', fontsize=16, color = 'r')
    
    plt.show()


'''

#%%
'''
for i in range(len(data[0]))[::20]:
    plt.plot(time, data[:,i]*1e9)#,  alpha=0.2)
    config_plot('Time (min)', 'Current (nA)',
                setlimits=False, limits=[0, 8, -50, 50])
plt.show()
'''

#%%plot pulses over time in heatmap
#reshape spec_mat into columns
'''
Xf, Yf, Zf = np.array([]), np.array([]), np.array([])

#loop over each spectrum
for i in range(len(data[0])):
    #create arrays of X, Y, and Z values
    Xf = np.append(Xf, np.repeat(i, len(data)))
    Yf = np.append(Yf, time)
    Zf = np.append(Zf, data[:, i]*1e9)
    
#create x, y, and z points to be used in heatmap
xf = np.linspace(Xf.min(),Xf.max(),100)
yf = np.linspace(Yf.min(),Yf.max(),100)
zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
#create the contour plot
CSf = plt.contourf(xf, yf, zf, 500, cmap=plt.cm.rainbow,
                   #vmax=50, vmin=-50)
                   vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
plt.colorbar()
config_plot('Sequence number', 'Pulse time (min)')
plt.show()
'''