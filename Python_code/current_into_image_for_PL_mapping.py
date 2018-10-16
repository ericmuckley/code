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

from scipy.interpolate import splrep
from scipy.interpolate import splev




def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)





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
    
    
#%% import data      
vm1_raw = pd.read_table('exp_data\\FA75_-1V_mapping')
vp1_raw = pd.read_table('exp_data\\FA75_1V_mapping')
v0_raw = pd.read_table('exp_data\\FA75_0V_mapping')


#select current data to work with
data0 = vm1_raw

data0 = np.array(data0['current'])


#settings of image
im_height, im_width = 81, 233
im_aspect_ratio = im_height/im_width





new_w = int(np.sqrt(len(data0)/im_aspect_ratio))
new_h = int(im_aspect_ratio*new_w)

total_new_points = new_h*new_w



#%%
'''
current0 = np.array(raw_data['current'])
time0 = np.array(raw_data['min'])

plt.plot(current0)
plt.title('raw data')
plt.show()

time, current = vec_stretch(time0, vecy0=current0, vec_len=18873)

plt.plot(current)
plt.title('stretched data')
plt.show()

current_reshaped = np.reshape(current, (81, 233), order='C')

new_mat = np.zeros((81,233))

for i in range(len(current_reshaped)):
    plt.plot(np.arange(len(current_reshaped[i]))+.5*i, current_reshaped[i])
    
    #new_mat[i] = 
    
plt.title('all rows')
plt.show()

'''
