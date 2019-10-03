# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:31:21 2019

Read in PEDOT:PSS data from ultiple experimental sources for correlation
between changing material properties as a function of relative humidity

@author: a6q
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import splrep
from scipy.interpolate import splev

#from google.colab import drive
#drive.mount('/content/gdrive')

def plot_setup(labels=['X', 'Y'], size=16, setlimits=False, limits=[0,1,0,1]):
    # set axes labels, ranges, and size of labels for a matplotlib plot
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))

def arr_stretch(arr, new_len=100, new_xlims=None, vec_scale='lin'):
    # Stretches or compresses an n-D array by using a spline fit.
    # Array should be shape [[x1, y1, ...ym], ...[xn, yn, ...yn]] where the
    # first column in array is x-values, next columns are y values.
    # If no x values exist, use x = np.arange(len(arr)) as x values.
    # Accepts linear or log x-values, and new x_limits.
    # check whether array should be stretched using a linear or log scale
    if vec_scale == 'lin':
        s = np.linspace
    if vec_scale == 'log':
        s = np.geomspace
    # get new x-limits for the stretched array
    if new_xlims is None:
        new_x1, new_x2 = arr[0, 0], arr[-1, 0]
    else:
        new_x1, new_x2 = new_xlims[0], new_xlims[1]
    # create new x values
    arrx = s(new_x1, new_x2, new_len)
    # create new empty array to hold stretched values
    stretched_array = np.zeros((new_len, len(arr[0])))
    stretched_array[:, 0] = arrx 
    # for each y-column, calculate parameters of degree-3 spline fit
    for col in range(1, len(arr[0])):
        spline_params = splrep(arr[:, 0], arr[:, col], k=3, s=15)
        # calculate spline at new x values
        arry = splev(arrx, spline_params)
        # populate stretched data into stretched array
        stretched_array[:, col] = arry
    return stretched_array

#@title Default title text
data_filepath = r'C:\Users\a6q\exp_data\pedotpss_multimode.xlsx'
#data_filepath = 'gdrive/My Drive/pedotpss_multimode.xlsx'
dict_raw = dict(pd.read_excel(data_filepath, None))
dict_clean = {}

new_spec_len = 100
new_rh_list = np.linspace(2, 96, num=95)

# loop over each functional mode in file and stretch the measured data array
for mode in dict_raw:
    print('measurement = %s' % str(mode))

    # set linear or log scale for interpolation
    if 'eis_' in mode:
        vec_scale='log'
    else:
        vec_scale = 'lin'
    
    # stretch array along spectrum axis
    spec_stretched = arr_stretch(dict_raw[mode].values, new_len=new_spec_len,
                                vec_scale=vec_scale)
    # transponse and add RH values as x-values
    spec_stretched_transpose = np.insert(
            spec_stretched[:, 1:].T, 0, list(dict_raw[mode])[1:], axis=1)
    # stretch along RH axis
    rh_stretched = arr_stretch(spec_stretched_transpose,
                               new_len=len(new_rh_list),
                               new_xlims=[new_rh_list[0], new_rh_list[-1]])

    # add x column and RH headers to stetched array and store as dataframe
    dict_clean[mode] = pd.DataFrame(data=np.insert(
        rh_stretched[:, 1:].T, 0, spec_stretched[:, 0], axis=1),
            columns=np.insert(new_rh_list.astype(str), 0, 'x'))

    
    
    # plot new and stretched versions
    for dic0, title0 in zip([dict_raw, dict_clean], ['RAW', 'CLEAN']):

        # line plots
        for col in range(1, len(list(dic0[mode]))):
            colors = cm.rainbow(np.linspace(0, 1, len(list(dic0[mode]))))[::-1]
            
            if 'eis_' in mode:
                plt.semilogx(dic0[mode].iloc[:, 0],
                         dic0[mode].iloc[:, col],
                         color=colors[col])
            else:
                plt.semilogx(dic0[mode].iloc[:, 0],
                         dic0[mode].iloc[:, col],
                         color=colors[col])
            
            
            plot_setup(labels=['X', str(mode)])
        plt.title(title0+' - '+str(mode))
        plt.show()

        
        
        # heatmap plots
        nrows, ncols = len(dic0[mode]), len(dic0[mode].iloc[0])
        plt.imshow(dic0[mode].iloc[1:, 1:], aspect='auto', cmap='rainbow')
        plt.colorbar()
        plt.title(title0+' - '+str(mode))
        plt.show()
        print('rows, columns = %s, %s' %(nrows, ncols))