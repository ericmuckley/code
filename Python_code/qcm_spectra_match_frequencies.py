# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 09:58:08 2019
Resamples QCM spectra with dynamic x-axes for construction of single
matrix with shared x-axis.
"""

#%%  USER INPUTS

filename = 'exp_data\\2019-07-31_14-14__qcm_n=11_spectra.csv'

# specify length of the new array to hold all spectra
new_length = 1000


#%% some functions 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from scipy.interpolate import griddata


def plot_setup(labels=['X', 'Y'], size=16, setlimits=False, limits=[0,1,0,1],
               title='', save=False, filename='plot.jpg'):
    # This can be called with Matplotlib for setting axes labels,
    # setting axes ranges, and setting the font size of plot labels.
    # Should be called between plt.plot() and plt.show() commands.
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    plt.title(title, fontsize=size)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()


def arr_resample(arr, new_len=100, new_xlims=[], vec_scale='lin', k=1, s=0):
    '''
    Resamples (stetches/compresses) a 2D array by using a spline fit.
    Array should be shape [[x1, y1, ...ym], ...[xn, yn, ...yn]] where the
    # first column in array is x-values and following next columns are
    y values. If no x values exist, insert column np.arange(len(arr))
    as x values.
    Accepts linear or log x-values, and new x_limits.
    k and s are degree and smoothing factor of the interpolation spline.'''
    # first, check whether array should be resampled using
    # a linear or log scale:
    if vec_scale == 'lin':
        new_scale = np.linspace
    if vec_scale == 'log':
        new_scale = np.geomspace
    # get new x-limits for the resampled array
    if len(new_xlims) > 1:
        new_x1, new_x2 = new_xlims[0], new_xlims[1]
    else:
        new_x1, new_x2 = arr[0, 0], arr[-1, 0]

    # create new x values
    arrx = new_scale(new_x1, new_x2, new_len)
    # create new empty array to hold resampled values
    stretched_array = np.zeros((new_len, len(arr[0])))
    stretched_array[:, 0] = arrx 
    # for each y-column, calculate parameters of degree-3 spline fit
    for col in range(1, len(arr[0])):
        spline_params = splrep(arr[:, 0], arr[:, col], k=int(k), s=0)
        # calculate spline at new x values
        arry = splev(arrx, spline_params)
        # populate stretched data into resampled array
        stretched_array[:, col] = arry
    return stretched_array


def heatmap_from_array(arr):
    '''
    Creates a heatmap using matplotlib from a 2D array. Array should have
    first column as independent variable and subsequent columns as
    dependent variable.'''
    #create x, y, and z points to be used in heatmap
    xf = np.arange(len(arr[0]) - 1) + 1
    yf = arr[:, 0]
    Xf, Yf, Zf = np.array([]), np.array([]), np.array([])
    # loop over each column in array
    for i in range(1, len(arr[0])):
        #create arrays of X, Y, and Z values
        Xf = np.append(Xf, np.repeat(i, len(arr)))
        Yf = np.append(Yf, yf)
        Zf = np.append(Zf, arr[:, i])
    # create grid
    zf = griddata((Xf, Yf), Zf,
                  (xf[None, :], yf[:, None]),
                  method='cubic')
    #create the contour plot
    plt.contourf(xf, yf, zf, 200, cmap=plt.cm.seismic,
                 vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
    plt.colorbar()

#%%
data = pd.read_csv(filename)

# initialize lowest and highest frequency values
low_freq, high_freq = -np.inf, np.inf

# loop over each 3rd column in data file, skipping the reactance columns
for col in range(0, len(list(data)), 3):
    
    # array of frequencies and drop empty rows
    farr = data.iloc[:, col].dropna().values
    # array of series resistances and drop empty rows
    rarr = data.iloc[:, col + 1].dropna().values
    
    # find low and high frequency values
    if np.min(farr) > low_freq:
        low_freq = np.min(farr)
    if np.max(farr) < high_freq:
        high_freq = np.max(farr)
    
    plt.plot(farr, rarr)
    plot_setup(title='Raw spectra',
               labels=['Frequency', 'Series resistance (Ohm)'])
plt.axvline(x=low_freq)
plt.axvline(x=high_freq)
plt.show()
    





#%% resample each spectrum and combine in a single array
''' Now that we have the full range of frequency values, we can
resample each spectrum by fitting to a spline and instert into a single
array with a shared frequency range. ''' 
  
new_arr = np.reshape(np.linspace(low_freq, high_freq, new_length), (-1, 1))
headers = ['freq']
specnum = 0
f0list = []
# loop over each 3rd column in data file, skipping the reactance columns
for col in range(0, len(list(data)), 3):

    # array of frequencies and drop empty rows
    farr = data.iloc[:, col].dropna().values
    # array of series resistances and drop empty rows
    rarr = data.iloc[:, col + 1].dropna().values

    new_spec = arr_resample(np.stack((farr, rarr), axis=1),
                            new_len=new_length,
                new_xlims=[low_freq, high_freq])
    
    f0list.append(farr[np.argmax(rarr)])
    # only add to array if spectrum is at least half the length of the longest
    # spectrum. this avoids including small chopped-off peaks
    
    if farr.max() - farr.min() > 0.5*(high_freq - low_freq):
        new_arr = np.column_stack((new_arr, new_spec[:, 1]))
        plt.plot(new_spec[:, 0], new_spec[:, 1])
        headers.append('spec'+str(specnum).zfill(3))
        specnum += 1
        
plot_setup(title='Resampled spectra',
           labels=['Frequency', 'Series resistance (Ohm)'])
plt.show()


# save the matched arrays
df = pd.DataFrame(columns=headers, data=new_arr)
df.to_csv('MATCHED_FREQS.csv', index=False)


# plot heatmap
heatmap_from_array(new_arr)
plot_setup(title='All resampled spectra',
           labels=['Spectrum number', 'Frequency'])
plt.show()
    

    
    
    
    
    