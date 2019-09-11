# -*- coding: utf-8 -*-
'''
This script performs 2D correlation spectroscopy based on the code at
https://github.com/shigemorita/2Dpy.

Data files should contain spectra where the first column is the wavenumber
and subsequent columns are the spectra intensities. The first row should
be headers.

Each data file should contain only one type of spectra (i.e. FTIR) and the
columns in each data file should all be the same length. Data in different
files can be of different lengths because all spectra will be resampled
using a spline to a user-defined designated length.

The script outputs the average spectra from each file with standard
deviations, and outputs csv files containing the matrix of synchronous
and asynchronous correlations.
'''

#%% USER INPUTS

# set hetero-correlation
hetero = True

# set input file names (inputfile2 is not used if hetero=True)
inputfile1 = "FTIR 7 days milk.csv"
inputfile2 = "Raman 7 days milk 09082019.csv"

# set the new length to resample each spectrum
new_len = 500

# set dynamic correlation
dynamic = True

# number of contours on the contour plot
num_contour = 10


#%%

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep
from scipy.interpolate import splev


def contourplot(spec, title=''):
    # create a contour plot using matplotlib
    x = spec.columns[0:].astype(float)
    y = spec.index[0:].astype(float)
    z = spec.values
    zmax = np.absolute(z).max()
    plt.figure(figsize=(4,4))
    plt.title(title)
    plt.contour(x, y, z, num_contour, colors='black', lw=0.5, alpha=0.5)
    plt.pcolormesh(x, y, z, cmap='jet', vmin=-1*zmax, vmax=zmax)


def get_correlations(spec1, spec2):
    '''
    Perform 2D correlation spectroscopy between two sets of spectra.
    Returns synchronous and asynchronous correlations. Uses the
    Hilbert-Noda transformation to find asynchronous correlation.
    '''
    def hilbert_noda_transform(spec):
        # Returns the Hilber-Noda transformation matrix of a spectrum.
        noda = np.zeros((len(spec), len(spec)))
        for i in range(len(spec)):
            for j in range(len(spec)):
                if i != j:
                    noda[i, j] = 1/math.pi/(j - i)
        return noda

    # find synchronous correlation
    sy = pd.DataFrame(spec1.values.T @ spec2.values/(len(spec1) - 1))
    sy.index = spec1.columns
    sy.columns = spec2.columns
    sy = sy.T
    # find asynchronous correlation
    noda = hilbert_noda_transform(spec1)
    asy = pd.DataFrame(spec1.values.T @ noda @ spec2.values/(len(spec1) - 1))
    asy.index = spec1.columns
    asy.columns = spec2.columns
    asy = asy.T
    return sy, asy


def arr_resample(arr, new_len=100, new_xlims=None,
                 vec_scale='lin', k=5, s=0):
    '''
    Resamples (stetches/compresses) a 2D array by using a spline fit.
    Array should be shape [[x1, y1, ...ym], ...[xn, yn, ...yn]] where the
    # first column in array is x-values and following next columns are
    y values. If no x values exist, insert column np.arange(len(arr))
    as x values.
    Accepts linear or log x-values, and new x_limits.
    k and s are degree and smoothing factor of the interpolation spline.
    '''
    # first, check whether array should be resampled using
    # a linear or log scale:
    if vec_scale == 'lin':
        new_scale = np.linspace
    if vec_scale == 'log':
        new_scale = np.geomspace
    # get new x-limits for the resampled array
    if new_xlims is None:
        new_x1, new_x2 = arr[0, 0], arr[-1, 0]
    else:
        new_x1, new_x2 = new_xlims[0], new_xlims[1]
    # create new x values
    arrx = new_scale(new_x1, new_x2, new_len)
    # create new empty array to hold resampled values
    stretched_array = np.zeros((new_len, len(arr[0])))
    stretched_array[:, 0] = arrx 
    # for each y-column, calculate parameters of degree-3 spline fit
    for col in range(1, len(arr[0])):
        spline_params = splrep(arr[:, 0], arr[:, col], k=int(k), s=s)
        # calculate spline at new x values
        arry = splev(arrx, spline_params)
        # populate stretched data into resampled array
        stretched_array[:, col] = arry
    return stretched_array


def plot_df(df, title=''):
    # plot each column of a pandas dataframe with the 1st column as x-values
    for col in df.columns[1:]:
        plt.plot(df.iloc[:, 0], df[col], label=col)
    plt.title(title)
    plt.legend()
    plt.show()


#%% Read the first file

# import data
spec1 = pd.read_csv(inputfile1)

# plot the raw spectra     
plot_df(spec1, title='Spectra-1 Original')

# resample the spectra to be the same length as new_len
spec1_resample = arr_resample(spec1.sort_values('wn').values, new_len=new_len)

# redefine dataframe with resampled data
spec1 = pd.DataFrame(data=spec1_resample, columns=list(spec1))

# plot resampled spectra
plot_df(spec1, title='Spectra-1 Resampled')


#%% Read the second file

# change second set of spectra based on "hetero" option
inputfile2 = inputfile2 if hetero else inputfile1
    
# import data
spec2 = pd.read_csv(inputfile2, header=0)

# plot the raw spectra     
plot_df(spec2, title='Spectra-2 Original')

# resample the spectra to be the same length as new_len
spec2_resample = arr_resample(spec2.sort_values('wn').values, new_len=new_len)

# redefine dataframe with resampled data
spec2 = pd.DataFrame(data=spec2_resample, columns=list(spec2))


# plot resampled spectra
plot_df(spec2, title='Spectra-2 Resampled')



#%% Get mean and standard deviation of each set of spectra

# create dataframe for mean and standard deviation of each spectrum
mean_df = pd.DataFrame(data=np.zeros((new_len, 6)),
                       columns=['spec1_wn', 'spec1_mean', 'spec1_std',
                                'spec2_wn', 'spec2_mean', 'spec2_std'])

# populate dataframe with the means and standard deviations
mean_df['spec1_wn'] = spec1['wn'].values
mean_df['spec1_mean'] = spec1[spec1.columns[1:]].mean(axis=1).values
mean_df['spec1_std'] = spec1[spec1.columns[1:]].std(axis=1).values
mean_df['spec2_wn'] = spec2['wn'].values
mean_df['spec2_mean'] = spec2[spec2.columns[1:]].mean(axis=1).values
mean_df['spec2_std'] = spec2[spec2.columns[1:]].std(axis=1).values  
  
# save mean spectra to csv file
mean_df.to_csv('correlation_spectra_MEAN.csv', index=False)




#%% Perform correlations

# set wavenumber as the new row index of each dataset
spec1.set_index('wn', inplace=True)
spec2.set_index('wn', inplace=True)

# transpose datasets
spec1 = spec1.T
spec2 = spec2.T

if dynamic:
 # subtract mean from each set of spectra
 spec1 = spec1 - spec1.mean()
 spec2 = spec2 - spec2.mean()

# get synchronous and asynchronous correlations
sy, asy = get_correlations(spec1, spec2)

# plot correlations
contourplot(sy, title='Synchronous correlation')
contourplot(asy, title='Asynchronous correlation')

# save correlation matrices to csv files
sy.to_csv('correlation_matrix_SYNCHRONOUS.csv')
asy.to_csv('correlation_matrix_ASYNCHRONOUS.csv')

