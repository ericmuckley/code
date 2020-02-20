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


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import splrep
from scipy.interpolate import splev


def contourplot(spec, title='', contours=10):
    # create a contour plot using matplotlib
    fontsize=16
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.labelsize'] = fontsize
    plt.rcParams['ytick.labelsize'] = fontsize
    x = spec.columns[0:].astype(float)
    y = spec.index[0:].astype(float)
    z = spec.values
    zmax = np.absolute(z).max()
    plt.figure(figsize=(6, 6))
    plt.title(title, fontsize=fontsize)
    plt.contour(x, y, z, contours, colors='black', lw=0.5, alpha=0.5)
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
                 scaletype='lin', k=5, s=0):
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
    if scaletype == 'lin':
        new_scale = np.linspace
    if scaletype == 'log':
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
    plt.show()


def corr_spec_2d(df1, df2=None, new_len=200, dynamic=True, contours=20,
                 save_means=False, save_matrices=False, scale='lin',
                 plot_spec=True, plot_mean=True):
    '''Performs 2D correlation spectroscopy using Pandas DataFrames.
    If a second dataframe is specified, hetero-correlation will be used.
    Each dataframe should have labeled columns as spectra, with the
    first column as the independent variable array.
    Arguments:    
    new_len: sets the new length of the spectra resampled by spline fit
    hetero: sets hetero-correlation
    dynamic: subtracts the mean of each spectra
    contours: sets the number of contour lines on the contour plots
    save_means: saves the mean spectra to csv file
    save_matrices: saves the correlation matrices to file
    '''
    # plot spectra in the first dataframe
    if plot_spec:
        plot_df(df1, title='Spectra-1 Original')
    # resample the spectra to be the same length as new_len
    df1_resample = arr_resample(df1.sort_values(df1.columns[0]).values,
                                new_len=new_len,
                                scaletype=scale)
    # redefine dataframe with resampled data
    df1 = pd.DataFrame(data=df1_resample, columns=list(df1))
    # plot original and resampled spectra
    if plot_spec:
        plot_df(df1, title='Spectra-1 Resampled')


    # create dataframe for mean and standard deviation of each spectrum
    mean_df = pd.DataFrame(data=np.zeros((new_len, 3)),
                           columns=['spec1_x', 'spec1_mean', 'spec1_std'])
    # populate mean dataframe with the means and standard deviations
    mean_df['spec1_x'] = df1[df1.columns[0]].values
    mean_df['spec1_mean'] = df1[df1.columns[1:]].mean(axis=1).values
    mean_df['spec1_std'] = df1[df1.columns[1:]].std(axis=1).values
    
    # plot mean spectra
    if plot_mean:
        plt.plot(mean_df['spec1_x'], mean_df['spec1_mean'], label='mean')
        plt.plot(mean_df['spec1_x'], mean_df['spec1_std'], label='std.')
        plt.title('Z')
        plt.legend()
        plt.show()
    



    # check if a second dataframe was passed to the function
    if df2 is not None:
        # plot the spectra
        if plot_spec:
            plot_df(df2, title='Spectra-2 Original')

        # resample the spectra to be the same length as new_len
        df2_resample = arr_resample(df2.sort_values(df2.columns[0]).values,
                                    new_len=new_len,
                                    scaletype=scale)
        # redefine dataframe with resampled data
        df2 = pd.DataFrame(data=df2_resample, columns=list(df1))
        # plot resampled spectra
        if plot_spec:
            plot_df(df2, title='Spectra-2 Resampled')


        mean_df['spec2_x'] = df2[df2.columns[0]].values
        mean_df['spec2_mean'] = df2[df2.columns[1:]].mean(axis=1).values
        mean_df['spec2_std'] = df2[df2.columns[1:]].std(axis=1).values 

        if plot_mean:
            plt.plot(mean_df['spec2_x'], mean_df['spec2_mean'], label='mean')
            plt.plot(mean_df['spec2_x'], mean_df['spec2_std'], label='std.')
            plt.title('Phase')
            plt.legend()
            plt.show()

    else:
        # if second dataframe was not passed, use the first
        df2 = df1.copy(deep=True)

      
    # save mean spectra to csv file
    mean_df.to_csv('correlation_spectra_MEAN.csv', index=False)

    # set wavenumber as the new row index of each dataset
    df1.set_index(df1.columns[0], inplace=True)
    df2.set_index(df2.columns[0], inplace=True)
    # transpose datasets
    df1 = df1.T
    df2 = df2.T
    if dynamic:
        # subtract mean from each set of spectra
        df1 = df1 - df1.mean()
        df2 = df2 - df2.mean()
    # get synchronous and asynchronous correlations
    sy, asy = get_correlations(df1, df2)
    # plot correlations
    contourplot(sy, title='Synchronous correlation', contours=contours)
    contourplot(asy, title='Asynchronous correlation', contours=contours)
    if save_matrices:
        # save correlation matrices to csv files
        sy.to_csv('correlation_matrix_SYNCHRONOUS.csv')
        asy.to_csv('correlation_matrix_ASYNCHRONOUS.csv')
    return sy, asy, mean_df


#%%

file1 = r'C:\Users\a6q\exp_data\wrinkled-ws2-z-increasingrh.csv'
file2 = r'C:\Users\a6q\exp_data\wrinkled-ws2-phase-increasingrh.csv'

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

df1[df1.columns[0]] = np.log10(df1[df1.columns[0]])
df2[df2.columns[0]] = np.log10(df2[df2.columns[0]])



#%%
sy, asy, mean_df = corr_spec_2d(df1, df1)


