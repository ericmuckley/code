# -*- coding: utf-8 -*-
# Created on Tue Aug 13 17:23:26 2019

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from glob import glob
from scipy.optimize import curve_fit

def plot_setup(labels=['X', 'Y'], size=16, setlimits=False,
               limits=[0,1,0,1], scales=['linear', 'linear'],
               title='', save=False, filename='plot.jpg'):
    #This can be called with Matplotlib for setting axes labels, setting
    #axes ranges and scale types, and  font size of plot labels. Function
    #should be called between plt.plot() and plt.show() commands.
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    plt.title(title, fontsize=size)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    plt.xscale(scales[0])
    plt.yscale(scales[1])    
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()

def stack_cvs(df):
    ''' Takes a Pandas DataFrame (df) with multiple CVs stacked vertically
    in 2 columns: one bias and one current column.
    Creates a new array of the stacked CVs with the first column as the
    bias and each subsequent column for each current sweep.'''
    # find indices of the start of each cv scan
    min_ind = np.where(df.iloc[:, 0] == df.iloc[:, 0].min())[0]

    # create array to hold all cv scans, with biases as first column
    cv_arr = np.array(df['E'].iloc[min_ind[0]:min_ind[1]])
    
    # stack all cv scans into one array, each scan in a new column
    for i in range(len(min_ind)):
        if i < len(min_ind) - 1:
            cv_arr = np.column_stack(
                    (cv_arr, df['I'].iloc[min_ind[i]:min_ind[i+1]]))
    return cv_arr[:, [0, 2]]


def multigauss(x, *params):
    # function with multiple gaussian peaks
    y = np.zeros_like(x)
    # for each gauss peak get the center, amplitude, and width
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    # return the sum of all the peaks   
    return y


def df_to_heatmap(df, vmin=0, vmax=100, fontsize=14, title=None,
                  savefig=False, filename='fig.jpg'):
    '''
    Plot a heatmap from 2D data in a Pandas DataFrame. The y-axis labels 
    should be index names, and x-axis labels should be column names.
    '''
    plt.rcParams['xtick.labelsize'] = fontsize 
    plt.rcParams['ytick.labelsize'] = fontsize
    #plt.xlabel(str(axis_titles[0]), fontsize=fontsize)
    #plt.ylabel(str(axis_titles[1]), fontsize=fontsize)
    plt.pcolor(df, cmap='jet', vmin=vmin, vmax=vmax)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index, fontsize=fontsize)
    plt.xticks(np.arange(0.5, len(df.columns), 1),
               df.columns, rotation='vertical', fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    fig = plt.gcf()
    fig.set_size_inches(9, 8)
    plt.colorbar()
    if savefig:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()
    plt.show()

def norm_arr(arr):
    '''Normalize an array so its values stretch from 0 to 1.'''
    arr2 = np.subtract(arr, np.min(arr))
    arr2 = arr2 / np.max(arr2)
    return arr2
    

#%% find files

filelist = glob(r'C:\Users\a6q\exp_data\cv_au_deposition\/*')

#%% create dataframe for all CV curves
cvdf = pd.DataFrame()

for filename in filelist:
    
    # get the sample name and info from the file name
    sample = filename.split('.dat')[0].split('\\')[-1]
    
    # import the file
    df = pd.read_table(filename, skiprows=39, sep=',',
                       header=None, names=['E', 'I'])

    # select last CV curve from each file 
    cv_arr = stack_cvs(df)
    # stack CV curve into dataframe
    cvdf['pot_'+sample] = cv_arr[:, 0]
    cvdf[sample] = cv_arr[:, 1]

    # show full CV curve
    plt.plot(cv_arr[:, 0], cv_arr[:, 1])
plt.show()



#%% fit peaks

peakdict = {}

# guess the positions of the CV peaks
peak_locs = [1.2, 1.3, 1.4]
guess = []
lowbounds = []
highbounds = []
# create guess for each detected peak
for i in peak_locs:
    guess = guess + [i, 1, 0.05]
    lowbounds = lowbounds + [0.5, 0, 0]
    highbounds = highbounds + [1.5, 1, .5]


ind1, ind2 = 1340, 1625 
for i in range(0, len(cvdf.columns), 2):
    x = cvdf.iloc[ind1:ind2, i]
    y = norm_arr(cvdf.iloc[ind1:ind2, i+1])
    
    
    sample = cvdf.columns[i]
    peakdict[sample] = {}
    peakdict[sample]['au'] = float(sample.split('Au[')[1].split(']_HSO')[0])
    peakdict[sample]['hso'] = float(sample.split('HSO[')[1].split(']_I')[0])
    peakdict[sample]['curr'] = float(sample.split('I[')[1].split(']')[0])

    # fit area of interest to multi-peak fit
    popt, _ = curve_fit(multigauss, x, y, p0=guess, bounds=(lowbounds, highbounds))
                    
    # print peak parameters
    param_arr = np.reshape(popt, ((-1, 3)))
    for params_i, params0 in enumerate(param_arr):
        print('peak #%i [center, amplitude, width]:' %params_i)
        print(params0)
        peakdict[sample]['peak-'+str(1+params_i)] = params0

    # loop over each fitted peak and plot it
    for p0 in range(0, len(popt), 3):
        peak0 = multigauss(x, popt[p0+0], popt[p0+1], popt[p0+2])
        plt.plot(x, peak0, label='peak-'+str(int(p0/3)), alpha=0.5)
    # plot total spectum fit
    tot_spec_fit = multigauss(x, *popt)
    plt.plot(x, tot_spec_fit, label='fit', c='r', lw=5, alpha=0.3)
    
    plt.plot(x, y, c='k', label='data')
    plt.legend()    
    plt.show()



#%% examine correlations in the peak characteristics

# ratios of peak 2 amplitude to peak 1 amplitude
for sample in peakdict:
    s = peakdict[sample]
    plt.scatter(s['au'], s['curr'],
                s=500*(s['peak-2'][1]/s['peak-1'][1])**2,
                c='b', alpha=0.4)
plot_setup(labels=['Au conc', 'Current'],
           title='Size = peak-2/peak-1 amplitude')
plt.show()

for sample in peakdict:
    s = peakdict[sample]
    plt.scatter(s['au'], s['hso'],
                s=500*(s['peak-2'][1]/s['peak-1'][1])**2,
                c='b', alpha=0.4)
plot_setup(labels=['Au conc', 'HSO'],
           title='Size = peak-2/peak-1 amplitude')
plt.show()    

for sample in peakdict:
    s = peakdict[sample]
    plt.scatter(s['curr'], s['hso'],
                s=500*(s['peak-2'][1]/s['peak-1'][1])**2,
                c='b', alpha=0.4)
plot_setup(labels=['Current', 'HSO'],
           title='Size = peak-2/peak-1 amplitude')
plt.show()    





# ratios of peak 3 amplitude to peak 1 amplitude
for sample in peakdict:
    s = peakdict[sample]
    plt.scatter(s['au'], s['curr'],
                s=500*(s['peak-3'][1]/s['peak-1'][1])**2,
                c='b', alpha=0.4)
plot_setup(labels=['Au conc', 'Current'],
           title='Size = peak-3/peak-1 amplitude')
plt.show()

for sample in peakdict:
    s = peakdict[sample]
    plt.scatter(s['au'], s['hso'],
                s=500*(s['peak-3'][1]/s['peak-1'][1])**2,
                c='b', alpha=0.4)
plot_setup(labels=['Au conc', 'HSO'],
           title='Size = peak-3/peak-1 amplitude')
plt.show()    

for sample in peakdict:
    s = peakdict[sample]
    plt.scatter(s['curr'], s['hso'],
                s=500*(s['peak-3'][1]/s['peak-1'][1])**2,
                c='b', alpha=0.4)
plot_setup(labels=['Current', 'HSO'],
           title='Size = peak-3/peak-1 amplitude')
plt.show()  



# ratios of peak 3 amplitude to peak 2 amplitude
for sample in peakdict:
    s = peakdict[sample]
    plt.scatter(s['au'], s['curr'],
                s=500*(s['peak-3'][1]/s['peak-2'][1])**2,
                c='b', alpha=0.4)
plot_setup(labels=['Au conc', 'Current'],
           title='Size = peak-3/peak-2 amplitude')
plt.show()

for sample in peakdict:
    s = peakdict[sample]
    plt.scatter(s['au'], s['hso'],
                s=500*(s['peak-3'][1]/s['peak-2'][1])**2,
                c='b', alpha=0.4)
plot_setup(labels=['Au conc', 'HSO'],
           title='Size = peak-3/peak-2 amplitude')
plt.show()    

for sample in peakdict:
    s = peakdict[sample]
    plt.scatter(s['curr'], s['hso'],
                s=500*(s['peak-3'][1]/s['peak-2'][1])**2,
                c='b', alpha=0.4)
plot_setup(labels=['Current', 'HSO'],
           title='Size = peak-3/peak-2 amplitude')
plt.show()  





#%% save all results in a single dataframe
results = np.empty((0, 12))

for sample in peakdict:
    s = peakdict[sample]
    results = np.vstack((results, 
            np.concatenate((np.array([s['au'], s['curr'], s['hso']]),
                           s['peak-1'], s['peak-2'], s['peak-3']))))
    
resultsdf = pd.DataFrame(columns=['au', 'curr', 'hso', 'p1c', 'p1a','p1w',
                                  'p2c', 'p2a', 'p2w','p3c', 'p3a', 'p3w'],
                        data=results)




