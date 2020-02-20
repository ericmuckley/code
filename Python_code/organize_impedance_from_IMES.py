# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 10:01:22 2019

@author: a6q
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib import colors, ticker, cm
from scipy.signal import medfilt, medfilt2d
from scipy.interpolate import splrep, splev

def df_to_heatmap(df, vmin=None, vmax=None, fontsize=14,
                  first_col_as_index=False,
                  title=None, save=False, filename='fig.jpg'):
    '''
    Plot a heatmap from 2D data in a Pandas DataFrame. The y-axis labels 
    should be index names, and x-axis labels should be column names.
    '''
    plt.rcParams['xtick.labelsize'] = fontsize 
    plt.rcParams['ytick.labelsize'] = fontsize
    #plt.xlabel(str(axis_titles[0]), fontsize=fontsize)
    #plt.ylabel(str(axis_titles[1]), fontsize=fontsize)
    if first_col_as_index:
        df = df.set_index(df.columns[0])
    if not vmin:
        vmin = df.values.min()
    if not vmax:
        vmax = df.values.max()    
    if title:
        plt.title(title, fontsize=fontsize)
    fig = plt.gcf()    
    plt.pcolor(df, cmap='jet', vmin=vmin, vmax=vmax)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index, fontsize=fontsize)
    plt.xticks(np.arange(0.5, len(df.columns), 1),
               df.columns, rotation='vertical', fontsize=fontsize)
    fig.set_size_inches(9, 8)
    plt.colorbar()
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()
    plt.show()


def get_spline(x, y, k=3, s=None, x_new=None):
    """Get a spline using x and y values with polynomial factor k.
    Interpolate at new x-values if desired."""
    # get spline knot parameters
    spline_params = splrep(x, y, k=int(k), s=s)
    # calculate spline at new x values
    if x_new:
        return splev(x_new, spline_params)
    else:
        return splev(x, spline_params)


#%%
filename = r'C:\Users\a6q\exp_data\2020-01-15_15-39__eis.csv' 

df = pd.read_csv(filename).dropna()
freqs = df.iloc[:, 0]

# create dataframes for each variable
dfz = pd.DataFrame({'frequency': freqs})
dfphi = pd.DataFrame({'frequency': freqs})
dfny = pd.DataFrame({'frequency': freqs})
  

# loop over each column and build smaller dataframes for each variable
# here we assume the columns are [freq, Z, phi, reZ, imZ]
for i in range(0, len(df.columns), 5):
    
    # get each type of spectrum 
    z0 = df.iloc[:, i + 1]
    phi0 = df.iloc[:, i + 2]
    nyx0 = df.iloc[:, i + 3]  /1e6
    nyy0 =  df.iloc[:, i + 4] / 1e6
    
    
    spline_fits = True
    if spline_fits:
        z0 = get_spline(freqs, z0)
        phi0 = get_spline(freqs, phi0)
        nyx0 = get_spline(freqs, nyx0)
        nyy0 = get_spline(freqs, nyy0)

    
    # add each spectrum to its own dataframe
    dfz['z_'+str(int((i+5)/5)).zfill(3)] = z0
    dfphi['phi_'+str(int((i+5)/5)).zfill(3)] = phi0
    dfny['rez_'+str(int((i+5)/5)).zfill(3)] = nyx0
    dfny['imz_'+str(int((i+5)/5)).zfill(3)] = nyy0


#%% plot results
    
# plot z traces
[plt.loglog(freqs, dfz.iloc[:, i]) for i in range(1, len(dfz.columns))]
plt.show()
# plot phi traces
[plt.semilogx(freqs, dfphi.iloc[:, i]) for i in range(1, len(dfz.columns))]
plt.show()
# plot nyquist traces
[plt.plot(dfny.iloc[:, i], dfny.iloc[:, i+1]) for i in range(1, len(dfny.columns), 2)]
plt.show()
# plot heatmaps of Z and phi
df_to_heatmap(dfz, title='Z', first_col_as_index=True)
df_to_heatmap(dfphi, title='Phi', first_col_as_index=True)















