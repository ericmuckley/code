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


#%%
filename = r'C:\Users\a6q\exp_data\2019-10-09-WS2-5layer-wrinkled slow\2019-10-09_16-34__eis.csv' 

df = pd.read_csv(filename).dropna()

# create dataframes for each variable
dfz = pd.DataFrame({'frequency': df.iloc[:, 0]})
dfphi = pd.DataFrame({'frequency': df.iloc[:, 0]})
dfny = pd.DataFrame({'frequency': df.iloc[:, 0]})
  

# loop over each column and build smaller dataframes for each variable
# here we assume the columns are [freq, Z, phi, reZ, imZ]
for i in range(0, len(df.columns), 5):
    # impedance data
    dfz['z_'+str(int((i+5)/5)).zfill(3)] = df.iloc[:, i + 1]
    # phase data
    dfphi['phi_'+str(int((i+5)/5)).zfill(3)] = df.iloc[:, i + 2]
    # nyquist data
    dfny['rez_'+str(int((i+5)/5)).zfill(3)] = df.iloc[:, i + 3]
    dfny['imz_'+str(int((i+5)/5)).zfill(3)] = df.iloc[:, i + 4]


#%% plot results
[plt.loglog(dfz.iloc[:, 0], dfz.iloc[:, i]) for i in range(1, len(dfz.columns))]
plt.show()
    
[plt.semilogx(dfphi.iloc[:, 0], dfphi.iloc[:, i]) for i in range(1, len(dfz.columns))]
plt.show()


for i in range(1, len(dfny.columns), 2):
    plt.plot(dfny.iloc[:, i], dfny.iloc[:, i+1])
plt.show()


df_to_heatmap(dfz, title='Z', first_col_as_index=True)
df_to_heatmap(dfphi, title='Phi', first_col_as_index=True)
