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

def df_to_heatmap(df, vmin=0, vmax=100, fontsize=14, title=None,
                  save=False, filename='fig.jpg'):
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
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()
    plt.show()


filename = r'C:\Users\a6q\exp_data\2019-10-02_WS2_5layer_wrinkled\2019-10-02_15-44__eis.csv' 
df = pd.read_csv(filename).dropna()

# create dataframes for each variable
dfz = pd.DataFrame({'frequency': df.iloc[:, 0]})
dfphi = pd.DataFrame({'frequency': df.iloc[:, 0]})
dfny = pd.DataFrame({'frequency': df.iloc[:, 0]})


# smooth columns
for i in range(0, len(df.columns)):
    # dont smooth frequency columns
    if i % 5 != 0:
        df.iloc[:, i] = medfilt(df.iloc[:, i].values, kernel_size=7)

        # get spline parameters
        #spline_params = splrep(df.iloc[:, 0], df.iloc[:, i], k=3, s=1e12)
        # calculate spline at new x values
        #df.iloc[:, i] = splev(df.iloc[:, 0], spline_params)
        

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


'''
# median filtering of 2D arrays
dfz = pd.DataFrame(data=medfilt2d(dfz.values, kernel_size=3),
                   columns=dfz.columns)

dfphi = pd.DataFrame(data=medfilt2d(dfphi.values, kernel_size=3),
                   columns=dfphi.columns)
'''



# plot results
[plt.loglog(dfz.iloc[:, 0], dfz.iloc[:, i]) for i in range(1, len(dfz.columns))]
plt.show()
    
[plt.semilogx(dfphi.iloc[:, 0], dfphi.iloc[:, i]) for i in range(1, len(dfz.columns))]
plt.show()


for i in range(1, len(dfny.columns), 2):
    plt.plot(dfny.iloc[:, i], dfny.iloc[:, i+1])
plt.show()


df_to_heatmap(dfz, vmin=100, vmax=2e6, title='Z')
df_to_heatmap(dfphi, vmin=-80, vmax=0, title='Phi')
