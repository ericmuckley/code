# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:50:52 2019

@author: a6q
"""

from scipy import interpolate
from scipy.interpolate import bisplrep
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_setup(labels=['X', 'Y'], size=16, setlimits=False,
               limits=[0,1,0,1], scales=['linear', 'linear'],
               title='', save=False, filename='plot.jpg'):
    ''' This can be called with Matplotlib for setting axes labels, setting
    axes ranges and scale types, and  font size of plot labels. Function
    should be called between plt.plot() and plt.show() commands.'''
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


def examine_df(df):
    ''' Examine min and max of each column in a Pandas DataFame.'''
    print('reading %i rows...' % len(df))
    for col in df:
        print('%s ---> min, max, mean: %0.3e, %0.3e, %0.3e' % (
                col, df[col].min(), df[col].max(), df[col].mean()))


def get_unique_df(df, cols=[]):
    ''' Get unique values of each column in cols in a Pandas DataFrame.'''
    npoints = len(np.unique(df[df.columns[0]]))
    uniquearr = np.empty((npoints, 0))
    for col in cols:
        uniquearr = np.column_stack((uniquearr, np.unique(df[col])))
    uniquedf = pd.DataFrame(columns=cols, data=uniquearr)
    return uniquedf




#%% import data
df = pd.read_hdf('voigt_surf90.hdf', key='all')

#%% examine the data
examine_df(df)
uniquedf = get_unique_df(df, cols=['mu', 'eta', 'rho', 'h'])

#%%


x, y = np.log10(uniquedf['mu']), np.log10(uniquedf['eta'])


df0 = df[df['rho'] == uniquedf['rho'].iloc[20]]
df0 = df0[df0['h'] == uniquedf['h'].iloc[20]]

# create grid from x and y values
X, Y = np.meshgrid(x, y)

# designate z values and reshape to match grid shape
zs = df0['df']
Z = zs.reshape(X.shape)

plt.pcolormesh(X, Y, Z)#, vmin=0, vmax=10)
plt.colorbar()
plt.show()


'''
df0 = df[df['rho'] == uniquedf['rho'].iloc[0]]
df0 = df0[df0['h'] == uniquedf['h'].iloc[0]]


spine2d = bisplrep(df0['mu'], df0['eta'], df0['df'])

interp = interpolate.interp2d(df0['mu'], df0['eta'], df0['df'], kind='cubic')

counter=0

for rho0 in uniquedf['rho']:
    for h0 in uniquedf['h']:

counter+=1
'''  
    

#%% plot results

'''
results2 = results[results['h'] == hlist[5]]

plotting = False

if plotting:
    
    # set fontsize on plots
    fs = 16

    for i, rho in enumerate(rholist[5:10]):
    
        rdf = results2[results2['rho'] == rho]
        
        
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(mulist, etalist)
        zs = rdf['dd']
        Z = zs.reshape(X.shape)
        
        ax.plot_surface((X), (Y), Z,
                        cmap=cm.jet, linewidth=0.1, alpha=0.7,
                        vmin=np.amin(Z), vmax=np.amax(Z))
    
        ax.set_xlabel('$\mu$ (Pa)', fontsize=fs)
        ax.set_ylabel('$\eta$ (Pa s)', fontsize=fs)
        ax.set_zlabel('$\Delta$f (Hz/cm$^2$)', fontsize=fs)
        #ax.set_yticks(np.log(yticks))
        #ax.xaxis._set_scale('log')
        # rotate the axes
        ax.view_init(30, -40)
        
        plt.rcParams['xtick.labelsize'] = fs 
        plt.rcParams['ytick.labelsize'] = fs
        plt.title('density: '+str(rho), fontsize=fs)
        
        # First remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        
        # Bonus: To get rid of the grid as well:
        ax.grid(False)
        
        
        plotname = 'exp_data\\voinova_3dplots\\voinova_3d_plot_'+str(i).zfill(2)+'.jpg'
        #fig.savefig(plotname, dpi=120, bbox_inches='tight')
    
        plt.show()


'''