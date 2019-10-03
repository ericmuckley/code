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
    print('%i rows' % len(df))
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


def h5store(filename, key, df, metadata={}):
    '''Store pandas dataframes into an HDF5 file using a key and metadata.
    Metadata can be a dictionary containing different values. Unpack the
    metadata and connect ot the HDF keys using h5load and
    h5metadata functions.'''
    store = pd.HDFStore(filename)
    store.put(key, df)
    store.get_storer(key).attrs.metadata = metadata
    store.close()


def h5load(filename, key):
    '''Retrieve data and metadata from an HDF5 file using a key'''
    with pd.HDFStore(filename) as store:
        data = store[key]
        metadata = store.get_storer(key).attrs.metadata
    return data, metadata


def h5metadata(filename):
    '''Create dictionary which ocnnects HDF5 keys and their metadata.'''
    h5_md = {}
    for h5_key in pd.HDFStore(filename).keys():
        # get dataframe and metadata from file
        df, md = h5load(filename, h5_key)
        h5_md[h5_key] = {}
        for md_key in md:
            h5_md[h5_key][md_key] = md[md_key]
    return h5_md


def sample_evenly(array, n=3):
    ''''Sample evenly spaced n elements from an array including endpoints.
    Returns the sampled array elements and their indices.'''
    m = len(array)
    increment = int(m / (n - 1))
    inds = [0]
    [inds.append(i * increment) for i in range(1, n-1)]
    inds.append(len(array) - 1)
    return array[inds], inds


# set fontsize on plots
fs = 12

#%% import data from HDF file, one key at a time
filename = r'exp_data\voigt_100.h5'

# create dictionary connecting HDF keys and their metadata
h5md = h5metadata(filename)

#%%

hlist = np.unique([h5md[key]['h']for key in h5md])
rholist = np.unique([h5md[key]['rho']for key in h5md])

#hsamples, _ = sample_evenly(hlist, 5)
hsamples = hlist[[0, 3, 6, 9]]
rhosamples = rholist[[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]]



#%% loop over dataframes and create plots

for rho in rhosamples:

    fig = plt.figure(figsize=(14, 6))
    
    
    # loop over film thicknesses (left to right)
    for h_i, h in enumerate(hsamples):
        
        # get the dict key which corresponds to the rho and h values
        key = [key for key, md in h5md.items() if md == {
                'rho':rho, 'h': h}][0]
        
        # get dataframe and metadata from file
        df, _ = h5load(filename, key)
        #rho, h = md['rho'], md['h']
        print('----------------------------------------------------------')
        print('rho: %0.2e, h: %0.2e' % (rho, h))
        #examine_df(df)
    
    
    
    
        z_var = 'dd'
        #df = df[df['df'] >= -5000]
        #df = df[df['df'] <= 1000]
        x = np.log10(df['mu'])
        y = np.log10(df['eta'])
        z = df[z_var].values
    
        ax = fig.add_subplot(2, len(hsamples), h_i+1, projection='3d')
        ax.set_xlabel('\nLog($\mu$) (Pa)', fontsize=fs)
        ax.set_ylabel('\nLog($\eta$) (Pa s)', fontsize=fs)
        
        title_str = str(int(h*1e9))+' nm, '+str(int(rho))+' g/cm$^3$'
        plt.title(title_str, fontsize=fs)
        # remove fill
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # set color to white
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        # set axis limits
        ax.set_xlim3d(np.min(x), np.max(x))
        ax.set_ylim3d(np.min(y), np.max(y))
        ax.set_zlim3d(0, 400)
        ax.set_zlabel('\n\n$\Delta$D (x10$^{-6}$)', fontsize=fs)
        ax.dist = 13
        #surf = ax.plot_wireframe(x, y, z, linewidth=0.2)
        surf = ax.plot_trisurf(x, y, z, linewidth=0, cmap=cm.jet) 
        #fig.colorbar(surf, shrink=0.8, aspect=10)
        #plt.savefig('teste.pdf')
        
        
        
        
        
        
        
        
        


        z_var = 'df'
        #df = df[df['df'] >= -5000]
        #df = df[df['df'] <= 1000]
        x = np.log10(df['mu'])
        y = np.log10(df['eta'])
        z = df[z_var].values
    
        ax = fig.add_subplot(2, len(hsamples),
                             h_i+1+len(hsamples), projection='3d')
        ax.set_xlabel('\nLog($\mu$) (Pa)', fontsize=fs)
        ax.set_ylabel('\nLog($\eta$) (Pa s)', fontsize=fs)
        
        #title_str = str(int(rho))+' g/cm$^3$, '+str(int(h*1e9))+' nm'
        #plt.title(title_str, fontsize=fs)
        # remove fill
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # set color to white
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        # set axis limits
        ax.set_xlim3d(np.min(x), np.max(x))
        ax.set_ylim3d(np.min(y), np.max(y))
        ax.set_zlim3d(-800, 100)
        ax.set_zlabel('\n\n$\Delta$f (Hz/cm$^{2}$)', fontsize=fs)
        ax.dist = 13
        #surf = ax.plot_wireframe(x, y, z, linewidth=0.2)
        surf = ax.plot_trisurf(x, y, z, linewidth=0, cmap=cm.jet) 
        #fig.colorbar(surf, shrink=0.8, aspect=10)
        #plt.savefig('teste.pdf')      
        
        
        
        
        
        
        
    plotfilename = 'voigt_surf_plots\\rho='+str(int(rho))+'.jpg' 
    plt.tight_layout()
    fig.savefig(plotfilename, dpi=120, bbox_inches='tight')
    #plt.show()
    
        
pd.HDFStore(filename).close()

#%%
'''
        fig = plt.figure(figsize=(8,5))
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(uniquedf['mu'], uniquedf['eta'])
        Z = df['df'].values.reshape(X.shape)
        
        ax.plot_surface(np.log10(X), np.log10(Y), Z,
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
        plt.title(str(rho)+' g/cm$^3$, '+str(int(h*1e9))+' nm', fontsize=fs)
        
        # remove fill
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        # set color to white
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')
        # set axis limits
        #ax.set_xlim3d(np.min(X), np.max(X))
        #ax.set_ylim3d(np.min(Y), np.max(Y))
        #ax.set_zlim3d(-10000,000)
        
        
        # Bonus: To get rid of the grid as well:
        ax.grid(False)
        
        
        #plotname = 'exp_data\\voinova_3dplots\\voinova_3d_plot_'+str(i).zfill(2)+'.jpg'
        #fig.savefig(plotname, dpi=120, bbox_inches='tight')
    
        plt.show()
'''
    


#%%

'''
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