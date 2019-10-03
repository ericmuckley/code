# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:22:52 2019

@author: a6q
"""

#from scipy.interpolate import griddata
#from matplotlib.colors import ListedColormap
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import sys

'''
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
'''

def voigt(mu_f, eta_f, rho_f, h_f=1e-6, n=1, f0=5e6):
    ''' 
    The Voinova equations come from eqns (15) in the paper by 
    Voinova: Vionova, M.V., Rodahl, M., Jonson, M. and Kasemo, B., 1999.
    Viscoelastic acoustic response of layered polymer films at fluid-solid
    interfaces: continuum mechanics approach. Physica Scripta, 59(5), p.391.
    Reference: https://github.com/88tpm/QCMD/blob/master
    /Mass-specific%20activity/Internal%20functions/voigt_rel.m
    Solves for Delta f and Delta d of thin adlayer on quartz resonator.
    Differs from voigt because calculates relative to unloaded resonator.
    Input
        mu_f = shear modulus of film in Pa
        eta_f = shear viscosity of film in Pa s
        rho_f = density of film in kg m-3
        h_f = thickness of film in m
        n = crystal harmonic number
        f0 = fundamental resonant frequency of crystal in Hz      
    Output
        df = frequency change of resonator
        dd =  dissipation change of resonator
    '''
    w = 2*np.pi*f0*n  # angular frequency
    mu_q = 2.947e10 # shear modulus of AT-cut quatz in Pa
    rho_q = 2648 # density of quartz (kg/m^3)
    h_q = np.sqrt(mu_q/rho_q)/(2*f0) #t hickness of quartz
    
    rho_b = 1.1839 #density of bulk air (25 C) in kg/m^3
    eta_b = 18.6e-6 #viscosity of bulk air (25 C) in Pa s
    # rho_b = 1000 #density of bulk water in kg/m^3
    # eta_b = 8.9e-4 #viscosity of bulk water in Pa s
    
    # eqn 14
    kappa_f = eta_f-(1j*mu_f/w)
    # eqn 13
    x_f = np.sqrt(-rho_f*np.square(w)/(mu_f + 1j*w*eta_f))
    x_b = np.sqrt(1j*rho_b*w/eta_b)
    # eqn 11 after simplification with h1 = h2 and h3 = infinity
    A = (kappa_f*x_f+eta_b*x_b)/(kappa_f*x_f-eta_b*x_b)
    # eqn 16
    beta = kappa_f*x_f*(1-A*np.exp(2*x_f*h_f))/(1+A*np.exp(2*x_f*h_f))
    beta0 = kappa_f*x_f*(1-A)/(1+A)
    # eqn 15
    df = np.imag((beta-beta0)/(2*np.pi*rho_q*h_q))
    dd = -np.real((beta-beta0)/(np.pi*f0*n*rho_q*h_q))*1e6
    return [df, dd]


def h5store(filename, key, df, **kwargs):
    '''Store pandas dataframes into an HDF5 file using a key and metadata'''
    store = pd.HDFStore(filename)
    store.put(key, df)
    store.get_storer(key).attrs.metadata = kwargs
    store.close()


def h5load(filename, key):
    '''Retrieve data and metadata from an HDF5 file using a key'''
    with pd.HDFStore(filename) as store:
        data = store[key]
        metadata = store.get_storer(key).attrs.metadata
        store.close()
    return data, metadata

#%% USER INPUTS

# set size of grid in which to search for solutions
if len(sys.argv) > 1:
    step_num = sys.argv[1]
else: 
    step_num = 30
      
filename = r'exp_data\voigt_'+str(step_num)+'.h5'

# film density
rholist = np.linspace(500, 2000, step_num).astype(float)
# film thickness
hlist = np.linspace(0.01e-6, 1e-6, step_num).astype(float)
# shear modulus
mulist = np.logspace(0, 8, step_num).astype(float)
# viscosity
etalist = np.logspace(1, -5, step_num).astype(float)


#%%
# mulist = np.linspace(1e4, 1e9, step_num).astype(float)
# etalist = np.linspace(0, 1e-6, step_num).astype(float)

start_time = time.time()
tot_points = len(mulist)*len(etalist)*len(rholist)*len(hlist)
print('grid contains %d points' %tot_points)

row = 0
df_num = 0

# loop over each thickness and density
for rho0 in rholist:
    for h0 in hlist:
        
        # initialize empty array to hold results
        arr = np.empty((step_num**2, 4))
        df_row = 0
        
        # loop over each viscosity and modulus and calculate and dF and dD
        for mu0 in mulist:
            for eta0 in etalist:

                calc = voigt(mu0, eta0, rho_f=rho0, h_f=h0)
                arr[df_row] = [mu0, eta0] + calc
                
                row += 1
                df_row += 1
                
        # store results in a dataframe
        df0 = pd.DataFrame(
                data=arr,
                columns=['mu', 'eta', 'df', 'dd'])
        
        # store dataframe with rho and h metadata
        h5store(filename, 's'+str(df_num).zfill(9),
                df0, **{'rho': rho0, 'h': h0})

        df_num += 1 

        if df_num % 100 == 0:
            print('sample %i / %i' %(row, tot_points))
            			
            runtime = time.time() - start_time
            print('runtime: = %i sec (%0.2f min)' %(
                int(runtime), float(runtime/60)))
        
pd.HDFStore(filename).close()             

runtime = time.time() - start_time
print('total runtime = %i sec (%0.2f min)' %(
        int(runtime), float(runtime/60)))
print('grid contains %d points' %tot_points)


#%% inspect irrelevant samples
'''
baddf = results[ results['dd'] > 100]
print('bad eta range: %0.2e, %0.2e' % (baddf['eta'].min(), baddf['eta'].max()))
print('bad mu range: %0.2e, %0.2e' % (baddf['mu'].min(), baddf['mu'].max()))
print('bad rho range: %0.2e, %0.2e' % (baddf['rho'].min(), baddf['rho'].max()))
print('bad h range: %0.2e, %0.2e' % (baddf['h'].min(), baddf['h'].max()))
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