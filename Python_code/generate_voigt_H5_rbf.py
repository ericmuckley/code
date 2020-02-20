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
        deltaf = frequency change of resonator
        deltad =  dissipation change of resonator
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
    deltaf = float(np.imag((beta-beta0)/(2*np.pi*rho_q*h_q)))
    deltad = float(-np.real((beta-beta0)/(np.pi*f0*n*rho_q*h_q))*1e6)
    return deltaf, deltad


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

def create_2dcombos(arr1, arr2):
    '''Create an array of all possible combinations of elements of
    each array.'''
    return np.array(np.meshgrid(arr1, arr2)).T.reshape(-1, 2)
 
def create_4dcombos(arr1, arr2, arr3, arr4):
    '''Create an array of all possible combinations of the elements of
    4 arrays.'''
    return np.array(np.meshgrid(arr1, arr2, arr3, arr4)).T.reshape(-1, 4)


#%% USER INPUTS

# set size of grid in which to build surfaces
if len(sys.argv) > 1:
    step_num = sys.argv[1]
else: 
    step_num = 25
      
filename = r'exp_data\voigt_rfb_'+str(step_num)+'.h5'

# film density
rholist = np.linspace(500, 2000, step_num).astype(float)
# film thickness
hlist = np.linspace(0.01e-6, 1e-6, step_num).astype(float)
# shear modulus
mulist = np.logspace(1, 6, step_num).astype(float)
# viscosity
etalist = np.logspace(1, -6, step_num).astype(float)
# get array of all combinations of input variables
combos = create_4dcombos(mulist, etalist, rholist, hlist)

#%%

start_time = time.time()
tot_points = len(mulist)*len(etalist)*len(rholist)*len(hlist)
print('grid contains %d points' %tot_points)

row = 0
df_num = 0

#%%

arr = np.empty((len(combos), 2))

# loop over each combination of variables and calculate dF and dD
for row_i, row in enumerate(combos):
    deltaf, deltad = voigt(row[0], row[1], rho_f=row[2], h_f=row[3])
    arr[row_i] = [deltaf, deltad]



'''
# loop over each thickness and density
for rho0 in rholist:
    for h0 in hlist:
        
        # initialize empty array to hold results
        arr = np.empty((step_num**2, 4))
        
        # create all combinations mu and eta
        mu_eta_combos = create_2dcombos(mulist, etalist)
        
        # loop over each mu and eta combo and calculate dF and dD
        for row_i, row in enumerate(mu_eta_combos):
            deltaf, deltad = voigt(row[0], row[1], rho_f=rho0, h_f=h0)
            arr[row_i] = list(row) + [deltaf, deltad]

        # radial basis function interpolation of data
        #linint = LinearNDInterpolator(arr[:, :2], arr[:, 2:])

        sbs = SBS(arr[:, 0], arr[:, 1], arr[:, 3])
        
        # store results in a dataframe
        df0 = pd.DataFrame(data=arr, columns=['mu', 'eta', 'df', 'dd'])

'''
'''        
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

'''
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



