# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:22:52 2019

@author: a6q
"""

from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
#import cv2

import matplotlib.gridspec as gridspec
ls = 16
plt.rcParams['xtick.labelsize'] = ls
plt.rcParams['ytick.labelsize'] = ls
plt.rcParams['figure.figsize'] = (6,6)


def voigt(mu_f, eta_f, rho_f, h_f):
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
                
    Output
        df = frequency change of resonator
        dd =  dissipation change of resonator
    '''
    
    # fundamental resonant frequency of crystal in Hz
    f0 = 5e6
    n = 3 # crystal harmonic number
    w = 2*np.pi*f0*n  # angular frequency
    
    mu_q = 2.947e10 #shear modulus of AT-cut quatz in Pa
    rho_q = 2648 #density of quartz (kg/m^3)
    h_q = np.sqrt(mu_q/rho_q)/(2*f0) #thickness of quartz
    
    # shear modulus and density of bulk air
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

def find_diff2D(p1, p2):
    #finds the distance between two ordered pairs of points in 2D
    #each point p1 and p2 must be an ordered pair with two elements
    xdiff = np.square(p2[0] - p1[0])
    ydiff = np.square(p2[1] - p1[1])
    diff = np.sqrt(xdiff + ydiff)
    return diff


#%% create arrays of possible values

# set size of grid in which to search for solutions
step_num = 150
print('grid contains %d points' %step_num**2)
#mu_range = np.logspace(1, 15, step_num).astype(float)
mu_range = np.linspace(1e5, 1e7, step_num).astype(float)
#eta_range = np.logspace(-12, 12, step_num).astype(float)
eta_range = np.linspace(0, 1e-6, step_num).astype(float)

exp_data = pd.read_csv('exp_data\\pedotpss_qcm_AI_data.csv')


#%% calculate delta f and delta D at each grid point
starttime = time.time()
results0 = []

time_sol = []
mu_sol= []
eta_sol = []


#iterate over each time step
for timestep in range(len(exp_data)):
    print('time step '+format(1+timestep)+' / '+format(len(exp_data)))
    df_exp = exp_data['df7'].iloc[timestep]*1e3
    dd_exp = exp_data['dd7'].iloc[timestep]
    time0 = exp_data['rh'].iloc[timestep]

    #lists for df and dd "matches"
    dfm = []
    ddm = []
    
    rho_f = 1e3
    h_f = 100e-9
    
    #----------iterate over each point in grid--------------------------
    for i in range(len(mu_range)-1):
        for j in range(len(eta_range)):
            #calculate delta f and delta d at each grid point  
            df0, dd0 = voigt(mu_range[i], eta_range[j], rho_f, h_f)
            df1, dd1 = voigt(mu_range[i+1], eta_range[j], rho_f, h_f)
            results0.append([mu_range[i], eta_range[j], df0, dd0])
            
            #check if surface intersects planes of experimental df and dd
            if df1 <= df_exp <= df0 or df0 <= df_exp <= df1:
                dfm.append([mu_range[i], eta_range[j]])
            if dd1 <= dd_exp <= dd0 or dd0 <= dd_exp <= dd1:  
                 ddm.append([mu_range[i], eta_range[j]])
                 
    dfm = np.array(dfm)    
    ddm = np.array(ddm)


    # set up multi-panel plot--------------------------------------------
    gs1 = gridspec.GridSpec(2,1)
    gs2 = gridspec.GridSpec(2,1)
    gs1.update(left=0.08, right=0.55)#, wspace=0.05)
    gs2.update(left=0.68, right=0.98)#, wspace=0.05)
    ax_int = plt.subplot(gs1[:])
    ax_mu = plt.subplot(gs2[0])
    ax_eta = plt.subplot(gs2[1])
    plt.setp(ax_mu.get_xticklabels(), visible=False)
    #plt.setp([a.get_xticklabels() for a in share_axes2[:-1]], visible=False)
    plt.subplots_adjust(hspace=0, bottom=.15, top=0.95, right=0.3, left=.1)
    #---------------------------------------------------------------------


    #--------------------------plot intersection surfaces ------------------
    if len(dfm) > 0:
        ax_int.scatter(1e-6*dfm[:,0], 1e6*dfm[:,1],
                   s=4, c='k', alpha=.4, label='$\Delta$f solution')
    if len(ddm) > 0:
        ax_int.scatter(1e-6*ddm[:,0], 1e6*ddm[:,1],
                   s=4, c='r', alpha=.7, label='$\Delta$D solution')
        
    #-------find intersection of df and dd contours at each timestep --------    
    if len(dfm) > 0 and len(ddm) > 0:
        all_diffs = []    
        for dfm_val in dfm:
            for ddm_val in ddm:
                #find point closest to both contours 
                all_diffs.append([dfm_val[0], ddm_val[1], 
                                  find_diff2D(dfm_val, ddm_val)])
    
        all_diffs = np.array(all_diffs)
        min_diff_ind = np.argmin(all_diffs[:,2])
        mu_sol0 = all_diffs[min_diff_ind][0]
        eta_sol0 = all_diffs[min_diff_ind][1]
        time_sol.append(time0)
        mu_sol.append(mu_sol0)
        eta_sol.append(eta_sol0)

        #plot current intersection point
        ax_int.scatter(1e-6*mu_sol0, 1e6*eta_sol0, s=150,
                       label='intersection', marker='*', c='g')


        #plot trajectory of intersection point
        #if len(intersect_mu) > 1:
        ax_int.scatter(1e-6*np.array(mu_sol), 1e6*np.array(eta_sol),
                    label='trajectory', s=5, c='b')
    
        #plot eta and mu over time
        ax_mu.plot(time_sol, 1e-6*(np.array(mu_sol) - mu_sol[0]))
        ax_mu.scatter(time_sol, 1e-6*(np.array(mu_sol)- mu_sol[0]))
        #ax_mu.set_xlabel('RH (%)', fontsize=ls)
        ax_mu.set_ylabel('$\Delta\mu$ (MPa)', fontsize=ls)
        
        
        ax_eta.plot(time_sol, 1e6*(np.array(eta_sol)-eta_sol[0]))
        ax_eta.scatter(time_sol, 1e6*(np.array(eta_sol)-eta_sol[0]))
        ax_eta.set_xlabel('Time', fontsize=ls)
        ax_eta.set_ylabel('$\Delta\eta$ ($\mu$Pa s)', fontsize=ls)
        #ax_eta.set_xlim([0, 100])
        #ax_mu.set_xlim([0, 100])
    
          
        #format plot
        #ax_int.set_xlim([1,2.2])
        #ax_int.set_ylim([0,140])
        #ax_int.set_yscale('log')
        #ax_int.set_xscale('log')  
        ax_int.set_xlabel('$\mu$ (MPa)', fontsize=ls)
        ax_int.set_ylabel('$\eta$ ($\mu$Pa s)', fontsize=ls)
        lgnd = ax_int.legend(loc='upper right', fontsize=ls)
    
        lgnd.legendHandles[0]._sizes = [20]
        lgnd.legendHandles[1]._sizes = [20]
        lgnd.legendHandles[2]._sizes = [60]
        lgnd.legendHandles[3]._sizes = [20]
        
        #ax_int.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        #ax_int.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #ax_mu.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    
    
        #save plot as image file          
        #plt.tight_layout()
        fig0 = plt.gcf() # get current figure
        fig0.set_size_inches(12, 6)
    
        '''
        #save plot as image file  
        if rh0 < 10:
            save_pic_filename = 'exp_data\\save_figs2\\fig_0'+format(rh0)+'.jpg'
        else:
            save_pic_filename = 'exp_data\\save_figs2\\fig_'+format(rh0)+'.jpg'
        plt.savefig(save_pic_filename, format='jpg', dpi=150)
    
        '''
        
        plt.show()
    
        #close figure from memory
        plt.close(fig0)
        #close all figures from memory
        plt.close('all')
    

