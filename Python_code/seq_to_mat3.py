# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 14:39:47 2018

@author: a6q
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from itertools import groupby
from timeit import default_timer as timer


#%%
def seq_to_mat(seq0, points_per_minute=6):
    '''
    Creates a matrix out of 2-column sequence data: [times, pessures].
    Outputs: [times, relative times, pressures, relative pressures,
    net pressure change since initial step].
    Outputs 2D numpy array and Pandas dataframe populated with 
    'points_per_minute' number of points per minute of the original sequence.
    '''

    #initialize values for each column##########################################
    #absolute time
    abs_t = np.linspace(0, np.sum(seq0[:,0]),
                        num = np.multiply(np.sum(seq0[:,0]),
                                          points_per_minute).astype(int))
    #relative time (time since last pressure change)
    rel_t = np.zeros((np.sum(seq0[0,0])*points_per_minute).astype(int))
    #initial pressure
    abs_p = np.ones_like(rel_t) * seq0[0,1]
    #relative pressure (pressure difference since last pressure change)
    rel_p = np.zeros_like(rel_t)
    #delta pressure (net change in pressure since initial pressure)
    net_delta_p = np.zeros_like(rel_t)
    
    #loop over steps in sequence and populate columns###########################
    for i in range(1, len(seq0)):
        
        #numper of points to append during this step
        point_num = np.multiply(seq0[i,0], points_per_minute).astype(int)
        
        #append each column with appropriate values
        rel_t = np.append(rel_t, np.arange(point_num))
        abs_p = np.append(abs_p, np.full(point_num, seq0[i,1]))
        rel_p = np.append(rel_p, np.full(point_num, seq0[i,1] - seq0[i-1,1]))
        net_delta_p = np.append(net_delta_p, 
                                np.full(point_num, seq0[i,1] - abs_p[0]))
        
    #put all columns toether
    seq_mat = np.array([abs_t, rel_t, abs_p, rel_p, net_delta_p]).T
    
    #construct dataframe to display results
    seq_mat_df = pd.DataFrame(seq_mat, 
                          columns=['abs_time', 'rel_time', 'abs_pressure',
                                   'rel_pressure', 'net_delta_p'])
    return seq_mat, seq_mat_df

#%% simulate expected signal using pressure sequence data

def simulate_signal(seq_mat, a0, tau, drift_factor, sig_offset):
    '''
    Creates simulated sensor response using sequence matrix:
    [abs_time, rel_time, abs_pressure, rel_pressure]
    
    Fit parameters for the exponential decays are: [tau (time constant),
    a0 (amplitude a.k.a. pre-exponential factor), 
    sig_offset (vertical offset of exponential),
    drift_factor (scaling factor for linear drift)]
    '''
    #build simulated signal from seq_out matrix
    sim_sig = np.array([0])
    for i in range(1, len(seq_mat)):
        sim_sig = np.append(sim_sig, sig_offset
                            + drift_factor*seq_mat[i,0] 
                            - a0*np.exp(-(seq_mat[i,1])/tau))*np.sign(seq_mat[i,3])
    return sim_sig


#%% get sequence matrix

#seq0 = pd.read_table('sample_seq.txt', header=None).values
seq0 = np.array([[10,2], [10,60], [10,2], [10,40], [10,60], [10,20], [10,2]])

seq_mat, seq_df = seq_to_mat(seq0, points_per_minute=6)

    
#%% calculate exponential signal

#build simulated signal from seq_out matrix

drift = 0.01
amp = 0.5
tau = 10
offset = 5

#offset + drift*seq_df['abs_time'] +
    
exp_sig0 =  np.sign(seq_df['rel_pressure'])*seq_df['abs_pressure'] * (1 - np.exp(-seq_df['rel_time']/tau))



plt.plot(seq_mat[:,0], exp_sig0)
plt.plot(seq_mat[:,0], seq_mat[:,2], linewidth=1)
plt.show()






