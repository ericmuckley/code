# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 16:38:42 2018

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

# find setpoint changes in the array of setpoints, which allows use
# of "setpoint difference" and "relative time" inputs for NN model

def find_setpoint_changes(setpoint0):
    #find indices of setpoint changes
    ei = np.array([i for i in range(len(setpoint0)-1) if setpoint0[i] !=
                   setpoint0[i+1]])
    
    #find values of each setpoint
    setpoint_vals = [k for k,g in groupby(last_file['pressure'])]
    
    #find changes in each setpoint
    setpoint_diff_vals = np.insert(np.ediff1d(setpoint_vals),0,0)
    
    #find length of each setpoint and add initial and final setpoint lengths
    step_len = np.append(np.insert(np.ediff1d(ei),0,ei[0]),
                         len(setpoint0) - ei[-1])

    #create array of setpoint differences
    setpoint_diff = np.array([])
    rel_time= np.array([])
    for i in range(len(step_len)):
        setpoint_diff = np.append(setpoint_diff,
                                  np.repeat(setpoint_diff_vals[i],
                                            step_len[i]))
        
        rel_time= np.append(rel_time, np.arange(step_len[i]))
        
    #returns arrays of relative times and setpoint changes    
    return rel_time, setpoint_diff 



#%%

def seq_to_mat(seq_in, tot_pnts, var_num):
    
    '''
    Creates a matrix out of 2-column sequence data: [times, pessures].
    Outputs: [times, relative times, pressures, relative pressures].
    Output contains var_num number of columns each with tot_pnts length.
    '''
    
    #populate output matrix
    tot_time = np.sum(seq_in[:,0])
    seq_out = np.zeros((tot_pnts+2,var_num))
    
    #absolute time, and time step dt
    abs_t, dt = np.linspace(0, tot_time, tot_pnts, retstep=True)
    seq_out[:(len(abs_t)),0] = abs_t
    
    abs_p_col = np.array([])
    rel_p_col = np.array([])
    rel_t = np.array([])
    step_lengths = []
    
    #loop over each step in sequence
    for i in range(len(seq_in)):
        #length of current step, in data points
        step_length = int(np.around(tot_pnts * seq_in[i,0] / tot_time))
        step_lengths.append(step_length)
        
        #store values for each column of output matrix
        rel_t = np.append(rel_t, abs_t[:step_length])
        
        #absolute pressure change
        abs_p_col = np.append(abs_p_col, np.full(step_length, seq_in[i,1]))
        #relative presure change
        rel_p = np.ediff1d(seq_in[:,1])[i-1]
        rel_p_col = np.append(rel_p_col, np.full(step_length, rel_p))
    
    #first points of pressure change should = 0 
    rel_p_col[:step_lengths[0]] = np.full(step_lengths[0], 0)
    
    #populate output matrix
    seq_out[:len(rel_t),1] = rel_t
    seq_out[:len(abs_p_col),2] = abs_p_col
    seq_out[:len(rel_p_col),3] = rel_p_col
    
    #fix rounding which causes wrong step size by removing last point
    seq_out = seq_out[:tot_pnts-2,:]
    
    return seq_out


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


#%% simulate a signal
'''
a0 = 0.1
tau = 30
drift_factor = -.00001
sig_offset = -5

noise = 10*np.random.rand(len(seq_mat))
sim_sig0 = noise + simulate_signal(seq_mat, a0, tau, drift_factor, sig_offset)
'''

#%% import file
NN_output_folder = glob.glob('C:\\Users\\a6q\\NN_output_2018-03-16/*')
NN_output_folder.sort(key=os.path.getmtime)

# look at last output file to get raw time, pressure, signal 
last_file_raw = pd.read_csv(NN_output_folder[-1])

#find lenth of non-nan points so we can remove all nans at the end of files
new_len = last_file_raw['signal'].notnull().sum() - 10

last_file = last_file_raw.iloc[0:new_len-15000]
time = last_file['time']/60
pressure = last_file['pressure']
signal = last_file['signal']+100



#%% create matrix out of sequence data:

#construct seq_mat out of time and pressure data
rel_time0, setpoint_diff0 = find_setpoint_changes(pressure)

seq_mat = np.array([np.array(time),
            rel_time0,
            np.array(pressure),
            setpoint_diff0]).T
  


time0 = timer()



#%% fit a signal to exponential decay model
fit_guess = [1e-2, 20, -3e-4, signal[0]]
lowbounds = [-100, 0.01, -10, -1000]
highbounds = [100, 1000, 10, 1000]
fit_bounds = (lowbounds, highbounds)


#fit the signal to exponential decay model
fit_params, fit_errors = curve_fit(simulate_signal, seq_mat, signal,
                                   p0=fit_guess, bounds=fit_bounds)

print('fit params (a0, tau, drift, offset) = '+format(fit_params))

#calculate fit line
fit_line = simulate_signal(seq_mat, *fit_params)

plt.plot(seq_mat[:,0], fit_line, linewidth=2, c='k', label='fit')
plt.scatter(seq_mat[:,0], signal, s=2, c='r', label='signal')
plt.plot(seq_mat[:,0], seq_mat[:,2], label='pressure')
plt.legend()
plt.show()


time1 = timer()
print('total time = '+format(time1-time0)+' sec')


#%% add fit line to sequence

df_export = pd.DataFrame(seq_mat)
df_export.columns = ['abs_time', 'rel_time', 'abs_pressure', 'rel_presure']

df_export['signal'] = signal
df_export['fit_line'] = fit_line

df_export.to_csv('exported_NN_fit2.csv', index=False)