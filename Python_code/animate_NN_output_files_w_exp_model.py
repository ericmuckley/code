# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:20:12 2018
@author: a6q
"""
import os, csv, glob, numpy as np, pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
label_size = 20 #make size of axis tick labels larger
plt.rcParams['xtick.labelsize'] = label_size 
plt.rcParams['ytick.labelsize'] = label_size

from scipy.optimize import curve_fit

from itertools import groupby

import PIL.Image
import imageio
def register_extension(id, extension):
    PIL.Image.EXTENSION[extension.lower()] = id.upper()
PIL.Image.register_extension = register_extension
def register_extensions(id, extensions):
    for extension in extensions:
        register_extension(id, extension)
PIL.Image.register_extensions = register_extensions


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
        sim_sig = np.append(sim_sig, sim_sig[i-1]
        + drift_factor*np.sqrt(seq_mat[i,0])
        - a0*seq_mat[i,3]*np.exp(-(seq_mat[i,1])/tau))
        
    return sim_sig + sig_offset





#%% create matrix out of sequence data:
seq_raw = pd.read_table('2018-03-16_RH_SEQUENCE.txt', header=None).values   
tot_pnts = 600
var_num = 4

seq_mat = seq_to_mat(seq_raw, tot_pnts, var_num)

  



#%% import NN output data files
NN_output_folder = glob.glob('C:\\Users\\a6q\\NN_output_2018-03-16/*')[0::190]
NN_output_folder.sort(key=os.path.getmtime)
print('found ' + format(len(NN_output_folder)) + ' NN output files') 




#%% look at last output file to get raw time, pressure, signal 
last_file_raw = pd.read_csv(NN_output_folder[-1])

#find lenth of non-nan points so we can remove all nans at the end of files
new_len = last_file_raw['signal'].notnull().sum() - 10

last_file = last_file_raw.iloc[0:new_len]
time = last_file['time']/60
pressure = last_file['pressure']
signal = last_file['signal']


#%% set fit guess bounds
fit_guess = [1, 10, 0, 0]
lowbounds = [-1000, 0.01, -1e-5, -1000]
highbounds = [1000, 1000, 1e-5, 1000]
fit_bounds = (lowbounds, highbounds)


#%% loop over every NN output file

for i in range(len(NN_output_folder)):
    
    
    #ANN prediction#######################################
    
    NN_output_data_full = pd.read_csv(NN_output_folder[i])
    NN_output_data = NN_output_data_full.iloc[0:new_len]
    
    model = NN_output_data['model']
    prediction = NN_output_data['prediction']
    error = NN_output_data['error']
    temp_signal = NN_output_data['signal']
        
    #find magnitude of largest signal
    signal_mag = np.amax(np.abs(signal))
    
    #calculate total error for each model/prediction
    model_plus_pred = np.add(np.nan_to_num(model), np.nan_to_num(prediction))
    deviation_raw = np.subtract(model_plus_pred, signal)
    deviation_percent = 100*np.divide(deviation_raw, signal_mag)
    
    
    
    
    
    
    
    #exponential decay simulated fit###########################
    
    #construct seq_mat out of time and pressure data
    rel_time0, setpoint_diff0 = find_setpoint_changes(pressure)
    
    seq_mat0 = np.array([np.array(time),
                rel_time0,
                np.array(pressure),
                setpoint_diff0]).T
    
    
    #designate with signal we are fitting and remove nans
    temp_signal_short = temp_signal[~pd.isnull(temp_signal)]

    
    
    #find fit params to exponential decay model
    fit_params, fit_errors = curve_fit(simulate_signal, 
                                       seq_mat0[:len(temp_signal_short)],
                                       temp_signal_short,
                                       p0=fit_guess,
                                       bounds=fit_bounds,
                                       ftol=1e-3, xtol=1e-3)
    
    print('fit params (a0, tau, drift, offset) = '+format(fit_params))
    
    #calculate fit line
    sim_fit = simulate_signal(seq_mat0, *fit_params)
    
    #find error between fitted and actual signal
    deviation_raw_fit = np.subtract(sim_fit, signal)
    
    deviation_percent_fit = 100*np.divide(deviation_raw_fit, signal_mag)
        
    
    
    '''
    plt.plot(seq_mat0[:,0], sim_fit, linewidth=2, c='k', label='fit')
    
    plt.scatter(seq_mat0[:len(temp_signal_short),0],
                         temp_signal_short, s=2, c='r', label='exp.')
    plt.legend()
    plt.show()
    '''
     
        
    
    
    
    
    
    
    
    #plot results###############################################    
    
    #set up multi-plot figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(6,10))
    fig.subplots_adjust(hspace=0, bottom=.08, top=0.98, right=.95, left=.2)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    
    
    
    #plot error
    ax1.plot(time, np.abs(deviation_percent), linewidth=0.5,
             c='r', label='ANN')
    ax1.plot(time, np.abs(deviation_percent_fit), linewidth=0.5,
             c='b', label='exp. model')
    ax1.set_ylabel('Residual (%)', fontsize=label_size)
    ax1.set_xlim(0,55)
    ax1.set_ylim(0,100)
    ax1.legend(loc='upper left')
    #plt.legend()
    
    
    
    #plot model and prediction
    ax2.plot(time, model, linewidth=1, c='g', label='ANN model')
    ax2.plot(time, prediction, linewidth=1, c='r', label='ANN prediction')    
    
    ax2.plot(time, sim_fit, linewidth=1, c='b', label='exp. model')
    
    ax2.set_ylabel('Model', fontsize=label_size)
    ax2.set_ylim(-75,40)
    ax2.legend(loc='lower left')
    #plt.legend()
    
    #plot measured signal
    ax3.scatter(time, temp_signal, s=2, c='g')
    ax3.plot(time, signal, c='k', linewidth=.5)
    ax3.set_ylabel('Response', fontsize=label_size)
    #ax2.set_xlabel('Time (hours)', fontsize=label_size)
    ax3.set_ylim(-75, 40)
    #plot pressure sequence
    ax4.plot(time, pressure, c='b', linewidth=1)
    ax4.set_ylabel('RH (%)', fontsize=label_size)
    ax4.set_xlabel('Time (hours)', fontsize=label_size)
    ax4.set_ylim(0, 90)
    
    #save plot as image file
    save_pic_filename = 'gif_frames_2018-03-16\\NN_output_frame_'+format(i)+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)
    
    plt.show()
    #close figure from memory
    plt.close(fig)
#close all figures from memory
plt.close("all")





#%% find all files in the designated data folder and sort by time/date

'''
all_frames = glob.glob('C:\\Users\\a6q\\gif_frames_2018-03-16/*')


all_frames.sort(key=os.path.getmtime)
print('found ' + format(len(all_frames)) + ' images')

#create gif using all saved image files
pics = []
for filename in all_frames: pics.append(imageio.imread(filename))
imageio.mimsave('NN_output2_2018-03-16.gif', pics, duration=0.2)
'''
