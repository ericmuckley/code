# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:53:17 2019

@author: ericmuckley@gmail.com
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fontsize = 16
plt.rcParams['xtick.labelsize'] = fontsize 
plt.rcParams['ytick.labelsize'] = fontsize


def create_video(image_folder, video_name, fps=8, reverse=False):
    #create video out of images saved in a folder
    import cv2
    import os
    
    images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]
    if reverse: images = images[::-1]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()


def get_baseline(x, y, samples=4):
    # get baseline response of a signal using the first samples
    pass
    
def single_exp(x, a, tau, y0):
    return y0 + a*np.exp(-(x)/tau)

def is_stable(signal, old_window=3,  new_window=3):
    # check stability of signal by looking at most recent points (new_window)
    # and the average and std of previous points (old_window).
    # get older points to create baseline from average and std
    signal = np.array(signal)
    old_points = signal[-old_window-new_window:-new_window]
    old_mean = np.mean(old_points)
    old_std = np.std(old_points)
    old_range = [old_mean - old_std/2, old_mean + old_std/2]
    # get new points to test
    new_points = signal[-new_window:]
    new_mean = np.mean(new_points)
    new_std = np.std(new_points)
    new_range = [new_mean - new_std/2, new_mean + new_std/2]
    # decide whether new points are statistically different from old points
    # loop through each entry in new range and see if its in old range
    overlap = False
    for new_i in new_range:
        if old_range[0] <= new_i <= old_range[1]:
            overlap = True
    for old_i in old_range:
        if new_range[0] <= old_i <= new_range[1]:
            overlap = True        
    return bool(overlap)

def sim_response(x):
    # simulate complex response of material. input should be pressure or RH
    # between 2 and 100. returns magnitude of response which 
    # resembles type-2 isotherm.
    return np.log(x)/10+(x)**3/1000000




def exp_response(sig_df, amp1=1, amp2=1, tau=10, offset=0, drift=0, nonlin_exp=0):
    # Creates simulated sensor response using signal dictionary:
    # [abs_time, rel_time, abs_pressure, rel_pressure].
    # Fit parameters for the exponential decays are:
    # [amp (amplitude of exponential), tau (exponential time constant),
    # offset (vertical offset of exponential),
    # drift (scaling factor for linear drift),
    # nonlin_amp (amplitude of nonlinear correction to make)]

    # build baseline
    sim_baseline = offset + drift*sig_df['abs_t'] 
        
    # build exponential response to pressure changes
    sim_response = np.zeros(len(sig_df['abs_t']))

    for i in range(1, len(sig_df['abs_t'])):  
        #use last value of previous step as starting point for current step 
        if sig_df['step_t'][i] == 0:
            prev0 = sim_response[i-1]# + sim_response[i-1] - sim_response[i-2]
            
        # build exponential response
        sim_response[i] = amp1*sig_df['abs_p'][i]**nonlin_exp \
                            + amp2*(prev0 - sig_df['delta_p'][i] \
                            * np.exp(-sig_df['step_t'][i]/tau))

    return sim_baseline + sim_response



def next_point_to_measure(x,y):
    # determines the independent variable of the next measurement point
    # based on the largest difference (gap) in measurmeents of a dependent
    # variable. inputs are x and y, the previously measured independent and 
    # dependent variable values. This funtion only interpolates, it will not
    # suggest a measurement which is outside the range of input x values.
    #sort arrays in order smallest to largest
    x, y = np.sort(np.array([x, y]))
    #find differences between values of adjacent measurements
    diffs = np.diff(y)*6
    #find index of biggest difference
    big_diff_index = np.argmax(diffs)
    #get suggested next independent variable value
    next_x_val = np.mean((x[big_diff_index], x[big_diff_index+1]))
    return next_x_val







# dataframe to hold signal information
sig_df = pd.DataFrame(
        columns=['abs_t', 'step_t', 'abs_p', 'delta_p','response', 'stable'])

# dataframe to hold isotherm information
iso_df = pd.DataFrame(columns=['press', 'res'])

tot_window = 6
rh_list = np.array([2, 60, 90, 10, 2]).astype(float)
randoms = [5*(np.random.random()-0.5) for i in range(5000)]

# minimum number of points to collect at each pressure step
min_points_per_step = 8

# initialize counters
i, rh_i, step_i = 0, 0, 0

# iterate over each data point over time
while True:
    # append new row to dataframe
    sig_df.loc[i] = np.zeros(6)
    # append pressure value to dataframe
    sig_df['abs_t'].iloc[i] = i
    sig_df['abs_p'].iloc[i] = rh_list[rh_i]
    
    # caculate magnitude of most recent change in RH 
    if rh_i == 0:
        sig_df['delta_p'].iloc[i] = 0
    else:
        sig_df['delta_p'].iloc[i] = rh_list[rh_i] - rh_list[rh_i-1]
        
    # measure new response
    response0  = randoms[i] + single_exp(
            step_i,
            float(sig_df['delta_p'].iloc[i])*sim_response(rh_list[rh_i]),8, 0)

    if i < 3:
        pass
    else:
        response0 = response0 + float(sig_df['response'].iloc[-2])
    sig_df['response'].iloc[i] = response0
        

    # if we have enough sample points, test for stability of the response
    if step_i < min_points_per_step:
        sig_df['stable'].iloc[i] = False
    else:
        
        sig_df['stable'].iloc[i] = is_stable(sig_df['response'])



    # if sample is stable, save response and increment to next RH step
    if sig_df['stable'].iloc[i] == True:
        
        # save isotherm point
        iso_df.loc[rh_i] = np.zeros(2)
        iso_df['press'].iloc[rh_i] = sig_df['abs_p'].iloc[i]
        iso_df['res'].iloc[rh_i] = sig_df['response'].iloc[i]
        
        # reinitialize step counter
        step_i = 0
        rh_i += 1        
        if rh_i == len(rh_list):
            break
    else:
        step_i += 1

    
    # increment universal number of points measured        
    i += 1

    if i % 10 == 0:
    
        fig = plt.figure(figsize=(6, 10))
        ax1 = plt.subplot(311)
        ax2 = plt.subplot(312)
        ax3 = plt.subplot(313, sharex = ax2)
        
        # plot pressure over time
        #ax1.plot(time, rh_arr, c='b')
        #ax1.fill_between(time, 0, rh_arr, facecolor='b', alpha=0.5)
        ax1.plot(iso_df['press'], iso_df['res'], lw=0,
                 c='k', fillstyle='none', markersize=15)
        ax1.set_ylabel('Response', fontsize=fontsize)
        ax1.set_xlabel('RH (%)', fontsize=fontsize)
        ax1.set_xlim([0, 100])
        
        # plot pressure over time
        ax2.plot(sig_df['abs_t'], sig_df['abs_p'], c='b')
        ax2.fill_between(sig_df['abs_t'], 0, sig_df['abs_p'], facecolor='b', alpha=0.5)
        ax2.set_ylabel('RH (%)', fontsize=fontsize)
        ax2.set_ylim([0, 100])
        
        # separte response into stable and unstable        
        stable_df = sig_df[sig_df['stable'] == True]
        unstable_df = sig_df[sig_df['stable'] == False]
        # plot stable points
        ax3.scatter(stable_df['abs_t'], stable_df['response'],
                 marker='*', c='g', s=100, label='stable')
        # plot unstable points
        ax3.scatter(unstable_df['abs_t'], unstable_df['response'],
                 marker='o', c='r', facecolors='white',
                 s=8, alpha=0.3, label='unstable')

        ax3.set_xlabel('Time (min)', fontsize=fontsize)
        ax3.set_ylabel('Response', fontsize=fontsize)
        ax3.plot(sig_df['abs_t'], sig_df['response'],
                 c='k', alpha=1, lw=1, label='response')
        ax3.legend(fontsize=12)
        plt.tight_layout()

        # save figure
        save_pic_filename = 'exp_data\\autonomous_experiment_plots\\fig'+str(i).zfill(4)+'.jpg'
        #plt.savefig(save_pic_filename, format='jpg', dpi=250)
        plt.show()






