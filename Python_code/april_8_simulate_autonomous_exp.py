# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:53:17 2019
 
@author: ericmuckley@gmail.com
"""
import time
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



def single_exp(x, a, tau, y0):
    return y0 + a*np.exp(-(x)/tau)



def is_stable(signal, old_window=3,  new_window=3):
    # check stability of signal by looking at most recent points (new_window)
    # and the average and std of previous points (old_window).
    # get old points
    signal = np.array(signal)
    old_points = signal[-old_window-new_window:-new_window]
    old_mean = np.mean(old_points)
    old_std = np.std(old_points)
    old_range = [old_mean - old_std/5, old_mean + old_std/5]
    # get new points
    new_points = signal[-new_window:]
    new_mean = np.mean(new_points)
    new_std = np.std(new_points)
    new_range = [new_mean - new_std/5, new_mean + new_std/5]
    # decide whether new points are statistically different from old points
    # loop through entries in new range and see if they overlap with old range
    overlap = False
    if old_range[0] <= new_range[0] <= old_range[-1]:
        overlap = True
    if old_range[0] <= new_range[-1] <= old_range[-1]:
        overlap = True
    if new_range[0] <= old_range[0] <= new_range[-1]:
        overlap = True
    if new_range[0] <= old_range[-1] <= new_range[-1]:
        overlap = True
    return bool(overlap)





def get_iso(rh):
    # force response to follow complex isotherm
    iso = (np.log(rh)/10+(rh)**3/1000000)/10
    return iso

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


def exp_response(sig_df, i, tau=8):
    # simulate exponentially decaying response

    noise = 2 * (np.random.random() - 0.5)
    
    # add offset to enforce continuity between steps 
    for jj in range(len(sig_df)):  
        #use last value of previous step as starting point for current step
        if sig_df['step_t'].iloc[jj] == 0:
            prev = sig_df['response'].iloc[jj-1]
    
    # simulate exponential response of material
    exp_factor = np.exp(-sig_df['step_t'].iloc[i]/tau)
    exp_prefactor =  prev - sig_df['abs_p'].iloc[i]
    response = sig_df['abs_p'].iloc[i] +  exp_prefactor * exp_factor
    
    return  response + noise# + iso_factor



def exp_res0(seq_df, amp1=1, amp2=1, tau=10, offset=0, drift=0, nonlin_exp=0):
    '''
    Creates simulated sensor response using sequence dataframe:
    [abs_time, rel_time, abs_pressure, rel_pressure].
    Fit parameters for the exponential decays are:
    [amp (amplitude of exponential), tau (exponential time constant),
    offset (vertical offset of exponential),
    drift (scaling factor for linear drift),
    nonlin_amp (amplitude of nonlinear correction to make)]
    '''
    #-----------build baseline------------------------------------
    sim_baseline = offset + drift*seq_df['t_abs'] 
        
    #--------build exponential response to pressure changes-----------------
    sim_response = np.zeros(len(seq_df))
    for i in range(1, len(seq_df)):  
        #use last value of previous step as starting point for current step 
        if seq_df['t_rel'][i] == 0:
            prev0 = sim_response[i-1]# + sim_response[i-1] - sim_response[i-2]
            
        #build exponential response
        sim_response[i] = amp1*seq_df['p_abs'][i]**nonlin_exp + amp2*(prev0 -
                 seq_df['p_rel'][i]*np.exp(-seq_df['t_rel'][i]/tau))
                 
        #sim_response[i] =  seq_df['p_abs'][i] + (
        #        prev0 - seq_df['p_abs'][i])*np.exp(-seq_df['t_rel'][i]/tau1)

    return sim_baseline + sim_response












# dataframe to hold signal information
sig_df = pd.DataFrame(
        columns=['abs_t', 'step_t', 'abs_p', 'delta_p',
                 'response', 'iso', 'delta_iso', 'stable'])

# dataframe to hold isotherm information
iso_df = pd.DataFrame(columns=['press', 'res'])

rh_list = np.array([2, 95]).astype(float)

# minimum and maximum number of points to collect at each pressure step
min_step_points, max_step_points = 6, 150

# total number of pressure steps to measure
tot_steps = 18
# initialize counters
i, rh_i, step_i = 0, 0, 0 
new_rh = None

start_time = time.time()

# iterate over each data point over time
while True:

    # append new row to dataframe
    sig_df.loc[i] = np.zeros(len(list(sig_df)))
    # append pressure value to dataframe
    sig_df['abs_t'].iloc[i] = i
    sig_df['step_t'].iloc[i] = step_i
    sig_df['abs_p'].iloc[i] = rh_list[rh_i]

    # force response to follow complex isotherm
    sig_df['iso'].iloc[i] = get_iso(rh_list[rh_i])

    # caculate magnitude of most recent change in RH
    if rh_i == 0:
        sig_df['delta_p'].iloc[i] = 0
        sig_df['delta_iso'].iloc[i] = 0
    else:
        sig_df['delta_p'].iloc[i] = rh_list[rh_i] - rh_list[rh_i-1]
        sig_df['delta_iso'].iloc[i] = get_iso(
                rh_list[rh_i]) -  get_iso(rh_list[rh_i-1])



    # set response time
    tau = np.abs(sig_df['delta_p'].iloc[i]/3)
    if tau == 0:
        tau = 5
    
    
    # measure new response
    response0 = exp_response(sig_df, i, tau=tau)
    sig_df['response'].iloc[i] = response0



    # set response to be not stable by default
    sig_df['stable'].iloc[i] = False
    # if we have enough sample points, test for stability of the response
    if sig_df['step_t'].iloc[i] > min_step_points:
        sig_df['stable'].iloc[i] = is_stable(np.array(sig_df['response']))


    # if we have surpassed max number of points per step, set as stable
    if sig_df['step_t'].iloc[i] > max_step_points:
        sig_df['stable'].iloc[i] = True
        print('max step number reached, moving to next step')


    # if sample is stable, save response and increment to next RH step
    if sig_df['stable'].iloc[i] == True:
        print('stable response reached, moving to next step')
        
        
        # save isotherm point
        iso_df.loc[rh_i] = np.zeros(2)
        iso_df['press'].iloc[rh_i] = sig_df['abs_p'].iloc[i]
        iso_df['res'].iloc[rh_i] = sig_df['response'].iloc[i]

        # reinitialize step counter
        step_i = 0
        rh_i += 1   
        
        if rh_i == len(rh_list):
           

            # append to RH list
            if rh_i <= tot_steps:
                # get new RH level to measure
                new_rh = next_point_to_measure(iso_df['press'], iso_df['res'])
                rh_list = np.append(rh_list, new_rh)
              
            # end the experiment
            else:
                break


            
            
    else:
        step_i += 1

    # increment universal number of points measured       
    i += 1 










    # plot results
    if i % 5 == 0:
        
        #print(i)
        elapsed = (time.time() - start_time)/60
        print('# '+str(i)+', minutes elapsed: '+(str(np.round(elapsed, decimals=2))))
        
        # set up plot
        fig = plt.figure()#constrained_layout=True)
        fig.set_size_inches(10, 4, forward=True)
        
        #gs = fig.add_gridspec(2, 2)
        #axp = fig.add_subplot(gs[1, 0])        
        #axr = fig.add_subplot(gs[0, 0], sharex=axp)
        #axi = fig.add_subplot(gs[:, 1])
        
        axr = plt.subplot2grid((2, 2), (0, 0), colspan=1)
        axp = plt.subplot2grid((2, 2), (1, 0), colspan=1)
        axi = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
        
        # separate response into stable and unstable       
        stable_df = sig_df[sig_df['stable'] == True]
        unstable_df = sig_df[sig_df['stable'] == False]

        # plot unstable points
        axr.scatter(unstable_df['abs_t'], unstable_df['response'],
                 marker='o', c='k', facecolors='white',
                 s=10, alpha=0.2, label='unstable')
        axr.set_ylabel('Response', fontsize=fontsize)
        axr.plot(sig_df['abs_t'], sig_df['response'],
                 c='k', alpha=0.5, lw=1, label='response')
        
        # plot stable points
        axr.scatter(stable_df['abs_t'], stable_df['response'],
                 marker='|', c='r', s=200, label='stable')


        #axr.legend(fontsize=12)
        axr.xaxis.set_visible(False)
        
        
        # plot pressure over time
        axp.plot(sig_df['abs_t'], sig_df['abs_p'], c='b')
        axp.fill_between(sig_df['abs_t'], 0, sig_df['abs_p'], facecolor='b', alpha=0.5)
        axp.set_ylabel('P/P$_0$ (%)', fontsize=fontsize)
        axp.set_xlabel('Time (min)', fontsize=fontsize)
        axp.set_ylim([0, 99])



        # plot isotherm
        axi.scatter(iso_df['press'], iso_df['res'],  c='k', s=5, marker='o')
        axi.set_ylabel('Response', fontsize=fontsize)
        axi.set_xlabel('P/P$_0$ (%)', fontsize=fontsize)
        axi.set_xlim([0, 100])
        # plot new RH selection
        if new_rh is not None:
            plt.axvline(x=new_rh, c='b', lw=1, linestyle='--')

        

        
        fig.subplots_adjust(wspace=0.5, hspace=0, top=0.95, bottom=.2)
        #plt.tight_layout()
        # save figure
        save_pic_filename = 'exp_data\\autonomous_experiment_plots\\fig'+str(i).zfill(4)+'.jpg'
        plt.savefig(save_pic_filename, format='jpg', dpi=250)
        plt.show()

 
#%%

make_video = True
if make_video:
    create_video('exp_data\\autonomous_experiment_plots\\',
                 'C:\\Users\\a6q\\Desktop\\auto_experiment_vid.avi', fps=6)
