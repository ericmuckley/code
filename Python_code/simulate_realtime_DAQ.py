# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from timeit import default_timer as timer


from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
label_size = 18 #make size of axis tick labels larger
plt.rcParams['xtick.labelsize'] = label_size 
plt.rcParams['ytick.labelsize'] = label_size


#%% define fitting functions

def single_exp(t, A1, tau1, y0):
    return (1 - A1*np.exp(-(t) / tau1)) + y0

def double_exp(t, A1, tau1, A2, tau2, y0):
    return A1 * np.exp(-(t) / tau1) + A2 * np.exp(-(t) / tau2) + y0

def linear(t, m, b):
    return m * t + b 

def label_axes(xlabel, ylabel, size=18):
    #set axes labels and size
    from matplotlib import rcParams
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)


#%%

def seq_to_mat(seq0, points_per_minute=1):
    '''
    Creates a matrix out of 2-column sequence data: [times, pessures].
    Outputs: [times, relative times, pressures, relative pressures].
    Outputs Pandas dataframe populated with 'points_per_minute'
     number of points per minute of the original sequence.
    '''
    #initialize absolute time column
    t_abs, dt = np.linspace(0, np.sum(seq0[:,0]),
                        num = np.multiply(np.sum(seq0[:,0]),
                                          points_per_minute).astype(int),
                                          retstep=True)
                        
    #init dialize relative time (time since last pressure change)
    t_rel = np.zeros((np.sum(seq0[0,0])*points_per_minute).astype(int))
    #initialize initial pressure
    p_abs = np.ones_like(t_rel) * seq0[0,1]
    #relative pressure (pressure difference since last pressure change)
    p_rel = np.zeros_like(t_rel)
    
    #loop over steps in sequence and populate columns#######################
    for i in range(1, len(seq0)):
        #numper of points to append during this step
        point_num = np.multiply(seq0[i,0], points_per_minute).astype(int)
        #append each column with appropriate values
        t_rel = np.append(t_rel, np.linspace(0, dt*point_num, num=point_num))
        p_abs = np.append(p_abs, np.full(point_num, seq0[i,1]))
        p_rel = np.append(p_rel, np.full(point_num, seq0[i,1]-seq0[i-1,1]))

    #construct dataframe with results
    seq_mat_df = pd.DataFrame(np.array([t_abs, t_rel, p_abs, p_rel]).T, 
                          columns=['t_abs', 't_rel', 'p_abs','p_rel'],
                          dtype=np.float64)
    return seq_mat_df


#%% calculate percent error from residuals
    
def percent_error(measured, fit):
    '''
    Calculates percent error between measured and fitted signals.
    Percent differences is calculated from the ratio of residual to
    entire range of measured values.
    '''
    measured_range = np.abs(np.amax(measured) - np.amin(measured))
    residual = np.subtract(measured, fit)
    percent_error = 100*np.abs(np.divide(residual, measured_range))
    
    return percent_error
    


#%% simulate expected signal using pressure sequence data

def exp_response(seq_df,
                 amp1=1, amp2=1, tau=10, offset=0, drift=0, nonlin_exp=0):
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







#%% prepare sequence data
seq0 = pd.read_table('cupcts_data_SEQ.txt', header=None).values

#seq0 = pd.read_table('cupcts_h2opulse_deltam_SEQ.txt', header=None).values
seq_df = seq_to_mat(seq0, points_per_minute=3)#, points_per_minute=15) #convert sequence to matrix

#%% prepare measured data
#data_raw = pd.read_table('cupcts_h2opulse_deltam.txt')
data_raw = pd.read_table('cupcts_data.txt')

seq_df['sig_interp'] = np.interp(
        seq_df['t_abs'], data_raw['time'], data_raw['delta_m'])

#%% build up one prediction interval at a time and fit to exponential model
#number of different predictions
pred_steps = 40 

#guess for fit_params (for pedotpss dtaaset):
#guess = [18, .1, 2, 6, .03, .40]

timer0 = timer()



fit_param_mat = np.zeros((6, pred_steps))

for i in range(1, pred_steps+1):
    print('prediction '+format(i)+'/'+format(pred_steps))
    
    partial_seq = seq_df.copy()[:int(len(seq_df)/pred_steps)*i]
    
    
    # fit measured data to exponential model and save fit parameters
    fit_params, _  = curve_fit(exp_response,
                               partial_seq, partial_seq['sig_interp'])
    
    fit_param_mat[:,i-1] = np.array(fit_params)

    fit = exp_response(partial_seq, *fit_params)
    
    plt.plot(partial_seq['t_abs'], fit, label='pred '+format(i))

    plt.scatter(partial_seq['t_abs'], partial_seq['sig_interp'], s=3,
                alpha=0.3, c='k', label='measured') 
    plt.show()

timer1 = timer()
print('total fit time = '+format(int(timer1-timer0)+1)+' sec')   
   











#%% create fits using saved fit parameters 

time = seq_df['t_abs']

for i in range(1, pred_steps+1):
    print('prediction '+format(i)+'/'+format(pred_steps))
    
    interval_len = int(len(seq_df)/pred_steps)*i
    partial_seq = seq_df.copy()[:interval_len]
    

    fit = exp_response(seq_df, *fit_param_mat[:,i-1])
    
    p_error = percent_error(np.array(fit),
                                  np.array(seq_df['sig_interp']))


    elapsed_time = np.array(partial_seq['t_abs'])[-1]
    
    
    #plot results###############################################    
    #set up multi-plot figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(6,10))
    fig.subplots_adjust(hspace=0, bottom=.08, top=0.98, right=.95, left=.2)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    
    #plot error
    ax1.axvline(x=elapsed_time, c='k', linewidth=0.5, alpha=.5)
    ax1.plot(time, p_error, linewidth=0.5, c='r', label='exp. model')
    ax1.set_ylabel('Residual (%)', fontsize=label_size)
    ax1.set_xlim(-2,np.max(time))
    ax1.set_ylim(0,50)

    #plot model and prediction
    ax2.axvline(x=elapsed_time,  c='k', linewidth=0.5, alpha=.5)
    ax2.plot(time, fit, linewidth=1, c='b')
    ax2.set_ylabel('Model', fontsize=label_size)
    ax2.set_ylim(0,9)

    #plot measured signal
    ax3.axvline(x=elapsed_time,  c='k', linewidth=0.5, alpha=.5)
    ax3.scatter(partial_seq['t_abs'],
                partial_seq['sig_interp'], s=3, c='g')
    ax3.plot(time, seq_df['sig_interp'], c='k', linewidth=.5)
    ax3.set_ylabel('Response', fontsize=label_size)
    #ax2.set_xlabel('Time (hours)', fontsize=label_size)
    ax3.set_ylim(0, 9)
    
    #plot pressure sequence
    ax4.axvline(x=elapsed_time,  c='k', linewidth=0.5, alpha=.5)
    ax4.plot(time, seq_df['p_abs'], c='b', linewidth=1)
    ax4.set_ylabel('RH (%)', fontsize=label_size)
    ax4.set_xlabel('Time (min)', fontsize=label_size)
    ax4.set_ylim(0, 99)
    
    #save plot as image file
    #ave_pic_filename = 'gif_frames2018-04-06\\frame_'+format(i)+'.jpg'
    #plt.savefig(save_pic_filename, format='jpg', dpi=150)
    
    plt.show()
    plt.close(fig)
plt.close("all")



#%% try manually simulated signal
'''
#params:
#[amp1=1, amp2=1, tau=10, offset=0, drift=0, nonlin_exp=0]

manual_params = [18, .1, 2, 6, .03, .40]

sim_sig = exp_response(seq_df, *manual_params)

plt.scatter(time, sim_sig, s=3, c='g')
plt.plot(time, seq_df['sig_interp'], c='k', linewidth=.5)
plt.show()
'''


#%% predict response far in future



future_seq0 = pd.read_table('test_seq.txt', header=None).values
future_df_seq = seq_to_mat(future_seq0, points_per_minute=3)

for i in range(pred_steps):
    future_sig = exp_response(future_df_seq, *fit_param_mat[:,i])
    
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(6,6))
    fig.subplots_adjust(hspace=0, bottom=0.12, top=0.98, right=.95, left=.2)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    #plot future signal
    ax1.plot(future_df_seq['t_abs'], future_sig, linewidth=1, c='k')
    ax1.set_ylabel('Predicted signal', fontsize=label_size)
    ax1.set_xlim(-2,np.max(future_df_seq['t_abs']))
    ax1.set_ylim(-.5,11)
    #plot pressure
    ax2.plot(future_df_seq['t_abs'], future_df_seq['p_abs'],
             linewidth=2, c='b')
    ax2.set_ylabel('RH (%)', fontsize=label_size)
    ax2.set_xlabel('Time (hours)', fontsize=label_size)
    ax2.set_ylim(0,99)
    
    #save_pic_filename = 'far_future_predictions\\frame_'+format(i)+'.jpg'
    #plt.savefig(save_pic_filename, format='jpg', dpi=150)

    plt.show()
