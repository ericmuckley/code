# coding: utf-8

import numpy as np
import pandas as pd
from inspect import signature
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from timeit import default_timer as timer

#%% define fitting functions

def single_exp(t, A1, tau1, y0):
    return (1 - A1*np.exp(-(t) / tau1)) + y0

def double_exp(t, A1, tau1, A2, tau2, y0):
    return A1 * np.exp(-(t) / tau1) + A2 * np.exp(-(t) / tau2) + y0

def linear(t, m, b):
    return m * t + b 


#%% create data matrx out of sequence table

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
                          columns=['t_abs', 't_rel', 'p_abs','p_rel'])
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


#%% import sequence data
seq0 = pd.read_table('cupcts_data_SEQ.txt', header=None).values
#seq0 = pd.read_table('su8_data_SEQ.txt', header=None).values

#convert sequence to matrix
seq_df = seq_to_mat(seq0, points_per_minute=6)

#%% import measured response
data_raw = pd.read_table('cupcts_data.txt')
#data_raw = pd.read_table('su8_data.txt')
#data_raw['delta_f'] *= 200

#interpolate signal to fit inside seq_mat
seq_df['sig_interp'] = np.interp(
        seq_df['t_abs'] , data_raw['time'], data_raw['delta_m'])


#seq_df = seq_df[:600]

#%% fit measured data to exponential signal
timer0 = timer()

fit_params, _  = curve_fit(exp_response, seq_df, seq_df['sig_interp'])
fit = exp_response(seq_df, *fit_params)

percent_error = percent_error(seq_df['sig_interp'], fit)
print('avg. percent_err = '+format(np.average(percent_error)))

#%%plot results


#sim = exp_response(seq_df, amp1=.001, tau=1, offset=0, drift=0, nonlin_amp=0)

#plt.plot(seq_df['t_abs'], sim, linewidth=1, c='r',label='sim')
#plt.plot(seq_df['t_abs'], percent_error, c='r', label='percent error')
plt.scatter(seq_df['t_abs'], seq_df['sig_interp'], s=2, c='k',  label='exp.')
plt.plot(seq_df['t_abs'], fit, linewidth=1, c='g',
         label='fit')
#plt.plot(seq_df['t_abs'], seq_df['p_abs'], linewidth=1, c='b', label='rh')
plt.legend()
plt.show()

timer1 = timer()
print('fit time = '+format(int(timer1-timer0)+1)+' sec')






#%% predict response far in future

fit_params2  = np.copy(fit_params)
fit_params2[2] = 3


future_seq0 = pd.read_table('test_seq.txt', header=None).values
#future_seq0[:,0] *=10
                       
future_df_seq = seq_to_mat(future_seq0, points_per_minute=6)

future_sig = exp_response(future_df_seq, *fit_params)
future_sig2 = exp_response(future_df_seq, *fit_params2)


plt.plot(future_df_seq['t_abs'], future_df_seq['p_abs'])
plt.show()

plt.plot(future_df_seq['t_abs'], future_sig)
plt.plot(future_df_seq['t_abs'], future_sig2)
plt.show()






