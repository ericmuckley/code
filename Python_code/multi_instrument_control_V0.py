# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 09:16:15 2018

@author: a6q
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:34:18 2018

@author: a6q
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from scipy.optimize import curve_fit
import time
import scipy.interpolate as interp
from scipy.signal import savgol_filter

from pymeasure.instruments.keithley import Keithley2400

#sark110_MV is name of python file provided by Melchor Varela on Github at
# https://github.com/EA4FRB/sark110-python/blob/master/src/sark110.py
#import as SARK functions (sf)
import sarkfunctions as sf

#manually fix pyUSB installation for import of Ocean Optics Spectometer
import usb.backend.libusb1
backend = usb.backend.libusb1.get_backend(
        find_library=lambda x: "C:\\Users\\a6q\\libusb-1.0.dll")
dev = usb.core.find(backend=backend)
import seabreeze
seabreeze.use('pyseabreeze')
import seabreeze.spectrometers as sb





#%% functions

ls = 16
def label_axes(xlabel='x', ylabel='y', size=ls):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)



def get_conductance(rs, xs):
    # calculates series conductance from Rs and Xs
    #complex impedance    
    Z = np.add(rs, 1j*xs)
    #complex admittance
    Y = np.reciprocal(Z)
    #conductance
    G = np.real(Y)
    #susceptance
    #B = np.imag(Y)

    return G



def get_f_range(band_cent, band_points=750, bandwidth=50000):
    # construct range of frequencies usuing center frequency (band_cent),
    # total number of points in the band (bandpoints), and
    # total bandwidth in Hz (bandwidth)
    df = np.linspace(0, int(bandwidth), int(band_points)) - int(bandwidth/2)
    f_range = np.add(band_cent, df).astype(int)
    return f_range



def remove_outliers(spectra, num_of_outliers=6):
    # removes outlier points caused by poor buffering/sampling by SARK-110 
    spectra2 = np.copy(spectra)
    #replace lowest outlier points with average of adjacent points
    for i in range(num_of_outliers):  
        #find index of minimum point
        min_index = np.argmin(spectra2)
        #make sure minimum is not first or last point
        if min_index != 0 and min_index != len(spectra2)-1:
            #convert minimum point into average of its neighboring points
            spectra2[min_index] = (
                    spectra2[min_index-1] + spectra2[min_index+1]) / 2
    return spectra2



def single_lorentz(freq, Gp, Cp, Gmax, D0, f0):
    # Returns conductance spectrum with single peak.
    # Spectrum calculation is taken from Equation (2) in:
    # Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009. Analyzing spur-distorted 
    # impedance spectra for the QCM. Journal of Sensors, 2009.
    # Gp = parallel conductance offset
    # Cp = parallel susceptance offset
    # Gmax = maximum of conductance peak
    # D0 = dissipation
    # f0 = resonant frequency of peak (peak position) 
    #construct peak
    peak = Gmax / (1 + (1j/D0)*((freq/f0)-(f0/freq)))
    #add parallel offsets to peak
    Y = Gp + 1j*2*np.pi*freq*Cp + peak
    G = np.real(Y)
    return G


def single_RLC(fit_params0):
    # calculate equivalent circuit parameters from RLC fits
    # from Yoon et al., Analyzing Spur-Distorted Impedance 
    # Spectra for the QCM, Eqn. 3.
    #FIT PARAMS = [Gp, Cp, Gmax0, D0, f0]
    G0 = fit_params0[2]
    f0 = fit_params0[4]
    D0 = fit_params0[3]
    R = 1/G0
    L = 1 / (2 * np.pi * f0 * D0 * G0)
    C = 1 / (4 * np.pi**2 * f0**2 * L)
    return R, L, C

def single_exp(x, a, tau, y0):
    #single exponential function with y offset
    return a * np.exp(-(x) / tau) + y0



def get_peaks(vec, n=3):
    # Get indicies and heights of peaks in vector. n parameter specifies
    # how many points on each side of the peak should be strictly
    # increasing/decreasing in order for it to be considered a peak.
   
    peak_indices = []
    peak_vals = []
    
    for i in range(n, len(vec)-n):
        
        #differences between points at rising and falling edges of peak
        rising_diff = np.diff(vec[i-n:i+1])
        falling_diff = np.diff(vec[i:i+n+1])
        
        #check if rising edge increases and falling edge decreases
        if np.all(rising_diff>0) and np.all(falling_diff<0):
            peak_indices.append(i)
            peak_vals.append(vec[i])

    peak_indices = np.array(peak_indices).astype(int)
    peak_vals = np.array(peak_vals).astype(float)
    
    #sort by highest to lowest peaks
    peak_order = peak_vals.argsort()[::-1]
    peak_vals, peak_indices = peak_vals[peak_order], peak_indices[peak_order]
    peak_indices = peak_indices.astype(int)
    
    
    return peak_indices, peak_vals




def expand_dic(dic):
    '''increases size of dictionary which holds measurement parameters so
    that new parameters can be appended to each key after every measurement 
    loop'''
    #scan through each dictionary key
    for key in dic:
        #for 2D dictionary values, add a row
        if len(np.shape(dic[key])) == 2:
            dic[key] = np.vstack((dic[key],
               np.zeros((1, np.shape(dic[key])[1]))))
        #for 3D dictionary values, stack an aray to the 3rd dimension
        if len(np.shape(dic[key])) == 3:
            dic[key] = np.dstack((dic[key],
               np.zeros((np.shape(dic[key])[0],np.shape(dic[key])[1],
                         1))))
        else: pass


def plot_rlc_params(dic, overtone_list, iteration_num):
    #realtime plotting of RLC parameters, deltaF, and deltaD
    fig, axarr = plt.subplots(5, sharex=True)
    axarr[0].set_ylabel('$\Delta$f/n (Hz/cm$^{2}$)', fontsize=ls)
    axarr[1].set_ylabel('$\Delta$D/n (x10$^{6}$)', fontsize=ls)
    axarr[2].set_ylabel('$\Delta$R/n ($\Omega$)', fontsize=ls)
    axarr[3].set_ylabel('$\Delta$L/n (mH)', fontsize=ls)
    axarr[4].set_ylabel('$\Delta$C/n (fF)', fontsize=ls)
    axarr[4].set_xlabel('Elapsed time (minutes)', fontsize=ls)
    for h in range(len(h_list)):
        axarr[0].plot(dic['elapsed_time'][:i, 1],
                (np.array(dic['f_res'][:i, h]) - dic['f_res'][0,h])/h_list[h],
                label='n='+format(h_list[h]))
        axarr[1].plot(dic['elapsed_time'][:i, 1],
                (np.array(dic['D'][:i, h]) - dic['D'][0,h])*1e6/h_list[h],
                label='n='+format(h_list[h]))
        axarr[2].plot(dic['elapsed_time'][:i, 1],
                 (np.array(dic['R'][:i, h]) - dic['R'][0,h])/h_list[h],
                 label='n='+format(h_list[h]))
        axarr[3].plot(dic['elapsed_time'][:i, 1],
                (np.array(dic['L'][:i, h]) - dic['L'][0,h])*1e3/h_list[h],
                label='n='+format(h_list[h]))
        axarr[4].plot(dic['elapsed_time'][:i, 1],
                (np.array(dic['C'][:i, h]) - dic['C'][0,h])*1e15/h_list[h],
                label='n='+format(h_list[h]))
    axarr[4].legend(loc='upper left', ncol=2)
    fig.set_size_inches(6,10)
    fig.subplots_adjust(left=.25, right=.95, bottom=.08, top=.98, hspace=0)
    plot_param_filename = fig_dir+'qcm_params_'+str(i).rjust(4, '0')+'.png'        
    plt.savefig(plot_param_filename, format='png', dpi=150)
    plt.show()








#---------------------- KEITHLEY 2420 FUNCTIONS -----------------------------
def get_v_list(v_max=0.5, v_steps=11):
    #create list of voltages for IV and CV measurements
    v_range = np.linspace(0, v_max, num=v_steps)
    v_list_iv = np.concatenate((-v_range[::-1], v_range[1:]))
    v_list_cv = np.concatenate((v_range, v_range[::-1][1:],
                             -v_range[1:], -v_range[::-1][1:]))
    return v_list_iv, v_list_cv

def run_keithley(smu, v_list):
    #run electrical measurement using Keithley 2420 named 'smu'
    smu.enable_source()
    current_list = np.empty_like(v_list)
    #loop through each applied voltage
    for i,v0 in enumerate(v_list):
        #apply voltage
        smu.source_voltage = v0
        smu.apply_voltage
        time.sleep(0.1)
        #read current
        current_list[i] = smu.current
        #print('V: %.3fV, I: %.4f uA' %(v0, 1e6*current_list[i]))
    smu.source_voltage = 0
    smu.disable_source()    
    smu.shutdown()
    return current_list

def setup_keithley(address='GPIB::24'):
    #set up Keithley 2420 for electrical measurements
    smu = Keithley2400(address)
    smu.reset()
    smu.use_front_terminals()
    smu.measure_current(current=0.01) #set current compliance
    return smu





#------------Ocean Optics USB 4000 functions---------------------------------

def get_optical_spectrum(device, int_time=1e6, smoothing='savgol'):
    '''acquire optical spectrum from OceanOptics spectrometer 'device', using
    integration time 'int_time' and filtering'''
    #set measurement integration time in microseconds
    device.integration_time_micros(3e6)
    wavelengths = device.wavelengths()[5:-185]
    intensities = device.intensities(correct_dark_counts=False,
                                 correct_nonlinearity=False)[5:-185]
    #close ocnnection to spectrometer
    #sm.close()
    
    
    if smoothing=='savgol':
        intensities_smooth = savgol_filter(intensities, 11, 1)
    if smoothing=='spline':
        spline_fit = interp.UnivariateSpline(wavelengths,
                                             intensities,
                                             k=1, s=4e8)
        intensities_smooth = spline_fit(wavelengths)
    else: pass

    return wavelengths, intensities, intensities_smooth


#%% user inputs


#set which instruments are connected
QCM = True
SMU = True
SPEC = True


#set diectory for saving data
data_dir = 'C:\\Users\\a6q\\exp_data\\2018-09-04_pedotpss_data\\'
#set directory for saving figures
fig_dir = 'C:\\Users\\a6q\\exp_data\\2018-09-04_pedotpss_figs\\'
#filepath for sending current RH value to LabVIEW
rh_file = 'C:\\Users\\a6q\\Desktop\\LABVIEW_DATA\\CURRENT_RH_VALUE.txt'

#input fundamental crystal resonance in Hz
crystal_res = 5e6
band_cent0 = int(0.999*crystal_res)

#list of crystal harmonics to measure
h_list = [1,9]
#number of points to measure at each frequency band
band_points = 200

#list of RH values to sample
rh_list = np.linspace(2,6,1).astype(int)

#how many full measurement loops to make for each RH step
loops_per_step = 1


#get list of voltages for IV and CV measurements
v_list_iv, v_list_cv = get_v_list(v_max=2, v_steps=11)


#%% prepare experiment
    
if QCM:    
    #open SARK-110 device
    device = sf.sark_open()
if SMU:
    #connect and configure Keithley 2420
    smu = setup_keithley('GPIB::24')
if SPEC:
    #set spectrometer device
    sm = sb.Spectrometer(sb.list_devices()[0])
    #print(sm.pixels)


#initiation rh step sequence
rh_i = 0
rh_val = rh_list[rh_i]
#initialize RH file with first RH value
with open(rh_file, "w") as rh_file_out:
    print(format(rh_val), file=rh_file_out)

#matrix to store all calculated fit parameters
dic = {
       'elapsed_time': np.empty((0, 3)),
       'f_res': np.empty((0, len(h_list))),
       'D': np.empty((0, len(h_list))),
       'Gmax': np.empty((0, len(h_list))),
       'R': np.empty((0, len(h_list))),
       'L': np.empty((0, len(h_list))),
       'C': np.empty((0, len(h_list))),
       'bvd_fit_params': np.empty((5, len(h_list), 0))}

#make folders for saved files if they don't already exist
if not os.path.exists(data_dir): os.makedirs(data_dir)
if not os.path.exists(fig_dir): os.makedirs(fig_dir)

#set up array to store keithley 2420 data
keithley_iv_data = v_list_iv
keithley_cv_data = v_list_cv
iv_max_current = np.array([])
cv_max_current = np.array([])

#set up array to store Ocean Optics USB4000 optical spectra data
optical_data = np.empty((sm.pixels-190, 0))


#initiate step clock
start_exp_time = time.time()
start_step_time = time.time()
#initiate loop counter
i=0

#%% run experiment

while rh_i < len(rh_list):
    start_loop_time = time.time()
    print('loop %i, RH step %i/%i, %i%% RH' %(
            i, rh_i+1, len(rh_list), rh_val))
    
    #expand size of dic to hold set of parameters measured at each loop
    expand_dic(dic)
    
    #pad with zeros for easy sorting in saved filenames
    i_save=str(i).rjust(4, '0')
    rh_save = str(rh_list[rh_i]).rjust(2, '0')
    
    #save time, loop number, and RH
    dic['elapsed_time'][i] = np.array(
            [i, (time.time()-start_exp_time)/60, rh_val])
    
    
    
    if QCM:
        #------------------------QCM measurements-----------------------------
        #loop over harmonic number
        for n in range(len(h_list)):
            
            #pad with zeros for easy sorting in saved filenames
            n_save=str(h_list[n]).rjust(3, '0')
            
            
            #---------------------take spectral measurement---------------------
            band_width = 3000 + 10000*(h_list[n]-1)
    
            #determine measurement band & use first iteration to optimize it
            if i == 0:
                band_cent = int(band_cent0*h_list[n] - 8000*(h_list[n]-1))
                
            if i > 0:
                band_cent = int(dic['f_res'][i-1, n])
                #band_width = int(dic['D'][i-1, n]*1000*1e6)
        
            #set frequency range to measure
            f_vec = get_f_range(band_cent,
                                band_points=band_points,
                                bandwidth=band_width)
            
            #standard implementation
            #loop over frequencies to measure Rs, Xs, and calculate conductance       
            g_vec = [get_conductance(
                    *sf.sark_measure(device, int(f))) for f in f_vec]
             
            #remove outlier points that are artefacts of bad sampling by SARK-110
            g_vec = remove_outliers(g_vec)  
            
            #fit conductance to spline for smoothing
            g_spline_fit = interp.UnivariateSpline(f_vec, g_vec, s=1e-8)
            g_spline = g_spline_fit(f_vec)
            
            #calculate peak position and height
            f_res = f_vec[np.argmax(g_spline)]
            
            dic['f_res'][i, n] = f_res
            dic['Gmax'][i,n] = np.max(g_spline)
    
    
    
    
    
    
    
            #-----------fitting to BVD equivalent circuit model-------------------
            #for first measurement at this harmonic, construct guess for RLC fit
            if i == 0: bvd_guess = [0,0, np.max(g_spline), 1e-5, f_res]
            #for subsequent measurements, use previous fit params as guess    
            if i > 0:
                bvd_guess = dic['bvd_fit_params'][:,n,i-1]
                #for G max guess, use measured G max
                bvd_guess[2] = dic['Gmax'][i,n]
                #for res. freq. guess, use freq at G max
                bvd_guess[4] = dic['f_res'][i, n]
             
            #detect peaks to use positions to set fitting window
            peaks_to_fit = 1 # = [1+int(i/4) for i in h_list]]
            peak_inds, peak_vals = get_peaks(g_spline, 3)
            peak_inds = peak_inds[:peaks_to_fit]
            peak_inds = f_vec[peak_inds]
            peak_vals = peak_vals[:peaks_to_fit]
    
            #set window of curve to fit
            fit_win = 60
            f_fit_win= f_vec[:np.argmax(g_spline)+fit_win]
            g_fit_win = g_spline[:np.argmax(g_spline)+fit_win]
            
            #fit data to lorentz peak
            try: rlc_params,_ = curve_fit(single_lorentz,
                                       f_fit_win,#f_mat[:,n,i],
                                       g_fit_win,#g_mat[:,n,i],
                                       bounds=(0,np.inf),
                                       p0=bvd_guess)
            except RuntimeError:
                print('BVD FIT FAILED')
                rlc_params = bvd_guess
            
            #populate matrix of fit parameters
            dic['bvd_fit_params'][:,n,i] = rlc_params
            dic['D'][i,n] = rlc_params[3]
            #calculate RLC from Butterworth Van Dyke equivalent circuit 
            dic['R'][i,n], dic['L'][i,n], dic['C'][i,n] = single_RLC(rlc_params)
            #calculate fitted curve
            g_fit = single_lorentz(f_vec, *rlc_params)
            
            
            
            
            
            
            
            
            #--------------------plot spectral data----------------------------
            #plot experimental data
            plt.scatter(f_vec, 1e3*g_vec,
                        s=5, c = 'k', label='exp.data', marker='.')
            #plot detected peak positions
            plt.scatter(peak_inds, 1e3*peak_vals,
                        marker='+', c='g', s=60, label='detected peak')
            #plot spline fit
            plt.plot(f_vec, 1e3*g_spline, lw=1, c='c', label='spline')
            #plot RLC fit
            plt.plot(f_vec, 1e3*g_fit, lw=1, c='r', label='fit')
            #plot peak position history
            plt.scatter(dic['f_res'][:i,n], 1e3*dic['Gmax'][:i,n],
                        c='c', marker='.', s=5, label='peak trajectory')
            plt.xlim([np.min(f_vec),np.max(f_vec)])
            label_axes('Frequency (Hz)', 'Conductance (mS)')
            plt.legend(fontsize=12)
            plt.title('n=%i, i=%i, %i%% RH'%(
                    h_list[n], i, rh_list[rh_i]),fontsize=ls)
            plt.tight_layout()
            

            plot_spec_filename = fig_dir+'qcm_'+n_save+'_'+rh_save+'_'+i_save+'.png'        
            plt.savefig(plot_spec_filename, format='png', dpi=250)
            plt.show()
        
            #save data
            result_df = pd.DataFrame(
                    data=np.column_stack((f_vec, f_vec - dic['f_res'][0,n],
                                          g_vec, g_vec / np.max(g_vec), g_fit)),
                                    columns=['f_res', 'f_norm',
                                             'g', 'g_norm','g_fit'])
            result_df.to_csv(path_or_buf=data_dir+'qcm'+n_save+'_'+i_save+'.txt',
                    sep='\t', index=False)
        
        #---------------plot calculated fit parameters--------------------------
        if i > 1: plot_rlc_params(dic=dic, overtone_list=h_list, iteration_num=i)
        
        
    
    
    
    
    if SMU:
        #-------Keithley 2420 IV measurements---------------------------------
        #get current from keithley measurements
        print('measuring CV on Keithley 2420...')
        cv_list = run_keithley(smu, v_list_cv)
        keithley_cv_data = np.column_stack((keithley_cv_data, cv_list))
        cv_max_current = np.append(cv_max_current, np.max(cv_list))
        
        plt.plot(v_list_cv, cv_list, c='k')
        plt.title('CV %i, %i%% RH'%(i, rh_list[rh_i]),fontsize=ls)
        label_axes('Voltage (V)', 'Current (A)')
        plt.tight_layout()
        plot_cv_filename = fig_dir+'cv_'+rh_save+'_'+i_save+'.png'        
        plt.savefig(plot_cv_filename, format='png', dpi=250)
        plt.show()  
        
        time.sleep(3)
        print('measuring IV on Keithley 2420...')
        iv_list = run_keithley(smu, v_list_iv)
        keithley_iv_data = np.column_stack((keithley_iv_data, iv_list))
        iv_max_current = np.append(iv_max_current, np.max(iv_list))
        
        plt.plot(v_list_iv, iv_list, c='k')
        plt.title('IV %i, %i%% RH'%(i, rh_list[rh_i]),fontsize=ls)
        label_axes('Voltage (V)', 'Current (A)')
        plt.tight_layout()
        plot_iv_filename = fig_dir+'iv_'+rh_save+'_'+i_save+'.png'       
        plt.savefig(plot_iv_filename, format='png', dpi=250)
        plt.show() 
        
        with open(data_dir+'keithley_cv.pkl', 'wb') as handle:
            pickle.dump(keithley_cv_data, handle, pickle.HIGHEST_PROTOCOL)
        with open(data_dir+'keithley_iv.pkl', 'wb') as handle:
            pickle.dump(keithley_iv_data, handle, pickle.HIGHEST_PROTOCOL)

    
    
    
    if SPEC:
        #-----optical measurements using Ocean Optics USB 4000 spectrometer---
        nm, intensity, intensity_smooth = get_optical_spectrum(sm)
        
        nm = np.reshape(nm, (len(nm), 1))
        intensity = np.reshape(intensity, (len(nm), 1))
        intensity_smooth = np.reshape(intensity_smooth, (len(nm), 1))
        
        
        if i == 0: optical_data = np.hstack((optical_data, nm,
                                             intensity_smooth))
        if i > 0: optical_data = np.hstack((optical_data, intensity_smooth))
        
        plt.scatter(nm, intensity, s=2,c='k', alpha=0.2, label='data')
        plt.plot(nm, intensity_smooth, c='r', label='smoothed')
        label_axes('Wavelength', 'Intensity')
        plt.title('Optical %i, %i%% RH'%(i, rh_list[rh_i]),fontsize=ls)
        plt.tight_layout()
        plot_optical_filename = fig_dir+'optical_'+rh_save+'_'+i_save+'.png'       
        plt.savefig(plot_optical_filename, format='png', dpi=250)
        plt.show()
    
        with open(data_dir+'optical_data.pkl', 'wb') as handle:
            pickle.dump(optical_data, handle, pickle.HIGHEST_PROTOCOL)
    
    
    
    
    
    
    
    
    
    
    #-----------------save all data to dictionary file-----------------------
    with open(data_dir+'data_dictionary.pkl', 'wb') as handle:
        pickle.dump(dic, handle, pickle.HIGHEST_PROTOCOL)
    
    loop_duration = (time.time() - start_loop_time)/60
    step_time  = (time.time() - start_step_time)/60
    print('loop duration: %.2f min' %loop_duration)
    print('RH step duration: %.2f min' %step_time)
    print('total time elapsed: %.2f minutes' %(
            (time.time()-start_exp_time)/60))


    
    #-------------------incrementing the rh value-----------------------------
    if (step_time > loops_per_step*loop_duration) or False:
        #if this is not the last step
        if rh_i < len(rh_list)-1:
            rh_i += 1
            rh_val = rh_list[rh_i]
            #reset step clock
            start_step_time = time.time()
            print('NEW RH VALUE: %i%% RH' %(rh_val))
            with open(rh_file, "w") as rh_file_out:
                print(format(rh_val), file=rh_file_out) 
            #wait while RH changes
            #time.sleep(180)
        #if this is the last step
        else: break
    i+=1 #next measurement cycle
    
    

#close spectrometer
sm.close()
    
#reinitialize RH   
with open(rh_file, "w") as rh_file_out:
                print(format(2), file=rh_file_out)  
    
#%% plot all keithley data
if SMU:
    for i in range(1, len(keithley_cv_data[0])):
        plt.plot(v_list_cv, keithley_cv_data[:,i])
    label_axes('Voltage (V)', 'Current (A)')
    plt.show()
    for i in range(1, len(keithley_iv_data[0])):
        plt.plot(v_list_iv, keithley_iv_data[:,i])
    label_axes('Voltage (V)', 'Current (A)')
    plt.show()

    plt.plot(dic['elapsed_time'][:,1], iv_max_current)
    label_axes('Time (min)', 'Max. IV current (A)')
    plt.show()
    
    plt.plot(dic['elapsed_time'][:,1], cv_max_current)
    label_axes('Time (min)', 'Max. CV current (A)')
    plt.show()



#%% plot all spectrometer data
if SPEC:
    for i in range(1, len(optical_data[0])):
        plt.plot(optical_data[:,0], optical_data[:,i])
    label_axes('Wavelength (nm)', 'Intensity (counts)')
    plt.show()
    
    
#%%
'''
#load pickled dictionary
with open(data_dir+'data_dictionary.pkl', 'rb') as handle:
    dic2 = pickle.load(handle)
''' 
