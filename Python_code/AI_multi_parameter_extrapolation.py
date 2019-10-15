# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:34:50 2018

@author: a6q
"""
import sys, os, numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline



def config_plot(xlabel='x', ylabel='y', size=12,
                setlimits=False, limits=[0,1,0,1]):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    #set axis limits
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))



def multiparameter_extrapolate(input_array, future_times, splineorder=1):
    '''Extrapolate a series of parameters to "future_times" using a spline of
    order "splineorder". Input array should have x values as first column and
    parameters to extrapolate as subsequent columns. Output is array of
    extrapolated parameters with future times as first column.'''
    #new array for extrapolated parameters
    predicted_params = future_times
    #loop over each parameter to extrapolate
    for i in range(1, len(input_array[0])):
        spline = InterpolatedUnivariateSpline(input_array[:,0],
                                              input_array[:,i],
                                              k=splineorder)
        prediction0 = spline(future_times) #create extrapolation using spline
        predicted_params = np.column_stack((predicted_params, prediction0))
        
        #plt.scatter(params0[:,0], params0[:,i])
        #plt.plot(params0[:,0], spline(params0[:,0]))
        #plt.show()
        
    return predicted_params



def singleBvD_reY(freq, Gp, Cp, Gmax00, D00, f00):
    # Returns admittance spectrum with single peak.
    # Spectrum calculation is taken from Equation (2) in:
    # Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009. Analyzing spur-distorted 
    # impedance spectra for the QCM. Journal of Sensors, 2009.
    # inputs:
    # Gp = conductance offset
    # Cp = susceptance offset
    # Gmax00 = maximum of conductance peak
    # D00 = dissipation
    # f00 = resonant frequency of peak (peak position) 
    #construct peak
    peak = Gmax00 / (1 + (1j/D00)*((freq/f00)-(f00/freq)))
    #add offsets to spectrum
    Y = Gp + 1j*2*np.pi*freq*Cp + peak
    G = np.real(Y)
    return G



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



#%%

freq = np.linspace(4.979e6, 4.98025e6, 1000)


rh0 = np.linspace(2, 98, 30)
rand = 0.1*np.random.random(len(rh0))

f0 = 4.98e6 - (.05+rand/1000)*np.square(rh0)
gp0 = np.full(len(rh0), 0.01)+rand/10
cp0 = np.full(len(rh0), 0.01)+rand/10
gmax0 = 100 - 9*np.power(rh0, 1/2)+rand/3
d0 = 1e-9 + 1e-8*np.square(rh0)



params0 = np.column_stack((rh0, gp0, cp0, gmax0, d0, f0))


config_plot('$\Delta$f (Hz/cm$^2$)', 'Conductance (mS)',
            setlimits=True, limits=[-700, 100, -0.5, 75])

for i in params0:
    peak = singleBvD_reY(freq, *i[1:])
    plt.plot(freq - 4.98e6, peak)

plt.show()




#%%

save_first_params = np.empty((len(params0[0]), 0))

#loop over time (how many peaks are used for training)
for i in range(len(rh0)):
    print(i)
    if i < 3:
        peak_old = singleBvD_reY(freq, *params0[i,1:])
        plt.plot(freq - 4.98e6, peak_old, c='k', alpha=0.3,
                 label='measured data')
    else:
        new_params = multiparameter_extrapolate(params0[:i],
                                                params0[i:,0],
                                                splineorder=2)
        #loop over each training peak
        for j in range(i):
            peak_old = singleBvD_reY(freq, *params0[j,1:])
            
            if j ==0:
                plt.plot(freq - 4.98e6, peak_old, c='k', alpha=0.3,
                         label='measured data')
            else:
                plt.plot(freq - 4.98e6, peak_old, c='k', alpha=0.3)
                
        #loop over each predicted peak
        for kk, new_params0 in enumerate(new_params):
            pred_peak0 = singleBvD_reY(freq, *new_params0[1:])
            
            
            if kk == 0:
                plt.plot(freq - 4.98e6, pred_peak0, c='g',
                         alpha = 1-kk/len(new_params), label='prediction')
            else: 
                plt.plot(freq - 4.98e6, pred_peak0, c='g',
                         alpha = 1-kk/len(new_params))
        plt.text(-670, 40, 'predicted spectra: '+str(len(rh0)-i), fontsize=10)
    config_plot('$\Delta$f (Hz/cm$^2$)', 'Conductance (mS)',
            setlimits=True, limits=[-700, 100, -0.5, 75])
     
    plt.text(-670, 48, 'training spectra: '+str(i+1), fontsize=10)
    
    plt.title('%i%% RH' %rh0[i], fontsize=12)
    plt.legend(loc='upper left', fontsize=10)
    
    plt.gcf().set_size_inches(4,3)
    save_pic_filename = 'exp_data\\save_predicted_peaks\\'+str(i).zfill(3)+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=250)

    plt.show()



#%%

make_video = True
if make_video:
    create_video('exp_data\\save_predicted_peaks\\',
                 'C:\\Users\\a6q\\Desktop\\predicted_single_peak.avi', fps=4)
















