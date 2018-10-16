import glob, os, sys
import numpy as np
import pandas as pd
from inspect import signature
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from timeit import default_timer as timer
from scipy import signal
import scipy.interpolate as inter
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.signal import savgol_filter

def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    from matplotlib import rcParams
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    

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



def get_eis_params(data0):
    '''Calculates impedance parameters'''
    freq0 = np.array(data0['Freq(MHz)']*1e6)
    #complex impedance    
    Z = np.add(data0['Rs'], 1j*data0['Xs'])
    #complex admittance
    Y = np.reciprocal(Z)
    #conductance
    G = np.real(Y)
    #susceptance
    #B = np.imag(Y)
    #conductance shift
    #Gp = np.min(G)
    #susceptance shift
    #Cp = np.min(B)

    return freq0, G




def remove_outliers(spectra, num_of_outliers=5):
    '''Remove the low outliers from spectra - these outliers are an
    artifact of SARK-110 measurements '''
    
    spectra2 = np.copy(spectra)
    
    #replace lowest outlier points with average of adjacent points
    for i in range(num_of_outliers):  
        
        #find index of minimum point
        min_index = np.argmin(spectra2)
        
        #make sure minimum is not first or last point
        if min_index != 0 and min_index != len(spectra2) - 1:
        
            #remove the mimimum points 
            spectra2[min_index] = (
                    spectra2[min_index-1] + spectra2[min_index+1]) / 2
            
    return spectra2



def get_rlc(fit_params0):
    # calculate equivalent circuit parameters from BvD fits
    # from Yoon et al., Analyzing Spur-Distorted Impedance 
    # Spectra for the QCM, Eqn. 3.
        
    #R_fit = []; L_fit = []; C_fit = []
    #deconstruct popt to calculate equivalent circuit parameters
    '''
    for k in range(len(guess0)):
        G0 = popt_real[3*k+2] 
        D0 = popt_real[3*k+3] 
        f0 = popt_real[3*k+4] 
        L0 = 1 / (2 * np.pi * f0 * D0 * G0)
        C0 = 1 / (4 * np.pi**2 * f0**2 * L0)
        R_fit.append(1/G0)
        L_fit.append(L0)
        C_fit.append(C0)
    '''
    
    
    #FOR SINGLE BvD PEAK:
    #FIT PARAMS = [Gp, Cp, Gmax00, D00, f00]
    
    G0 = fit_params0[2]
    f0 = fit_params0[4]
    D0 = fit_params0[3]
    
    R = 1/G0
    L = 1 / (2 * np.pi * f0 * D0 * G0)
    C = 1 / (4 * np.pi**2 * f0**2 * L)

    return R, L, C, D0






#%% import data from folder
    
folder = glob.glob(
        'C:\\Users\\a6q\\Desktop\\Wanyi_perovskite_thin_films\\Wanyi_QCM/*')

print('found ' + format(len(folder)) + ' spectra') 


#indices to start, stop, and every nth points to skip per data file:
index1, index2, skip_n = 5, -1, 1


peak_params = []
file_params = []



#%% loop over every data file 

#loop over each file
for i in range(len(folder)):
    print('spectrum '+format(i+1)+' / '+format(len(folder)))
    
    #get metadata from filenames
    envir = folder[i].split('_')[4].split('\\')[1]
    sample = folder[i].split('_')[5]
    harmonic = folder[i].split('_')[6].split('.')[0]
    
    
    
    #open file
    data = pd.read_csv(folder[i], skiprows=1).iloc[index1:index2:skip_n,:]
    freq0, G0 = get_eis_params(data)
    
    
    
    #create spline fit for conducance
    #G0 = remove_outliers(G0)
    #spline_fit = inter.UnivariateSpline(freq0, G0, s=1e-5)
    #G0_spline = spline_fit(freq0)
    #Savitzky-Golay filter

    G0_savgol = savgol_filter(G0, 23, 2)#, mode='nearest')
    #G0 = G0_savgol

    #plt.plot(freq0, G0_spline, c='k', label='spline')
    #plt.plot(freq0, G0_savgol, c='y', label='savgol')
    
    
    #guesses for fitting
    Gmax_guess = np.max(G0)
    D_guess = 1e-6
    f_guess = freq0[np.argmax(G0)]
    guess0 = [1e-6, 1e-6, Gmax_guess, D_guess, f_guess]
    print(guess0)
    
    
    
    #fit to real conductance (G)
    winsize = 10
    
    if G0[0] > G0[-1]:
        freq_window = freq0[np.argmax(G0)-winsize:]
        G_window = G0[np.argmax(G0)-winsize:]
        
        
        
    if G0[0] <= G0[-1]:
        freq_window = freq0[:np.argmax(G0)+winsize]
        G_window = G0[:np.argmax(G0)+winsize]

    
    
    
    
    fit_params, _ = curve_fit(singleBvD_reY, freq_window,
                              G_window,
                              p0=guess0,
                             ftol=1e-14, xtol=1e-14)#, bounds=(0, np.inf))
    G_fit = singleBvD_reY(freq0,#freq_window
                          *fit_params)    
    
    plt.plot(freq0,#,freq_window
             G_fit, c='r', label='BVD fit')
    plt.scatter(freq0, G0, alpha=.2, label='data', marker='.')
    plt.title(format(harmonic)+', '+format(sample)+', '+format(envir), fontsize=18)
    label_axes('Frequency (Hz)', 'G (S)')
    plt.legend(fontsize=12)
    plt.show()
    
    
    R, L, C, D = get_rlc(fit_params)
    
    
    peak_params.append([sample, envir, harmonic, f_guess,
                        R, L, C, D])
    
params_all = np.array(peak_params)  
 

#%%  

params_air = params_all[0:21,:]
params_vac = params_all[21:40,:]
params_all2 = np.reshape(params_all,(int(len(params_all)/2),-1), order='F')
 
#df = pd.(data=peak_params, columns=[]

#%%    
'''    
    data_raw = pd.read_csv(folder[i], skiprows=1)
    data = data_raw.iloc[index1:index2:skip_n,:]



    #calculate parameters################################################
    f = np.array(data['Freq(MHz)']*1e6)
    #complex impedance
    data['Z'] = np.add(data['Rs'], 1j*data['Xs'])
    #complex admittance
    data['Y'] = np.reciprocal(data['Z'])
    #conductance
    data['G']  = np.real(data['Y'])
    #susceptance
    data['B'] = np.imag(data['Y'])
    #conductance shift
    Gp = np.min(data['G'])
    #susceptance shift
    Cp = np.min(data['B'])
    
    #plot raw data
    #plot2(f, data['G'],'Frequency (MHz)', 'Conductance')
    #plot2(f, data['B'],'Frequency (MHz)', 'Susceptance')


    plt.plot(f, data['G']*1000)
    label_axes('Frequency (Hz)', 'Conductance (mS)')
    plt.show()



    
    #---------detect peaks and use them as guesses for peak fitting ---------
    peaks_detected0,_ = detect_peaks(data['G'], 0.000001, f)
    #print('raw peaks = '+format(peaks_detected0))
    #sort detected peaks by amplitude high to low    
    #peaks_detected = peaks_detected0[peaks_detected0[:,1].argsort()[::-1]]
    peaks_detected = peaks_detected0
    

    print('peaks found = '+format(len(peaks_detected[:,0])))
    print(peaks_detected)
    
    
    #plot peaks found by peak detection algorithm
    plt.scatter(peaks_detected[:,0], peaks_detected[:,1]*1000, label='peak', s=100, c='g', marker='*')
    
    
    #print('sorted peaks = ') #print coords of highest peaks
    #print(peaks_detected[:3,:])
    
    
    
    
    
    # construct guess parameters for BvD fitting#############################

    # guess starts with Gp and Cp
    guess = np.array([np.min(data['G']), 1e-6])
    
    
    
    #append peak params to guess
    
    g_max_guess = list(peaks_detected[:,1])
    f_guess = list(peaks_detected[:,0])
    #f_guess =[1.49596400e+07, 1.49778800e+07, 1.50056400e+07, 1.50402800e+07]
    
    #g_max_guess =[2.18869274e-02, 1.10357466e-03, 3.50685304e-03, 5.02348525e-04]
    
    
    #D_guess = [8e-4, 4e-4]
    D_guess = np.full(len(g_max_guess), 5e-4)
    
    
    for j in range(len(f_guess)):
        guess = np.append(guess, [g_max_guess[j], D_guess[j], f_guess[j]])

    #print('guess = '+format(guess))








    # fit BvD circuit to experimental data ##################################
    
    #fit to real conductance (G)
    popt_real, _ = curve_fit(multiBvD_reY, f, data['G'], p0=guess,
                             ftol=1e-14, xtol=1e-14, bounds=(0, np.inf))
    G_fit = multiBvD_reY(f, *popt_real)
    
    #fit to imaginary susceptance (B)
    popt_imag, _ = curve_fit(multiBvD_imagY, f, data['B'], p0=guess,
                             ftol=1e-14, xtol=1e-14)#, bounds=fitbounds)
    B_fit = multiBvD_imagY(f, *popt_imag)


    #print('real params = '+format(popt_real))
    #print('imag params = '+format(popt_imag))






    # calculate equivalent circuit parameters from BvD fits ##################
    # from Yoon et al., Analyzing Spur-Distorted Impedance 
    # Spectra for the QCM, Eqn. 3.
        
    R_fit = []; L_fit = []; C_fit = []
    #deconstruct popt to calculate equivalent circuit parameters
    for k in range(len(f_guess)):
        G0 = popt_real[3*k+2] 
        D0 = popt_real[3*k+3] 
        f0 = popt_real[3*k+4] 
        L0 = 1 / (2 * np.pi * f0 * D0 * G0)
        C0 = 1 / (4 * np.pi**2 * f0**2 * L0)
        R_fit.append(1/G0)
        L_fit.append(L0)
        C_fit.append(C0)
    
    
    #append fit parameters to lists
    all_R_fit.append(R_fit)
    all_C_fit.append(C_fit)
    all_L_fit.append(L_fit)
    
    fitted_pressures.append(pressures[i])







    # plot experimental data and fits #######################################
    
    #plot measured real conductance (G)
    plt.scatter(f, data['G']*1e3, s=3, c='k', alpha=0.3, label='exp.')
    #plot fit to real conductance (G)
    plt.plot(f, G_fit*1e3, color='r', label='fit')
    plt.title(format(pressures[i])+'% RH', fontsize=18)
    #plt.text(1.494e7, 2.25, 'HI')
    label_axes('Frequency (Hz)', 'G (mS)')
    plt.legend(fontsize=12)
    plt.show()


    #save_pic_filename = 'frames_peak_fitting\\frame_'+format(i)+'.jpg'
    #plt.savefig(save_pic_filename, format='jpg', dpi=150)



        
    #plot measured imaginary susceptance (B)
    plt.scatter(f, data['B']*1e3, s=3, c='k', alpha=0.3, label='exp.')
    #plot fit to imaginary susceptance (B)
    plt.plot(f, B_fit*1e3, color='r', label='fit')
    plt.title(format(pressures[i])+'% RH', fontsize=18)
    label_axes('Frequency (Hz)', 'B (mS)')
    plt.legend(fontsize=12)
    plt.show()

    



all_R_fit = np.array(all_R_fit)
all_C_fit = np.array(all_C_fit)
all_L_fit = np.array(all_L_fit)    
    

'''


#%% plot results of fitting over time

'''
R L C significnce from SRS QCM200 manual:
    
R (resistor) corresponds to the dissipation of the oscillation energy from
mounting structures and from the medium in contact with the crystal 
(i.e. losses induced by a viscous solution).

C (capacitor) corresponds to the stored energy in the oscillation and is 
related to the elasticity of the quartz and the surrounding medium.

L (inductor) corresponds to the inertial component of the oscillation, 
which is related to the mass displaced during the vibration.


plt.plot(fitted_pressures, all_C_fit) 
plt.scatter(fitted_pressures, all_C_fit) 
label_axes('RH (%)', 'C (F)')
plt.show()

plt.plot(fitted_pressures, all_L_fit)
plt.scatter(fitted_pressures, all_L_fit)
label_axes('RH (%)', 'L (H)')
plt.show()

plt.plot(fitted_pressures, all_R_fit)
plt.scatter(fitted_pressures, all_R_fit)

label_axes('RH (%)', 'R ($\Omega$)')
plt.show()
'''


    
#%%check to see if data can be reproduced after converting to circuit params 
# from Yoon et al., Analyzing Spur-Distorted Impedance 
# Spectra for the QCM, Eqn. 1.  
'''
peak0 = 1 / (1j*2*np.pi*f*L_fit[0] + (1/(1j*2*np.pi*f*C_fit[0])) + R_fit[0])
#peak1 = 1 / (1j*2*np.pi*f*L_fit[1] + (1/(1j*2*np.pi*f*C_fit[1])) + R_fit[1])

Ysim = popt_real[0] + 1j*2*np.pi*f*popt_imag[1] + peak0 #+ peak1

#plot fit to real conductance (G)
plt.scatter(f, data['G']*1e3, s=3, c='k', alpha=0.3, label='exp.')
plot2(f, Ysim.real*1e3, 'Frequency (MHz)', 'G (mS)', color='r', label='fit')


#plot fit to imaginary susceptance (B)
plt.scatter(f, data['B']*1e3, s=3, c='k', alpha=0.3, label='exp.')
plot2(f, Ysim.imag*1e3, 'Frequency (MHz)', 'B (mS)', color='r', label='fit')
'''

