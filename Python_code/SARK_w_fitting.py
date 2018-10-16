import csv, glob, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from matplotlib import rcParams
#make size of axis tick labels larger
labelsize = 18
plt.rcParams['xtick.labelsize'] = labelsize 
plt.rcParams['ytick.labelsize'] = labelsize

#%% define fitting functions

def lorentz(x, x0, w, intensity):
    lorentz_numerator = intensity + np.square(w)
    lorentz_denominator = np.pi*w*(np.square(x-x0)+np.square(w))
    return np.divide(lorentz_numerator, lorentz_denominator)

def gauss(x, mean, stdev, intensity):
    prefactor = intensity * np.divide(1, np.sqrt(2*np.pi*np.square(stdev)))
    exponent = -np.divide(np.square(x-mean),2*np.square(stdev))
    return prefactor * np.exp(exponent)

#%% pressure file
'''
pressure_file = pd.read_table('C:\\Users\\a6q\\2018-03-09mxeneRH', sep='\t')
pressure = pressure_file['abs_pressure']
time_raw = pressure_file['abs_time']
time0 = time_raw - np.min(time_raw)
'''
#%% find all impedance files in the designated folder
folder = glob.glob('C:\\Users\\a6q\\2018-04-10_pedotpss_labeled/*')#[0::3]
#sort files by time/date
folder.sort(key=os.path.getmtime)
print('found ' + format(len(folder)) + ' impedance files') 

#%% find size of each impedance file
data_example_full = pd.read_excel(folder[0], skiprows=3)
#skip first couple data points (sometimes they are overly noisy)
data_skip = 15 
data_example = data_example_full.values[data_skip:,:]
#frequencies in kHz
freq = data_example[:,0]*1e3 
#find resonant frequency of first spectrum
res_freq0 = freq[np.argmax(data_example_full['Rs'])]

#%% initlaize arrays for storing calculated values
var_list = np.empty((len(freq), len(folder)+1))
#set first column of saved spectra matrix to be frequency column
var_list[:,0] = freq
var_fit_list = np.copy(var_list)

var_max_list = np.array([])
var_FWHM_list = np.array([])
res_freq_list = np.array([])

#%% loop over each impedance file

starttime = time.time()

#set function to use for peak fitting
fitting_function = lorentz

for i in range(len(folder)):
    #print iteration number every 5 iterations
    #if i%5 == 0: print('iteration %d' %i)
    
    print(folder[i])
    
    data_raw = pd.read_excel(folder[i], skiprows=3)
    #set which variable to analyze
    var_spectrum_raw = 1e3*np.array(data_raw['Gp'])[data_skip:]
    
    #apply median filter to each spectrum
    var_spectrum_filt = var_spectrum_raw#medfilt(G_spectrum_raw, kernel_size=21)
    #subtract background signal
    var_spectrum = var_spectrum_filt# - np.min(var_spectrum_filt)
    plt.scatter(freq-res_freq0, var_spectrum, s=1)
    
    #fit each spectrum using fitting_function
    popt, pcov = curve_fit(fitting_function, freq-res_freq0, var_spectrum)
    plt.plot(freq-res_freq0, fitting_function(freq-res_freq0, *popt), linewidth=0.5)
    
    #save each spectrum and its fit
    var_list[:,i+1] = var_spectrum
    var_fit_list[:,i+1] = fitting_function(freq-res_freq0, *popt)
    
    #find resonant frequency of each spectrum
    res_freq_list = np.append(res_freq_list, popt[0])
    #find intensity of each peak
    var_max_list = np.append(var_max_list, popt[2])
    #find peak width
    var_FWHM_list = np.append(var_FWHM_list, popt[1])
    
plt.xlabel('Delta F (kHz)', fontsize=labelsize)
plt.ylabel('Delta signal', fontsize=labelsize)  
plt.show()


#%% plot results

#align pressure times and impedance times
sample_time = np.max(time0)*np.arange(len(folder))/len(folder)
#shift so time = 0 is the beginning
res_freq_list_corr = res_freq_list - res_freq_list[0]

fig, ax1 = plt.subplots()
ax1.plot(time0, pressure, linewidth=0.5, c='b')
ax1.set_xlabel('Time (min)', fontsize=labelsize)
ax1.set_ylabel('Pressure', color='b', fontsize=labelsize)

ax2 = ax1.twinx()
ax2.plot(sample_time, res_freq_list_corr, linewidth=0.5, c='r')
ax2.set_ylabel('Delta f (kHz)', color='r', fontsize=labelsize)
ax2.tick_params('y', colors='r')
plt.show()





fig, ax1 = plt.subplots()
ax1.plot(time0, pressure, linewidth=0.5, c='b')
ax1.set_xlabel('Time (min)', fontsize=labelsize)
ax1.set_ylabel('Pressure', color='b', fontsize=labelsize)

ax2 = ax1.twinx()
ax2.plot(sample_time,var_max_list, linewidth=0.5, c='r')
ax2.set_ylabel('Peak max', color='r', fontsize=labelsize)
ax2.tick_params('y', colors='r')
plt.show()





fig, ax1 = plt.subplots()
ax1.plot(time0, pressure, linewidth=0.5, c='b')
ax1.set_xlabel('Time (min)', fontsize=labelsize)
ax1.set_ylabel('Pressure', color='b', fontsize=labelsize)

ax2 = ax1.twinx()
ax2.plot(sample_time, var_FWHM_list, linewidth=0.5, c='r')
ax2.set_ylabel('FWHM', color='r', fontsize=labelsize)
ax2.tick_params('y', colors='r')
plt.show()


#%% export results to file
# format arrays to save so they are all the same length

with open('SARK_analysis_output.csv','w') as save_file:
    writer = csv.writer(save_file, lineterminator='\n')
    for row in var_fit_list:
        writer.writerow(row)
save_file.close()


endtime = time.time()
tottime = (endtime-starttime)/60
print('elapsed time = %.2f minutes' %tottime)
