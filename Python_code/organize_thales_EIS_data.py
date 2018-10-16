import csv, glob, os, sys, numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import timeit
from scipy import optimize
from numpy import NaN, Inf, arange, isscalar, asarray, array
from timeit import default_timer as timer


def label_axes(xlabel='x', ylabel='y', size=16):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)


#%% imnport data

#filename with Bode plots
bodefile = 'exp_data\\2018-10-04_pp_qcm_bodelist.csv' 

#read in file
bodedata = pd.read_csv(bodefile, skiprows = 16, skip_blank_lines=True,
                       error_bad_lines=False, warn_bad_lines=False, sep=',') 

#str to flatt, coerce to NaN, drop NaN
bodedata = bodedata.apply(pd.to_numeric, errors='coerce').dropna() 

#convert data columns to numpy arrays
pointnum = np.array(bodedata['Number']).astype(float)
freq = np.array(bodedata['Frequency/Hz']).astype(float)
ztot = np.array(bodedata['Impedance/Ohm']).astype(float)
phasedeg = np.array(bodedata['Phase/deg']).astype(float)


#%% organize data

pointnum = np.unique(pointnum) #retain only one array of point numbers

freq = freq[0:len(pointnum)] #retain only one array of frequencies
#freq = np.flipud(freq) #(comment or uncomment this as needed) to flip frquency values 

#reshape to separate spectra
ztot = np.reshape(ztot, (len(pointnum), -1), order='F')
phasedeg = np.reshape(phasedeg, (len(pointnum), -1), order='F')
omega = 2*np.pi*freq #define angular frequency 
spectranum = len(ztot[0]) #number of spectra present

#delay in minutes between spectra measurements
delay = 15
time = np.arange(spectranum)*delay + delay
time = time/60

# calculate Re(Z) and Im(Z) using phase information
phase = np.divide(np.multiply(np.pi, phasedeg), 180)
rez = np.multiply(ztot, np.cos(phase))
imz = np.multiply(ztot, np.sin(phase))

#%% plot results 

#plot Bode Z data
[plt.loglog(freq, ztot[:,i]) for i in range(spectranum)]
label_axes('Frequency (Hz)', 'Z ($\Omega$)')
plt.show()

#plot Bode phase data
[plt.semilogx(freq, phasedeg[:,i]) for i in range(spectranum)]
label_axes('Frequency (Hz)', 'Phase (deg)')
plt.show()

#create Nyquist plots
[plt.scatter(rez[:,i], imz[:,i], s=2) for i in range(spectranum)]
label_axes('Re(Z) ($\Omega$)', 'Im(Z) ($\Omega$)')
plt.show()

#get z values at lowest frequencies measured
lowfreqz = ztot[-1,:]
plt.semilogy(time, lowfreqz)
label_axes('Time (hrs)', 'Z @ lowest freq. ($\Omega$)')
plt.show()



#%% format data into Origin-ready matrices

all_bode_z = freq
all_bode_phase = freq
all_nyquist = freq

for i in range(spectranum):
    all_bode_z = np.column_stack((all_bode_z, ztot[:,i]))
    all_bode_phase = np.column_stack((all_bode_phase, phasedeg[:,i]))
    
    all_nyquist = np.column_stack((all_nyquist, rez[:,i]))
    all_nyquist = np.column_stack((all_nyquist, imz[:,i]))


#%% save results in csv file
'''
#open CSV file for saving spectra
fileheaders = ['frequency'] # build list of file headers
savedata = np.copy(freq) #build list of data

for i in range(spectranum):
    fileheaders.append('Ztot_' + format(i))
    fileheaders.append('ReZ_' + format(i))
    fileheaders.append('ImZ_' + format(i))
    fileheaders.append('Phase_' + format(i))
    
    savedata = np.append(savedata, ztot[i])
    savedata = np.append(savedata, rez[i])
    savedata = np.append(savedata, imz[i])
    savedata = np.append(savedata, phase[i])
    
savedata = savedata.reshape(-1, len(pointnum))
savedata = np.transpose(savedata)

with open('EISspectra.csv', 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(fileheaders) #write headers
    for row in savedata:
        writer.writerow(row) 
'''

'''
#open CSV file for saving low-frequency Z values
zheaders = ['time', 'low-freq. Z (Ohms)'] 
savedata2 = np.append(time, lowfreqz)
savedata2 = savedata2.reshape(-1, spectranum)
savedata2 = np.transpose(savedata2)

with open('EISlowFreqZ.csv', 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(zheaders) #write headers
    for row in savedata2:
        writer.writerow(row)
'''