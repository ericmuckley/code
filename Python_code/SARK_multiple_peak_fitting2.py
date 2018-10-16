import csv, glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.signal import medfilt
from scipy.optimize import curve_fit
import sys

from matplotlib import rcParams
labelsize = 18 #make size of axis tick labels larger
plt.rcParams['xtick.labelsize'] = labelsize 
plt.rcParams['ytick.labelsize'] = labelsize

#%% peak finding algorithm 
def detect_peaks(v, delta, x = None): 
    ''' detect peaks with x = vector of x values, v = vector of y values,
    delta = minimum peak height to detect.
    from https://gist.github.com/endolith/250860
    '''
    maxtab = []
    mintab = []
    if x is None:
        x = np.arange(len(v))
    v = np.asarray(v) 
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')  
    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')   
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN   
    lookformax = True
    
    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]       
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True              
    return np.array(maxtab), np.array(mintab)





#%% define lorentz function

def lorentz(x, x0, w, amp):
    lorentz_numerator = amp * np.square(w)
    lorentz_denominator = np.pi*w*(np.square(x-x0)+np.square(w))
    return np.divide(lorentz_numerator, lorentz_denominator)

def multilorentz(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp  = params[i+1]
        wid = params[i+2]
        numerator = np.square(wid) * amp
        denominator = np.pi*wid*(np.square(x-ctr)+np.square(wid))
        y = y + np.divide(numerator, denominator)
    return y

def gauss(x, mean, stdev, intensity):
    prefactor = intensity * np.divide(1, np.sqrt(2*np.pi*np.square(stdev)))
    exponent = -np.divide(np.square(x-mean),2*np.square(stdev))
    return prefactor * np.exp(exponent)

def multigauss(x, *params):
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        ctr = params[i]
        amp = params[i+1]
        wid = params[i+2]
        y = y + amp * np.exp( -((x - ctr)/wid)**2)
    return y


#%% set fitting parameters

fitting_function = multilorentz
peaks = 3 #specify number of peaks

lowbounds=np.zeros(peaks*3)
highbounds=np.copy(lowbounds)

if fitting_function == multigauss:
    
    #for Gp 
    guess = [24.93, 1, 0.01, 
             24.95, 1, 0.01,
             24.96, .3, 0.005,
             #44.813, .1, 0.004,
             24.98, 1, 0.001]
    for i in range(0,len(lowbounds), 3):
        lowbounds[i] = 44.76 #center low
        lowbounds[i+1] = 10 #amplitude low
        lowbounds[i+2] = 0.00005 #width low
        highbounds[i] = 44.86 #center high
        highbounds[i+1] = 100 #amplitude high
        highbounds[i+2] = 0.04 #width high
    
    
    
if fitting_function == multilorentz:
    
    #for Gp 
    guess = [24.93, 4, 0.005, 
             24.95, .5, 0.005,
             24.97, .3, 0.005]
    for i in range(0,len(lowbounds), 3):
        lowbounds[i] = 24.81 #center low
        lowbounds[i+1] = 0.0001 #amplitude low
        lowbounds[i+2] = 0.00005 #width low
        highbounds[i] = 25 #center high
        highbounds[i+1] = 20 #amplitude high
        highbounds[i+2] = 0.1 #width high
    

fitbounds = (lowbounds, highbounds)


#%% find all impedance files in the designated folder and sort by time/date
folder = sorted(glob.glob('C:\\Users\\a6q\\2018-05-01pedotpss_labeled/*'))#[0::30]
#folder.sort(key=os.path.getmtime)
print('found ' + format(len(folder)) + ' impedance files') 

#%% find size of each data file

#indices to start, stop, and every nth points to skip per data file:
index1, index2, skip_n = 1200, 4100, 1


data_example_full = pd.read_csv(folder[0], skiprows=1)
data_example = data_example_full.values[index1:index2:skip_n,:]

freq = data_example[:,0]#*1e3 #frequencies in kHz

#get pressure values based on filenames
pressures = np.array([
        i.split('\\')[-1].split('.')[0] for i in folder]).astype(float)

#%% iterate over each data file
starttime = time.time()
var_list = np.zeros((len(freq), len(folder)+1))
var_list[:,0] = freq
var_max_list = np.array([])
res_freq_list = np.array([])
all_centers = np.zeros((len(folder), peaks))
all_amplitudes = np.zeros((len(folder), peaks))
all_widths = np.zeros((len(folder), peaks))
single_peaks = np.zeros((len(freq), peaks))

#iterate over data files
for i in range(len(folder)):
    print('spectrum '+format(i+1)+' / '+format(len(folder))+', '+
          format(pressures[i])+'% RH ----------------------------')

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
    
    
    spec = data['G']
    
    
    
    
    #---------detect peaks and use them as guesses for peak fitting ---------
    peaks_detected0,_ = detect_peaks(spec, .00002, freq)
    #print('raw peaks = '+format(peaks_detected0))
    #sort detected peaks by amplitude high to low    
    peaks_detected = peaks_detected0[peaks_detected0[:,1].argsort()[::-1]]
    
    #print('sorted peaks = ') #print coords of highest peaks
    #print(peaks_detected[:3,:])
    
    guess = [peaks_detected[0,0], peaks_detected[0,1], .02,#1*(i+1)*0.001, 
             peaks_detected[1,0], peaks_detected[1,1], .02,#1*(i+1)*0.001,
             peaks_detected[2,0], peaks_detected[2,1], .02]#1*(i+1)*0.001]

    ''' for 90%, and use 0.002 for peak detection limit
    guess = [24.9, .02, .05,#1*(i+1)*0.001, 
         24.93, .05, .05,#1*(i+1)*0.001,
         24.96, .02, .05]#1*(i+1)*0.001]
   '''
    
   
   
    var_list[:,i+1] = spec
    res_freq_list = np.append(res_freq_list, freq[np.argmax(spec)])
    
    plt.scatter(freq, spec, c='k', s=3, alpha=.3, label='data')
    
    
    
    
    
    
    
    #--- fit spectra ----------------------------------------------------
    popt0, pcov = curve_fit(fitting_function, freq, spec,
                           p0=guess)#, bounds=fitbounds)#, ftol=5e-6, xtol=1e-7)
    
    #sort peaks in order from low to high
    popt = np.reshape(popt0, (peaks, -1))
    popt = popt[popt[:,0].argsort()]
    popt = np.reshape(popt, (len(popt0)))

    calculated_fit = fitting_function(freq, *popt)
    
    
    plt.plot(freq, calculated_fit, c='r', label='fit')
    #print('standard errors from fitting = '+format(np.sqrt(np.diag(pcov))))
    
    # save fit parameters
    centers0 = np.array([]); amplitudes0 = np.array([]); widths0 = np.array([])
    for j in range(0,len(popt),3):
        centers0 = np.append(centers0, popt[j])
        amplitudes0 = np.append(amplitudes0, popt[j+1])
        widths0 = np.append(widths0, popt[j+2])    




    #plot deconvoluted peaks----------------------------------------------
    for k in range(len(centers0)):
        if fitting_function == multilorentz:
            single_peak = np.divide(amplitudes0[k]*np.square(widths0[k]),
                                    np.pi*widths0[k]*(np.square(freq-centers0[k])+np.square(widths0[k])))
            
        if fitting_function == multigauss:
            single_peak = amplitudes0[k]*np.exp(-((freq-centers0[k])/widths0[k])**2)
         
        single_peaks[:,k] = single_peak
        plt.plot(freq, single_peak, linewidth=1, label=format(k))
    
    #save deconvoluted peak info----------------------------------------------
    all_centers[i,:] = centers0
    all_amplitudes[i,:] = amplitudes0
    all_widths[i,:] = widths0
    
    
    plt.title(format(int(pressures[i]))+'% RH', fontsize=18)
    #plt.xlim((24.812, 24.99))
    #plt.ylim((-.05, 3))
    plt.xlabel('Delta F (MHz)', fontsize=labelsize)    
    plt.ylabel('Signal', fontsize=labelsize)
    #plt.legend()
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    #save_pic_filename = 'frames_peak_fitting\\frame_'+format(i)+'.jpg'
    #plt.savefig(save_pic_filename, format='jpg', dpi=150)
    plt.show()
    plt.close()
    

    
endtime = time.time()
tottime = (endtime-starttime)/60
print('elapsed time = %.2f minutes' %tottime)





#%% show trends in peak parameters
spec_num = np.arange(len(folder))+1

for i in range(len(all_centers[0])):
    plt.plot(pressures, all_centers[:,i])
    plt.scatter(pressures, all_centers[:,i], label=format(i))
plt.xlabel('RH (%)', fontsize=labelsize)
plt.ylabel('Center', fontsize=labelsize)
plt.legend(); plt.show()

for i in range(len(all_centers[0])):
    plt.plot(pressures, all_amplitudes[:,i])
    plt.scatter(pressures, all_amplitudes[:,i], label=format(i))
plt.xlabel('RH (%)', fontsize=labelsize)
plt.ylabel('Amplitude', fontsize=labelsize)
plt.legend(); plt.show()

for i in range(len(all_centers[0])):
    plt.plot(pressures, all_widths[:,i])
    plt.scatter(pressures, all_widths[:,i], label=format(i))
plt.xlabel('RH (%)', fontsize=labelsize)
plt.ylabel('Width', fontsize=labelsize)
plt.legend(); plt.show()



#%%
plt.plot(var_list[:,0], var_list[:,14])
plt.plot(var_list[:,0], var_list[:,15])
plt.plot(var_list[:,0], var_list[:,16])
plt.plot(var_list[:,0], var_list[:,17])
plt.plot(var_list[:,0], var_list[:,18])
plt.show()


#%% save frames to make gif:
'''
pressures = [2,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90]

#rcParams['figure.figsize'] = 6, 8

plt.figure(figsize=(6,6))
for i in range(len(folder)):
    print('spectrum '+format(100+i+1)+' / '+format(len(folder)))

    #organize data from file
    data_raw = pd.read_excel(folder[i], skiprows=3)
    freq=data_raw['Freq']
    spec = 1e3*np.array(data_raw['Gp'])
    spec = spec - np.min(spec)
    
    plt.figure(figsize=(6,6))
    plt.plot(freq, spec, c='k', lw=1)
    plt.xlabel('Frequency (Hz)', fontsize=18)
    plt.ylabel('Conductance (mS)', fontsize=18)
    plt.xlim((24.9, 24.99))
    plt.ylim((-.05, 6))

    
    plt.title(format(pressures[i])+' % RH', fontsize=18)
    
    #save plot as image file
    save_pic_filename = 'gif_frames_2018-04-13pedotpss\\frame_'+format(i)+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)
    

    
    plt.show()
plt.close()
plt.close("all")
'''



