# coding: utf-8
import glob, os, sys
import numpy as np
import pandas as pd
from inspect import signature
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from timeit import default_timer as timer
from scipy import signal





#%%

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




def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    

def plot2(x, y, xlabel='x', ylabel='y',show=True,size=18,label='', color='k'):
    #set axes labels and size
    from matplotlib import rcParams
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.plot(x,y, c=color, label=str(label))
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    plt.legend(fontsize=size-4)
    if show == True:
        plt.show()
    else: pass








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






def multiBvD_reY(freq, *params):
    # Returns admittance spectrum with multiple peaks.
    # Spectrum calculation is taken from Equation (2) in:
    # Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009. Analyzing spur-distorted 
    # impedance spectra for the QCM. Journal of Sensors, 2009.
    # inputs:
    # Gp = conductance offset
    # Cp = susceptance offset
    # Gmax00 = maximum of conductance peak
    # D00 = dissipation
    # f00 = resonant frequency of peak (peak position) 
    
    #flat admittance signal
    Y = np.zeros_like(freq)
    
    #conductance and susceptance offsets 
    Gp = params[0]
    Cp = params[1]
        
    #create Gmax, D, and f for each peak
    for i in range(0, len(params)-2, 3):
        Gmax00 = params[i+2]
        D00  = params[i+3]
        f00 = params[i+4]
        #construct peak
        peak = Gmax00 / (1 + (1j/D00)*((freq/f00)-(f00/freq)))
        #add peak to spectrum
        Y = Y + peak
        
    #add offsets to spectrum
    Y = Y + Gp + 1j*2*np.pi*f*Cp

    return np.real(Y)



def multiBvD_imagY(freq, *params):
    # Returns admittance spectrum with multiple peaks.
    # Spectrum calculation is taken from Equation (2) in:
    # Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009. Analyzing spur-distorted 
    # impedance spectra for the QCM. Journal of Sensors, 2009.
    # inputs:
    # Gp = conductance offset
    # Cp = susceptance offset
    # Gmax00 = maximum of conductance peak
    # D00 = dissipation
    # f00 = resonant frequency of peak (peak position) 
    
    #flat admittance signal
    Y = np.zeros_like(freq)
    
    #conductance and susceptance offsets 
    Gp = params[0]
    Cp = params[1]
        
    #create Gmax, D, and f for each peak
    for i in range(0, len(params)-2, 3):
        Gmax00 = params[i+2]
        D00  = params[i+3]
        f00 = params[i+4]
        #construct peak
        peak = Gmax00 / (1 + (1j/D00)*((freq/f00)-(f00/freq)))
        #add peak to spectrum
        Y = Y + peak
        
    #add offsets to spectrum
    Y = Y + Gp + 1j*2*np.pi*f*Cp

    return np.imag(Y)





def sark_time_series(folder, col='Rs'):
    '''Reads in xlxs files produced from SARK-110 impedance
    analyzer and builds a Pandas dataframe out of them, with frequency
    as first column. This function reads every file inside the 
    'folder' variable.
    '''
    folder.sort(key=os.path.getmtime)
    #get frequencies
    freq = pd.read_csv(folder[-1], skiprows=1)['Freq(MHz)']
    #set up matrix to populate
    series = np.zeros((len(freq), len(folder)+1))
    series[:,0] = freq
    for i in range(len(folder)):
        print('spectrum '+format(i+1)+' / '+format(len(folder)))
        data0 = pd.read_csv(folder[i], skiprows=1)
        #populate columns of matrix
        series[:,i+1] = np.array(data0[col])
    return series






#%% import data from folder
folder = glob.glob('C:\\Users\\a6q\\2018-05-01pedotpss_labeled/*')#[0::30]
folder.sort(key=os.path.getmtime)
print('found ' + format(len(folder)) + ' spectra') 


#%% examine data

#indices to start, stop, and every nth points to skip per data file:
index1, index2, skip_n = 1200, 4100, 1

#get pressure values based on filenames
pressures = np.array([
        i.split('\\')[-1].split('.')[0] for i in folder]).astype(float)

#pressures = [60, 62.5, 65, 67.5, 70, 72.5, 75, 77.5, 80, 82.5, 85, 87.5,
#             90, 92.5, 95]


#Rs_mat = sark_time_series(folder, 'Rs')
#Xs_mat = sark_time_series(folder, 'Xs')



#%% loop over every data file 

all_R_fit = []; all_C_fit= []; all_L_fit = []

fitted_pressures = []


for i in range(23):#len(folder)):#[::6]
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
    
    #plot raw data
    #plot2(f, data['G'],'Frequency (MHz)', 'Conductance')
    #plot2(f, data['B'],'Frequency (MHz)', 'Susceptance')








    
    #---------detect peaks and use them as guesses for peak fitting ---------
    peaks_detected0,_ = detect_peaks(data['G'], .00005, f)
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
'''

plt.plot(fitted_pressures, all_C_fit) 
label_axes('RH (%)', 'C (F)')
plt.show()

plt.plot(fitted_pressures, all_L_fit)
label_axes('RH (%)', 'L (H)')
plt.show()

plt.plot(fitted_pressures, all_R_fit)
label_axes('RH (%)', 'R ($\Omega$)')
plt.show()



    
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

