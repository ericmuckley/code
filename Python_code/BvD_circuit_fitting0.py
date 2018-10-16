# coding: utf-8
import glob, os, sys
import numpy as np
import pandas as pd
from inspect import signature
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from timeit import default_timer as timer
from scipy import signal





#%%

def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    from matplotlib import rcParams
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    

def plot2(x, y, xlabel='x', ylabel='y',show=True,size=18,label='',color='k'):
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
        peak = Gmax00 / (1 + 1j*(1/D00)*((freq/f00)-(f00/freq)))
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
        peak = Gmax00 / (1 + 1j*(1/D00)*((freq/f00)-(f00/freq)))
        #add peak to spectrum
        Y = Y + peak
        
    #add offsets to spectrum
    Y = Y + Gp + 1j*2*np.pi*f*Cp

    return np.imag(Y)





#%% import data from folder
folder = glob.glob('C:\\Users\\a6q\\2018-04-30pedotpss15MHz_labeled/*')#[0::30]
folder.sort(key=os.path.getmtime)
print('found ' + format(len(folder)) + ' spectra') 


#%% examine data

#indices to start, stop, and every nth points to skip per data file:
index1, index2, skip_n = 3, -1, 2

data_raw = pd.read_csv(folder[5], skiprows=1)
data = data_raw.iloc[index1:index2:skip_n,:]
#rename frequency column
data = data.rename(columns={'Freq(MHz)': 'freq'}) 



#%%calculate parameters

f = np.array(data['freq']*1e6)
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

#plot2(f, data['G'],'Frequency (MHz)', 'Conductance')
#plot2(f, data['B'],'Frequency (MHz)', 'Susceptance')




#%% construct guess parameters for BvD fitting
 
# guess starts with Gp and Cp
guess = np.array([np.min(data['G']), 1e-6])

#append peak params to guess
g_max_guess = [2e-4, 0.5e-4]
D_guess = [8e-4, 4e-4]
f_guess = [14.95e6, 14.97e6]
for i in range(len(f_guess)):
    guess = np.append(guess, [g_max_guess[i], D_guess[i], f_guess[i]])



#%% fit BvD circuit to experimental data

#fit to real conductance (G)
popt_real, _ = curve_fit(multiBvD_reY, f, data['G'], p0=guess)
G_fit = multiBvD_reY(f, *popt_real)

#fit to imaginary susceptance (B)
popt_imag, _ = curve_fit(multiBvD_imagY, f, data['B'], p0=guess)
B_fit = multiBvD_imagY(f, *popt_imag)



#%% plot fits to experimental data

#plot fit to real conductance (G)
plt.scatter(f, data['G']*1e3, s=3, c='k', alpha=0.3, label='exp.')
plot2(f, G_fit*1e3, 'Frequency (MHz)', 'G (mS)', color='r', label='fit')

#plot fit to imaginary susceptance (B)
plt.scatter(f, data['B']*1e3, s=3, c='k', alpha=0.3, label='exp.')
plot2(f, B_fit*1e3, 'Frequency (MHz)', 'B (mS)', color='r', label='fit')





#%% calculate equivalent circuit parameters from BvD fits
# from Yoon et al., Analyzing Spur-Distorted Impedance 
# Spectra for the QCM, Eqn. 3.

R_fit = []; L_fit = []; C_fit = []
#deconstruct popt to calculate equivalent circuit parameters
for i in range(len(f_guess)):
    G0 = popt_real[3*i+2] 
    D0 = popt_real[3*i+3] 
    f0 = popt_real[3*i+4] 
    L0 = 1 / (2 * np.pi * f0 * D0 * G0)
    C0 = 1 / (4 * np.pi**2 * f0**2 * L0)
    R_fit.append(1/G0)
    L_fit.append(L0)
    C_fit.append(C0)
    
   
    

    
    
    
    
#%%check to see if data can be reproduced after converting to circuit params 
# from Yoon et al., Analyzing Spur-Distorted Impedance 
# Spectra for the QCM, Eqn. 1.  
'''
peak0 = 1 / (1j*2*np.pi*f*L_fit[0] + (1/(1j*2*np.pi*f*C_fit[0])) + R_fit[0])
peak1 = 1 / (1j*2*np.pi*f*L_fit[1] + (1/(1j*2*np.pi*f*C_fit[1])) + R_fit[1])

Ysim = popt_real[0] + 1j*2*np.pi*f*popt_imag[1] + peak0 + peak1

print('Y sim')
#plot fit to real conductance (G)
plt.scatter(f, data['G']*1e3, s=3, c='k', alpha=0.3, label='exp.')
plot2(f, Ysim.real*1e3, 'Frequency (MHz)', 'G (mS)', color='r', label='fit')


#plot fit to imaginary susceptance (B)
plt.scatter(f, data['B']*1e3, s=3, c='k', alpha=0.3, label='exp.')
plot2(f, Ysim.imag*1e3, 'Frequency (MHz)', 'B (mS)', color='r', label='fit')
'''

