
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import Symbol, solve, nsolve
from scipy.optimize import fsolve
from scipy.optimize import leastsq
import math
from timeit import default_timer as timer

#make size of axis tick labels larger
fsize = 15
plt.rcParams['xtick.labelsize'], plt.rcParams['ytick.labelsize'] = fsize, fsize

#%% define functions

# calculate viscosity of liquid in contact with quartz oscillator
# using frequency shift (delta_f), overtone (n), and liquid density (rho).
# From Kanazawa-Gordon equation (4) in Kanazawa, K.K. and Gordon, J.G., 1985.
# Frequency of a quartz microbalance in contact with liquid. Analytical 
# Chemistry, 57(8), pp.1770-1771.
def compute_eta(delta_f, n, f0, rho):
    mu_q = 2.947e10 #shear modulus of quartz (kg/m*s^2, or Pa)
    rho_q = 2648 #density of quartz (kg/m^3)
    eta_numerator = np.square(delta_f) * np.pi * mu_q * rho_q
    eta_denominator = n * rho * np.power(f0, 3)  
    
    return np.divide(eta_numerator, eta_denominator)
    

def compute_eta_rho(delta_f, n, f0):
    mu_q = 2.947e10 #shear modulus of quartz (kg/m*s^2, or Pa)
    rho_q = 2648 #density of quartz (kg/m^3)
    numerator = np.square(delta_f) * np.pi * mu_q * rho_q
    denominator = n * np.power(f0, 3)  
    
    return np.divide(numerator, denominator)



#%% import the QCM-D data
import_data = pd.read_csv('2018-01-19_MXene_QCMD.csv')

#%% define frequency shift and resonant frequency at each overtone

time = import_data['time']
rh = import_data['rh']
delta_fs = import_data[['df1', 'df3', 'df5', 'df7', 'df9']].values
delta_Ds = import_data[['dd1', 'dd3', 'dd5', 'dd7', 'dd9']].values

res_fs = np.array([4984790, 14936300, 24888000, 
                   34836700, 44784300]).astype(float)
  
f0 = 5e6
#assume constant film density in kg/m^3
film_rho = 2000 

#%% Sauerbrey fits for freq shift
delta_fs_S = np.zeros_like(delta_fs)
delta_fs_Sdev = np.zeros_like(delta_fs)
calc_eta_mat = np.zeros_like(delta_fs)

for i in range(len(res_fs)):
    #multiply freq shifts by overtone number; this is predicted by Sauerbrey
    delta_fs_S[:,i] = delta_fs[:,0]*(2*i+1)
    #find deviation between Sauerbrey model and actual measured frequencies
    delta_fs_Sdev[:,i] = np.subtract(delta_fs[:,i], delta_fs_S[:,i])

    #calculate viscosity using deviant frequency shifts
    calc_eta_mat[:,i] = compute_eta_rho(delta_fs_Sdev[:,i], #delta_f
                           1,#2*i+1, #n
                           res_fs[i]) #f0


#%% plot results

#plot RH
plt.plot(time, rh)
plt.xlabel('Time (hours)', fontsize=fsize)
plt.ylabel('RH (%)', fontsize=fsize)
plt.show()  
  
#plot delta f
for i in range(len(res_fs)):
    plt.scatter(time, delta_fs[:,i], alpha=0.3, label=format(2*i+1))
    plt.plot(time, delta_fs_S[:,i], linewidth=2)
plt.xlabel('Time (hours)', fontsize=fsize)
plt.ylabel('$\Delta$f (Hz/cm$^2$)', fontsize=fsize)
plt.legend(fontsize=12)
plt.show()   
 
#plot delta f deviation from Sauerbrey
for i in range(len(res_fs)):
    plt.plot(time, delta_fs_Sdev[:,i], linewidth=2)
plt.xlabel('Time (hours)', fontsize=fsize)
plt.ylabel('Dev. from Sauerbrey (Hz/cm$^2$)', fontsize=fsize)
plt.show()

#plot delta D
for i in range(len(res_fs)):
    plt.plot(time, 1e6*delta_Ds[:,i], label=format(2*i+1))
plt.xlabel('Time (hours)', fontsize=fsize)
plt.ylabel('$\Delta$D (x10$^{-6}$)', fontsize=fsize)
plt.legend(fontsize=fsize)
plt.show()   

#plot claculated eta
for i in range(len(res_fs)):
    plt.plot(time, calc_eta_mat[:,i], label=format(2*i+1))
plt.xlabel('Time (hours)', fontsize=fsize)
plt.ylabel('$\eta$ x rho', fontsize=fsize)
plt.legend(fontsize=fsize)
plt.show() 
    









