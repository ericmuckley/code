# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:13:40 2019

@author: a6q
"""

import sys, glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import figure


import time
import datetime
from scipy.optimize import curve_fit

import scipy.signal as filt
from scipy.signal import savgol_filter

import scipy.interpolate as inter
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.interpolate import griddata



def get_eis_params(data0):
    # Calculates impedance parameters from SARK-110 csv file
    freq0 = np.array(data0['Freq(MHz)']*1e6)
    # complex impedance    
    Z = np.add(data0['Rs'], 1j*data0['Xs'])
    # complex admittance
    Y = np.reciprocal(Z)
    # conductance
    G = np.real(Y)
    # susceptance
    # B = np.imag(Y)
    # conductance shift
    # Gp = np.min(G)
    # susceptance shift
    # Cp = np.min(B)

    return freq0, G


def bvd_peak(freq, Gp, Cp, Gmax, D, f0):
    # Returns real part (conductance) of admittance spectrum with single peak,
    # equivalent to Butterworth van Dyke equivalent circuit model fitting.
    # Spectrum calculation is taken from Equation (2) in:
    # Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009. Analyzing spur-distorted 
    # impedance spectra for the QCM. Journal of Sensors, 2009.
    # Gp = conductance offset
    # Cp = susceptance offset
    # Gmax = maximum of conductance peak
    # D = dissipation
    # f0 = resonant frequency of peak (peak position) 
    peak = Gmax / (1 + (1j/D)*((freq/f0)-(f0/freq)))  # construct peak
    Y = Gp + peak+ 1j * 2 * np.pi * freq * Cp   # add offsets to spectrum
    G = np.real(Y)
    return G

def normalize_vec(vec):
    # normalize a vector from 0 to 1
    vec2 = vec - min(vec)
    vec2 = vec2 / max(vec2)
    return vec2

#%%


raw_df = pd.read_csv(r'C:\Users\a6q\exp_data\example_qcm_spectrum.csv',
                     skiprows=1)

_ , g = get_eis_params(raw_df)

df = raw_df.copy()
df['g'] = g

freq = np.array(df['Freq(MHz)'])*1e6

specs = ['Rs', 'Xs', 'g']
 
for s0 in specs:
    y = df[s0]
    y = normalize_vec(y)
    plt.plot(freq, y, label=s0)

# resonant frequency
f0 = freq[np.argmax(g)]
# frequency of highest reactance value
f0_xs = freq[np.argmax(df['Xs'])]
# estimation FWHM of the resonant peak
fwhm = 2 * np.abs(f0 - f0_xs)
# create guess for dissipation
D_guess = fwhm / f0

# construct guess for peak fitting: [Gp, Cp, Gmax, D, f0]
guess = np.array([0, 0, np.amax(g), D_guess, f0])
 
#use previous fit params as guess for next ones
#if int(i/band_num) != 0: guess = popt
#print('peak fit guess = '+format(guess))


def perform_bvd_fit(freq, g, guess):
    # perform Butterworth van Dyke fitting of QCM spectrum
    # perform fit
    popt, _ = curve_fit(bvd_peak, freq, g, p0=guess)
    # get fitted peak model
    fit = bvd_peak(freq, *popt)
    return popt, fit
    


fit_params, fit = perform_bvd_fit(freq, g, guess)

plt.plot(freq, fit, label='fit')


plt.legend()
plt.show()







