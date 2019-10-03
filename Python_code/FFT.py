# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:17:10 2019
@author: ericmuckley@gmail.com

This module attempts to use noise of a QCM crystal to measure resonance peaks
by deconvoluting the noise spectrum using FFT.

The frequency resolution of the transform (the difference between the
frequencies of adjacent points in the calculated frequency spectrum) is
the reciprocal of the time duration of the signal.

Nyquist-Shannon sampling theorem:
    If a function X(t) contains no frequencies higher than B Hz, then X(t)
    can be determined by a set of points spaced 1/2B seconds apart.


For QCM with 100 MHz range, with 1 Hz resolution, we need 
~200 million samples per second (5 ns sampling frequency for 1 second).

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
from matplotlib import rcParams
fs = 18
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs

data_filename = 'C:\\Users\\a6q\\exp_data\\C100us_500KS_bareAu00001.txt'
#data_filename = 'C:\\Users\\a6q\\exp_data\\au.txt'
raw_data = pd.read_csv(data_filename, sep='\t', skiprows=4)



# number of samples
ns = len(raw_data)

# time increment per point
#dt = np.round(np.diff(raw_data['Time']), decimals=9)[-1]
dt = np.diff(raw_data['Time'])[-1]

# perform transform
sig_raw = raw_data['Ampl']
trans = scipy.fftpack.fft(sig_raw)

# remove 0th point and scale
sig_trans = 2.0/ns * np.abs(trans[0:int(ns/2)])[1:]

# format frequencies
time_trans = np.linspace(0.0, 1.0/(2.0*dt), int(ns/2))[1:]

print('Most sampled frequencies (MHz):')
print(time_trans[np.argsort(sig_trans)[::-1]][:20]/1e6)



plt.plot(time_trans, sig_trans)
plt.xlabel('Frequency (Hz)', fontsize=fs)
plt.ylabel('Amplitude', fontsize=fs)
plt.show()

plt.semilogy(time_trans, sig_trans)
plt.xlabel('Frequency (Hz)', fontsize=fs)
plt.ylabel('Amplitude', fontsize=fs)
plt.show()
