# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:29:50 2018

@author: a6q
"""
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def label_axes(xlabel='x', ylabel='y', size=16):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)

def linear(t, m, b):
    return m * t + b

def single_exp(t, A1, tau1, y0):
    return A1 * np.exp(-(t) / tau1) + y0

def double_exp(t, A1, tau1, A2, tau2, y0):
    return A1 * np.exp(-(t) / tau1) + A2 * np.exp(-(t) / tau2) + y0

def triple_exp(t, A1, tau1, t01, A2, tau2, t02, A3, tau3, t03, y0):
    return A1*np.exp(-(t-t01)/tau1) + A2*np.exp(-(t-t02)/tau2) + A3*np.exp(-(t-t03)/tau3) + y0

def rsquared(y, fit):
    #r^2 value for quantifying fit quality
    return 1 - (np.sum((y - fit)**2) / np.sum((y - np.mean(y))**2))

def MSE(y,y0):
    if len(y.shape) > 1:
        y = np.squeeze(y)
    if len(y0.shape) > 1:
        y0 = np.squeeze(y0)
    return (np.average((y - y0)**2))/np.average(y)*100





#%% import data
filename = r'C:\Users\a6q\exp_data\sect_FA70.txt'
data = pd.read_csv(filename, header=None).values

time = (np.arange(28)+1)*4.15
distance = (np.arange(100)+1)*0.3125

taus = np.array([])
errors = np.array([])

for i in range(len(data[0])):
    
    sig = data[:,i]
    
    
    #construct guess for fit
    guess = [.01, 50, np.min(sig)]
    
    #fit data
    popt, pcov = curve_fit(single_exp, time, sig, p0=guess)#, ftol=1e-14, xtol=1e-14,)
    
    fit = single_exp(time, *popt)
    
    taus = np.append(taus, popt[1])
    errors = np.append(errors, MSE(sig, fit))
    
    
    
    plt.scatter(time, sig)
    plt.plot(time, fit)
    label_axes('Time (sec)', 'Signal')
    plt.show()



