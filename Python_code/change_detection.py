import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.signal import savgol_filter

def single_exp(x, a, tau, y0):
    return y0 + a*np.exp(-(x)/tau)




def get_baseline(x, y, samples=4):
    # get baseline response of a signal using the first samples
    pass



def get_slope(raw_sig, window=6):
    # get the slope of signal (raw_sig) using only the most
    # recent points (window).
    # get x values for slope calculation
    x = np.arange(window)
    # get sample of raw y values for slope calculation
    sample = raw_sig[-window:]
    # standard deviation in sample
    std = np.std(sample)
    # calculate slope
    slope = np.abs(linregress(x, sample)[0])
    return slope, std
   
    
def get_deviance(raw_sig, window=6):
    # get the deviance of a point from signal (raw_sig) using only the most
    # recent points (window).
    raw_sig = np.array(raw_sig)
    # get sample of raw signal values
    sample = raw_sig[-window:-1]
    mean = np.mean(sample)
    std = np.std(sample)
    # sample to test
    x = raw_sig[-1]
    
    # deviance between sample to test and mean
    dev = np.abs(x - mean - std)
    return dev
    




df = pd.read_table('exp_data\\cupcts_data.txt')[::10]
#df = pd.read_table('exp_data\\mxene_step_data.txt')[::1]



response0 = df['delta_m']
slopes = []
stds = [] 
devs = []

slope_limit = 0.01
window_len = 6
samples = 6

fig = plt.figure(figsize=(12, 5))


# loop over window of interest in response
for i in range(samples, len(response0)):

    response = response0[:i]
    time = df['time'][:i]
    
    slope0, std0 = get_slope(response)
    dev0 = get_deviance(response)
    stds.append(std0)
    slopes.append(slope0)
    devs.append(dev0)


plt.scatter(time, (response/30), c='k', alpha=0.5, s=3, label='signal')

plt.plot(time[samples-1:], slopes, c='b', lw=1, label='slope')
#plt.plot(time[samples-1:], stds, c='g', lw=1, label='std')
plt.plot(time[samples-1:], devs, c='r', lw=1, label='devs')
plt.axhline(y=0, c='k', lw=1)
plt.legend()
plt.show()


#plt.figure(figsize=(12, 5))
#plt.plot(stds)


#plt.figure(figsize=(12, 5))
#plt.plot(slopes)