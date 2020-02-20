
# set the full filename of the CSV file you want to fit
filename = r'C:\Users\a6q\exp_data\2019-11-21_BB5\2019-11-21_09-00__qcm_n=1_spectra.csv'

#exp_data\2019-11-14_au4\2019-11-14_10-20__qcm_n=1_spectra.csv'


#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit

def singleBvD_reY(freq, Gp, Cp, Gmax00, D00, f00):
    """Returns admittance spectrum with single peak.
    Spectrum calculation is taken from Equation (2) in:
    Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009. Analyzing spur-distorted 
    impedance spectra for the QCM. Journal of Sensors, 2009.
    inputs:
    Gp = conductance offset
    Cp = susceptance offset
    Gmax00 = maximum of conductance peak
    D00 = dissipation
    f00 = resonant frequency of peak (peak position)
    """
    # construct peak
    peak = Gmax00 / (1 + (1j/D00)*((freq/f00)-(f00/freq)))
    # add offsets to spectrum
    Y = Gp + 1j*2*np.pi*freq*Cp + peak
    G = np.real(Y)
    return G

def plot_setup(labels=['X', 'Y'], size=16, setlimits=False, limits=[0,1,0,1],
               title=None):
    # set axes labels, ranges, and size of labels for a matplotlib plot
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    if title:
        plt.title(str(title), fontsize=size)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))


#%% import data

data = pd.read_csv(filename)
data = data.dropna()

#%% loop over each spectrum and fit

tot_spec = int(len(data.columns) / 3)
fitresults = np.zeros((tot_spec, 2))
spec_num = 0

# last frequency we want to fit in the spectrum
# (use this to cut off spurious peaks)
last_freq = 1.484e15

for i in range(0, len(data.columns), 3):
    print('spectrum %i / %i' %(i+1, len(data.columns)))
    
    # find last idx to probe based on highest frequency variable
    last_idx = len(data.iloc[:, i][data.iloc[:, i] < last_freq])
    
    # get frequency and series resistance
    f = data.iloc[:last_idx, i]
    rs = data.iloc[:last_idx, i + 1]
    
    # construct guess for peak fit [Gp, Cp, Gmax00, D00, f00]
    guess = [0, 0, np.amax(rs), 1e-4, f[np.argmax(rs)]]
    # fit data
    try:
        popt, _ = curve_fit(singleBvD_reY, f, rs, p0=guess)
        print('FIT SUCCESSFUL')
    except RuntimeError:
        print('FIT FAILED')
        popt = guess

    # get parameters from fit
    f0, d = popt[4], np.abs(popt[3])
    print('F0: %s, D: %s' %(str(f0), str(d)))

    # plot spectrum and fit
    fit = singleBvD_reY(f, *popt)
    plt.scatter(f, rs, s=4, c='k', marker='o', alpha=0.3)
    plt.plot(f, fit, c='r', lw=0.75)
    plot_setup(labels=['Freq.', 'Rs'])
    plt.show()

    # save results to array
    fitresults[spec_num, 0] = f0
    fitresults[spec_num, 1] = d
    
    spec_num += 1
    
# plot delta F over time
plt.plot(fitresults[:, 0])
plt.title('f')
plt.show()

# plot delta D over time
plt.plot(fitresults[:, 1])
plt.title('D')
plt.show()

results = pd.DataFrame(columns=['F', 'D'], data=fitresults)



