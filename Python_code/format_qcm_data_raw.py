import glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.signal as filt
import time
import datetime
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from scipy.interpolate import splrep
from scipy.interpolate import splev
import scipy.interpolate as inter

import pickle

#%% import experimental data

'''
#open pickle file
with open('exp_data\\2018-08-01_pp_data_dictionary.pkl',
          'rb') as handle:
    dic = pickle.load(handle)

f_res = dic['f_res']
'''

#%%

raw_data = pd.read_table('exp_data\\pp_qcm_ml_params_aug17.txt')

length = 60

#%% increase density of points
new_rh = np.arange(4, 193)/2
raw_data2 = pd.DataFrame(data=new_rh, columns=['rh'])

for col in raw_data:
    if col != 'rh':
        #perform spline fit to interpolate data
        spline_params = inter.UnivariateSpline(raw_data['rh'],
                                               raw_data[col], s=8e-9)
        spline = spline_params(new_rh)
        
        plt.scatter(raw_data['rh'], raw_data[col])
        plt.scatter(new_rh, spline, s=2, c='r')
        plt.show()
    
        raw_data2[col] = spline
#save df as pickle file
raw_data2.to_pickle('exp_data\\pp_qcm_ml_small_full.pkl')

#%% fit to spline for interpolation





#%% shape into matrix for combining with electrical data

#save original small df as pickle file
raw_data.to_pickle('exp_data\\pp_qcm_ml_small.pkl')

#%% format large array
new_array0 = np.array([])

for i in range(len(raw_data)):

    new_row = raw_data.iloc[i]

    for j in range(length):
        
        new_array0 = np.append(new_array0, new_row)

new_array = np.reshape(new_array0, (-1, len(list(raw_data))))

df = pd.DataFrame(data=new_array, columns=list(raw_data))

df.to_pickle('exp_data\\pp_qcm_ml.pkl')

