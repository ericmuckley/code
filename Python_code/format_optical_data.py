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

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

from scipy.stats import spearmanr
from scipy.stats import pearsonr

from minepy import MINE
from minepy import pstats, cstats



def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    
    
    
def vec_stretch(vecx0, vecy0=None, vec_len=100, vec_scale='lin'):
    '''Stretches or compresses x and y values to a new length
    by interpolating using a 3-degree spline fit.
    For only stretching one array, leave vecy0 == None.'''

    #check whether original x scale is linear or log
    if vec_scale == 'lin': s = np.linspace
    if vec_scale == 'log': s = np.geomspace
    
    #create new x values
    vecx0 = np.array(vecx0)
    vecx = s(vecx0[0], vecx0[-1], vec_len)
    
    #if only resizing one array
    if np.all(np.array(vecy0)) == None:
        return vecx
    
    #if resizing two arrays
    if np.all(np.array(vecy0)) != None:        
        #calculate parameters of degree-3 spline fit to original data
        spline_params = splrep(vecx0, vecy0)
        
        #calculate spline at new x values
        vecy = splev(vecx, spline_params)
        
        return vecx, vecy 
    
    
    
#%% import data
n_raw = pd.read_table('exp_data\\pp_n.txt') 
k_raw = pd.read_table('exp_data\\pp_k.txt')

rh_old = np.array(list(n_raw)[1:]).astype(float)
#rh_new = np.arange(4, 193)/2
rh_new = np.linspace(2, 96, 189)
nm_old = np.array(n_raw['nm'])







#%% stretch along wavelength axis
new_len = 60

n_array2 = np.zeros((new_len, len(list(n_raw))))
k_array2 = np.zeros((new_len, len(list(k_raw))))

for i in range(1, len(list(n_raw))):
    
    n_array2[:,0], n_array2[:,i] = vec_stretch(nm_old,
           vecy0=n_raw.iloc[:,i],
           vec_len=new_len)
    
    k_array2[:,0], k_array2[:,i] = vec_stretch(nm_old,
           vecy0=k_raw.iloc[:,i],
           vec_len=new_len)



nm_new = n_array2[:,0]


#%%stretch along rh axis
n_array3 = np.zeros((len(rh_new),new_len))
k_array3 = np.zeros((len(rh_new),new_len))

for i in range(len(n_array2)):
    
    n_spline_fit = inter.UnivariateSpline(rh_old, n_array2[i, 1:], s=1e-2)
    n_spline = n_spline_fit(rh_new)
    
    k_spline_fit = inter.UnivariateSpline(rh_old, k_array2[i, 1:], s=1e-5)
    k_spline = k_spline_fit(rh_new)

    n_array3[:,i] = n_spline
    k_array3[:,i] = k_spline




#%% plot results

for i in range(len(n_array3))[::2]:
    #plt.plot(n_array2[:,0], n_array3[i,:])
    plt.plot(n_array3[i,:])
plt.title('n')
plt.show()

for i in range(len(n_array3))[::2]:
    #plt.plot(n_array2[:,0], n_array3[i,:])
    plt.plot(k_array3[i,:])
plt.title('k')
plt.show()



#transpose for copying into and plotting in origin
k_array3_origin = np.transpose(k_array3)
n_array3_origin = np.transpose(n_array3)



#%% save data to small pickle files
#index 38 = 800 nm
optical_df_small = pd.DataFrame(data=rh_new, columns=['rh'])
optical_df_small['n820'] = n_array3[:,38]
optical_df_small['k820'] = k_array3[:,38]

plt.plot(rh_new, n_array3[:,40])
plt.title('n')
plt.show()

plt.plot(rh_new, k_array3[:,40])
plt.title('k')
plt.show()

optical_df_small.to_pickle('exp_data\\pp_optical_ml_small_full.pkl')


#%% save data to pickle files
new_nm_labels = np.array(k_array2[:,0]).astype(int).astype(str)
new_nm_labels_k = ['k' + s for s in new_nm_labels]
new_nm_labels_n = ['n' + s for s in new_nm_labels]


optical_df_big = pd.DataFrame(data=k_array3, columns=new_nm_labels_k)
optical_df_big.insert(0, 'rh', rh_new)


for i in range(len(n_array3[0])):
    optical_df_big[new_nm_labels_n[i]] = n_array3[:,i]
    

optical_df_big.to_pickle('exp_data\\pp_optical_ml.pkl')


#%%




