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
    


#%%


iv_raw_data = pd.read_table('exp_data\\pedotpss_IV.txt')
rh_list = np.array(list(iv_raw_data)[2:]).astype(int)

iv_raw_data = np.array(iv_raw_data)


#correct for offset at v=0
iv_mat_corr = np.copy(iv_raw_data)

for i in range(1, len(iv_raw_data[0])):
    plt.plot(iv_raw_data[:,0], iv_raw_data[:,i])
    
    iv_mat_corr[:,i] = 1e9*(iv_raw_data[:,i] - iv_raw_data[200,i])
    
plt.show()

#plot corrected IV
for i in range(1, len(iv_mat_corr[0])):    
    plt.plot(iv_mat_corr[:,0], iv_mat_corr[:,i])
plt.show()


#squeeze vec to shorter amount of points
new_iv_len = 100
iv_mat = np.zeros((new_iv_len, len(rh_list)+2))

#plot corrected squeezed IV
for i in range(1, len(iv_mat_corr[0]),10):    
    
    
    iv_mat[:,0], iv_mat[:,i] = vec_stretch(iv_mat_corr[:,0],
            vecy0=iv_mat_corr[:,i], vec_len=new_iv_len)
    
    
    plt.plot(iv_mat[:,0], iv_mat[:,i])
plt.show()



#%% organize data files
 
#indices to start, stop, and every nth points to skip per data file:
#index1, index2, skip_n = 0, -1, 1    





#%% load impedance files

# -1 VDC impedance
eism1 = pd.read_csv('exp_data\\2018-06-06pedotpssEIS_-1VDC_ML.csv')
# 0 VDC impedance
eis0 = pd.read_csv('exp_data\\2018-06-06pedotpssEIS_0VDC_ML.csv')
# +1 VDC impedance
eisp1 = pd.read_csv('exp_data\\2018-06-06pedotpssEIS_1VDC_ML.csv')


#get rh values from eis dataframes
rh = np.unique([int(i.split('_')[1]) for i in list(eis0)[2:]])



#%% build dataframe to hold all formatted data

#length of spectra: this can set dataset lengths using vec_stretch function
spec_len = 60

#create dataframe for formatted data
df = pd.DataFrame(np.repeat(rh, spec_len), columns=['rh'])

#change size of eis frequency array
f_eis = vec_stretch(eis0['frequency'], vec_len=spec_len, vec_scale='log')

#add eis frequency to dataframe
df['f_eis'] = np.tile(f_eis, int(len(df)/len(f_eis)))




#%%

#empty arrays for z and phi at +0,+1,-1 VDC

zlistm1, philistm1 = np.array([]), np.array([])
zlist0, philist0 = np.array([]), np.array([])
zlistp1, philistp1 = np.array([]), np.array([])  

#loop over each column in each dataframe

#-1 VDC
for col in eism1:
    #check if column is Z or Phi and add to appropriate list
    if 'z' in col:
        _,z0 = vec_stretch(eism1['frequency'], vecy0=eism1[col],
                         vec_len=spec_len, vec_scale='log')
        zlistm1 = np.append(zlistm1, z0)
        
    if 'phi' in col:
        _,phi0 = vec_stretch(eism1['frequency'], vecy0=eism1[col],
                         vec_len=spec_len, vec_scale='log')
        philistm1 = np.append(philistm1, phi0)

#O VDC
for col in eis0:
    #check if column is Z or Phi and add to appropriate list
    if 'z' in col:
        _,z0 = vec_stretch(eis0['frequency'], vecy0=eis0[col],
                         vec_len=spec_len, vec_scale='log')
        zlist0 = np.append(zlist0, z0)
        
    if 'phi' in col:
        _,phi0 = vec_stretch(eis0['frequency'], vecy0=eis0[col],
                         vec_len=spec_len, vec_scale='log')
        philist0 = np.append(philist0, phi0)
     
# +1 VDC     
for col in eisp1:
    #check if column is Z or Phi and add to appropriate list
    if 'z' in col:
        _,z0 = vec_stretch(eisp1['frequency'], vecy0=eisp1[col],
                         vec_len=spec_len, vec_scale='log')
        zlistp1 = np.append(zlistp1, z0)
        
    if 'phi' in col:
        _,phi0 = vec_stretch(eisp1['frequency'], vecy0=eisp1[col],
                         vec_len=spec_len, vec_scale='log')
        philistp1 = np.append(philistp1, phi0)

#add z and phi lists to df:
df['z0'], df['z-1'], df['z1'] = zlist0, zlistm1, zlistp1
df['phi0'], df['phi-1'], df['phi1'] = philist0, philistm1, philistp1






        
