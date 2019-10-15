import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.signal as filt
import datetime
import time
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from scipy.interpolate import splrep
from scipy.interpolate import splev
import scipy.interpolate as inter



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
#import file
iv_df = pd.read_table('C:\\Users\\a6q\\exp_data\\2019-01-28_PSS_IV_short')
#extract voltage bias
bias = iv_df['V'].values
#convert currents into array
iv_array = iv_df[list(iv_df)[1:]].values
#correct for instument offset at 0 bias
for i in range(len(iv_array[0])):
    iv_array[:,i] = (iv_array[:,i] - iv_array[50,i])*1e9


#get time strings from file
time_strings = list(iv_df)[1:]
#convert time strings datetime objects
datetimes = [datetime.datetime.strptime(i,
                '%Y-%m-%d %H:%M:%S.%f') for i in time_strings]
#get elapsed time in hours
elapsed_time = [((i - datetimes[0]).seconds/3600) for i in datetimes]
    
    
    
##datetimes2 = datetimes - datetimes[0]

#d#atetimes3 = [datetimes2[i].seconds for i in datetimes2]


#plot each IV curve
[plt.plot(bias, iv_array[:,i]) for i in range(len(iv_array[0]))]
label_axes(xlabel='Bias (V)', ylabel='Current (A)')
plt.show()

#plot initial current
plt.plot(iv_array[0,:])
label_axes(xlabel='Time', ylabel='Current @ initial bias (A)')
plt.show()

#plot final current
plt.plot(iv_array[-1,:])
label_axes(xlabel='Time', ylabel='Current @ final bias (A)')
plt.show()


#%% plot raw IV curves
'''
iv_raw_data0 = pd.read_table('exp_data\\pedotpss_IV.txt')

#drop 0% RH column:
iv_raw_data = iv_raw_data0.drop(['0'], axis=1)

rh_list = np.array(list(iv_raw_data)[1:]).astype(int)

iv_raw_data = np.array(iv_raw_data)


#correct for offset at v=0
iv_mat_corr = np.copy(iv_raw_data)

for i in range(1, len(iv_raw_data[0])):
    plt.plot(iv_raw_data[:,0], iv_raw_data[:,i])
    
    iv_mat_corr[:,i] = 1e9*(iv_raw_data[:,i] - iv_raw_data[200,i])
plt.title('Raw data', fontsize=18)    
plt.show()

'''

#%% plot corrected IV
'''
for i in range(1, len(iv_mat_corr[0])):    
    plt.plot(iv_mat_corr[:,0], iv_mat_corr[:,i])
plt.title('Corrected data', fontsize=18)
plt.show()

'''
#%% squeeze vec to shorter amount of points
'''
new_iv_len = 60
iv_mat = np.zeros((new_iv_len, len(rh_list)+1))

#plot corrected squeezed IV
for i in range(1, len(iv_mat_corr[0])):    
    
    iv_mat[:,0], iv_mat[:,i] = vec_stretch(iv_mat_corr[:,0],
            vecy0=iv_mat_corr[:,i], vec_len=new_iv_len)
    
    #fix last current point
    iv_mat[-1,i] = iv_mat[-2,i] + 1.5*(iv_mat[-2,i] - iv_mat[-3,i])
    
    plt.plot(iv_mat[:,0], iv_mat[:,i], label=format(i))
label_axes('Voltage (V)', 'Current (nA)')
plt.title('Corrected squeezed data', fontsize=18)
plt.show()
'''
#%% plot trend in each point vs RH
'''
for i in range(0, len(iv_mat), 5):
    plt.plot(rh_list, iv_mat[i,1:], label=format(iv_mat[i,0]))
label_axes('RH (%)', 'Current (nA)')
plt.title('Current vs RH', fontsize=18)
plt.legend()        
plt.show()
'''




#%% reshape into columns and construct dataframe
'''
#create dataframe for formatted data
iv_df = pd.DataFrame(np.repeat(rh_list, new_iv_len), columns=['rh'])


current_list = np.reshape(iv_mat[:,1:], (-1,), order='F')

#add eis frequency to dataframe
iv_df['v'] = np.tile(iv_mat[:,0], int(len(iv_df)/len(iv_mat[:,0])))
iv_df['i'] = current_list

'''

#%% save dataframe into pickle

#iv_df.to_pickle('exp_data\\pp_iv_ml.pkl')



#%% small dataframe with only certain bias values

'''
iv_df_small = pd.DataFrame(data=rh_list, columns=['rh'])

#get columns for current measured at +/-2 V
iv_df_small['iv_m2'] = iv_mat_corr[0,1:]
iv_df_small['iv_p2'] = iv_mat_corr[-1,1:]

# save dataframe into pickle
iv_df_small.to_pickle('exp_data\\pp_iv_ml_small.pkl')




'''






#%% increase density of points
'''
new_rh = np.arange(4, 193)/2
raw_data2 = pd.DataFrame(data=new_rh, columns=['rh'])

for col in iv_df_small:
    if col != 'rh':
        #perform spline fit to interpolate data
        spline_params = inter.UnivariateSpline(iv_df_small['rh'],
                                               iv_df_small[col], s=8e-9)
        spline = spline_params(new_rh)
        
        plt.scatter(iv_df_small['rh'], iv_df_small[col])
        plt.scatter(new_rh, spline, s=2, c='r')
        plt.show()
    
        raw_data2[col] = spline
#save df as pickle file
raw_data2.to_pickle('exp_data\\pp_iv_ml_small_full.pkl')
'''

