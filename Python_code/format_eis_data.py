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
    

def get_eis_params(data0):
    '''Calculates impedance parameters'''
    freq0 = np.array(data0['Freq(MHz)']*1e6)
    #complex impedance    
    Z = np.add(data0['Rs'], 1j*data0['Xs'])
    #complex admittance
    Y = np.reciprocal(Z)
    #conductance
    G = np.real(Y)
    #susceptance
    #B = np.imag(Y)
    #conductance shift
    #Gp = np.min(G)
    #susceptance shift
    #Cp = np.min(B)

    return freq0, G


'''
import pickle

a = {'hello': 'world'}

with open('filename.pickle', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)

print a == b
'''


#%% load impedance files


# load impedance file
eis_data0 = pd.read_csv(
        'exp_data\\2018-06-06pedotpssEIS_0VDC_ML.csv').iloc[:31,:]

#get rh values from eis data
rh = np.unique([int(i.split('_')[1]) for i in list(eis_data0)[2:]])



#%% build dataframe to hold all formatted data

#length of spectra: this can set dataset lengths using vec_stretch function
spec_len = 60

#create dataframe for formatted data
eis_df = pd.DataFrame(np.repeat(rh, spec_len), columns=['rh'])

#change size of eis frequency array
f_eis = vec_stretch(eis_data0['frequency'], vec_len=spec_len, vec_scale='log')

#add eis frequency to dataframe
eis_df['f_eis'] = np.tile(f_eis, int(len(eis_df)/len(f_eis)))



#%% create 2D arrays for Z and phase

z_2d = f_eis
phi_2d = f_eis



#%%

#empty arrays for z and phi at +0,+1,-1 VDC
zlist0, philist0 = np.array([]), np.array([])

#loop over each column in each dataframe


#O VDC
for col in eis_data0:
    #check if column is Z or Phi and add to appropriate list
    if 'z' in col:
        _,z0 = vec_stretch(eis_data0['frequency'], vecy0=eis_data0[col],
                         vec_len=spec_len, vec_scale='log')
        zlist0 = np.append(zlist0, z0)
        
        #plt.loglog(f_eis, z0)
        #plt.show()
        
        
        z_2d = np.column_stack((z_2d, z0))
        
        
        
    if 'phi' in col:
        _,phi0 = vec_stretch(eis_data0['frequency'], vecy0=eis_data0[col],
                         vec_len=spec_len, vec_scale='log')
        philist0 = np.append(philist0, phi0)
     
        
        #plt.semilogx(f_eis, phi0)
        #plt.show()
        
        
        phi_2d = np.column_stack((phi_2d, phi0))
        
        
        
        
#add z and phi lists to df:
eis_df['eis_z_mag'], eis_df['eis_phi'] = zlist0, philist0

#%% calculate other EIS parameters

#calculate vacuum capacitance of sample
vac_permittivity = 8.854e-12 #F/m
#vacuum capacitance
vc = vac_permittivity# * 20e-6 #F

'''
Calculates impedance parameters
#complex impedance    
Z = np.add(data0['Rs'], 1j*data0['Xs'])
#complex admittance
Y = np.reciprocal(Z)
#conductance
G = np.real(Y)
#susceptance
#B = np.imag(Y)
#dielectric constant 
epsilon_re = -z_imag / [(z_re^2 + z_im^2)*omega*C0]
'''

z_re_2d = f_eis
z_im_2d = f_eis
z_nyquist = f_eis
y_2d = f_eis
g_2d = f_eis
b_2d = f_eis
y_mag_2d = f_eis
y_phase_2d = f_eis

#dielectric constants and electric moduli
ep_re_2d = f_eis
ep_im_2d = f_eis
m_re_2d = f_eis
m_im_2d = f_eis


#loop over ever RH value
for i in range(1, len(rh)+1):
    
    #calculate all EIS parameters from impedance and phase
    z_re0 = z_2d[:,i] * np.cos(phi_2d[:,i])
    z_im0 = z_2d[:,i] * np.sin(phi_2d[:,i])
    
    y0 = np.reciprocal(np.add(z_re0, 1j*z_im0))
    g0 = np.real(y0)
    b0 = np.imag(y0)
    y_mag0 = np.sqrt(np.square(g0) + np.square(b0))
    y_phase0 = np.arctan(b0/g0)
    
    w = f_eis*2*np.pi
    ep_re0 = -z_im0 / ((np.square(z_re0) + np.square(z_im0)) *w*vc)
    ep_im0 = -z_re0 / ((np.square(z_re0) + np.square(z_im0)) *w*vc)
    m_re0 = ep_re0 / (np.square(ep_re0) + np.square(ep_im0))
    m_im0 = ep_im0 / (np.square(ep_re0) + np.square(ep_im0))
    
    
    
    #put all parameters together into 2D arrays for plotting in origin
    ep_re_2d = np.column_stack((ep_re_2d, ep_re0))
    ep_im_2d = np.column_stack((ep_im_2d, ep_im0))
    m_re_2d = np.column_stack((m_re_2d, m_re0))
    m_im_2d = np.column_stack((m_im_2d, m_im0))
    
    y_phase_2d = np.column_stack((y_phase_2d, y_phase0))
    y_mag_2d = np.column_stack((y_mag_2d, y_mag0))
    y_2d = np.column_stack((y_2d, y0))
    g_2d = np.column_stack((g_2d, g0))
    b_2d = np.column_stack((b_2d, b0))
    z_re_2d = np.column_stack((z_re_2d, z_re0))
    z_im_2d = np.column_stack((z_im_2d, z_im0))
    z_nyquist = np.column_stack((z_nyquist, z_re0, z_im0))

    #plt.plot(z_re0, z_im0)
    #plt.show()


#%% make single lists out of 2D arrays so we can combine all the lists
# into a single dataframe
    
    
eis_df['eis_z_real'] = np.reshape(z_re_2d[:, 1:], (-1,), order='F')
eis_df['eis_z_imag'] = np.reshape(z_im_2d[:, 1:], (-1,), order='F')

eis_df['eis_ep_real'] = np.reshape(ep_re_2d[:, 1:], (-1,), order='F')
eis_df['eis_ep_imag'] = np.reshape(ep_im_2d[:, 1:], (-1,), order='F')

eis_df['eis_m_real'] = np.reshape(m_re_2d[:, 1:], (-1,), order='F')
eis_df['eis_m_imag'] = np.reshape(m_im_2d[:, 1:], (-1,), order='F')
 
eis_df['eis_y_mag'] = np.reshape(y_mag_2d[:, 1:], (-1,), order='F')
eis_df['eis_y_real'] = np.reshape(g_2d[:, 1:], (-1,), order='F')
eis_df['eis_y_imag'] = np.reshape(b_2d[:, 1:], (-1,), order='F')    


#%% scale columns
scale = True
if scale:   
    #eis_df['rh'] /= 100
    #eis_df['f_eis'] /= 100
    eis_df['eis_z_mag'] /= 1e6
    #eis_df['phi']
    eis_df['eis_z_real'] /= 1e6
    eis_df['eis_z_imag'] /= 1e6
    eis_df['eis_ep_real'] /= 1e3
    eis_df['eis_ep_imag'] /= 1e3
    eis_df['eis_m_real'] /= 1e-3
    eis_df['eis_m_imag'] /= 1e-3
    eis_df['eis_y_mag'] /= 1e-6
    eis_df['eis_y_real'] /= 1e-6
    eis_df['eis_y_imag'] /= 1e-6

#%% save data to a pickle file

eis_df.to_pickle('exp_data\\pp_eis_ml.pkl')


#%% save data to a small pickle file

#eis_df_small = pd.DataFrame(data=rh, columns=['rh'])



eis_df_small0 = eis_df.loc[eis_df['f_eis'] == 0.1]

eis_df_small = eis_df_small0.drop('f_eis', 1)



eis_df_small.to_pickle('exp_data\\pp_eis_ml_small.pkl')








#%% increase density of points
new_rh = np.arange(4, 193)/2
raw_data2 = pd.DataFrame(data=new_rh, columns=['rh'])

for col in eis_df_small:
    if col != 'rh':
        #perform spline fit to interpolate data
        spline_params = inter.UnivariateSpline(eis_df_small['rh'],
                                               eis_df_small[col], s=8e-9)
        spline = spline_params(new_rh)
        
        plt.scatter(eis_df_small['rh'], eis_df_small[col])
        plt.scatter(new_rh, spline, s=2, c='r')
        plt.show()
    
        raw_data2[col] = spline
#save df as pickle file
raw_data2.to_pickle('exp_data\\pp_eis_ml_small_full.pkl')







