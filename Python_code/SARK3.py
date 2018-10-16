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





def label_axes(xlabel='x', ylabel='y', size=16):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)




    
def get_time_table(filename, pressure_col_name='p_abs'): #'RH stpnt'
    '''Read file which contains timestamps and changing pressures. The 
    function retuns a dataframe with times and corresponding pressures.
    '''
    data = pd.read_table(str(filename))
    pressure_col_name = str(pressure_col_name)
    
    p_raw = np.array(data[pressure_col_name])
    p_indices = np.array([])
    time_table = []
    
    for i in range(len(data)-1):
        #get indices of times to keep
        if p_raw[i] != p_raw[i+1]:
            p_indices = np.append(p_indices, i).astype(int)
    
            time_table.append([data['date_time'].iloc[i],
                               data[str(pressure_col_name)].iloc[i]])          
    time_table = pd.DataFrame(time_table, columns=['time', 'pressure'])
    
    #add column for formatted timestamps
    ts = [datetime.datetime.strptime(step,'%Y-%m-%d %H:%M:%S.%f'
                                     ) for step in time_table['time']]
    time_table['ts'] = ts
    
    return time_table
     







def get_band_info(data_folder, new_spec_len, index1, index2, skip_n):
    ''' Find the number of distinct frequency bands measured by
    looking at the first frequency value in the data files '''
    
    #sort data files by time modified
    data_folder.sort(key=os.path.getmtime)
    
    first_freq = []
    freq_array = []

    for file in data_folder[:20]:
        
        file0 = pd.read_csv(file, skiprows=1).iloc[index1:index2:skip_n,:]
    
        #find frequency values
        freq_array0 = file0['Freq(MHz)']

        
        freq_array0 = vec_stretch(freq_array0, vec_len=new_spec_len)
        
        freq_array.append(freq_array0)
        
        #find first frequency in frequency column to determine band
        first_freq.append(int(freq_array0[0]))
    
    #find number of unique bands measured
    band_num = len(np.unique(first_freq))

    #get list of unique bands measured
    band_list = np.sort(np.unique(first_freq))
    
    #find actual frequency values which correspond to each band
    freq_array = np.sort(np.unique(np.array(freq_array)))
    freq_mat = np.reshape(freq_array, (-1, band_num), order='F')
    
    #make dictionary of frequency values
    freq_dict= dict(zip(band_list, freq_mat.T))
    
    #return number of bands, list of bands, number of points in each band,
    #and a dictionary of frequencies associated with each band
    return band_num, band_list, freq_dict









def get_good_files(time_table, data_folder, band_num=1):
    '''Collect the data files from "datafolder" which were created
    just before the pressure changes occure in the "time_table"
    time/pressure table. The number of files saved for each pressure
    step is equal to "band_num".'''
    
    #sort data files by time modified
    data_folder.sort(key=os.path.getmtime)
    
    # get timestamp for each data file
    data_time = [datetime.datetime.strptime(time.ctime(os.path.getmtime(
            file)),'%a %b  %d %H:%M:%S %Y') for file in data_folder]
    
    #make list of good files which were measured just pressure changes
    good_files = []   

    #loop over each pressure
    for i in range(len(time_table)):
        
        #loop over each data file in folder
        for j in range(band_num, len(data_folder)): 
    
            #check if data file was created before timestep changed
            if data_time[j] < time_table['ts'].iloc[i]:
                
                #if single band
                if band_num == 1:
                    good_file0 = data_folder[j] 
                    
                #if multiple bands, get files just before pressure change
                if band_num > 1:
                    good_file0 = data_folder[j-band_num:j]
            
        good_files.append(good_file0) 

    #reshape to accomodate multiple bands
    if band_num == 1: good_files = np.reshape(good_files, (-1, 1))
    
    return  np.array(good_files)






def get_band_col(band_list, data0):
    '''Determine which frequency band a data file belongs to in order to
    place it in the appropriate column in an array.'''
    
    #if only one band, place data in single column
    if len(band_list) == 1: band_col = 0
    
    #if multiple bands, place data in column determined by the
    #first frequency value in the data file
    if len(band_list) > 1:
        
        #scan through the bands in band_list
        for j in range(len(band_list)-1):
            
            #begnning and intermediate bands
            if band_list[j] <= data0['Freq(MHz)'].iloc[0] < band_list[j+1]:
                band_col = j
                
            #last band
            if data0['Freq(MHz)'].iloc[0] >= band_list[-1]:
                band_col = len(band_list)-1
                
    return band_col










def multiBvD_reY(freq, *params):
    # Returns admittance spectrum with multiple peaks.
    # Spectrum calculation is taken from Equation (2) in:
    # Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009. Analyzing spur-distorted 
    # impedance spectra for the QCM. Journal of Sensors, 2009.
    # inputs:
    # Gp = conductance offset
    # Cp = susceptance offset
    # Gmax00 = maximum of conductance peak
    # D00 = dissipation
    # f00 = resonant frequency of peak (peak position) 
    
    #flat admittance signal
    Y = np.zeros_like(freq)
    
    #conductance and susceptance offsets 
    Gp = params[0]
    Cp = params[1]
        
    #create Gmax, D, and f for each peak
    for b in range(0, len(params)-2, 3):
        Gmax00 = params[b+2]
        D00  = params[b+3]
        f00 = params[b+4]
        #construct peak
        peak = Gmax00 / (1 + (1j/D00)*((freq/f00)-(f00/freq)))
        #add peak to spectrum
        Y = Y + peak
        
    #add offsets to spectrum
    Y = Y + Gp + 1j*2*np.pi*freq*Cp

    return np.real(Y)






def singleBvD_reY(freq, Gp, Cp, Gmax00, D00, f00):
    # Returns admittance spectrum with single peak.
    # Spectrum calculation is taken from Equation (2) in:
    # Yoon, S.M., Cho, N.J. and Kanazawa, K., 2009. Analyzing spur-distorted 
    # impedance spectra for the QCM. Journal of Sensors, 2009.
    # inputs:
    # Gp = conductance offset
    # Cp = susceptance offset
    # Gmax00 = maximum of conductance peak
    # D00 = dissipation
    # f00 = resonant frequency of peak (peak position) 
    
    #construct peak
    peak = Gmax00 / (1 + (1j/D00)*((freq/f00)-(f00/freq)))

    #add offsets to spectrum
    Y = Gp + 1j*2*np.pi*freq*Cp + peak

    G = np.real(Y)

    return G





def get_singlebvd_rlc(fit_params0):
    # calculate equivalent circuit parameters from BvD fits
    # from Yoon et al., Analyzing Spur-Distorted Impedance 
    # Spectra for the QCM, Eqn. 3.
        
    #R_fit = []; L_fit = []; C_fit = []
    #deconstruct popt to calculate equivalent circuit parameters
    '''
    for k in range(len(guess0)):
        G0 = popt_real[3*k+2] 
        D0 = popt_real[3*k+3] 
        f0 = popt_real[3*k+4] 
        L0 = 1 / (2 * np.pi * f0 * D0 * G0)
        C0 = 1 / (4 * np.pi**2 * f0**2 * L0)
        R_fit.append(1/G0)
        L_fit.append(L0)
        C_fit.append(C0)
    '''
    
    
    #FOR SINGLE BvD PEAK:
    #FIT PARAMS = [Gp, Cp, Gmax00, D00, f00]
    
    G0 = fit_params0[2]
    f0 = fit_params0[4]
    D0 = fit_params0[3]
    
    R = 1/G0
    L = 1 / (2 * np.pi * f0 * D0 * G0)
    C = 1 / (4 * np.pi**2 * f0**2 * L)

    return R, L, C, D0






def get_spectrum_dict(band_list, new_spec_len, time_table, freq_dict):
    #create dictionary to hold spectra, where keys are the harmonics, and
    #the first column in each dict entry is the frequency array
    spec_dict = {}
    for key in band_list:
            spec_dict[key] = np.zeros((new_spec_len, len(time_table)+1))
    
            spec_dict[key][:,0] = freq_dict[key]
    
    return spec_dict




def remove_outliers(spectra, num_of_outliers=5):
    '''Remove the low outliers from spectra - these outliers are an
    artifact of SARK-110 measurements '''
    
    spectra2 = np.copy(spectra)
    
    #replace lowest outlier points with average of adjacent points
    for i in range(num_of_outliers):  
        
        #find index of minimum point
        min_index = np.argmin(spectra2)
        
        #make sure minimum is not first or last point
        if min_index != 0 and min_index != len(spectra2) - 1:
        
            #remove the mimimum points 
            spectra2[min_index] = (
                    spectra2[min_index-1] + spectra2[min_index+1]) / 2
                    

    #replace errant points by average of adjacent points
    for i in range(1, len(spectra2)-1):  

        #base difference between adjacent points
        base_diff = np.abs(spectra2[i+1] - spectra2[i-1])
        
        #get average of adjacent points
        adj_avg = (spectra2[i+1] + spectra2[i-1])/2
        
        #check if point is errant
        if np.abs(spectra2[i] - np.abs(adj_avg)) > 2*base_diff:
            spectra2[i] = adj_avg
            
    return spectra2






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






def get_peaks(vec, n=3):
    '''get indicies and heights of peaks in vector. n parameter specifies
        how many points on each side of the peak should be strictly
        increasing/decreasing in order for it to be consiidered a peak.'''
   
    peak_indices = []
    peak_vals = []
    
    for i in range(n, len(vec)-n):
        
        #differences between points at rising and falling edges of peak
        rising_diff = np.diff(vec[i-n:i+1])
        falling_diff = np.diff(vec[i:i+n+1])
        
        #check if rising edge increases and falling edge decreases
        if np.all(rising_diff>0) and np.all(falling_diff<0):
            peak_indices.append(i)
            peak_vals.append(vec[i])

    peak_indices = np.array(peak_indices).astype(int)
    peak_vals = np.array(peak_vals).astype(float)
    

    return peak_indices, peak_vals










#%% USER INPUTS
 
#indices to start, stop, and every nth points to skip per data file:
index1, index2, skip_n = 6, -1, 1    

#folder with measured data files
data_folder = glob.glob(
        'C:\\Users\\a6q\\exp_data\\2018-07-09mxene/*')

print('found ' + format(len(data_folder)) + ' data files')

#allow expansion/contraction of spectra length
new_spec_len = 10000

#file with pressure data   
pressure_file = 'exp_data\\2018-07-09_rh'

#determine whether to fit to BvD circuit:
bvd_fit = False
#set number of peaks to fit
peaks_to_fit = 7



#%%organize input data

#get times for each pressure
time_table = get_time_table(pressure_file)
rh_list = list((np.array(time_table.pressure).astype(int)))

#find number of measured frequency bands, list of bands,
#and dictionary of frequency values associated with each band
band_num, band_list, freq_dict = get_band_info(
        data_folder, new_spec_len, index1, index2, skip_n)

#get good data files saved just before pressure changes 
good_files = get_good_files(time_table, data_folder, band_num=band_num)

#get dictionary for saving all the measured spectra
spec_dict = get_spectrum_dict(band_list, new_spec_len, time_table, freq_dict)



#%% examine each data file

#make empty arrays for max and resonant frequencies 
f_res = np.zeros((len(time_table), band_num))
G_max = np.zeros((len(time_table), band_num))

#BvD equivalent circuit RLC parameters 
R_mat = np.zeros((len(time_table), band_num))
L_mat = np.zeros((len(time_table), band_num))
C_mat = np.zeros((len(time_table), band_num))
D_mat = np.zeros((len(time_table), band_num))
peak_num_mat = np.zeros((len(time_table), band_num))
peak_ind_mat = np.zeros((len(rh_list), peaks_to_fit))
peak_val_mat = np.zeros((len(rh_list), peaks_to_fit))


#loop over each frequency band
for band in range(len(band_list)): 
   
    #peaks found from BvD fitting
    single_peaks = np.zeros((new_spec_len, peaks_to_fit))
    
    print('frequency band '+format(band+1)+' / '+format(band_num))
    
    
    
    #loop over each pressure (each individual data file)
    for i in range(len(time_table)): 
        
        print('rh = '+format(rh_list[i]))
        
        #read data
        data0 = pd.read_csv(good_files[i][band],
                skiprows=1).iloc[index1:index2:skip_n,:]
      
        #get conductance (G)
        freq0, G0 = get_eis_params(data0)
        freq0 *= 1e-6
        
        #squeeze each vector to a specified length
        freq0, G0 = vec_stretch(freq0, vecy0=G0, vec_len=new_spec_len)
        
        
        #create spline fit for conducance
        spline_fit = inter.UnivariateSpline(freq0, G0, s=1e-9)
        G0_spline = spline_fit(freq0)
        
        G0 = G0_spline
        
        #normalize conductance
        G0 -= np.min(G0)
        G0 /= np.max(G0)
        
        #normalize frequency to f0 = 0
        f_res0 = freq0[np.argmax(G0)]

        #Savitzky-Golay filter
        #G0_savgol = savgol_filter(G0, 13, 2)#, mode='nearest')
        
        #determine frequency band by the first frequency value
        band_col = get_band_col(band_list, data0)
        
        #save resonant and maximum of spectrum
        f_res[i][band_col] = f_res0
        G_max[i][band_col] = np.max(G0)
        
        #save spectra into matrices
        spec_dict[band_list[band_col]][:, i+1] = G0
        
        
        
        

        
        
        
        #try to detect peaks and use them as guesses for peak fitting#########
        peak_inds, peak_vals = get_peaks(G0_spline, n=3)
        
        '''
        #see if enough peaks were found. if not, guess where they are
        if len(peak_inds) < peaks_to_fit:
            peak_inds = np.append(
                    f_res0, [f_res0+50*i for i in range(peaks_to_fit-1)])
            peak_vals = np.append(
                    np.max(G0), np.full(peaks_to_fit-1, np.max(G0)/2))
        '''
            
            
            
        peak_num_mat[i][band_col] = len(peak_inds)

        #sort by highest to lowest peaks
        peak_order = peak_vals.argsort()[::-1]
        peak_vals, peak_inds = peak_vals[peak_order], peak_inds[peak_order]
        peak_vals = peak_vals[:peaks_to_fit]
        peak_inds = peak_inds[:peaks_to_fit].astype(int)




        #fig, _ = plt.subplots(2, 1, figsize=(6,10))
        #fig.subplots_adjust(hspace=0, bottom=.08, top=0.98, right=.95, left=.2)





        #fit BvD circuit to experimentally measured real conductance (G)######
        if bvd_fit == True and band_col == 2:#len(peak_inds)>2 
            
            #construct guess parameters for BvD fits starting with Gp and Cp
            guess = np.array([1e-6, 1e-6])
            
            #append peak params to guess
            g_max_guess = peak_vals
            D_guess = np.full(len(peak_inds), 1e-5)
            
            #construct guess: [Gp, Cp, Gmax00, D00, f00]
            for k in range(len(peak_inds)):
                guess = np.append(guess, [peak_vals[k],
                                          D_guess[k], freq0[peak_inds[k]]])
    
            #fit peaks
            popt, _ = curve_fit(multiBvD_reY, freq0, G0,
                                       p0=guess, ftol=1e-14,
                                       xtol=1e-14, bounds=(0,np.inf))
            
            #print('peak fit guess = '+format(guess))
            #print('fitted peak parameters = '+format(popt))
            
            '''
            #organize fit parameters################################
            centers0, amps0, widths0 = np.array([]), np.array([]), np.array([])
        
            for y in range(0,len(popt),5):
                amp0 = np.append(amps0, popt[y+2])
                center0 = np.append(centers0, popt[y])
                width0 = np.append(widths0, popt[y+2])    
            '''
        
        
            
            #plot deconvoluted peaks######################################
            plt.figure(figsize=(6,5))
            for p in range(0, len(popt)-2, 3):
                peak0 = multiBvD_reY(freq0,
                        popt[0], popt[1], popt[p+2], popt[p+3], popt[p+4])
                 
                peak_ind_mat[i,int(p/3)] = popt[p+4]
                peak_val_mat[i,int(p/3)] = popt[p+2]
                
                
                single_peaks[:,int(p/3)] = peak0
                
                #plot fitted peaks
                plt.plot(freq0, peak0, label='peak-'+format(1+int(p/3)), lw=1)



            G_fit = multiBvD_reY(freq0, *popt)
            plt.plot(freq0, G_fit, label='BvD fit', c='r', lw=1)
            
            for p in range(peaks_to_fit):
                plt.scatter(peak_ind_mat[:, p], peak_val_mat[:,p],
                            marker='o', s=10, c='c', alpha=.5)
            
            
            '''
            #calculate R L C from fit parameters
            R0, L0, C0, D0 = get_singlebvd_rlc(fit_params0)
            
            R_mat[i][band_col] = R0
            L_mat[i][band_col] = L0
            C_mat[i][band_col] = C0*1e15
            D_mat[i][band_col] = D0*1e6
    
            print('Calculated R L C parameters from BvD fit:')
            print(format(R0)+', '+format(L0)+', '+format(C0))
            '''
            
            #plot peak positions
            plt.scatter(freq0[peak_inds], peak_vals,
                    label='local max.', s=20, c='c', marker='o')
             
                
                
            #plot raw data    
            plt.scatter(freq0, G0, label='exp. data', s=5, alpha=0.2,
                        c='k')    
                
            #plot spline fit
            #plt.plot(freq0*1e-6, 1e3*G0_spline, c='c', lw=1, label='spline')
                
            #plot sav gol fit
            #plt.plot(freq0*1e-6, 1e3*G0_savgol, c='g', lw=1, label='sav-gol')

            
            
            plt.title(format(rh_list[i])+' % RH', fontsize=16)#+', band = '+format(band_list[band_col]), fontsize=18)
            label_axes('Frequency (Hz)', 'Conductance (mS)')
            plt.legend(fontsize=10, ncol=2, loc='upper right')


            plt.axis((24.9, np.max(freq0), -0.02, 1.02))
            plt.tight_layout()
    
            save_pic_filename = 'exp_data\\save_figs\\fig'+format(
                    rh_list[i])+'.jpg'
            plt.savefig(save_pic_filename, format='jpg', dpi=150)
            
            plt.show()





#%% plot spectra and save to txt files

#loop over bands in spectra dictionary
for band in band_list:

    spec_mat = spec_dict[band]
    
    #subtract f0 from F spectra so they are normalized to delta F = 0 @ t = 0
    spec_dict[band][:,0] = spec_dict[band][:,0] - spec_dict[band][
            np.argmax(spec_dict[band][:,1]),0]
        
    #loop over each spectra
    for i in range(1, len(time_table)+1):
                
        plt.plot(spec_mat[:,0], spec_mat[:,i],
                 label=format(int(time_table.pressure.iloc[i-1])))
    label_axes('Frequency (MHz)', 'G conductance (S)')
    #plt.legend()
    plt.show()
    

    #save spectra to file
    save_headers = list((np.array(time_table.pressure
                                          ).astype(int)).astype(str))
    save_headers[0:0] = 'f'
    save_headers = str(' '.join(save_headers)).replace(' ', '\t')
    
    save_spec_filename = 'exp_data\\norm_spec_mat_0'+format(band)+'.txt'
    np.savetxt(save_spec_filename, spec_mat, delimiter='\t',
               header=save_headers, fmt='%.10e', comments='')
    





#%% plot analysis results
 

#plot change in spectra maximum       
[plt.plot(time_table['pressure'], G_max[:,i] - G_max[0,i],
          label=band_list[i]) for i in range(band_num)]      
label_axes('RH (%)', '$\Delta$ G max ($S$)')
plt.legend(fontsize=14)
plt.show()       




#plot change in resonance frequency
[plt.plot(time_table['pressure'], f_res[:,i] - f_res[0,i],
          label=band_list[i]) for i in range(band_num)]      
label_axes('RH (%)', '$\Delta$ Res. freq. (Hz)')
plt.legend(fontsize=14)
plt.show()




#if BvD fitting was used:
if bvd_fit == True:

    #plot change in R from BVD circuit
    [plt.plot(time_table['pressure'], R_mat[:,i] - R_mat[0,i],
              label=band_list[i]) for i in range(band_num)]      
    label_axes('RH (%)', 'BvD: $\Delta$ R ($\Omega$)')
    plt.legend(fontsize=14)
    plt.show()
    
    #plot change in L from BVD circuit
    [plt.plot(time_table['pressure'], L_mat[:,i] - L_mat[0,i],
              label=band_list[i]) for i in range(band_num)]      
    label_axes('RH (%)', 'BvD: $\Delta$ L (H)')
    plt.legend(fontsize=14)
    plt.show()
    
    #plot change in C from BVD circuit
    [plt.plot(time_table['pressure'], C_mat[:,i] - C_mat[0,i],
              label=band_list[i]) for i in range(band_num)]      
    label_axes('RH (%)', 'BvD: $\Delta$ C (F)')
    plt.legend(fontsize=14)
    plt.show()


    #plot change in D from BVD circuit
    [plt.plot(time_table['pressure'], D_mat[:,i] - D_mat[0,i],
              label=band_list[i]) for i in range(band_num)]      
    label_axes('RH (%)', 'BvD: $\Delta$ D (F)')
    plt.legend(fontsize=14)
    plt.show()





#plot number of peaks detected
[plt.plot(time_table['pressure'], peak_num_mat[:,i],
          label=band_list[i]) for i in range(band_num)]      
label_axes('RH (%)', 'Number of peaks found')
plt.legend(fontsize=14)
plt.show()


#%% save figures for creating videos
'''
#band to save
band = 54

#save spectra to file
rh_list = list((np.array(time_table.pressure).astype(int)))

#loop over RH
for i in range(0, len(rh_list)):
       
    plt.plot(spec_dict[band][:,0]*1000, spec_dict[band][:,i+1],c='navy',lw=1)
    label_axes('$\Delta$F (kHz/cm$^2$)', 'Norm. conductance')
    
    plt.title(format(rh_list[i])+'% RH', fontsize=18)

    plt.axis((-100, 100, 0, 1))
    plt.tight_layout()
    
    save_pic_filename = 'exp_data\\save_figs\\fig'+format(rh_list[i])+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)

    plt.show()
'''





#plt.savefig('sample_save_spyder_fig.jpg', format='jpg', dpi=1000)


#%% plot heat maps of QCM response

#turn heatmaps on or off
plot_heatmap = False


if plot_heatmap == True:
    
    #loop over each band in band dictionary
    for key in spec_dict:
    
        #reshape spec_mat into columns
        Xf, Yf, Zf = np.array([]), np.array([]), np.array([])
        
        #loop over each pressure
        for i in range(len(time_table)):
            
            #set X values = pressure 
            Xf = np.append(Xf, np.repeat(
                    np.array(time_table['pressure'].iloc[i]), new_spec_len))
            #set Y values = frequency
            Yf = np.append(Yf, spec_dict[key][:,0])
            #set Z values = G
            Zf = np.append(Zf, spec_dict[key][:,i+1])
        
        
        
        # create x-y points to be used in heatmap
        xf = np.linspace(Xf.min(),Xf.max(),100)
        yf = np.linspace(Yf.min(),Yf.max(),100)
        # Z is a matrix of x-y values
        zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
        # Create the contour plot
        CSf = plt.contourf(xf, yf, zf, 100, cmap=plt.cm.rainbow, 
                           vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
        plt.colorbar()
        label_axes('RH (%)', 'F$_0$ (Hz/cm$^2$)')
        plt.show()
    




#%% manual peak construction
        
      
'''       
guess = [1.00e-06, 1.00e-06, 1e3*1.00e+00, 5, 4.11e+02]

G_fit = multiBvD_reY(freq0*1e6, *guess)

plt.scatter(freq0*1e6, G0*1e3)
plt.plot(freq0*1e6, G_fit*1e3)   
        
'''       


for p in range(peaks_to_fit):
    plt.plot(peak_ind_mat[:, p], peak_val_mat[:,p])
plt.axis((24.9, np.max(freq0), -0.02, 1.02))
plt.show()


#%%


what_mat = np.zeros_like(good_files).astype(str)
what_mat2 = np.zeros_like(good_files).astype(str)
for col in range(len(good_files[0])):
    for row in range(len(good_files)):

        print('row = '+format(row))
        print('col = '+format(col))
        
        #read data
        data00 = pd.read_csv(good_files[row][col],
                skiprows=1).iloc[index1:index2:skip_n,:]
    
        what_mat[row][col] = str(int(data00['Freq(MHz)'].iloc[0]))
        what_mat2[row][col] = str(rh_list[row])