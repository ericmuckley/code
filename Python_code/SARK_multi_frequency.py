# -*- coding: utf-8 -*-

import glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import shutil
from scipy.signal import savgol_filter

def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    from matplotlib import rcParams
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)





def get_time_table(filename,
                   pressure_col_name='RH stpnt'):
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
    
            time_table.append([data['date/time'].iloc[i],
                               data[pressure_col_name].iloc[i]])          
    time_table = pd.DataFrame(time_table, columns=['time', 'pressure'])
    
    #add column for formatted timestamps
    ts = [datetime.datetime.strptime(step,'%Y-%m-%d %H:%M:%S.%f'
                                     ) for step in time_table['time']]
    time_table['ts'] = ts
    
    return time_table





def get_good_files(datafolder, time_table, band_num=1):
    '''Collect the data files from "datafolder" which were created
    just before the pressure changes occure in the "time_table"
    time/pressure table. The number of files saved for each pressure
    step is equal to "band_num".'''
    
    #sort data files by time modified
    datafolder.sort(key=os.path.getmtime)
    
    # get timestamp for each data file
    d_time = [datetime.datetime.strptime(time.ctime(os.path.getmtime(file)),
        '%a %b  %d %H:%M:%S %Y') for file in datafolder]

    
    #make list of files which were measured before the pressure changes
    good_files = []   
   
    #loop over each pressure
    for i in range(len(time_table)):
    
        #time to associate with each pressure
        p_time = time_table['ts'].iloc[i]
    
        #loop over each data file in folder
        for j in range(band_num, len(datafolder)): 
    
            #check if data file was created before timestep changed
            if d_time[j] < p_time:
                last_index = j
    
            #get files which were saved just before pressure change
            good_files0 = datafolder[last_index-band_num:last_index]   
        good_files.append(good_files0)   
       
    return np.array(good_files)








#%% find times for each pressure   
time_table = get_time_table('2018-05-30_multi_freq', 'RH stpnt')



#%% find all data files in the designated folder and sort by date/time
datafoldername = 'C:\\Users\\a6q\\2018-05-30pedotpss_multi'
band_num = 4 #number of frequency bands measured



#%% look at data files
datafolder = glob.glob(datafoldername + '/*')#[0::30]
print('found ' + format(len(datafolder)) + ' data files') 


   
#%%  match data files with timestamped pressure values
good_files = get_good_files(datafolder, time_table, band_num=4)



#%% copy and re-label all "good" files based on pressure and frequency

#make new directory for files separated by frequency range
folder_sep = datafoldername +'_labeled_by_freq_and_RH'
if not os.path.exists(folder_sep): os.makedirs(folder_sep)
    
    
#indices to start, stop, and every nth points to skip per data file:
index1, index2, skip_n = 5, -1, 1

#loop over each pressure step
for step in range(len(good_files)):
    
    print('pressure step '+format(step+1)+' / '+format(len(good_files)))
    
    #loop over each frequency band
    for band in range(len(good_files[0])):

        #read data file
        filename0 = os.path.basename(good_files[step,band]).split('.')[0]
        
        data0 = pd.read_csv(good_files[step, band],
                               skiprows=1).iloc[index1:index2:skip_n,:]
        
        freq = np.array(data0['Freq(MHz)'])

        rh = np.array(time_table['pressure'])[step].astype(int)
        if rh < 10: rh = '0'+format(rh)

        
        
        
        
        
        #find harmonic and pressure level of each file
        n0 = int(file_labels[i].split('_')[0])
        p0 = float(file_labels[i].split('_')[1])
    
        rs_raw = np.array(file0['Rs'])# / np.max(np.array(file0['Rs']))
        rs_filt = savgol_filter(rs_raw, 13, 2, mode='nearest')
    
    
        plt.plot(file0['Freq(MHz)'], rs_raw)
        plt.plot(file0['Freq(MHz)'], rs_filt)
        label_axes('Frequency (MHz)', 'Rs (Ohm)')
        plt.title('n = '+format(n0)+', RH (%) = '+format(p0), fontsize=18)
    
        plt.show()
        
        
        
        
        
        



        #check the frequency band and copy and rename files based on frequency
        if freq[0] < 4:
            
            n0 = 0
            
            shutil.copy2(good_files[step,band],
                         folder_sep+'\\0_'+format(rh)+'.csv')
            
            
            
            
        if freq[0] > 4 and freq[0] < 5:
            
            n0 = 1
            
            shutil.copy2(good_files[step,band],
                         folder_sep+'\\1_'+format(rh)+'.csv')
            
            
            
            
        if freq[0] > 14 and freq[0] < 15:
            
            n0 = 3
            
            shutil.copy2(good_files[step,band],
                         folder_sep+'\\3_'+format(rh)+'.csv')
            
            
            
            
        if freq[0] > 24 and freq[0] < 25:
            
            n0 = 5
            
            shutil.copy2(good_files[step,band],
                         folder_sep+'\\5_'+format(rh)+'.csv')

       
        
#%% iterate through each labeled file

labeled_files = glob.glob(folder_sep + '/*')

#remove path and file type extension from filename
file_labels = np.array([os.path.splitext(
        os.path.basename(file))[0] for file in labeled_files])



#spec_mat0 = 


res0 = []; res1 = []; res3 = []; res5 = []


for i in range(len(labeled_files)):#[0::10]:
    
    #call file
    file0 = pd.read_csv(labeled_files[i],
                        skiprows=1).iloc[index1:index2:skip_n,:]
    
    #find harmonic and pressure level of each file
    n0 = int(file_labels[i].split('_')[0])
    p0 = float(file_labels[i].split('_')[1])

    rs_raw = np.array(file0['Rs'])# / np.max(np.array(file0['Rs']))
    rs_filt = savgol_filter(rs_raw, 13, 2, mode='nearest')


    plt.plot(file0['Freq(MHz)'], rs_raw)
    plt.plot(file0['Freq(MHz)'], rs_filt)
    label_axes('Frequency (MHz)', 'Rs (Ohm)')
    plt.title('n = '+format(n0)+', RH (%) = '+format(p0), fontsize=18)

    plt.show()


    res_freq = file0['Freq(MHz)'][rs_filt.argmax()]

    #collect resonance freqs into matrix
    if n0==0: res0.append(res_freq)
    if n0==1: res1.append(res_freq)
    if n0==3: res3.append(res_freq)
    if n0==5: res5.append(res_freq)


 
#%% plot results
res_lists = [res0, res1, res3, res5]

#normalize delta f
for i in range(len(res_lists)):
    res_lists[i] = res_lists[i] - res_lists[i][0]


for i in res_lists:
    plt.plot(time_table['pressure'], i)
    label_axes('RH (%)', 'Res. Freq. (MHz)')
    plt.show()

#%%



plt.scatter(res_lists[0]*1e6, res_lists[1]*1e6)
plt.scatter(res_lists[0]*1e6, res_lists[2]*1e6)
plt.scatter(res_lists[0]*1e6, res_lists[3]*1e6)
plt.show()
    
    
