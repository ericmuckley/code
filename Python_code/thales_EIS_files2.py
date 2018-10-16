# -*- coding: utf-8 -*-


import csv, glob, os, sys, time, numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import datetime
from scipy import optimize
from numpy import NaN, Inf, arange, isscalar, asarray, array
from timeit import default_timer as timer
from scipy.signal import savgol_filter
import shutil

#make size of axis tick labels larger
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15 






def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    from matplotlib import rcParams
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)





def get_time_table(filename, time_col_name, pressure_col_name):
    
    '''Read file which contains timestamps and changing pressures. The 
    function retuns a dataframe with times and corresponding pressures.
    '''
    data = pd.read_table(str(filename))
    
    p_raw = np.array(data[pressure_col_name])
    
    time_table = []
    
    for i in range(len(data)-1):
        
        #check if pressure changes
        if p_raw[i] != p_raw[i+1]:
    
            time_table.append([data[str(time_col_name)].iloc[i],
                               data[str(pressure_col_name)].iloc[i]])
                
    time_table = pd.DataFrame(time_table, columns=['time', 'pressure'])
    
    #add column for formatted timestamps
    ts = [datetime.datetime.strptime(step,'%Y-%m-%d %H:%M:%S.%f'
                                     ) for step in time_table['time']]
    time_table['ts'] = ts
    
    
    return time_table









def get_good_files(time_table, data_folder):
    '''Collect the data files from "datafolder" which were created
    just before the pressure changes occure in the "time_table"
    time/pressure table.'''
    
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
        for j in range(len(data_folder)): 
    
            #check if data file was created before timestep changed
            if data_time[j] < time_table['ts'].iloc[i]:
                
                good_file0 = data_folder[j] 
            
        good_files.append(good_file0) 

    
    return  np.array(good_files)








def rename_ism_files(data_folder_path, time_table):
    '''Take time-series .ism files created by Thales, match them
    with pressures/RH values, and copy the .ism files to a new folder
    with new filenames to reflect the pressure/RH for each file.'''
    
    #make new directory for files separated by pressure
    folder_labeled = data_folder_path + '_labeled'
    
    if not os.path.exists(folder_labeled): os.makedirs(folder_labeled)
    
    
    #copy good files into new folder and rename by pressure level
    
    for i in range(len(time_table)):
    
        #get pressure in proper format
        rh0 = np.array(time_table['pressure'])[i].astype(int)
    
        rh = str(rh0)
        if rh0 < 10: rh = '0'+format(rh0)

    
        new_filename = os.path.join(folder_labeled, rh + '.ism')
    
        shutil.copy2(good_files[i], new_filename)













def save_eis_spectra(bodefile):
    '''Take an exported .csv file from Thales software which contains 
    data from a list of Bode plot files. This function saves that exported
    list into matrices for import into Origin for plotting.'''
    
    
    #read in file
    rawbodedata = pd.read_csv(bodefile, skiprows = 16,
                              skip_blank_lines=True,
                              error_bad_lines=False,
                              warn_bad_lines=False, sep=',')
    
    #str to flt, coerce to NaN, drop NaN
    bodedata = rawbodedata.apply(pd.to_numeric, errors='coerce').dropna() 
    
    #convert data columns to numpy arrays
    pointnum = np.array(bodedata['Number']).astype(float)
    freq = np.array(bodedata['Frequency/Hz']).astype(float)
    ztot = np.array(bodedata['Impedance/Ohm']).astype(float)
    phasedeg = np.array(bodedata['Phase/deg']).astype(float)
    
    #retain only one array of point numbers
    pointnum = np.unique(pointnum) 
    #retain only one array of frequencies
    freq = freq[0:len(pointnum)] 
    #(comment or uncomment this as needed) to flip frquency values 
    #freq = np.flipud(freq) 
    
    #reshape z array to separate spectra
    ztot = ztot.reshape(-1, len(pointnum)) 
    #reshape phase array to separate spectra
    phasedeg = phasedeg.reshape(-1, len(pointnum)) 
    
    
    
    
    #plot Bode Z data
    for i in ztot: plt.loglog(freq,i) 
    label_axes('Frequency (Hz)', 'Z ($\Omega$)')
    plt.show()
    
    #plot Bode phase data
    for i in phasedeg: plt.semilogx(freq,-i) 
    plt.xlabel('Frequency (Hz)', fontsize=15); plt.ylabel('Phase (deg)', fontsize=15)
    plt.show()
    
    #calculate Re(Z) and Im(Z) using phase information
    phase = np.divide(np.multiply(np.pi,phasedeg),180)
    rez = np.multiply(ztot, np.cos(phase))
    imz = np.multiply(ztot, np.sin(phase))
    
    #create Nyquist plots
    for i in range(len(rez)): plt.scatter(rez[i],imz[i], s=2)  
    label_axes('Re(Z) ($\Omega$)', 'Im(Z) ($\Omega$)')
    plt.show()
    
    # get Z values at lowest frequencies measured
    lowfreqz = ztot[:,-1] 
    plt.plot(np.arange(len(ztot))+1, lowfreqz)
    label_axes('Spectrum', 'Z @ lowest freq. ($\Omega$)')
    plt.show()
    
    
    
    
    # save results in csv file
    
    fileheaders = ['frequency'] # build list of file headers
    savedata = np.copy(freq) #build list of data
    
    for i in range(len(ztot)):
        fileheaders.append('Ztot_' + format(i))
        fileheaders.append('ReZ_' + format(i))
        fileheaders.append('ImZ_' + format(i))
        fileheaders.append('Phase_' + format(i))
        
        savedata = np.append(savedata, ztot[i])
        savedata = np.append(savedata, rez[i])
        savedata = np.append(savedata, imz[i])
        savedata = np.append(savedata, phase[i])
        
    savedata = savedata.reshape(-1, len(pointnum))
    savedata = np.transpose(savedata)
    
    with open('EISspectra.csv', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(fileheaders) #write headers
        for row in savedata:
            writer.writerow(row) 
    
    # open CSV file for saving low-frequency Z values
    zheaders = ['time', 'low-freq. Z (Ohms)'] 
    savedata2 = np.append(time, lowfreqz)
    savedata2 = savedata2.reshape(-1, len(ztot))
    savedata2 = np.transpose(savedata2)
    
    with open('EISlowFreqZ.csv', 'w') as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerow(zheaders) #write headers
        for row in savedata2:
            writer.writerow(row)





















#%%

data_folder_path = 'C:\\Users\\a6q\\exp_data\\2018-06-13pedotpss_EIS_-1VDC'
data_folder = glob.glob(data_folder_path + '/*')

print('found ' + format(len(data_folder)) + ' data files')



#file with pressure data   
pressure_file = 'exp_data\\2018-06-14_rh_chamber'


#get times for each pressure
time_table = get_time_table(pressure_file, 'date_time', 'p_abs')


#get files which correspond to each pressure level
good_files = get_good_files(time_table, data_folder)


#copy .ism files into new folder and label by pressure level
rename_ism_files(data_folder_path, time_table)


#create matrices of impedance results to export
#THIS FUNCTION CANNOT BE RUN UNTIL THALES SOFTWARE IS USED TO EXPORT
#CSV LIST OF ALL FILES CREATES IN 'rename_ism_files" FUNCTION

#file epxorted by Thales which contains list of bode plots
bodefile = '2017-10-27 pdse2 h2o steps bode list.csv'

try: save_eis_spectra(bodefile)
except: print('BODE LIST FILE IS NOT FOUND - CREATE USING THALES SOFTWARE')

