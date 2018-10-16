import glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as filt
import time
from scipy.signal import medfilt
from scipy.optimize import curve_fit

def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    from matplotlib import rcParams
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    
  
    
    


#%% find all impedance files in the designated folder and sort by time/date
folder = glob.glob('C:\\Users\\a6q\\2018-05-03pedotpss_lowfreq_labeled/*')#[0::30]
folder.sort(key=os.path.getmtime)
print('found ' + format(len(folder)) + ' spectra') 

#%% find size of each data file

#indices to start, stop, and every nth points to skip per data file:
index1, index2, skip_n = 600, 4100, 1

data_example_full = pd.read_csv(folder[0], skiprows=1)

data_example = data_example_full.iloc[index1:index2:skip_n,:]

freq = np.array(data_example['Freq(MHz)']) #frequencies in MHz



#%% get pressures from filenames

p_list = np.array([file.split('\\')[-1].split('.')[0] for file in folder]).astype(float)


#%% organize data

starttime = time.time()
var_list = np.empty((len(freq), len(folder)+1))
var_list[:,0] = freq
var_max_list = np.array([])
res_freq_list = np.array([])


for i in range(len(folder)): #loop over each file in folder
    print('spectrum '+format(i+1)+' / '+format(len(folder)))
    
    
    data_raw = pd.read_csv(folder[i], skiprows=1).iloc[index1:index2:skip_n,:]
    var_spectrum = np.array(data_raw['Rs'])
    
    #remove minimum outlier points
    min_index = np.argmin(var_spectrum)
    var_spectrum[min_index] = var_spectrum[min_index+1]
    min_index = np.argmin(var_spectrum)
    var_spectrum[min_index] = var_spectrum[min_index+1]
       
    
    #normalize
    var_spectrum = var_spectrum - np.min(var_spectrum)
    #var_spectrum = var_spectrum  / np.max(var_spectrum)
    
    
    #add to table
    var_list[:,i+1] = var_spectrum
    
    var_max = np.max(var_spectrum)
    var_max_list = np.append(var_max_list, var_max)
    res_freq_list = np.append(res_freq_list, freq[np.argmax(var_spectrum)])
    
    rh0 = p_list[i]
    
    plt.plot(freq, var_spectrum, c='k')
    label_axes('$\Delta$F (Hz)', 'Signal')
    plt.title(format(rh0)+'% RH', fontsize=18)
    plt.show()


endtime = time.time()
tottime = (endtime-starttime)/60
print('elapsed time = %.2f minutes' %tottime)




#%% plot results


res_freq_list_corr = res_freq_list - res_freq_list[0]


plt.plot(p_list, res_freq_list_corr)
plt.scatter(p_list, res_freq_list_corr)
label_axes('RH', '$\Delta$F (Hz)')
plt.show()

plt.plot(p_list, var_max_list)
plt.scatter(p_list, var_max_list)
label_axes('RH', 'Max signal')

#plt.savefig('sample_save_spyder_fig.jpg', format='jpg', dpi=1000)
plt.show()



plt.plot(res_freq_list_corr, var_max_list)
plt.scatter(res_freq_list_corr, var_max_list)
label_axes('$\Delta$F (Hz)', 'Max signal')
plt.show()





