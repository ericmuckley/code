# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:19:19 2018

@author: a6q
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import figure
from scipy.optimize import curve_fit
import numpy.polynomial.polynomial as poly


def config_plot(xlabel='x', ylabel='y', size=12,
               setlimits=False, limits=[0,1,0,1]):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)
    #set axis limits
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))


#%% USER INPUTS

filedir = r'C:\Users\a6q\Desktop\AI-controlled experiment\good_data\2019-07-31_14-14_pedotpss'
filename = r'2019-07-31_14-14__cv.csv'

#%% load in data file
data = pd.read_csv(filedir + '\\' + filename).dropna()
# extract bias values and drop bias columns
bias = data[data.columns[0]].values
data = data.drop(data.filter(regex='bias').columns, axis=1)
rates = np.unique([h.split('V/s')[0].split('__')[1] for h in list(data)])

# remove redundant starting and ending data so we have a full complete loop
start_i = np.argmax(bias)
end_i = np.argmax(bias[np.arange(len(bias))!=np.argmax(bias)]) + 1

#%% loop over each set of cv curves

max_current = np.empty((int(len(data.columns)/len(rates)), len(rates)))
min_current = np.empty((int(len(data.columns)/len(rates)), len(rates)))
tot_area = np.empty((int(len(data.columns)/len(rates)), len(rates)))
hi_curr_arr = np.empty((int(2*start_i)+1, int(len(data.columns)/len(rates))))
lo_curr_arr = np.empty((int(2*start_i)+1, int(len(data.columns)/len(rates))))

# loop over each et of CV curves
for i in range(0, len(data.columns), len(rates)):
   #  print('sweep %i' %int(i/len(rates)))   
    
    # loop over each sweep rate in the set
    for j in range(len(rates)):
        # format current array
        current = data[data.columns[i+j]].values*1e6
        current -= current[0]
        current[-1] = current[0]
        
        # calculate area of CV loop
        low_area = np.trapz(
                current[start_i:start_i*3], x=bias[start_i:start_i*3])
        high_area = np.trapz(
                current[start_i*3:start_i*5], x=bias[start_i*3:start_i*5])

        # save data to arrays
        tot_area[int(i/len(rates)), j] = high_area + low_area       
        max_current[int(i/len(rates)), j] = current.max()
        min_current[int(i/len(rates)), j] = current.min()
        hi_curr_arr[:, int(i/len(rates))] = current[start_i-1:start_i*3] 
        lo_curr_arr[:, int(i/len(rates))] = current[start_i*3-1:start_i*5]
        
        
        if j==0:
            plt.plot(bias[start_i:end_i],
                     current[start_i:end_i],
                     label=str(rates[j])+' V/s')
    #plt.title(str(int(i/len(rates))))
    #plt.legend()
plt.show()


plt.plot(max_current)
plt.title('max current')
plt.show()
    
plt.plot(min_current)
plt.title('min current')
plt.show()

for i in range(len(rates)):
    plt.plot(tot_area[:, i], label=rates[i])
plt.title('tot. area')
plt.legend()
plt.show()


plt.plot(hi_curr_arr)
plt.title('hi current')
plt.show()

plt.plot(lo_curr_arr)
plt.title('lo current')
plt.show()

#%% loop pover each CV file and plot 
'''
#loop over each file
for i, file in enumerate(good_files):
    print('file %i/%i' %(i+1, len(good_files)+1))
    data0 = pd.read_table(file)
    #bias = data0['V']
    
    #loop over each sweep
    for j, col in enumerate(list(data0)[1:]):
        current = np.array(data0[col])*1e6
        current -= current[0]
        current = current[start_i:end_i]
        current[-1] = current[0]

        plt.plot(bias, current, lw=0.3,
                 marker='o',
                 markersize=0.6,
                 label=sweep_rates[j])
        
        
        
        low_current = current[start_i:start_i*3]
        high_current = current[start_i*3:start_i*5]
        
        low_area = np.trapz(low_current, x=bias[start_i:start_i*3])
        high_area = np.trapz(high_current, x=bias[start_i*3:start_i*5])

        tot_area = high_area - low_area       
        #tot_area = np.trapz(current, x=bias)
        
        
        
        
        #print(tot_area)
        dic['areas'][i,j] = tot_area
        
        dic[col] = np.column_stack((dic[col], current))
    
        plt.fill_between(bias, current, 0, alpha=0.09)
    
    #config_plot('Bias (V)', 'Current (nA)',
    #           setlimits=True, limits=[-2.1, 2.1, -1, 1])

    plt.legend(fontsize=8.3, loc='upper left', ncol=3,
               handletextpad=0.1).get_frame().set_linewidth(0.0)
    plt.gcf().set_size_inches(4,3)
    
    plt.axhline(y=0, color='k', alpha=0.2, linewidth=0.5)
    plt.axvline(x=0, color='k', alpha=0.2, linewidth=0.5)
    
    save_pic_filename = 'exp_data\\save_CV_plots_2019-01-28_pss\\fig'+str(i).zfill(3)+'.jpg'
    #plt.savefig(save_pic_filename, format='jpg', dpi=250)
    plt.show()

    
#%% plot area under curves

for i in range(len(sweep_rates)):
    plt.plot(np.arange(len(dic['areas'])), dic['areas'][:,i], lw=0.5,
             marker='o', markersize=1.5, label=sweep_rates[i])  
config_plot('RH (%)', 'Area (uA V)')
plt.legend(fontsize=12)
plt.show()


#%% plot each sweep rate over time, and maximum current over time

max_currents = np.zeros((len(good_files), len(sweep_rates0)))

# get all sweeps at a given sweeprate
for rate_i, rate in enumerate(sweep_rates0):
    for col in range(1, len(dic[rate][0])):
        
        bias = dic[rate][:, 0]
        current = dic[rate][:, col]
        
        max_current0 = np.amax(current)
        max_currents[col-1, rate_i] = max_current0
        
        
        plt.plot(bias, current)
    plt.title(rate)
    config_plot('Bias', 'Current')
    plt.show()
      

# plot all max currents
for col in range(len(max_currents[0])):
    plt.plot(np.arange(len(max_currents))+1, max_currents[:, col],
             label=sweep_rates[col])
plt.legend()
plt.xlabel('Time')
plt.ylabel('max current')
plt.show()

'''        
