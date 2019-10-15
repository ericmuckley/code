# -*- coding: utf-8 -*-
# Created on Tue Aug 13 17:23:26 2019

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot_setup(labels=['X', 'Y'], size=16, setlimits=False, limits=[0,1,0,1]):
    # This function can be used with Matplotlib for setting axes labels,
    # setting axes ranges, and setting the font size of plot labels
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    # plt.rcParams.update({'figure.autolayout': True})
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))

def stack_cvs(df):
    # takes a Pandas DataFrame (df) with multiple CVs stacked vertically
    # in 2 columns: one bias and one current column.
    # creates a new array of the stacked CVs with the first column as the
    # bias and each subsequent column for each current sweep.
    # find indices of the start of each cv scan
    min_ind = [
            i for i in range(len(df)) if df[
                    df.columns[0]].iloc[i] == np.min(df[df.columns[0]])]
    
    # create array to hold all cv scans, with biases as first column
    cv_arr = np.array(df['E'].iloc[min_ind[0]:min_ind[1]])
    
    # stack all cv scans into one array, each scan in a new column
    for i in range(len(min_ind)):
        if i < len(min_ind) - 1:
            cv_arr = np.column_stack(
                    (cv_arr, -df['I'].iloc[min_ind[i]:min_ind[i+1]]))
    return cv_arr


#%%

# data file to import
filename = r'C:\Users\a6q\exp_data\cv\B_CV_1_CV.dat'


#%%


# import the file
df = pd.read_table(filename, skiprows=39, sep=',',
                   header=None, names=['E', 'I'])



# plot all raw scans
plt.plot(df['E'], 1e6*df['I'])
plot_setup(labels=['E (V)', 'I (uA)'])
plt.show()


curr = 1e6*df['I'].values
pos_curr = [i for i in curr if i>0]#+-[int(len(curr)/2):]
plt.plot(pos_curr)


'''
# get array of cv scans stacked by column
cv_arr = stack_cvs(df)

# colors for plotting
colors = cm.rainbow(np.linspace(0, 1, len(cv_arr[0])))

areas = []

# loop through each cv scan and plot it
for cv_i in range(1, len(cv_arr[0])):
    area = np.trapz(cv_arr[:, cv_i], x=cv_arr[:, 0])
    areas.append(area)
    
    print('CV #%s, area: %s' %(cv_i, area))
    
    # plot each cv curve
    plt.plot(cv_arr[:, 0], 1e6*cv_arr[:, cv_i],
             color=colors[cv_i], label=cv_i, lw=1)
plot_setup(labels=['E (V)', 'I (uA)'])
plt.legend(ncol=3)
plt.show()
'''
'''
# plot cv areas
plt.plot(np.arange(len(areas))+1, areas, marker='o')
plot_setup(labels=['CV number', 'CV area'])
plt.show()
'''
#%%
'''
xc = 300
yc = .4
# loop through each cv scan and plot it
for cv_i in range(1, len(cv_arr[0])):

    # plot each cv curve
    plt.plot(1e6*cv_arr[:, cv_i],
             color=colors[cv_i], label=cv_i, lw=1)
    
# plt.axvline(x=xc, c='k')
plt.axhline(y=yc, c='k', lw=1)
plot_setup(labels=['Time', 'I (uA)'])
plt.legend(ncol=3)
plt.show()

'''














