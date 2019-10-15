# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:41:43 2019

Analyze Marek's time-dependent casein QCM-D data

@author: ericmuckley@gmail.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import ListedColormap
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


from matplotlib import rcParams
plt.rcParams['xtick.labelsize'] = 16 
plt.rcParams['ytick.labelsize'] = 16

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in
    Parameters
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional
    Returns: xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.
    Parameters
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# %% plot delta D over time for each file
    
with open('exp_data\\marek_qcmd_2019.pkl', 'rb') as handle:
    dic = pickle.load(handle)


key_i = 0
# loop over each file
for key in dic:
    
    print('processing sheet %i / %i (name: %s)' % (key_i, len(dic), key))
    
    
    
    # get current array
    arr = dic[key]
    
    # read in data
    time = dic[key][:, 0, 0]
    for i in range(np.shape(dic[key]))
    deltaf = dic[key][:, 0, 0]
    deltad = 
    
    colormap = plt.cm.rainbow(np.linspace(0, 1, 6))[::-1]


    # populate data dictionary by looping over each measured parameter
    for param_i, param in enumerate(params):

        # populate time column
        data_dict[sheet_name][:, 0, param_i] = data[
                'time'][:num_of_samples]/3600

        # populate data columns by looping over each harmonic
        for n_i, n in enumerate(np.unique(data['n'])):

            # populate data column
            data_dict[sheet_name][:, n_i + 1, param_i] = data[
                    data['n'] == n][param] - data[data['n'] == n][param].iloc[0]
            
            # find "n_mags" number of largest abs magnitudes in data column
            abs_mags = np.abs(data_dict[sheet_name][:, n_i + 1, param_i])
            param_max_ind = np.argsort(abs_mags)[::-1][:n_mags]
            param_max = data_dict[sheet_name][param_max_ind, n_i + 1, param_i]
            mag_dict[sheet_name][n_i*n_mags:n_i*n_mags+n_mags, param_i] = param_max
            
            
            
'''            
            if param == 'freq':
                # plot delta f vs time
                plt.plot(data_dict[sheet_name][:, 0, param_i],
                data_dict[sheet_name][:, n_i + 1, param_i], label=n,
                         c=colormap[n_i])


        if param == 'freq':
            plt.title(sheet_name, fontsize=16)
            plt.legend(fontsize=8, ncol=2)
            plt.ylabel('Delta '+param, fontsize=16)
            plt.xlabel('Time (hours)', fontsize=16)
            plt.tight_layout()
            save_pic_filename = 'exp_data\\casein_plots0\\fig'+str(
                    file_i).zfill(3)+'.jpg'
    
            # plt.savefig(save_pic_filename, format='jpg', dpi=250)
            plt.gcf().set_size_inches(4, 3)
            plt.show()




'''
    key_i += 1

