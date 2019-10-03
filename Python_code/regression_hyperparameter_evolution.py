# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 14:14:05 2019
@author: ericmuckley@gmail.com

Investigate the behavior of regression hyperparameters when they are
allowed to change values based on their a gradient decent.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR, NuSVR, SVR
from sklearn.model_selection import ParameterGrid
from sklearn.utils.extmath import cartesian
from scipy.interpolate import griddata


def plot_setup(labels=['X', 'Y'], size=16, setlimits=False, limits=[0,1,0,1],
               title='', save=False, filename='plot.jpg',
               scales=['linear', 'linear']):
    # This can be called with Matplotlib for setting axes labels,
    # setting axes ranges, and setting the font size of plot labels.
    # Should be called between plt.plot() and plt.show() commands.
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    plt.title(title, fontsize=size)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    plt.xscale(scales[0])
    plt.yscale(scales[1])    
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()


def heatmap_from_array(arr):
    '''
    Creates a heatmap using matplotlib from a 2D array. Array should have
    first column as independent variable and subsequent columns as
    despendent variable.'''
    #create x, y, and z points to be used in heatmap
    xf = np.arange(len(arr[0]) - 1) + 1
    yf = arr[:, 0]
    Xf, Yf, Zf = np.array([]), np.array([]), np.array([])
    # loop over each column in array
    for i in range(1, len(arr[0])):
        #create arrays of X, Y, and Z values
        Xf = np.append(Xf, np.repeat(i, len(arr)))
        Yf = np.append(Yf, yf)
        Zf = np.append(Zf, arr[:, i])
    # create grid
    zf = griddata((Xf, Yf), Zf,
                  (xf[None, :], yf[:, None]),
                  method='cubic')
    #create the contour plot
    plt.contourf(xf, yf, zf, 200, cmap=plt.cm.seismic,
                 vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
    plt.colorbar()



#%%

data = load_boston()
X = data['data']
y = data['target']

model = SVR()

clist = [1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e10, 1e15]
epsilonlist = [0, .5, 1, 5, 10, 20, 30, 40, 50]
hplist = cartesian((clist, epsilonlist))

#hp_dict = {'C': paramlist1, 'epsilon': paramlist2}

#hp_list = list(ParameterGrid(hp_dict))

# loop over each set model hyperparameters 

scores = []

for hp0 in hplist:
    model.set_params(C=hp0[0], epsilon=hp0[1])
    model.fit(X, y)

    score0 = abs(model.score(X, y))
    scores.append(score0)

    print(hp0[0], hp0[1], score0*200)
    plt.scatter(hp0[0], hp0[1], s=score0*200, c='b', alpha=0.3)
plot_setup(scales=['log', 'linear'], labels=['C', 'Epsilon'])
plt.plot()    


'''   
for hp in hp_list:
    print(hp)
    model.set_params(**hp)
    model.fit(X, y)
    print(model.score(X, y))
'''