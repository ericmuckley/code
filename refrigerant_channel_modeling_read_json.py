# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 11:02:37 2020

@author: ericmuckley@gmail.com
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

# change matplotlib settings to make plots look nicer
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3

def plot_setup(labels=['X', 'Y'], fsize=18,
               setlimits=False, limits=[0,1,0,1],
               title='', legend=False,
               save=False, filename='plot.jpg', dpi=200):
    """Creates a custom plot configuration to make graphs look nice.
    This can be called with matplotlib for setting axes labels,
    titles, axes ranges, and the font size of plot labels.
    This should be called between plt.plot() and plt.show() commands."""
    plt.xlabel(str(labels[0]), fontsize=fsize)
    plt.ylabel(str(labels[1]), fontsize=fsize)
    plt.title(title, fontsize=fsize)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    if legend:
        plt.legend(fontsize=fsize-4)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        #plt.tight_layout()


def summarize_models(folder):
    """Summarize Monte Carlo models which have been saved inside
    a folder in the form of .json files. Take all the dta across
    different files and compile into a single dictionary."""
    filepaths = sorted([n for n in glob(folder+r'/*') if '.json' in n])
    # create dictionary of results by looping over each file
    d = {}
    for fi, f in enumerate(filepaths):
        # load dictionary from json file
        print('processing model {}/{}'.format(fi+1, len(filepaths)))
        with open(f, 'r') as fp:
            data = json.load(fp)
        # extract label and trial number from filename
        L = os.path.split(f)[1].split('.')[0].split('__trial_')[0]
        t = int(os.path.split(f)[1].split('.')[0].split('__trial_')[1])
        # create new dictionary for each model
        if L not in d:
            d[L] = {
                'img': None,
                'sim': {},
                'avg_err': [],
                'bub_num': [],
                'df': pd.DataFrame()}
        # populate dictionary with model information
        d[L]['img'] = data['img']
        d[L]['sim'][t] = data['sim']
        d[L]['df']['rad_trial_'+str(t)] = pd.Series(data['rad'])
        d[L]['bub_num'].append(len(pd.Series(data['rad']).dropna()))
        d[L]['df']['x_trial_'+str(t)] = pd.Series(np.array(data['cent'])[:, 0])
        d[L]['df']['y_trial_'+str(t)] = pd.Series(np.array(data['cent'])[:, 1])
        d[L]['avg_err'].append(data['tot_err_percent'])
    return d


def get_mean_and_std(arr):
    """Get mean and standard deviation of an array."""
    return np.round(np.mean(arr), 3), np.round(np.std(arr), 3)

def plot_channel_image(image, title=None, vmin=None, vmax=None):
    """Plot image of the channel, colored by void fraction."""
    plt.imshow(image, origin='lower', cmap='jet', vmin=vmin, vmax=vmax)
    fig = plt.gcf()
    fig.set_size_inches(11,11)
    plt.axis('off')
    if title is not None:
        plt.title(title, fontsize=16)
    plt.show()


# %%
# folder which contains the json models 
folder = r'C:\Users\a6q\exp_data\void_fraction_models'

# get dictionary of all model results
d = summarize_models(folder)

# %%

"""
Each model is stored in a dictionary with four entries:
'img': the original image
'sim': the simulated image
'avg_err': percent differece between the original and simulated image
'df': dataframe containing simulated bubble radii and (X, Y) positions
"""

hist_bin_num = 10
rad_histograms = np.empty((hist_bin_num, 0))
    
# loop over each model and compile results for plotting
for L in d:
    
    # plot histogram of bubble radii
    rad_cols = [c for c in d[L]['df'].columns if 'rad' in c]
    all_rad = d[L]['df'][rad_cols].to_numpy()
    all_rad = np.reshape(all_rad, (-1, 1))
    histogram = plt.hist(all_rad, bins=hist_bin_num, label=L)
    plot_setup(title=L+' histogram of bubble radii',
               labels=['Radius', 'Frequency'])
    plt.yscale('log', nonposy='clip')
    plt.show()

    # report deviation between original and simulated image
    errors = d[L]['avg_err']
    #err_mean, err_std  = get_mean_and_std(errors)

    rad_histograms = np.column_stack((rad_histograms, histogram[0]))
    
    # find index of the best simulation
    best_sim_index = np.argmin(errors)
    # plot original and simulated images
    img, sim = np.array(d[L]['img']), np.array(d[L]['sim'][best_sim_index])
    plot_channel_image(img, title=L+' measured image', vmin=0, vmax=1)
    plot_channel_image(sim, title=L+' simulated image', vmin=0, vmax=1)

    # print out report of results
    print(L)
    print('{} trials'.format(len(d[L]['sim'])))
    print('average bubbles per channel: {} ± {}'.format(
        int(np.mean(d[L]['bub_num'])), int(np.std(d[L]['bub_num']))))
    print('average error: {} ± {} %'.format(
        round(np.mean(errors), 3), round(np.std(errors), 3)))
    print('best error: {} %'.format(round(np.min(errors), 3)))
    print('===============================================================')

# %%

for col in range(len(rad_histograms[0])):
    L = list(d)[col]
    plt.plot(histogram[1][:-1], rad_histograms[:, col], marker='o', label=L)
   
plot_setup(title='Frequency of bubble radii',
           labels=['Frequency', 'Radius'])
plt.yscale('log', nonposy='clip')
plt.legend()
plt.show()




    