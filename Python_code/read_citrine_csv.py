# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:24:53 2019
@author: Eric Muckley (ericmuckley@gmail.com)
"""


import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_setup(labels=['X', 'Y'], size=16, setlimits=False, limits=[0,1,0,1],
               title='', save=False, filename='plot.jpg'):
    """Creates a custom plot configuration to make graphs look nice.
    This can be called with Matplotlib for setting axes labels,
    axes ranges, and the font size of plot labels.
    Functoin should be called between plt.plot() and plt.show() commands."""
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    plt.title(title, fontsize=size)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()

#%%




# set file for import
filename = r'C:\Users\a6q\exp_data\citrination-export.csv'

# import data
data_raw = pd.read_csv(filename)

# convert numeric columns to floats
data_raw[data_raw.columns[2:]] = data_raw[data_raw.columns[2:]].apply(
                                            pd.to_numeric, errors='coerce')
# remove columns with all nan values
df = data_raw.dropna(axis=1, how='all')
# remove rows with any nan values
df = df.dropna(how='any')

# create a column for boiling-to-melting temperature ratio
#df['Boiling to Melting Ratio'] = np.divide(df['Property Boiling Point'],
#                                             df['Property Melting Point'])







#%%


props = df.columns[2:]

prop_pairs = list(itertools.combinations(props, 2))

for combo in prop_pairs:

    plt.scatter(df[combo[0]], df[combo[1]], s=5, c='k')
    plot_setup(labels=[combo[0], combo[1]])
    plt.show()
    


#%%

# perform t-SNE dimensionality reduction on numerical features
props_data = df[props].values
tsne = TSNE(n_components=2).fit_transform(props_data)
plt.scatter(tsne[:, 0], tsne[:, 1], s=5, c='k')
plot_setup(labels=['Component-1', 'Component-2'], title='t-SNE analysis')
plt.show()






