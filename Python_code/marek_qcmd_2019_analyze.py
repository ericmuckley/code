# -*- coding: utf-8 -*-
"""
Created on Fri May 10 11:41:43 2019

Analyze Marek's casein QCM-D data using only the 3-highest signal magnitudes

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


def scale_vector(vec0):
    # scale a vector by its mean and standard deviation
    vec = (vec0 - np.mean(vec0)) / np.std(vec0)
    return vec


# %% run LDA on proteolysis data
with open('exp_data\\marek_qcmd_2019_n_mag3.pkl', 'rb') as handle:
    mag_df = pickle.load(handle)

# keep only casein removal data
df = mag_df.loc[~mag_df['note'].str.contains('assembly')]
df = df.loc[~mag_df['note'].str.contains('bubbles')]
df = df.loc[~mag_df['note'].str.contains('K ')]
df = df.loc[~mag_df['note'].str.contains('PBS ')]
# create normalized frequency column
df['f/n'] = df['freq'] / df['n']
# remove outliers
df = df[df['f/n'].between(-40, 40)]

df['class'] = np.full(len(df), 0)

for i in range(len(df)):
    '''
    if 'B ' in str(df['note'].iloc[i]):
        df['class'].iloc[i] = 'B'
    if 'K ' in str(df['note'].iloc[i]):
        df['class'].iloc[i] = 'K'
    if 'wash ' in str(df['note'].iloc[i]):
        df['class'].iloc[i] = 'PBS'    
        
    '''    
    if 'plasmin' in str(df['note'].iloc[i]):
        df['class'].iloc[i] = 'plasmin'
    if 'thrombin' in str(df['note'].iloc[i]):
        df['class'].iloc[i] = 'thrombin'
    if 'trypsin' in str(df['note'].iloc[i]):
        df['class'].iloc[i] = 'trypsin'      
    if 'PBS' in str(df['note'].iloc[i]):
        df['class'].iloc[i] = 'PBS'      
        
        
#    if 'wash' in str(df['note'].iloc[i]):
#        df['class'].iloc[i] = 'PBS'


# get integer classes from string classes
classes = pd.factorize(df['class'])[0]
#classes = np.array(np.random.rand(len(input_df))*3).astype(int)
# remove redundant inputs for the model
input_df = df.drop(columns=['class', 'note', 'file', 'n', 'freq'])


# scale data
for col in input_df:
    input_df[col] = scale_vector(input_df[col])


# identify inputs and outputs
X = input_df.values
y = classes
# create LDA transformation
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit(X, y).transform(X)

plt.figure()
colors = ['k', 'm', 'g', 'c', 'b']#['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors,
                                 np.arange(len(colors)),
                                 np.unique(df['class'])):
    print(i)
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], alpha=.8, color=color,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('LDA')

plt.show()


#%% plot decision boundaries on LDA-transformed data

draw_boundaries = True

if draw_boundaries:
    # perform classification
    X = X_lda
    y = classes
    C = 1e6  # SVM regularization parameter
    models = (svm.SVC(kernel='linear', C=C, degree=2),
              svm.SVC(kernel='rbf', gamma=0.01, C=C),
              DecisionTreeClassifier(max_depth=4))
    models = (clf.fit(X, y) for clf in models)
    
    # title for the plots
    titles = ('SVM, linear kernel',
              'SVM, RBF kernel',
             'Decision Tree')
    
    # Setup grid for plotting.
    fig, sub = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.4, hspace=0.1)
    
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, h=0.02)
    
    fig = plt.gcf()
    fig.set_size_inches(12,5)
    
    for clf, title, ax, in zip(models, titles, sub.flatten()):
        
        score = round(clf.score(X,y), 3)
        print('%s score: %.4f' %(title, clf.score(X,y)))
    
        plot_contours(ax, clf, xx, yy, alpha=0.2, cmap=ListedColormap(['k', 'm', 'g', 'c', 'b']))#cmap=plt.cm.RdYlBu)
        ax.scatter(X0, X1, c=y, s=30, edgecolors='k',
                   cmap=ListedColormap(['k', 'm', 'g', 'c', 'b']))
        
        #ax.set_xlim(-50, -20)
        #ax.set_ylim(1.5, 7)
        ax.set_xlabel('LD-1', fontsize=18)
        ax.set_ylabel('LD-2', fontsize=18)
        ax.set_title(title, fontsize=18)
        
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.2)
    plt.show()

# %% decision boundaries on raw data
'''
classify_assembly = False
if classify_assembly:
    with open('exp_data\\marek_qcmd_2019_n_mag3.pkl', 'rb') as handle:
        mag_df = pickle.load(handle)
        
    # select only specific rows
    df = mag_df.loc[mag_df['note'].str.contains('assembly')]
    # cteate normalized frequency column
    df['f/n'] = df['freq'] / df['n']
    # remove outliers
    input_df = df[df['D'].between(1, 7)]
    input_df = input_df[input_df['f/n'].between(-50, -20)]
    # get integer classes from string classes
    classes = pd.factorize(input_df['note'])[0]
    # get inputs for the model
    input_df = input_df.drop(columns=['note', 'file', 'n', 'freq'])
    
    
    # show raw data
    plt.scatter(input_df['f/n'], input_df['D'])
    plt.show()
    
    # perform classification
    X = input_df.values[:, [6, 1]]
    y = classes
    C = 1e5  # SVM regularization parameter
    models = (svm.SVC(kernel='poly', C=1e6, degree=2),
              svm.SVC(kernel='rbf', gamma=0.01, C=1e5),
              DecisionTreeClassifier(max_depth=5))
    models = (clf.fit(X, y) for clf in models)
    
    # title for the plots
    titles = ('SVM, quadratic kernel',
              'SVM, RBF kernel',
             'Decision Tree')
    
    # Setup grid for plotting.
    fig, sub = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1, h=1)
    
    fig = plt.gcf()
    fig.set_size_inches(12,5)
    
    for clf, title, ax, in zip(models, titles, sub.flatten()):
        
        score = round(clf.score(X,y), 3)
        
        plot_contours(ax, clf, xx, yy, alpha=0.2, cmap=ListedColormap(['k', 'm', 'b']))#cmap=plt.cm.RdYlBu)
        ax.scatter(X0, X1, c=y, s=10, edgecolors='k',
                   cmap=ListedColormap(['k', 'm', 'b']))
        
        ax.set_xlim(-50, -20)
        ax.set_ylim(1.5, 7)
        ax.set_xlabel('$\Delta$f/n (Hz/cm$^{2}$)', fontsize=18)
        ax.set_ylabel('$\Delta$D (x10$^{-6}$)', fontsize=18)
        ax.set_title(title, fontsize=18)
        print('%s score: %.4f' %(title, clf.score(X,y)))
    
    plt.subplots_adjust(left=0.08, right=0.95, top=0.9, bottom=0.2)
    
    plt.show()
'''
