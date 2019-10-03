# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:11:20 2019

@author: a6q
"""
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def open_pickle(filename):
    '''
    Opens serialized Python pickle file as a dictionary.
    Filename should be something like 'saved_data.pkl'.
    '''
    with open(filename, 'rb') as handle:
        dic = pickle.load(handle)
    return dic


def bar_plot(
    y, xlabels, axis_titles=['X', 'Y'], fontsize=18, figsize=(10, 6),
    setlimits=False, ylimits=[0, 1], colors=None, title=None,
    save=False, filename='fig.jpg'):
    ''' Plots a bar graph using matplotlib with y-values and x-labels.
    For colors, pass a matplotlib colormap such as colors=cm.jet(y/20).'''
    plt.rcParams['xtick.labelsize'] = fontsize 
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.xlabel(str(axis_titles[0]), fontsize=fontsize)
    plt.ylabel(str(axis_titles[1]), fontsize=fontsize)
    plt.xlim((-1, len(y)))
    if setlimits:
        plt.ylim((ylimits[0], ylimits[1]))
    else:
        plt.ylim((0, max(y)))
    if title:
        plt.title(str(title), fontsize=fontsize)
    plt.bar(np.arange(len(y)), y, width=0.6, color=colors)
    plt.xticks(np.arange(0, len(y), 1), xlabels, rotation='vertical')    
    fig = plt.gcf()
    fig.set_size_inches(10, 6)
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()
    plt.show()


def df_to_heatmap(df, vmin=0, vmax=100, fontsize=14, title=None,
                  save=False, filename='fig.jpg'):
    '''
    Plot a heatmap from 2D data in a Pandas DataFrame. The y-axis labels 
    should be index names, and x-axis labels should be column names.
    '''
    plt.rcParams['xtick.labelsize'] = fontsize 
    plt.rcParams['ytick.labelsize'] = fontsize
    #plt.xlabel(str(axis_titles[0]), fontsize=fontsize)
    #plt.ylabel(str(axis_titles[1]), fontsize=fontsize)
    plt.pcolor(df, cmap='jet', vmin=vmin, vmax=vmax)
    plt.yticks(np.arange(0.5, len(df.index), 1), df.index, fontsize=fontsize)
    plt.xticks(np.arange(0.5, len(df.columns), 1),
               df.columns, rotation='vertical', fontsize=fontsize)
    if title:
        plt.title(title, fontsize=fontsize)
    fig = plt.gcf()
    fig.set_size_inches(9, 8)
    plt.colorbar()
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()
    plt.show()


def plot_setup(labels=['X', 'Y'], size=16, setlimits=False, limits=[0,1,0,1],
               title='', save=False, filename='plot.jpg'):
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
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()


#%% load data

# open the file
dic = open_pickle('C:\\users\\a6q\\exp_data\\pp_automl_dict.pkl')
df0 = pd.DataFrame.from_dict(dic)

# create dataframes
df_traintime = df0.copy(deep=True)
df_scorelist = df0.copy(deep=True)
df_lowerr = df0.copy(deep=True)

# save lengths of each hyperparameter list
hp_lengths = []
# populate dataframes
for row in df0.index:
    for col in df0.columns:
        
        scorelist = list(np.sort(df0.loc[row, col]['score'])[::-1])
        traintime = float(df0.loc[row, col]['traintime'])
        lowerr = scorelist[-1]
        
        hp_lengths.append(len(scorelist))

        df_scorelist.loc[row, col] = scorelist
        df_traintime.loc[row, col] = traintime
        df_lowerr.loc[row, col] = lowerr

#%% create iterator for custom sampling of hyperparameter list

custom_i = list(range(40))[::2] + \
        list(range(50, 160, 20)) + \
        list(range(200, 1200, 200))

#%% create error vs. training time plots     
   
traintime_plots = False
if traintime_plots:     
    fignum = 0  
    for col in df_traintime.columns:
        # get points to plot
        x = df_traintime[col].values.astype('float')
        y = df_lowerr[col].values.astype('float')
        xmin, xmax, ymin, ymax = 0.01, 10, 1, 100
        fignumstr = str(fignum).zfill(4)
        # add scatter points and labels to plot
        for i, txt in enumerate(df_traintime.index):
            if xmin <= x[i] <= xmax and ymin <= y[i] <= ymax:
                plt.scatter(x, y, c='b', s=8)
                plt.annotate(str(txt)[:3].upper(),
                             ((x[i], y[i])), fontsize=10)
        plt.xscale('log')
        plt.yscale('log')
        plot_setup(labels=['Training time (s)', 'Mean error (%)'], size=18,
                      setlimits=True, limits=[xmin, xmax, ymin, ymax],
                      title=col.upper(),
                      save=True, filename='exp_data/mpe_traintime_figs/'+fignumstr+'.jpg')
        plt.show()
        fignum += 1

#%% create heatmap

best_scores = np.empty((0, len(list(df_scorelist))))


create_heatmap = False
if create_heatmap:
    
    df_score0 = df_scorelist.copy(deep=True)
    
    for i in custom_i:
        
        for row in df_scorelist.index:
            for col in df_scorelist.columns:
        
                list0 = df_scorelist.loc[row, col]
                df_score0.loc[row, col] = list(list0)[i] if i < len(list0) else list0[-1]
   
        
        # rebuild dataframe so that cells are all floats
        df_score0 = pd.DataFrame(data=df_score0.values.astype('float'),
                          columns=list(df_score0), index=df_score0.index)
        
        
        best_scores0 = df_score0.min()
        
        best_scores = np.vstack((best_scores, best_scores0))
        
        
        '''
        filename = 'C:\\Users\\a6q\\exp_data\\pp_heatmaps\\'+str(i).zfill(4)+'.jpg'
        df_to_heatmap(
                df_score0, vmax=50,
                title='Mean error (%): hyperparameter config '+str(i).zfill(4),
                save=True, filename=filename)
        '''


#%% best scores all
'''
best_scoresall = np.empty((0, len(list(df_scorelist))))


for i in range(max(hp_lengths)):
    for row in df_scorelist.index:
        for col in df_scorelist.columns:
            
            df_score0.loc[row, col] = list(list0)[i] if i < len(list0) else list0[-1]
            best_scores0 = df_score0.min()
            
            best_scores = np.vstack((best_scores, best_scores0))
'''

#%% create best score by model bar graph
'''
bargraph = True
if bargraph:
    for row, y in enumerate(best_scores):
        
        filename = 'C:\\Users\\a6q\\exp_data\\pp_bargraphs\\'+str(row).zfill(4)+'.jpg'
        
        bar_plot(y, list(df0), axis_titles=['Property', 'Mean error (%)'],
                 setlimits=True, ylimits=[0, 20],
                 title='Hyperparameter config '+str(custom_i[row]).zfill(4),
                 save=True,
                 filename=filename)
'''

#%%

from glob import glob
import cv2

def create_video(image_list, video_name, fps=8, reverse=False):
    # create video out of images saved in a folder
    # frames per second (fps) and order of the images can be reversed 
    # using the **kwargs.
    if reverse: image_list = image_list[::-1]
    frame = cv2.imread(image_list[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    for image in image_list:
        video.write(cv2.imread(image))
    cv2.destroyAllWindows()
    video.release()
    return video

image_list = sorted(glob(r'C:\Users\a6q\exp_data\pp_bargraphs/*.jpg'))
#image_list = sorted(glob(r'C:\Users\a6q\exp_data\pp_heatmaps/*.jpg'))
#image_list = [i for i in image_list if 'error' in i]

create_vid = False
if create_vid:
    create_video(image_list,
            r'C:\Users\a6q\Desktop\pp_bargraphs.avi',                     
            fps=8)
