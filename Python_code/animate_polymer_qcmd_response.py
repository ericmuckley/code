# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:55:03 2020
@author: ericmuckley@gmail.com

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import cv2

# change matplotlib settings to make plots look nicer
fsize=18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3

'''
def plot_setup(labels=['X', 'Y'], fsize=18, setlimits=False, limits=[0,1,0,1],
               title='', legend=True, save=False, filename='plot.jpg'):
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
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()
'''

def create_video(imagelist, video_name='vid.avi', fps=8, reverse=False):
    """
    Create video out of a list of images saved in a folder.
    Specify name of the video in 'video_name'.
    Frames per second (fps) and order of the images can be reversed.
    """
    imagelist = sorted(imagelist)
    if reverse:
        imagelist = imagelist[::-1]
    # get size of the video frame
    frame = cv2.imread(imagelist[0])
    height, width, layers = frame.shape
    # initiate video
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    for i, img in enumerate(imagelist):
        print('writing frame %i / %i' %(i+1, len(imagelist)))
        img = cv2.imread(img)
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    return video


# %%

data_path = r'C:\Users\a6q\exp_data\bb3_df_for_vid.csv'
df = pd.read_csv(data_path)
harmonics = [int(h.split('df')[1]) for h in df.columns[1:]]
y = harmonics
times = np.array(df['time_min']).astype('float')

# plot all traces
[plt.plot(times, df.iloc[:, i]) for i in range(1, len(df.columns))]
plt.show()

# loop through each row of the data table 
for i in range(56):#len(df)):
    
    # get data for plot
    curr_time = df.iloc[i]['time_min'] - 6.25
    shifts = np.array(df.iloc[i][df.columns[1:]]).astype(float)
    x = np.arange(len(shifts))
    z = np.tile(shifts, (len(shifts), 1)).T
    
    # create plot
    plt.imshow(z, interpolation='gaussian', extent=[0, 30, 17, 1],
               cmap='seismic', vmin=-300, vmax=300)
    
    vapor_str = 'air' if curr_time < 0 else 'cyclohexane'
    plot_text = 'Exposure time (min): '+str(np.round(curr_time, 1))+'   Vapor: '+vapor_str
    plt.text(0, -3, plot_text, fontsize=fsize)
    plt.text(31, 4, '$\Delta$f/n (kHz/cm$^2$)', fontsize=fsize, rotation=90)
    plt.text(32, 19, 'Mass loading', fontsize=fsize)
    plt.text(32, 0, 'Delamination', fontsize=fsize)
    
    fig = plt.gcf()
    fig.set_size_inches((12, 4))
    
    
    plt.xlabel('Film-crystal interface', fontsize=fsize)
    plt.ylabel('Harmonic number', fontsize=fsize)
    plt.title('Film-vapor interface', fontsize=fsize)
    plt.colorbar()
    
    
    frame1 = plt.gca()
    frame1.axes.xaxis.set_ticklabels([])
    plt.yticks(np.arange(min(harmonics), max(harmonics)+2, 2))

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    
    fig_path = 'C:\\Users\\a6q\\exp_data\\qcmd_dynamics_figs\\'
    fig_name = str(i).zfill(3) + '.jpg'
    fig.savefig(fig_path + fig_name, dpi=120, bbox_inches='tight')
    
    plt.tight_layout()
    plt.show()
    

#%%


make_video = True
if make_video:
    imagelist = glob(r'C:\Users\a6q\exp_data\qcmd_dynamics_figs\/*.jpg')
    #image_list = [i for i in image_list if 'error' in i]
    
    video = create_video(
            imagelist,
            fps=8,
            video_name = r'C:\Users\a6q\Desktop\BB3_response.avi')
