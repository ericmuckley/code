# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 10:20:12 2018
@author: a6q
"""
import os, csv, glob, numpy as np, pandas as pd 
import matplotlib.pyplot as plt
from matplotlib import rcParams
#rcParams.update({'figure.autolayout': True})
label_size = 20 #make size of axis tick labels larger
plt.rcParams['xtick.labelsize'] = label_size 
plt.rcParams['ytick.labelsize'] = label_size

import PIL.Image
import imageio
def register_extension(id, extension):
    PIL.Image.EXTENSION[extension.lower()] = id.upper()
PIL.Image.register_extension = register_extension
def register_extensions(id, extensions):
    for extension in extensions:
        register_extension(id, extension)
PIL.Image.register_extensions = register_extensions



#%% import NN output data files
NN_output_folder = glob.glob('C:\\Users\\a6q\\NN_output_2018-03-16/*')[0::20]
NN_output_folder.sort(key=os.path.getmtime)
print('found ' + format(len(NN_output_folder)) + ' NN output files') 


#%% look at last output file to get raw time, pressure, signal 
last_file_raw = pd.read_csv(NN_output_folder[-1])

#find lenth of non-nan points so we can remove all nans at the end of files
new_len = last_file_raw['signal'].notnull().sum() - 400

last_file = last_file_raw.iloc[0:new_len]
time = last_file['time']/60
pressure = last_file['pressure']
signal = last_file['signal']


#%% loop over ever NN output file

for i in range(len(NN_output_folder)):
    
    NN_output_data_full = pd.read_csv(NN_output_folder[i])
    NN_output_data = NN_output_data_full.iloc[0:new_len]
    
    model = NN_output_data['model']
    prediction = NN_output_data['prediction']
    error = NN_output_data['error']
    temp_signal = NN_output_data['signal']
    
    #find magnitude of largest signal
    signal_mag = np.amax(np.abs(signal))
    
    #calculate total error for each model/prediction
    model_plus_pred = np.add(np.nan_to_num(model), np.nan_to_num(prediction))
    deviation_raw = np.subtract(model_plus_pred, signal)
    deviation_percent = 100*np.divide(deviation_raw, signal_mag)
    
    #set up multi-plot figure
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(6,10))
    fig.subplots_adjust(hspace=0, bottom=.08, top=0.98, right=.95, left=.2)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    #plot error
    ax1.plot(time, np.abs(deviation_percent), linewidth=0.5, c='r')
    ax1.set_ylabel('Error (%)', fontsize=label_size)
    ax1.set_xlim(0,55)
    ax1.set_ylim(0,100)
    #plot model and prediction
    ax2.plot(time, model, linewidth=1, c='b')
    ax2.plot(time, prediction, linewidth=1, c='r')    
    ax2.set_ylabel('Model/prediction', fontsize=label_size)
    ax2.set_ylim(-75,40)
    #plot mreasured signal
    ax3.scatter(time, temp_signal, s=2, c='g')
    ax3.plot(time, signal, c='k', linewidth=.5)
    ax3.set_ylabel('Response', fontsize=label_size)
    #ax2.set_xlabel('Time (hours)', fontsize=label_size)
    ax3.set_ylim(-75, 40)
    #plot pressure sequence
    ax4.plot(time, pressure, c='b', linewidth=1)
    ax4.set_ylabel('RH (%)', fontsize=label_size)
    ax4.set_xlabel('Time (hours)', fontsize=label_size)
    ax4.set_ylim(0, 90)
    
    #save plot as image file
    save_pic_filename = 'gif_frames\\NN_output_frame_'+format(i)+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)
    
    #close figure from memory
    plt.close(fig)
#close all figures from memory
plt.close("all")

#%% find all files in the designated data folder and sort by time/date
all_frames = glob.glob('C:\\Users\\a6q\\gif_frames/*')


all_frames.sort(key=os.path.getmtime)
print('found ' + format(len(all_frames)) + ' images')

#create gif using all saved image files
pics = []
for filename in all_frames: pics.append(imageio.imread(filename))
imageio.mimsave('NN_output2_2018-03-16.gif', pics, duration=0.2)


#%%




#%% combine images to form video
'''
import cv2
import os

image_folder = 'images'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, -1, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

'''

