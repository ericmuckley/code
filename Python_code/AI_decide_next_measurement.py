# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:03:15 2018

@author: a6q
"""

import numpy as np
import matplotlib.pyplot as plt
import time

#%%

def label_axes(xlabel='x', ylabel='y', size=16):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)


def sigmoid(x):
  return 1 / (1 + np.exp(-x+5))





def create_video(image_folder, video_name, fps=8, reverse=False):
    #create video out of images saved in a folder
    import cv2
    import os
    
    images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]
    if reverse: images = images[::-1]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()




def next_point_to_measure(x,y):
    '''determines the independent variable of the next measurement point
    based on the largest difference (gap) in measurmeents of a dependent
    variable. inputs are x and y, the previously measured independent and 
    dependent variable values. This funtion only interpolates, it will not
    suggest a measurmeent which is outside the range of input x values.'''
    #sort arrays in order smallest to largest
    x, y = np.sort(np.array([x, y]))
    #find differences between values of adjacent measurements
    diffs = np.diff(y)*6
    #find index of biggest difference
    big_diff_index = np.argmax(diffs)
    #get suggested next independent variable value
    next_x_val = np.mean((x[big_diff_index], x[big_diff_index+1]))
    return next_x_val



def find_max_bias(current_lim):
    '''find maximum bias voltage which should be applied in order to keep
    current under the specificed current limit.'''
    print('searching for bias limit...')
    #biases to test in Volts
    bias_try = np.linspace(0, 0.1, 201)
    bias_lim = bias_try[0]
    for try0 in bias_try:
        print(try0)
        #measure current here
        current0 = try0*3
        #if current goes over limit, stop and save bias limit
        if current0 >= current_lim:
            print('Current limit of %0.6f A was reached at %0.6f V bias.' %(
                    current_lim, try0))
            print('Max bias has been set to %0.6f V.' %bias_lim)
            break
        #if current does not go over limit, increase bias limit
        else: bias_lim = try0
    return bias_lim



#%%
    

bias_lim = find_max_bias(.005)



#%% stop scan based on high derivative or abs value
    

'''
measure_limit = .05

#independent variable
x_try = np.linspace(0, .001, 51)

y = np.array([])
x = np.array([])

for try0 in x_try:
    
    #dependent variable
    y0 = try0*100
    
    if y0 >= measure_limit:
        print('Current limit reached at %0.8f V.' %try0)
        plt.axhline(y=measure_limit, c='r', alpha=0.75, ls='dotted')
        break
    else:
        x = np.append(x, try0)
        y = np.append(y, y0)

plt.scatter(x, y, s=10, c='k', alpha=0.6)

label_axes('X', 'Y')
plt.show()
'''



#%% AI decide next measurement

find_next_measurement = False
if find_next_measurement:
    image_destination = 'C:\\Users\\a6q\\exp_data\\decide_next_measurement_images\\'
    
    #independent variable
    x = np.array([.02, 0.5, 0.95]).astype(float)
    
    for i in range(30):
        #measurement
        y = np.log(x-0.01) + np.power(x-.5, 4)*40
        y -= np.min(y)
        y = y / np.max(y)
        print(np.min(y))
        
        plt.scatter(x,y,
                    facecolors='none',
                    edgecolors='k',
                    marker='o', s=20)
        
        next_point = next_point_to_measure(x,y)
        
        if i != 0:
            plt.axvline(x=next_point, c='r', alpha=0.75, ls='dotted')
            
        label_axes('P/P$_{0}$', 'Response')
        plt.tight_layout()
        #plt.savefig(image_destination+str(i).zfill(3)+'.jpg', format='jpg', dpi=150)
        plt.show()
        
        x = np.append(x, next_point_to_measure(x,y))




#%% combine figs into video
'''

video_name = 'C:\\Users\\a6q\\Desktop\\AI_next_point_selection.avi'

make_video = False
if make_video: create_video(image_destination, video_name, fps=2, reverse=False)
'''