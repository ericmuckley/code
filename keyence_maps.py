# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:41:33 2020
@author: ericmuckley@gmail.com
"""

import cv2
import numpy as np
import pandas as pd
from glob import glob
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# change matplotlib settings to make plots look nicer
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3
plt.rc('font', **{'sans-serif' : 'Arial', 'family' : 'sans-serif'})

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


def tilt_correct(mat, x_correct=True, y_correct=True, remove_minimum=True,
                 reference=None):
    """Correct the tilt of a 2D matrix, such as an image. Use 'reference'
    to correct tilt based on a designated reference square within the matrix.
    Square should designate matrix indices as: [x1, y1, x2, y2]"""
    if reference is not None:
        x1, y1, x2, y2 = reference
        m = mat[y1:y2, x1:x2]
        if x_correct:
            avg_x_profile = np.mean(m, axis=0)
            x_slope,_,_,_,_ = stats.linregress(
                    np.arange(x2-x1), avg_x_profile)
            x_fit_line = np.arange(len(mat[0])) * x_slope
            mat -= x_fit_line
        if y_correct:
            avg_y_profile = np.mean(m, axis=1)            
            y_slope,_,_,_,_ = stats.linregress(
                    np.arange(x2-x1), avg_y_profile)
            y_fit_line = np.arange(len(mat)) * y_slope
            mat -= np.reshape(y_fit_line, (len(mat), 1))
        if remove_minimum:
            mat -= np.min(mat)
    else:
        if x_correct:
            avg_x_profile = np.mean(mat, axis=0)
            mat -= avg_x_profile
        if y_correct:
            avg_y_profile = np.mean(mat, axis=1)
            mat -= np.reshape(avg_y_profile, (len(avg_y_profile), 1))
        if remove_minimum:
            mat -= np.min(mat)
    return mat

#%%

filelist = glob(r'C:\Users\a6q\exp_data\BR_IL_keyence_maps\/*')



labels = []
max_heights = []

# loop over each file
for fi, f in enumerate(filelist):

    # get file label
    label = f.split('.csv')[0].split('\\')[-1]#.split(' 01')[0]
    labels.append(label)
    
    # get X, Y, Z scales
    scale_df = pd.read_csv(f, skiprows=39, nrows=2, header=None)
    xy_nm_per_pixel = scale_df.iloc[0, 1]
    z_nm_per_digit = scale_df.iloc[1, 1]
   
    # get matrix data
    mat = pd.read_csv(f, skiprows=49, header=None).values.astype(float)[::-1]
    # convert height to nanometers
    mat *= z_nm_per_digit/1e3 

    
    
    mat = tilt_correct(mat, reference=[0, 220, 80, 300])
    mean = np.mean(mat)
    std = np.std(mat)
    low_lim, high_lim = mean - 3*std, mean + 8*std
    mat = np.clip(mat, low_lim, high_lim)
    
    print('%s: MIN: %0.2f, MAX: %0.2f, AVG: %0.2f, RANGE: %0.2f' %(
            label, np.min(mat), np.max(mat), np.mean(mat), np.ptp(mat)))

    # save stats
    profile = np.median(mat[320:410, :], axis=0)
    profile -= np.min(profile)
    profiles = profile if fi == 0 else np.column_stack((profiles, profile))
    max_heights.append(np.max(profile))

    if fi in [0, 1]:
        mat -= 1.1
    if fi in [2]:
        mat -= 0.3
    

    plot_image = True
    if plot_image == True:
        '''
        plt.gca().add_patch(Rectangle(
                (0, 220), 80, 80, linewidth=1,
                edgecolor='r', facecolor='none'))
        '''
        plt.imshow(mat, interpolation='gaussian',
                   vmin=0,
                   vmax=0.8,
                   cmap='terrain', origin='lower',
                   extent=xy_nm_per_pixel/1e3 * np.array([0, 1024, 0, 768]))

        plt.colorbar()
        plt.text(-40, 250, label, fontsize=18)
        
        img_path = r'C:\\Users\\a6q\\exp_data\\Br_IL_keyence_images\\'
        plot_setup(title='Height (μm)',
                   labels=['Distance (μm)', 'Distance (μm)'],
                   filename=img_path+str(label)+'.jpg',
                   save=True, dpi=200)
        plt.show()

    



# %% show cross sections and max height over time 
     
for col in range(len(profiles[0])):
    plt.plot(profiles[:, col], label=labels[col])
plt.show()

plt.plot(max_heights, marker='o')
plt.show()

# %% make video 

make_video = False
if make_video:
    imagelist = glob(r'C:\Users\a6q\exp_data\Br_IL_keyence_images\/*.jpg')
    print(imagelist)
    #image_list = [i for i in image_list if 'error' in i]
    
    video = create_video(
            imagelist,
            fps=2,
            video_name = r'C:\Users\a6q\Desktop\keyence_imgs.avi')





