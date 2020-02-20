# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:46:57 2020
@author: ericmuckley@gmail.com

"""
from time import time
import glob
import numpy as np
import matplotlib.pyplot as plt
from ncempy.io import dm
from scipy import signal
from scipy.fftpack import fft, rfft
from matplotlib.patches import Rectangle
from scipy import interpolate

# change matplotlib settings to make plots look nicer
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3





def plot_setup(labels=['X', 'Y'], fsize=18, setlimits=False, limits=[0,1,0,1],
               title='', legend=False, save=False, filename='plot.jpg'):
    """Creates a custom plot configuration to make graphs look nice.
    This can be called with matplotlib for setting axes labels,
    titles, axes ranges, and the font size of plot labels.
    This should be called between plt.plot() and plt.show() commands."""
    plt.xlabel(str(labels[0]), fontsize=fsize)
    plt.ylabel(str(labels[1]), fontsize=fsize)
    plt.title(title, fontsize=fsize)
    fig = plt.gcf()
    fig.set_size_inches(19, 19)
    if legend:
        plt.legend(fontsize=fsize-4)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()



#%%
        
        
img_len_x, img_len_y = ((40, 30))
img = np.random.random((img_len_y, img_len_x))
img[10:, 20:] = img[10:, 20:] + np.square(img[10:, 20:])

remap = np.full((img_len_y, img_len_x), np.nan)

# choose number of pixels for sampling and oversampling
# oversamp gives an over-sampling buffer on each side of samp_pix
samp_num = 1
oversamp_num = 3


samp_origin_x = np.arange(0, img_len_x, samp_num)
samp_origin_y = np.arange(0, img_len_y, samp_num)
samp_grid = np.array(np.meshgrid(
        samp_origin_x, samp_origin_y)).T.reshape(-1,2)




# loop over each sampling window
# (x0, y0) = origin of sampling window
for x0, y0 in samp_grid:
    # find range of sampling window in each direction and
    # clip sampling range if it lies outside of image area
    x0, x1 = np.clip((x0-oversamp_num, x0+samp_num+oversamp_num), 0, img_len_x)
    y0, y1 = np.clip((y0-oversamp_num, y0+samp_num+oversamp_num), 0, img_len_y)


    # get oversampled image
    img0 = img[y0:y1, x0:x1]
    
    print(np.shape(img0))
    # acquire data about oversampled image
    result = np.mean(img0)
    
    
    # save result to sampled data
    remap[y0:y1, x0:x1] = result#img[y_range[0]:y_range[1], x_range[0]:x_range[1]]




plt.imshow(img, origin='lower')
plt.show()

plt.imshow(remap, origin='lower')
plt.show()




#%%


'''
def get_physical_image_size(data):
    """Get the physcial size of the image, using the dictionary of image
    data as an input."""
    nm_per_x_pixel, nm_per_y_pixel = data['pixelSize']
    x_span = nm_per_x_pixel*len(data['data'][0])
    y_span = nm_per_y_pixel*len(data['data'])
    return (x_span, y_span), nm_per_x_pixel


def scale_array(arr, lim=(0, 1)):
    """Scale values of an array inside new limits."""
    scale = lim[1] - lim[0]
    arr_scaled = scale*(arr-np.min(arr))/(np.max(arr)-np.min(arr))+lim[0]
    return arr_scaled

def get_window_grid(data, window_len):
    """Get a grid of pixel coordinates at which to position a sliding
    window across the entire image."""
    x_steps = np.arange(0, len(data['data'][0])-window_len, window_len)
    y_steps = np.arange(0, len(data['data'])-window_len, window_len)
    grid = np.array(np.meshgrid(x_steps, y_steps)).T.reshape(-1,2)
    return grid

def get_gradient(arr):
    """Get 2D gradient of 2D array."""
    grad_y, grad_x = np.gradient(arr)
    grad = {'x': grad_x,
            'y': grad_y,
            'xabs': np.abs(grad_x),
            'yabs': np.abs(grad_y),
            'med_xabs': np.median(np.abs(grad_x)),
            'med_yabs': np.median(np.abs(grad_y)),
            'max': np.max([np.abs(grad_x), np.abs(grad_y)]),
            'min': np.min([np.abs(grad_x), np.abs(grad_y)]),
            'x-y': np.median(np.abs(grad_x))-np.median(np.abs(grad_y))}
    return grad



# set the folder holding the image files   
data_folder = r'C:\Users\a6q\exp_data\2020-01-24_CeO2ZnO2_on_graphene'
# set sliding window size length in pixels
win_len = 30

# get list of all files in the folder and list of all images to examine
files = glob.glob(data_folder + r'/*')
images = [f for f in files if 'cw' in f.lower() and 'saed' not in f.lower()]
start_time = time()

# loop over each image in the folder
for filename in images[:1]:
   
    # read file
    image_label = filename.split('\\')[-1].split('.dm3')[0]
    print('\n------------------------------\nFile: {}'.format(image_label))
    data = dm.dmReader(filename)
    img_span, nm_per_pixel = get_physical_image_size(data)
    grid = get_window_grid(data, win_len)#[:60]
    
    orientation = np.zeros_like(data['data']).astype(float)
    
    plot_raw_image = True
    if plot_raw_image:
        plt.imshow(data['data'], zorder=1, origin='lower',
                   cmap='gray',
                   extent=[0, img_span[0], 0, img_span[1]])
        
        plot_setup(
                title=image_label,
                labels=['Distance (nm)', 'Distance (nm)'])

        # plot window square
        plt.gca().add_patch(
                Rectangle((nm_per_pixel*x0, nm_per_pixel*y0),
                nm_per_pixel*win_len, nm_per_pixel*win_len,
                linewidth=1, 
                edgecolor='white',
                facecolor='none', zorder=5))
        plot_setup(
                title=image_label,
                labels=['Distance (nm)', 'Distance (nm)'])
        plt.show()

    
    # loop sliding window over entire image
    for cord_i, cord in enumerate(grid):
        #if cord_i % 1000 == 0:
        #    print('{} / {}'.format(cord_i, len(grid)+1))
        
        x0, y0 = (cord)
        
        



        # get window area
        img0 = scale_array(data['data'][y0:y0+win_len, x0:x0+win_len])
        # plot window image
        #plt.imshow(img0, cmap='gray', origin='lower')
        #plt.show()
        
        # get 2D gradient statistics
        grad = get_gradient(img0)


        plt.imshow(grad['xabs'], vmin=grad['min'], vmax=grad['max'],
                   cmap='jet', origin='lower')
        plt.colorbar()
        plt.title('X gradient, median: {}'.format(grad['med_xabs']))
        plt.show()
        
        plt.imshow(grad['yabs'], vmin=grad['min'], vmax=grad['max'],
                   cmap='jet', origin='lower')
        plt.colorbar()
        plt.title('Y gradient, median: {}'.format(grad['med_yabs']))
        plt.show()            

        # save orientation data
        orientation[y0:y0+win_len, x0:x0+win_len] = grad['med_yabs']/grad['med_xabs']#['x-y']

        if grad['x-y'] > 0.01:
            orientation[y0:y0+win_len, x0:x0+win_len] = 1
        if grad['x-y'] < -0.01:
            orientation[y0:y0+win_len, x0:x0+win_len] = -1
        #if grad['x-y'] 



    # plot oritentation map
    plt.imshow(orientation, origin='lower', cmap='bwr',
               extent=[0, img_span[0], 0, img_span[1]], zorder=6, alpha=0.3) 
    plt.colorbar()
    plt.show()

print('total time: {} s'.format(np.round(time() - start_time, 2)))
'''