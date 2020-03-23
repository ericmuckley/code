# -*- coding: utf-8 -*-
"""refrigerant_channel_modeling.ipynb

Automatically generated by Colaboratory. Original file is located at:
https://colab.research.google.com/drive/1NOLLwT-XZHhCXqnlXs3Yi3oUflgNmdTz

This notebook is designed for analysis of neutron data of refrigerant
flowing through a microfluidic channel.

Created on Thu Feb 13 10:33:13 2020  
author: ericmuckley@gmail.com

"""

import json
import numpy as np
from glob import glob
from time import time
import matplotlib.pyplot as plt



def scale_image(image, scaling_factor=2):
    """Scale a 2D image using a scaling factor."""
    from skimage.transform import resize
    # get number of pixels in each dimension
    x_pixels = np.shape(image)[1]
    y_pixels = np.shape(image)[0]
    scaled_image  = resize(
        image, (scaling_factor*y_pixels, scaling_factor*x_pixels))
    return scaled_image


def get_bubble_profile(rad=10):
    """Get the 3D surface profile of a bubble using its radius,
    based on the equation of a sphere (x^2 + y^2 + z^2 = r^2)."""
    # create 2D x-y grid
    xgrid, ygrid = np.meshgrid(np.linspace(-rad, rad, num=1+2*rad),
                               np.linspace(-rad, rad, num=1+2*rad))
    # calculate value of 3D bubble surface based on equation of sphere
    z = np.sqrt(np.square(rad)-np.square(xgrid)-np.square(ygrid))
    # replace nan values with zeros
    return np.nan_to_num(z)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


"""### Trim empty space out of images
The empty space around each channel is removed, and the inner channel
region is bordered by red lines in the plot below. Finally, the
inner channel region is isolated from the rest of the image to isolate
the area of interest. The void faction of each region of interest is
calculated so that  blue=0 and red=1.
"""

# folder containing images
#img_folder_path = "refrigerant_images/*.*"
img_folder_path = r'C:\Users\a6q\Documents\Github\datasets\refrigerant_images/*'
all_images = glob(img_folder_path)
print('\n------------------------\nFound {} files:'.format(len(all_images)))
for img in all_images:
    print(img.split('\\')[-1])


# set vertical pixel limits that we want to select to remove aluminum area
rlims25 = [245, 280]
rlims75 = [225, 270]

# This section commented out because we are only looking at 75 lb flowrate
# images for now
# get list of all images and their titles
#img_names = [i for i in all_images if 'empty' not in i]
#labels = [i.split('/')[-1].split('.tif')[0] for i in img_names]
# get reference image as a 2D numpy array
#ref_img_name = [i for i in all_images if 'empty' in i][0]
#ref_img = plt.imread(ref_img_name)[:, :, 0].T[rlims75[0]:rlims75[1]]

# create dictionary to hold all results
d = {}

# loop over each non-reference image
# look at only 75 lb flow rate images for now
img_names = sorted([i for i in all_images if '75' in i])
labels = [i.split('\\')[-1].split('.tif')[0] for i in img_names]
for i, img in enumerate(img_names):
    
    # get flowrate and temperature from image names
    flowrate = int(labels[i].split('lb')[0])
    if flowrate == 25:
        img = plt.imread(img)[:, :, 0].T[rlims25[0]:rlims25[1]]
        temp = int(labels[i].split('00')[-1])
        xlims = [68, 428]
        ylims= [15, 25]
    if flowrate == 75:
        img = plt.imread(img)[:, :, 0].T[rlims75[0]:rlims75[1]]
        temp = int(labels[i].split('000')[-1])
        xlims = [97, 457]
        ylims = [20, 30]
        
    # cut image to only include channel interior
    channel_img = img[ylims[0]:ylims[1], xlims[0]:xlims[1]]

    # save results into dictionary
    d[labels[i]] = {
            'flowrate': flowrate,
            'temp': temp,
            'img_full': img,
            'img_channel': channel_img,
            'par_profile_full': np.mean(img, axis=0),
            'perp_profile_full': np.mean(img, axis=1),
            'par_profile_channel': np.mean(channel_img, axis=0),
            'perp_profile_channel': np.mean(channel_img, axis=1),
            'channel_min': np.min(channel_img),
            'channel_max': np.max(channel_img),
            'channel_mean': np.mean(channel_img),
            'channel_std': np.std(channel_img),
            'void_frac_img': channel_img/255} 
    print('Trimmed image pixel shape: {}'.format(np.shape(channel_img)))
    print('==============================================================')




"""## Prepare monte carlo model

### Define physical dimensions and constraints

First we determine the possible frequencies of occurence of each type of
bubble inside a given area. We take all combinations of possible bubbles, and
based on their diameters, remove possibilities which are not physically
relevant (i.e. multiple large bubbles together which will not fit inside a
given area.) The physical relevance is based on maximum and minimum
possible bubble diameters, number of distinct bubble sizes we want to search
for, and the physical channel dimensions.
"""

# define channel dimensions (width, height, depth) in μm
channel = {'w': 22000, 'h': 250, 'd': 750}

# define bubble radius limits and min buffer distance between bubbles in μm
bubble = {'min_rad': 1, 'max_rad': 60, 'buffer': 1}

"""promising parameters:
channel = {'w': 22000, 'h': 250, 'd': 750}
bubble = {'min_rad': 1, 'max_rad': 60, 'buffer': 1}
img = scale_image(img, scaling_factor=int(0.1*channel['w']/len(img[0])))
plot_channel_image(img, title='Rescaled image', vmin=0, vmax=1)
max_bubbles = 1000000
"""

starttime = time()

# set parameters for monte carlo simulation
max_bubbles = 50  # max number of bubbles to add to simulation
tot_trials = 2  # total trials to run for each image

#labels=[labels[-1]]
for L in labels:  # loop over each image

    # get target image, rescale so we can work with more pixels, and plot it
    img = d[L]['void_frac_img']
    img = scale_image(img, scaling_factor=int(0.1*channel['w']/len(img[0])))

    for trial_i in range(tot_trials):
        
        
        # create empty dictionary to hold simulation results
        sim = {'img': np.zeros(img.shape), 'cent': [], 'rad': []}
        simg = sim['img']
    
        # get limits of pixel coordinates
        xmax, ymax = len(simg[0])-1, len(simg)-1
    
        # generate array of random bubble centers and radii for seeding simulation
        cent_list = np.zeros((max_bubbles, 2)).astype(int)
        cent_list[:, 0] = np.random.randint(0, len(simg[0]), size=(max_bubbles))
        cent_list[:, 1] = np.random.randint(0, len(simg), size=(max_bubbles))
        rad_list = np.random.randint(
            bubble['min_rad'], bubble['max_rad'], size=(max_bubbles))
    
        # loop over bubbles and populate simulation
        for b in range(max_bubbles):
    
            # get random bubble radius and center and scale by channel depth
            rad = rad_list[b]
            cent = cent_list[b]
            bub_profile = get_bubble_profile(rad=rad) / channel['d']
    
            # get slice indices of image which will be populated by bubble
            x1, x2 = cent[0]-rad, cent[0]+rad+1
            y1, y2 = cent[1]-rad, cent[1]+rad+1
            # trim bubble and image indices if bubble extends outside image range
            if x1 < 0:
                bub_profile = bub_profile[:, -x1:]
                x1 = 0
            if x2 > xmax:
                bub_profile = bub_profile[:, :-(x2-xmax)]
                x2 = xmax
            if y1 < 0:
                bub_profile = bub_profile[-y1:]
                y1 = 0
            if y2 > ymax:
                bub_profile = bub_profile[:-(y2-ymax)]
                y2 = ymax
            
            # add new bubble profile to existing simulation if it will fit
            # first check if radius will fit at that specific center
            current_val = simg[cent[1], cent[0]]
            if current_val + 2*rad/channel['d'] <= img[cent[1], cent[0]]:
            # then check if the whole bubble will fit
                new_profile = np.zeros(simg.shape)
                new_profile[y1:y2, x1:x2] += bub_profile 
    
                # if entire bubble fits, save its stats and add it to model
                if np.all(simg + new_profile <= 1.05*img):
                    simg[y1:y2, x1:x2] += bub_profile 
                    sim['cent'].append(cent)
                    sim['rad'].append(rad)
    
        print('trial {}/{}:'.format(trial_i+1, tot_trials))
        tot_err = np.mean(np.abs(img - simg)/img)*100
        sim['tot_err_percent'] = float(tot_err)
        print('total bubbles: {}, runtime (min): {}, error (%): {}'.format(
            len(sim['rad']),
            round((time() - starttime)/60,1),
            round(tot_err, 2)))
        print('---------------------------------------------------------\n')

        # save model to file and download the file to local machine
        trial_label = L + '__trial_' + str(trial_i).zfill(3)
        sim['img'] = sim['img'].tolist()
        
        # save simulation dictionary to a json file
        with open(trial_label+'.json', 'w') as f:
            json.dump(sim, f, cls=NumpyEncoder)
        

        # load dictionary from json file
        #with open('data.json', 'r') as fp:
        #data = json.load(fp)



''''
# loop over every bubble and reconstruct the image
reconstructed_img = np.zeros_like(sim['img'])
for i in range(len(sim['rad'])):
    cent = sim['cent'][i]
    bub_profile = get_bubble_profile(rad=sim['rad'][i]) / channel['d']
    reconstructed_img = add_bubble_to_grid(
        reconstructed_img, bub_profile, rad=sim['rad'][i], cent=sim['cent'][i])
plot_channel_image(reconstructed_img, vmin=0, vmax=1)
'''



"""## Examine statistics of bubble distribution"""
'''
def get_stats(arr):
    """Return stats of an array in dictioary form."""
    d = {'min': np.min(arr), 'max': np.max(arr),
         'mean': np.mean(arr), 'median': np.median(arr), 'std': np.std(arr)}
    return d


def simulation_report(sim):
    """Create histograms which summarize statistics of the simulation."""
    # plot histogram of bubble radii
    d = get_stats(sim['rad'])
    print('Radius statistics:')
    [print('{}: {}'.format(key, val)) for key, val in d.items()]
    plt.hist(sim['rad'], bins=60)
    plot_setup(title='Distribution of bubble radii',
            labels=['Radius (μm)', 'Frequency'], size=(5, 3))
    plt.show()

    # plot histogram of bubble center y-values
    yvals = [pair[1] for pair in sim['cent']]
    plot_setup(title='Bubble positions perpendicular to channel',
            labels=['Position (μm)', 'Frequency'], size=(5, 3))
    plt.hist(yvals, bins=60)
    plt.show()
    # plot histogram of bubble center x-values
    xvals = [pair[0] for pair in sim['cent']]
    plot_setup(title='Bubble positions parallel to channel',
            labels=['Position (μm)', 'Frequency'], size=(5, 3))
    plt.hist(xvals, bins=60)
    plt.show()
'''
#simulation_report(sim)

