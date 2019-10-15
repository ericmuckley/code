#from __future__ import print_function, division, unicode_literals  # ensure python3 compatibility
import numpy as np  # fast math
#from warnings import warn
import matplotlib.pyplot as plt  # plotting
#import h5py  # reading the data file
#import os  # file operations
#from scipy import interpolate, stats  # various convenience tools
#from skimage import transform  # image processing and registration
#import subprocess
#import sys
import cv2
    
#from scipy import ndimage
#from scipy import misc
import glob
import scipy.ndimage
import scipy.interpolate as inter

def normalize_image(image):
    #Normalizes the provided image from 0 to 1
    return (image - np.amin(image)) / (np.amax(image) - np.amin(image))


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







#%%
 
    
#load images as numpy arrays
image_names = glob.glob(
        r'C:\Users\a6q\Desktop\lab on a QCM\images\2018-10-05_pedotpss_qcm_images_fast_50X_2sec_6/*')

image_destination = 'C:\\Users\\a6q\\exp_data\\corrected_images\\'


#Initiate SIFT detector for image matching
orb = cv2.ORB_create()

image_matching_on = False

line_profiles_on = True
profile_len = 300
line_profiles =  np.empty((profile_len, 0))
shift = []

#loop over every image in folder
for i, img in enumerate(image_names):
    print('image %i/%i' %(i+1, len(image_names)))
    
    #load image
    image0 = cv2.imread(img)
    
    #crop image by using numpy slicing of pixels
    #image0 = image0[900:-500, 1800:, :]
    image0 = image0[:2000, :1200, :]
    
    #resize to fraction of original
    #image0 = cv2.resize(image0, (0,0), fx = 0.5, fy = 0.5)
    #normalize image intensity
    #image0 = normalize_image(image0)
    
    
    #sharpen with a custom kernel
    #kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    #image0 = cv2.filter2D(image0, -1, kernel)
    
    #to flip red and blue color channels
    #image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)



    all_intensity = np.sum(image0, axis=2)
    all_intensity = all_intensity - np.amin(all_intensity)
    all_intensity = all_intensity / np.amax(all_intensity)
    #all_intensity = np.round(all_intensity)

    #extract line profiles
    if line_profiles_on:
        x1, y1 = 600, 30
        x2, y2 = 600, 1800
        
        profilex = np.linspace(x1, x2, profile_len)
        profiley = np.linspace(y1, y2, profile_len)
        
        profile0 = scipy.ndimage.map_coordinates(
                                                all_intensity,
                                                np.vstack((
                                                profiley, profilex)))
        
        #fit profile to psline to smooth it
        spline_params = inter.UnivariateSpline(np.arange(profile_len),
                                               profile0, s=1.5)
        spline = spline_params(np.arange(profile_len))        
        spline = spline - np.min(spline)
        spline = spline / np.max(spline[20:])
        
        
        line_profiles = np.column_stack((line_profiles, spline))
    
        plt.plot(profile0)
        plt.plot(spline)
        plt.show()
    
    
    
        for i in range(len(line_profiles[0])):
            plt.plot(np.arange(profile_len)+1, line_profiles[:,i], alpha=.3)
        plt.title('Line profiles', fontsize=18)
        plt.show()
    
    
        plt.plot([x1, x2], [y1, y2], c='r')
    
    plt.tight_layout()
    #plt.axis('off')
   
    plt.imshow(all_intensity)
    plt.show()


    shift0 = np.where(spline==1)[0][0]
    shift.append(shift0)
plt.plot(shift)
plt.show()

#%% heatmaps of interference
'''
#loop over each spectrum
for i in range(len(dic2[key][0])-1):
    #create arrays of X, Y, and Z values
    Xf = np.append(Xf, np.repeat(i, len(dic2[key][:,0])))
    Yf = np.append(Yf, dic2[key][:,0])
    Zf = np.append(Zf, dic2[key][:,i+1])
    
#create x, y, and z points to be used in heatmap
xf = np.linspace(Xf.min(),Xf.max(),100)
yf = np.linspace(Yf.min(),Yf.max(),100)
zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
#create the contour plot
CSf = plt.contourf(xf, yf, zf, 100, cmap=plt.cm.rainbow, 
                   vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
plt.colorbar()
label_axes('Time', 'F (MHz)')
plt.show()
'''   



'''
    if image_matching_on:
        #save first image as reference for subsequent images
        if i == 0:
            ref_image = image0 
            image_corr = image0
        
        if i > 0:
            # find the keypoints and descriptors with SIFT
            kp1, des1 = orb.detectAndCompute(ref_image, None)
            kp2, des2 = orb.detectAndCompute(image0, None)
    
            #create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                # Match descriptors.
            matches = bf.match(des1,des2)
                # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
                # Draw first 10 matches.
            img_match = cv2.drawMatches(ref_image, kp1,
                                   image0, kp2,
                                   matches[:6], None, flags=2)
        
        
            plt.imshow(img_match)
            plt.show()
    
    
    
            #extract location of good matches
            matchpoints1 = np.zeros((len(matches), 2), dtype=np.float32)
            matchpoints2 = np.zeros((len(matches), 2), dtype=np.float32)
    
            for j, match in enumerate(matches):
                matchpoints1[j, :] = kp1[match.queryIdx].pt
                matchpoints2[j, :] = kp2[match.trainIdx].pt
               
            #Find homography
            h, mask = cv2.findHomography(matchpoints1, matchpoints2, cv2.RANSAC)
            #Use homography
            height, width, channels = image0.shape
            image_corr = cv2.warpPerspective(image0, h, (width, height))
            
    else:
        image_corr = image0
        plt.imshow(image_corr)
    
        
    
    
    

    
    plt.tight_layout()
    plt.axis('off')
    #plt.savefig(image_destination+str(i).zfill(3)+'.jpg', format='jpg', dpi=250)
    plt.show()

    cv2.imwrite(image_destination+str(i).zfill(5)+'.jpg', image_corr)
'''


#%% combine figs into video

image_folder = image_destination
video_name = 'C:\\Users\\a6q\\Desktop\\pp_swelling.avi'

make_video = False
if make_video: create_video(image_destination, video_name, fps=8, reverse=True)