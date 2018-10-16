import glob, os, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.signal as filt
from scipy.signal import medfilt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import pickle
from scipy.interpolate import splrep
from scipy.interpolate import splev
import scipy.interpolate as inter



def label_axes(xlabel='x', ylabel='y', size=18):
    #set axes labels and size
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(xlabel), fontsize=size)
    plt.ylabel(str(ylabel), fontsize=size)





def vec_stretch(vecx0, vecy0=None, vec_len=100, vec_scale='lin'):
    '''Stretches or compresses x and y values to a new length
    by interpolating using a 3-degree spline fit.
    For only stretching one array, leave vecy0 == None.'''

    #check whether original x scale is linear or log
    if vec_scale == 'lin': s = np.linspace
    if vec_scale == 'log': s = np.geomspace
    
    #create new x values
    vecx0 = np.array(vecx0)
    vecx = s(vecx0[0], vecx0[-1], vec_len)
    
    #if only resizing one array
    if np.all(np.array(vecy0)) == None:
        return vecx
    
    #if resizing two arrays
    if np.all(np.array(vecy0)) != None:        
        #calculate parameters of degree-3 spline fit to original data
        spline_params = splrep(vecx0, vecy0)
        
        #calculate spline at new x values
        vecy = splev(vecx, spline_params)
        
        return vecx, vecy


#%% animate IV curves


'''       
iv_df = pd.read_table('exp_data\\pedotpss_IV.txt')
#save IV parameters to a dictionry for tracking over time
iv_params = {'high_i':[], 'low_i':[], 'mid_pos_i':[], 'mid_pos_v':[],
             'mid_neg_i':[], 'mid_neg_v':[]}


rh_vals = np.array(list(iv_df)[1:]).astype(int)

#loop over each IV curve
for i in range(1, len(iv_df.iloc[0])):
    print('%i/49' %i)
    rh0 = int(list(iv_df)[i])
    v0 = np.array(iv_df.iloc[:,0])
    i0 = np.array(iv_df.iloc[:,i])*1e9
    i0 = i0 - i0[np.argmin(np.abs(v0))]

    iv_params['high_i'].append(i0[-1])
    iv_params['low_i'].append(i0[0])
    
    
    plt.plot(v0, i0)
    plt.xlim([-2.05, 2.05])
    plt.ylim([-80, 370])
    
    plt.axvline(x=0, c='k', alpha=0.2, linewidth=1)
    plt.axhline(y=0, c='k', alpha=0.2, linewidth=1)
    
    plt.title('%i%% RH' %rh0, fontsize=18)
    plt.tight_layout()
    plt.gcf().set_size_inches(6, 6)
    label_axes('Voltage (V)', 'Current (nA)')
    
    #plot low current
    plt.scatter(np.repeat(v0[0], len(iv_params['low_i'])),
                iv_params['low_i'],
                s=3, c='c', marker='.')
    plt.scatter(np.repeat(v0[0], len(iv_params['low_i']))[-1],
                iv_params['low_i'][-1],
                s=40, c='c', marker='*')
    
    #plot high current
    plt.scatter(np.repeat(v0[-1], len(iv_params['high_i'])),
                iv_params['high_i'],
                s=3, c='r', marker='.')
    plt.scatter(np.repeat(v0[-1], len(iv_params['high_i']))[-1],
                iv_params['high_i'][-1],
                s=40, c='r', marker='*')
    
    
    
    if rh0 > 10:
        iv_params['mid_pos_i'].append(np.min(i0[180:300]))
        iv_params['mid_pos_v'].append(v0[200+np.argmin(i0[200:300])])
        
        #plot mid current positive
        plt.scatter(iv_params['mid_pos_v'],
                iv_params['mid_pos_i'],
                s=3, c='g', marker='.')
        #most recent point
        plt.scatter(iv_params['mid_pos_v'][-1],
                iv_params['mid_pos_i'][-1],
                s=40, c='g', marker='*')
        
    if rh0 > 26:
        iv_params['mid_neg_i'].append(np.min(i0[100:200]))
        iv_params['mid_neg_v'].append(v0[100+np.argmin(i0[100:200])])

        #plot mid current negative
        plt.scatter(iv_params['mid_neg_v'],
                iv_params['mid_neg_i'],
                s=3, c='y', marker='.')
    
        #most recent point
        plt.scatter(iv_params['mid_neg_v'][-1],
                iv_params['mid_neg_i'][-1],
                s=40, c='y', marker='*')

    
    
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
    save_pic_filename = 'gif_frames\\fig_'+format(str(rh0).zfill(2))+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)
    
    plt.show()


plt.plot(rh_vals, iv_params['high_i'])
label_axes('Voltage (V)', 'Current')
plt.title('high I')
plt.show()

plt.plot(rh_vals, iv_params['low_i'])
plt.title('Low I')
plt.show()
         
'''      
      








#%% animate impedance data

rh_vals =np.linspace(2, 96, 48).astype(int)
    
with open('exp_data\\pp_eis_ml.pkl', 'rb') as handle:
    eis_raw = pickle.load(handle)  
    

eis = pd.read_csv(
        'exp_data\\2018-06-06pedotpssEIS_0VDC_ML.csv').iloc[:31,:]
 

low_re = []
low_im = [] 
inflecx = [] 
inflecy = []
  
for i in range(2, len(list(eis))-1, 2):
    print('%i%% RH' %i)
    
    
    f0 = eis.iloc[:, 1]
    z0 = eis.iloc[:, i]
    phi0 = eis.iloc[:, i+1]
    zre0 = z0*np.cos(phi0)/1e6
    zim0 = z0*np.sin(phi0)/-1e6
    
    #find derivative to track inflection point
    d_re = np.diff(zre0)
    d_im = np.diff(zim0)
    deriv = np.divide(d_im, d_re)
    
    inflectionx = zre0[np.argmin(deriv)-1]
    inflectiony = zim0[np.argmin(deriv)-1]
    inflecx.append(inflectionx)
    inflecy.append(inflectiony)

    plt.scatter(inflectionx, inflectiony, c='c', marker='*', s=60)    
    plt.scatter(inflecx, inflecy, c='g', marker='.', s=15, alpha=.2)
    plt.plot(zre0, zim0, linewidth=2)
    label_axes('Re(Z) (M$\Omega$)', '-Im(Z) (M$\Omega$)')
    
    low_re.append(zre0[0])
    low_im.append(zim0[0])
    
    plt.scatter(low_re, low_im, c='r', marker='.', s=15, alpha=.2)
    plt.scatter(zre0[0], zim0[0], c='r', marker='*', s=60)

    
    plt.xlim([-2, 60])
    plt.ylim([-1, 52])
    
    plt.title('%i%% RH' %i, fontsize=18)
    plt.tight_layout()
    plt.gcf().set_size_inches(6, 6)
    

    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
    save_pic_filename = 'gif_frames\\fig_'+format(str(i).zfill(2))+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)
    
    plt.show()





#%% compile images into video

def create_video(image_folder, video_name, fps=8):
    #create video out of images saved in a folder
    import cv2
    images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(video_name, -1, fps, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()



image_folder = 'gif_frames'
video_name = 'C:\\Users\\a6q\\Desktop\\new_vid.avi'    
fps = 6 

create_video(image_folder, video_name, fps)








   

    
    
    
    
    
    
    
    
    
    
    
    
    