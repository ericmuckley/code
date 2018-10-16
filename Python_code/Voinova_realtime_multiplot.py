# -*- coding: utf-8 -*-
'''
Created on Wed Mar  7 17:28:41 2018
@author: a6q
'''
from scipy.interpolate import griddata
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
#import cv2

import matplotlib.gridspec as gridspec
ls = 16
plt.rcParams['xtick.labelsize'] = ls
plt.rcParams['ytick.labelsize'] = ls
plt.rcParams['figure.figsize'] = (6,6)



#%% caluclate delta f and delta D using Voinona Voigt model

def voigt(mu_f, eta_f):
    ''' The Voinova equations come from eqns (15) in the paper by 
        Voinova: Vionova, M.V., Rodahl, M., Jonson, M. and Kasemo, B., 1999.
        Viscoelastic acoustic response of layered polymer films at fluid-solid
        interfaces: continuum mechanics approach. Physica Scripta,
        59(5), p.391.
    
        Reference: https://github.com/88tpm/QCMD/blob/master
        /Mass-specific%20activity/Internal%20functions/voigt_rel.m
            
       Solves for Delta f and Delta d of thin adlayer on quartz resonator.
       Differs from voight because calculates relative to unloaded resonator.
       
      Input
           density1 = density of adlayer in kg m-3
           shear1 = elastic shear modulus of adlayer in Pa
           viscosity1 = dynamic viscosity of adlayer in Pa s
           thickness1 = thickness of adlayer in m
           densitybulk = density of bulk Newtonian fluid in kg m-3
           viscositybulk = dynamic viscosity of bulk Newtonian fluid in kg m-3
           f0 = resonator fundamental frequency in s-1
       Output
           frequency = frequency change of resonator
           dissipation =  dissipation change of resonator
    '''
    # eqn 14
    kappa_f = eta_f-(1j*mu_f/w)
    # eqn 13
    x_f = np.sqrt(-rho_f*np.square(w)/(mu_f + 1j*w*eta_f))
    x_b = np.sqrt(1j*rho_b*w/eta_b)
    # eqn 11 after simplification with h1 = h2 and h3 = infinity
    A = (kappa_f*x_f+eta_b*x_b)/(kappa_f*x_f-eta_b*x_b)
    # eqn 16
    beta = kappa_f*x_f*(1-A*np.exp(2*x_f*h_f))/(1+A*np.exp(2*x_f*h_f))
    beta0 = kappa_f*x_f*(1-A)/(1+A)
    # eqn 15
    df = np.imag((beta-beta0)/(2*np.pi*rho_q*h_q))
    dd = -np.real((beta-beta0)/(np.pi*f0*n*rho_q*h_q))*1e6
    
    return df, dd


#%% difference between two ordered pairs
    
def find_diff2D(p1, p2):
    #finds the distance between two ordered pairs of points in 2D
    #each point p1 and p2 must be an ordered pair with two elements
    xdiff = np.square(p2[0] - p1[0])
    ydiff = np.square(p2[1] - p1[1])
    diff = np.sqrt(xdiff + ydiff)
    return diff






#%% Constants

f0 = 5e6 #fundamental resonant frequency of crystal
n = 9 #harmonic number
w = 2*np.pi*f0*n #angular frequency 

mu_q = 2.947e10 #shear modulus of AT-cut quatz in Pa
rho_q = 2648 #density of quartz (kg/m^3)
h_q = np.sqrt(mu_q/rho_q)/(2*f0) #thickness of quartz

h_f = 5e-7 #film thickness in m
rho_f = 1000 #film density in kg/m^3

rho_b = 1.1839 #density of bulk air (25 C) in kg/m^3
eta_b = 18.6e-6 #viscosity of bulk air (25 C) in Pa s
#rho_b = 1000 #density of bulk water in kg/m^3
#eta_b = 8.9e-4 #viscosity of bulk water in Pa s



#%% set size of grid 
step_num = 300
print('grid contains %d points' %step_num**2)
#mu_range = np.logspace(1, 15, step_num).astype(float)
mu_range = np.linspace(1e6, 2.1e6, step_num).astype(float)
#eta_range = np.logspace(-12, 12, step_num).astype(float)
eta_range = np.linspace(0, 1.5e-4, step_num).astype(float)

exp_data = pd.read_table('exp_data\\pedotpss_7thovertone_dd_df.txt')




#%% calculate delta f and delta D at each grid point
starttime = time.time()
results0 = []

rh_sol = []
mu_sol= []
eta_sol = []


#iterate over each time step
for timestep in range(len(exp_data)):
    print('time step '+format(1+timestep)+' / '+format(len(exp_data)))
    df_exp = exp_data['df_exp'][timestep]
    dd_exp = exp_data['dd_exp'][timestep]
    rh0 = exp_data['rh'][timestep]


    #lists for df and dd "matches"
    dfm = []
    ddm = []
 
    
    
    
    #----------iterate over each point in grid--------------------------
    for i in range(len(mu_range)-1):
        for j in range(len(eta_range)):
            #calculate delta f and delta d at each grid point  
            df0, dd0 = voigt(mu_range[i], eta_range[j])
            df1, dd1 = voigt(mu_range[i+1], eta_range[j])
            results0.append([mu_range[i], eta_range[j], df0, dd0])
            
            #check if surface intersects planes of experimental df and dd
            if df1 <= df_exp <= df0 or df0 <= df_exp <= df1:
                dfm.append([mu_range[i], eta_range[j]])
            if dd1 <= dd_exp <= dd0 or dd0 <= dd_exp <= dd1:  
                 ddm.append([mu_range[i], eta_range[j]])
                 
    dfm = np.array(dfm)    
    ddm = np.array(ddm) 





    # set up multi-panel plot--------------------------------------------
    gs1 = gridspec.GridSpec(2,1)
    gs2 = gridspec.GridSpec(2,1)
    gs1.update(left=0.08, right=0.55)#, wspace=0.05)
    gs2.update(left=0.68, right=0.98)#, wspace=0.05)
    ax_int = plt.subplot(gs1[:])
    ax_mu = plt.subplot(gs2[0])
    ax_eta = plt.subplot(gs2[1])
    plt.setp(ax_mu.get_xticklabels(), visible=False)
    #plt.setp([a.get_xticklabels() for a in share_axes2[:-1]], visible=False)
    plt.subplots_adjust(hspace=0, bottom=.15, top=0.95, right=0.3, left=.1)
    #---------------------------------------------------------------------






    #--------------------------plot intersection surfaces ------------------
    if len(dfm) > 0:
        ax_int.scatter(1e-6*dfm[:,0], 1e6*dfm[:,1],
                   s=4, c='k', alpha=.4, label='$\Delta$f solution')
    if len(ddm) > 0:
        ax_int.scatter(1e-6*ddm[:,0], 1e6*ddm[:,1],
                   s=4, c='r', alpha=.7, label='$\Delta$D solution')
        
        
        
        
        
        
    #-------find intersection of df and dd contours at each timestep --------    
    if len(dfm) > 0 and len(ddm) > 0:
        all_diffs = []    
        for dfm_val in dfm:
            for ddm_val in ddm:
                #find point closest to both contours 
                all_diffs.append([dfm_val[0], ddm_val[1], 
                                  find_diff2D(dfm_val, ddm_val)])
        all_diffs = np.array(all_diffs)
        min_diff_ind = np.argmin(all_diffs[:,2])
        mu_sol0 = all_diffs[min_diff_ind][0]
        eta_sol0 = all_diffs[min_diff_ind][1]
        rh_sol.append(rh0)
        mu_sol.append(mu_sol0)
        eta_sol.append(eta_sol0)

        #plot current intersection point
        ax_int.scatter(1e-6*mu_sol0, 1e6*eta_sol0, s=150,
                       label='intersection', marker='*', c='g')



    #plot trajectory of intersection point
    #if len(intersect_mu) > 1:
    ax_int.scatter(1e-6*np.array(mu_sol), 1e6*np.array(eta_sol),
                label='trajectory', s=5, c='b')



    
    
    #plot eta and mu over time
    ax_mu.plot(rh_sol, 1e-6*(np.array(mu_sol) - mu_sol[0]))
    ax_mu.scatter(rh_sol, 1e-6*(np.array(mu_sol)- mu_sol[0]))
    #ax_mu.set_xlabel('RH (%)', fontsize=ls)
    ax_mu.set_ylabel('$\Delta\mu$ (MPa)', fontsize=ls)
    
    
    ax_eta.plot(rh_sol, 1e6*(np.array(eta_sol)-eta_sol[0]))
    ax_eta.scatter(rh_sol, 1e6*(np.array(eta_sol)-eta_sol[0]))
    ax_eta.set_xlabel('RH (%)', fontsize=ls)
    ax_eta.set_ylabel('$\Delta\eta$ ($\mu$Pa s)', fontsize=ls)
    ax_eta.set_xlim([0, 100])
    ax_mu.set_xlim([0, 100])
    

    
    
    
    
      
    #format plot
    ax_int.set_xlim([1,2.2])
    ax_int.set_ylim([0,140])
    #ax_int.set_yscale('log')
    #ax_int.set_xscale('log')  
    ax_int.set_xlabel('$\mu$ (MPa)', fontsize=ls)
    ax_int.set_ylabel('$\eta$ ($\mu$Pa s)', fontsize=ls)
    lgnd = ax_int.legend(loc='upper right', fontsize=ls)

    lgnd.legendHandles[0]._sizes = [20]
    lgnd.legendHandles[1]._sizes = [20]
    lgnd.legendHandles[2]._sizes = [60]
    lgnd.legendHandles[3]._sizes = [20]
    
    #ax_int.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    #ax_int.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    #ax_mu.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    ax_int.text(1.14, 132, format(rh0)+'% RH',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=20)



    #save plot as image file          
    #plt.tight_layout()
    fig0 = plt.gcf() # get current figure
    fig0.set_size_inches(12, 6)
    #save plot as image file   

    if rh0 < 10:
        save_pic_filename = 'exp_data\\save_figs2\\fig_0'+format(rh0)+'.jpg'
    else:
        save_pic_filename = 'exp_data\\save_figs2\\fig_'+format(rh0)+'.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)

    plt.show()

    #close figure from memory
    plt.close(fig0)
    #close all figures from memory
    plt.close('all')



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



image_folder = 'exp_data\\save_figs2'
video_name = 'C:\\Users\\a6q\\Desktop\\new_vid.avi'    
fps = 5 

create_video(image_folder, video_name, fps)


      

#%% organize results
'''
#construct dataframe with 3D surface 
voigt_surf_array = np.array(results0)
voigt_surf = pd.DataFrame(voigt_surf_array,
                          columns=['mu', 'eta', 'df','dd'], dtype=np.float64)

#voigt_surf['mu'] = np.array(voigt_surf['mu'])*1e-6
#voigt_surf['eta'] = np.array(voigt_surf['mu'])*1e6

# organize dd and df intersection points to find solution
sol = np.array([rh_sol, mu_sol, eta_sol]).T


'''

#%%


'''
# delta f surface
# set X, Y, and Z for plot
Xf, Yf, Zf, = voigt_surf['mu'], voigt_surf['eta'], voigt_surf['df']
# create x-y points to be used in heatmap
xf = np.linspace(Xf.min(),Xf.max(),75)
yf = np.linspace(Yf.min(),Yf.max(),75)
# Z is a matrix of x-y values
zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
# Create the contour plot
#plt.contourf(xf, yf, zf, 150, cmap=plt.cm.rainbow,
#                     vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))


plt.colorbar()
plt.set_xlabel('mu',fontsize=ls)
plt.set_ylabel('eta',fontsize=ls);
plt.title('delta f (Hz/cm^2)',fontsize=ls)

plt.show()





# set up multi-panel plot--------------------------------------------
gs1 = gridspec.GridSpec(2,1)
gs2 = gridspec.GridSpec(2,1)
gs1.update(left=0.08, right=0.55)#, wspace=0.05)
gs2.update(left=0.68, right=0.98)#, wspace=0.05)
ax_df = plt.subplot(gs1[:])
ax_dd = plt.subplot(gs2[:])
plt.subplots_adjust(hspace=0, bottom=.15, top=0.95, right=0.3, left=.1)
#---------------------------------------------------------------------



# delta f surface
# set X, Y, and Z for plot
Xf, Yf, Zf, = voigt_surf['mu'], voigt_surf['eta'], voigt_surf['df']
# create x-y points to be used in heatmap
xf = np.linspace(Xf.min(),Xf.max(),75)
yf = np.linspace(Yf.min(),Yf.max(),75)
# Z is a matrix of x-y values
zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
# Create the contour plot
ax_df.contourf(xf, yf, zf, 150, cmap=plt.cm.rainbow,
                     vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
ax_df.colorbar()
ax_df.set_xlabel('mu',fontsize=ls)
ax_df.set_ylabel('eta',fontsize=ls);
ax_df.title('delta f (Hz/cm^2)',fontsize=ls)

plt.show()
# delta d surface
# set X, Y, and Z for plot
Xd, Yd, Zd, = voigt_surf['mu'], voigt_surf['eta'], voigt_surf['dd']
# create x-y points to be used in heatmap
xd = np.linspace(Xd.min(),Xd.max(),1000)
yd = np.linspace(Yd.min(),Yd.max(),1000)
# Z is a matrix of x-y values
zd = griddata((Xd, Yd), Zd, (xd[None,:], yd[:,None]), method='cubic')
ax_dd.contourf(xd, yd, zd, 150, cmap=plt.cm.rainbow,
             vmax=np.nanmax(Zd), vmin=np.nanmin(Zd))
ax_dd.set_xlabel('mu',fontsize=ls)
ax_dd.set_ylabel('eta',fontsize=ls)
ax_dd.colorbar()
ax_dd.title('delta D (x10^-6)',fontsize=ls)
plt.show()
'''



