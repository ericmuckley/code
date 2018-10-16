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
labelsize=15 #make size of axis tick labels larger
plt.rcParams['xtick.labelsize'] = labelsize
plt.rcParams['ytick.labelsize'] = labelsize





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
step_num = 250
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
for timestep in range(0, len(exp_data), 5):
    print('time step '+format(1+timestep)+' / '+format(len(exp_data)))
    df_exp = exp_data['df_exp'][timestep]
    dd_exp = exp_data['dd_exp'][timestep]

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







    #--------------------------plot intersection surfaces ------------------
    fig = plt.figure()
    ax = plt.gca()
    if len(dfm) > 0:
        ax.scatter(dfm[:,0], dfm[:,1],
                   s=1, c='k', label='$\Delta$f')
    if len(ddm) > 0:
        ax.scatter(ddm[:,0], ddm[:,1],
                   s=1, c='r', label='$\Delta$D')
        
        
        
        
        
        
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
        rh_sol.append(exp_data['rh'][timestep])
        mu_sol.append(mu_sol0)
        eta_sol.append(eta_sol0)

        #plot current intersection point
        plt.scatter(mu_sol0, eta_sol0, s=150, marker='*', c='b')


    #plot trajectory of intersection point
    #if len(intersect_mu) > 1:
    plt.scatter(mu_sol, eta_sol,
                label='trajectory', s=5, c='b')



    
      
    #format plot
    ax.set_xlim([1e6,2.2e6])
    ax.set_ylim([0,1.5e-4])
    #ax.set_yscale('log')
    #ax.set_xscale('log')  
    ax.set_xlabel('$\mu$ (Pa)', fontsize=labelsize)
    ax.set_ylabel('$\eta$ (Pa s)', fontsize=labelsize)
    plt.legend(loc='upper right', fontsize=12)
    plt.title(format(exp_data['rh'][timestep])+'% RH', fontsize=labelsize)
    
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    #save plot to file
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    save_pic_filename = 'exp_data\\save_figs2\\'+format(
            exp_data['rh'][timestep])+'_RH.jpg'
    plt.savefig(save_pic_filename, format='jpg', dpi=150)
    
    plt.show()







            

#%% organize results

#construct dataframe with 3D surface 
voigt_surf_array = np.array(results0)
voigt_surf = pd.DataFrame(voigt_surf_array,
                          columns=['mu', 'eta', 'df','dd'], dtype=np.float64)

# organize dd and df intersection points to find solution
sol = np.array([rh_sol, mu_sol, eta_sol]).T

plt.plot(rh_sol, mu_sol)
plt.scatter(rh_sol, mu_sol)
plt.xlabel('RH (%)', fontsize=labelsize)
plt.ylabel('$\mu$ (Ps)', fontsize=labelsize)
plt.show()

plt.plot(rh_sol, 1e6*np.array(eta_sol))
plt.scatter(rh_sol, 1e6*np.array(eta_sol))
plt.xlabel('RH (%)', fontsize=labelsize)
plt.ylabel('$\eta$ ($\mu$Pa s)', fontsize=labelsize)
plt.show()



#%% save results
'''
voigt_surf.to_csv('voigt_surf.txt', sep='\t')
'''




#%% plot voigt surfaces 
'''
# delta f surface
# set X, Y, and Z for plot
Xf, Yf, Zf, = voigt_surf['mu'], voigt_surf['eta'], voigt_surf['df']
# create x-y points to be used in heatmap
xf = np.linspace(Xf.min(),Xf.max(),1000)
yf = np.linspace(Yf.min(),Yf.max(),1000)
# Z is a matrix of x-y values
zf = griddata((Xf, Yf), Zf, (xf[None,:], yf[:,None]), method='cubic')
# Create the contour plot
CSf = plt.contourf(xf, yf, zf, 150, cmap=plt.cm.rainbow, vmax=np.nanmax(Zf), vmin=np.nanmin(Zf))
plt.colorbar()
plt.xlabel('mu',fontsize=labelsize)
plt.ylabel('eta',fontsize=labelsize);
plt.title('delta f (Hz/cm^2)',fontsize=labelsize)
plt.show()

# delta d surface
# set X, Y, and Z for plot
Xd, Yd, Zd, = voigt_surf['mu'], voigt_surf['eta'], voigt_surf['dd']
# create x-y points to be used in heatmap
xd = np.linspace(Xd.min(),Xd.max(),1000)
yd = np.linspace(Yd.min(),Yd.max(),1000)
# Z is a matrix of x-y values
zd = griddata((Xd, Yd), Zd, (xd[None,:], yd[:,None]), method='cubic')
plt.contourf(xd, yd, zd, 150, cmap=plt.cm.rainbow, vmax=np.nanmax(Zd), vmin=np.nanmin(Zd))
plt.xlabel('mu',fontsize=labelsize)
plt.ylabel('eta',fontsize=labelsize)
plt.colorbar()
plt.title('delta D (x10^-6)',fontsize=labelsize)
plt.show()
'''










#%% show time required for analysis
'''
print('df min, max = '+format(voigt_surf['df'].min())+', '+
      format(voigt_surf['df'].max()))
print('dd min, max = '+format(voigt_surf['dd'].min())+', '+
      format(voigt_surf['dd'].max()))

endtime = time.time()
tottime = (endtime-starttime)/60
print('elapsed time = %.2f minutes' %tottime)
'''