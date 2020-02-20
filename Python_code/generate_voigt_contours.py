# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 08:46:25 2019

@author: ericmuckley@gmail.com

This script creates Voigt surface for QCM-D analysis.

"""
import matplotlib._cntr as cntr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString

def voigt(mu_f, eta_f, rho_f, h_f=1e-6, n=1, f0=5e6):
    ''' 
    The Voinova equations come from eqns (15) in the paper by 
    Voinova: Vionova, M.V., Rodahl, M., Jonson, M. and Kasemo, B., 1999.
    Viscoelastic acoustic response of layered polymer films at fluid-solid
    interfaces: continuum mechanics approach. Physica Scripta, 59(5), p.391.
    Reference: https://github.com/88tpm/QCMD/blob/master
    /Mass-specific%20activity/Internal%20functions/voigt_rel.m
    Solves for Delta f and Delta d of thin adlayer on quartz resonator.
    Differs from voigt because calculates relative to unloaded resonator.
    Input
        mu_f = shear modulus of film in Pa
        eta_f = shear viscosity of film in Pa s
        rho_f = density of film in kg m-3
        h_f = thickness of film in m
        n = crystal harmonic number
        f0 = fundamental resonant frequency of crystal in Hz      
    Output
        deltaf = frequency change of resonator
        deltad =  dissipation change of resonator
    '''
    w = 2*np.pi*f0*n  # angular frequency
    mu_q = 2.947e10 # shear modulus of AT-cut quatz in Pa
    rho_q = 2648 # density of quartz (kg/m^3)
    h_q = np.sqrt(mu_q/rho_q)/(2*f0) #t hickness of quartz
    
    rho_b = 1.1839 #density of bulk air (25 C) in kg/m^3
    eta_b = 18.6e-6 #viscosity of bulk air (25 C) in Pa s
    # rho_b = 1000 #density of bulk water in kg/m^3
    # eta_b = 8.9e-4 #viscosity of bulk water in Pa s
    
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
    deltaf = np.imag((beta-beta0)/(2*np.pi*rho_q*h_q))
    deltad = -np.real((beta-beta0)/(np.pi*f0*n*rho_q*h_q))*1e6
    return deltaf, deltad


def plot_setup(labels=['X', 'Y'], size=16, setlimits=False,
               limits=[0,1,0,1], scales=['linear', 'linear'],
               title='', save=False, filename='plot.jpg',
               colorbar=False):
    #This can be called with Matplotlib for setting axes labels, setting
    #axes ranges and scale types, and  font size of plot labels. Function
    #should be called between plt.plot() and plt.show() commands.
    plt.rcParams['xtick.labelsize'] = size 
    plt.rcParams['ytick.labelsize'] = size
    plt.xlabel(str(labels[0]), fontsize=size)
    plt.ylabel(str(labels[1]), fontsize=size)
    plt.title(title, fontsize=size)
    fig = plt.gcf()
    fig.set_size_inches(7, 5)
    plt.xscale(scales[0])
    plt.yscale(scales[1])   
    if colorbar:
        plt.colorbar()
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()


def get_mesh(arr1, arr2):
    # create a 2D mesh using two input arrays
    return np.meshgrid(arr1, arr2)


def create_2dcombos(arr1, arr2):
    '''Create an array of all possible combinations of elements of
    each array.'''
    return np.array(np.meshgrid(arr1, arr2)).T.reshape(-1, 2)


def get_intersections(curve1, curve2):
    '''Get intersection points of two curves by checking intersections
    of line segments hich connect each pair of points.'''
    coords = []
    for p1 in range(0, len(curve1)-1):
        line1 = LineString([(curve1[p1][0], curve1[p1][1]),
                           (curve1[p1 + 1][0], curve1[p1 + 1][1])])
        for p2 in range(0, len(curve2)-1):
            line2 = LineString([(curve2[p2][0], curve2[p2][1]),
                               (curve2[p2 + 1][0], curve2[p2 + 1][1])])
            if line1.intersection(line2):
                int_point = line1.intersection(line2)
                coords.append([int_point.x, int_point.y])
    return coords


    
def find_diff2D(p1, p2):
    #finds the distance between two ordered pairs of points in 2D
    #each point p1 and p2 must be an ordered pair with two elements
    xdiff = np.square(p2[0] - p1[0])
    ydiff = np.square(p2[1] - p1[1])
    diff = np.sqrt(xdiff + ydiff)
    return diff


def find_intersection_fast(curve1, curve2, threshold=0.05):
    '''Estimate the point of intersection between two curves quickly.
    Each curve should be a 2D array of ordered pairs.'''
    dist = np.inf
    cord = None
    # loop over each point in both curves
    for p1 in curve1:
        for p2 in curve2:
            # calculate distance between the two points
            xdist = np.square(p2[0] - p1[0])
            ydist = np.square(p2[1] - p1[1])
            dist0 = np.sqrt(xdist + ydist)
            if dist0 < dist and dist0 < threshold:
                dist = dist0
                cord = [np.mean([p1[0], p2[0]]), np.mean([p1[1], p2[1]])]
    return cord


def get_contours(x, y, z, level=0):
    '''Get the contour lines corresponding to z = level. Inputs should each
    be 2D meshes, made by x, y = np.meshgrid(x_list, y_list), z = f( x, y).'''
    c = cntr.Cntr(x, y, z)
    nlist = c.trace(level, level, 0)
    segs = nlist[:len(nlist)//2]
    return segs


#%%

exp_data = pd.read_csv(r'C:\Users\a6q\exp_data\lab_on_qcm_pp_QCMD.csv')
step_num = 10
# film density
rholist = np.linspace(500, 2000, step_num).astype(float)
# film thickness
hlist = np.linspace(10e-9, 800e-9, step_num).astype(float)
# shear modulus exponents
mulist = np.linspace(0, 8, step_num).astype(float)
# viscosity exponents
etalist = np.linspace(-1, -8, step_num).astype(float)

rholist = [1000]#np.linspace(500, 2000, 3).astype(float)
hlist = [500e-9]#np.linspace(10e-9, 1000e-9, 3).astype(float)
# set rho and h values to loop over
rho_h_combos = create_2dcombos(rholist, hlist)

# get 2D mesh of mu and eta values
mu_mesh, eta_mesh = get_mesh(mulist, etalist)
# get combinations of rho and h to test
rho_h_combos = create_2dcombos(rholist, hlist)

counter = 1

results_arr = np.empty((0, 6))


#%%

# loop over experimentally measured values
for exp_row in exp_data.values:
    rh, df_exp, dd_exp = exp_row[0], exp_row[1], exp_row[2]
    
    
    # calculate delta f and delta d across surface using Voigt function
    rho0, h0 = rholist[0], hlist[0]*(1+rh/200)
    
    df_surf, dd_surf = voigt(10**mu_mesh, 10**eta_mesh, rho_f=rho0, h_f=h0)
    rho_h_title = 'RH='+str(int(rh))+'%, h='+str(int(h0*1e9))+'nm'
   
    
    # get all solution contours
    df_conts = get_contours(mu_mesh, eta_mesh, df_surf, level=df_exp)
    dd_conts = get_contours(mu_mesh, eta_mesh, dd_surf, level=dd_exp)
    # sort contours to find low-order contours
    df_cont0 = sorted(df_conts, key=lambda x: np.max(x[:, 0]))[-1]
    if len(dd_conts) < 2:
        dd_cont0 = dd_conts[0]
    else:
        dd_cont0 = sorted(dd_conts, key=lambda x: np.max(x[:, 0]))[-2]
    # get solution coordinates
    cords = find_intersection_fast(df_cont0, dd_cont0)
    
    # plot all solution contours
    #[plt.plot(c[:, 0], c[:, 1], '--', c='r', lw=1) for c in df_conts]
    #[plt.plot(c[:, 0], c[:, 1], '--', c='b', lw=1) for c in dd_conts]
    # create delta D contour plot
    
    plt.contourf(mulist, etalist, np.log10(dd_surf), 200,
                 cmap=plt.cm.rainbow,
                 vmax=np.log10(dd_surf.max()),
                 vmin=np.log10(dd_surf.min()))
    plt.colorbar()
    # plot first-order contours
    plt.plot(df_cont0[:,0], df_cont0[:, 1], '--', lw=1, c='r')
    plt.plot(dd_cont0[:,0], dd_cont0[:, 1], '--', lw=1, c='b')
    # plot and save solution if one is found
    if cords:
        plt.scatter(cords[0], cords[1], marker='x', c='k', s=200)
        results_arr = np.vstack((results_arr, [cords[0], cords[1], rho0, h0, df_exp, dd_exp]))
        
    # plot previus solutions
    if len(results_arr) > 1:
        plt.scatter(results_arr[:, 0], results_arr[:, 1], marker='o', c='k', s=15, alpha=0.5)
        
        
    plot_setup(
            labels=['Log($\mu$) (Pa)', 'Log($\eta$) (Pa s)'],
            limits=[3, 6, -7, -2], setlimits=True,
            title='Log($\Delta$D) (x10$^6$) '+rho_h_title, save=True, 
            filename='C:\\Users\\a6q\\exp_data\\voigt_solutions\\'+str(counter).zfill(3)+'.jpg')
    
    plt.show()
    
    counter += 1
    
    
#%%
results = pd.DataFrame(columns=['mu', 'eta', 'rho', 'h', 'df', 'dd'],
                       data=results_arr)

















