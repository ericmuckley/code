# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 09:52:52 2019

@author: a6q
"""

import numpy as np
import time


def voigt(mu_f, eta_f, rho_f, h_f):
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
                
    Output
        df = frequency change of resonator
        dd =  dissipation change of resonator
    '''
    
    # fundamental resonant frequency of crystal in Hz
    f0 = 5e6
    n = 1 # crystal harmonic number
    w = 2*np.pi*f0*n  # angular frequency
    
    mu_q = 2.947e10 #shear modulus of AT-cut quatz in Pa
    rho_q = 2648 #density of quartz (kg/m^3)
    h_q = np.sqrt(mu_q/rho_q)/(2*f0) #thickness of quartz
    
    # shear modulus and density of bulk air
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
    df = np.imag((beta-beta0)/(2*np.pi*rho_q*h_q))
    dd = -np.real((beta-beta0)/(np.pi*f0*n*rho_q*h_q))*1e6
    
    return [df, dd]


def find_2d_dist(measured, modeled):
    # finds distance between two ordered pairs in 2D.
    # both measured and modeled variables should be length-2
    # arrays of the ordered pairs.
    # uses "distance" to mean perfent difference so variales with differing
    # scales can be used.
    dist_x = np.square((measured[0] - modeled[0])/measured[0])
    dist_y = np.square((measured[1] - modeled[1])/measured[1])
    tot_dist = np.sqrt(dist_x + dist_y)
    return tot_dist



#%% create arrays of possible values

start_time = time.time()
# create inputs for Voigt function

step_num = 10
mu_f = np.linspace(1e3, 1e9, step_num)
eta_f = np.linspace(1e-4, 1e-8, step_num)
rho_f = np.linspace(1e2, 5e3, step_num)
h_f = np.linspace(100e-9, 10e-6, step_num)
inputs = np.array(np.meshgrid(mu_f, eta_f, rho_f, h_f)).T.reshape(-1, 4)

# calculate df and dd solutions for each set of inputs
sol = np.array([voigt(*input0) for i, input0 in enumerate(inputs)])

# find extreme df and dd values from given inputs
df_lims = [np.amin(sol[:, 0]), np.amax(sol[:, 0])]
dd_lims = [np.amin(sol[:, 1]), np.amax(sol[:, 1])]
print('unconstrained min, max df: %f, %f Hz' %(df_lims[0], df_lims[1]))
print('unconstrained min, max dd: %f, %f' %(dd_lims[0], dd_lims[1]))
print('%i total solutions' %(len(sol)))

# create constraints which limit solutions to commonly-observable physical ones
df_constraints = [-100000, 10000]  # min df, max df
dd_constraints = [-1e-3, 100]  # min dd, max dd

# check which entries satisfy the constraints
inside_constraints = np.array(
        (sol[:, 0] > df_constraints[0]) &
        (sol[:, 0] < df_constraints[1]) &
        (sol[:, 1] > dd_constraints[0]) &
        (sol[:, 1] < dd_constraints[1]))
print('%i solutions inside constraints' %np.sum(inside_constraints))

# get possible solutions after constraints are imposed
constrained_sol = sol[inside_constraints]

print('-------------------------------------------------')
print('constrained min, max df: %f, %f' %(
        np.amin(constrained_sol[:, 0]),
        np.amax(constrained_sol[:, 0])))
print('constrained min, max dd: %f, %f' %(
        np.amin(constrained_sol[:, 1]),
        np.amax(constrained_sol[:, 1])))

# get inputs which satisfy the constraints
good_inputs = inputs[inside_constraints]

print('-------------------------------------------------')
# show extremes of inputs which can satisfy the constraints
print('constrained mu min, max = %f, %f'%(
        np.amin(good_inputs[:,0]), np.amax(good_inputs[:,0])))
print('constrained eta min, max = %f, %f'%(
        np.amin(good_inputs[:,1]), np.amax(good_inputs[:,1])))
print('constrained rho min, max = %f, %f'%(
        np.amin(good_inputs[:,2]), np.amax(good_inputs[:,2])))
print('constrained h min, max = %f, %f'%(
        np.amin(good_inputs[:,3]), np.amax(good_inputs[:,3])))

print('-----------------------------------')
tot_time = (time.time() - start_time)/60
print('total time (min): '+str(np.round(tot_time, decimals=3)))



# %% search for solutions

measured = [-1e3, 1.4e-6]  # measured df and dd point

err = 100000

# loop over each good solution
for i in range(len(good_inputs)):
    
    err0 = find_2d_dist(measured, good_inputs[i])
    
    if err0 < err:
        err = err0
        i_min = i
    else: pass

print(constrained_sol[i_min])
    







