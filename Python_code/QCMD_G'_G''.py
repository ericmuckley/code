# -*- coding: utf-8 -*-
"""
This script takes delta F and delta D QCM data, fits to the Kevin-Voigt
model to obtain viscosity and shear modulus of adlayer, and
calculates  G' and G'' (elastic and loss moduli).

For more information, see the publication
Liu, S.X. and Kim, J.T., 2009. Application of Kevin—Voigt model in
quantifying whey protein adsorption on polyethersulfone using QCM-D.
JALA: Journal of the Association for Laboratory Automation, 14(4),
pp.213-220.
https://journals.sagepub.com/doi/full/10.1016/j.jala.2009.01.003

Created on Tue Dec  3 12:18:48 2019
@author: ericmuckley@gmail.com
"""

import numpy as np
import matplotlib.pyplot as plt


# change matplotlib settings to make plots look nicer
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['xtick.minor.width'] = 3
plt.rcParams['xtick.major.width'] = 3
plt.rcParams['ytick.minor.width'] = 3
plt.rcParams['ytick.major.width'] = 3


def plot_setup(labels=['X', 'Y'], fsize=20, setlimits=False, limits=[0,1,0,1],
               title=None, legend=False, colorbar=False,
               save=False, filename='plot.jpg'):
    """Creates a custom plot configuration to make graphs look nice.
    This can be called with matplotlib for setting axes labels,
    titles, axes ranges, and the font size of plot labels.
    This should be called between plt.plot() and plt.show() commands."""
    plt.xlabel(str(labels[0]), fontsize=fsize)
    plt.ylabel(str(labels[1]), fontsize=fsize)
    fig = plt.gcf()
    fig.set_size_inches(6, 4)
    if title:
        plt.title(title, fontsize=fsize)
    if legend:
        plt.legend(fontsize=fsize-4)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if colorbar:
        plt.colorbar()
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()


def kevin_voigt(mu_f, eta_f, rho_f, h_f=1e-6, n=1, f0=5e6,
                medium='air'):
    """ 
    The Kevin-Voigt model comes from eqns (15) in the paper by 
    Voinova: Vionova, M.V., Rodahl, M., Jonson, M. and Kasemo, B., 1999.
    Viscoelastic acoustic response of layered polymer films at fluid-solid
    interfaces: continuum mechanics approach. Physica Scripta, 59(5), p.391.
    Reference: https://github.com/88tpm/QCMD/blob/master
    /Mass-specific%20activity/Internal%20functions/voigt_rel.m.
    
    This function solves for Delta f and Delta d of thin adlayer on QCM.
    It differs from voigt because it calculates relative to an
    unloaded resonator.
    Inputs
        mu_f = shear modulus of film in Pa
        eta_f = shear viscosity of film in Pa s
        rho_f = density of film in kg m-3
        h_f = thickness of film in m
        n = crystal harmonic number
        f0 = fundamental resonant frequency of crystal in Hz      
    Output
        deltaf = frequency change of resonator
        deltad =  dissipation change of resonator
    """
    
    # define properties of QCM crystal
    w = 2*np.pi*f0*n  # angular frequency
    mu_q = 2.947e10  # shear modulus of AT-cut quatz in Pa
    rho_q = 2648  # density of quartz (kg/m^3)
    h_q = np.sqrt(mu_q/rho_q)/(2*f0)  # thickness of quartz
    
    # define properties of medium
    if medium == 'air':
        rho_b = 1.1839  # density of bulk air (25 C) in kg/m^3
        eta_b = 18.6e-6  # viscosity of bulk air (25 C) in Pa s
    if medium == 'liquid':
        rho_b = 1000  # density of bulk water in kg/m^3
        eta_b = 8.9e-4  # viscosity of bulk water in Pa s
    
    # define equations from the Kevin-Voigt model in publication
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


# %% USER INPUTS

# film density in kg/m^3 (water is 1000 kg/m^3)
film_density = 100

# film thickness in meters
film_thickness = 100e-9

# low and high mu exponents to search. x --> mu = 10^x
mu_exp_low, mu_exp_high = 0, 6

# low and high eta exponents to search. x --> eta = 10^x
eta_exp_low, eta_exp_high = -2, -9

# number of steps to search in grid
step_num = 100

# %%

# get 2D mesh grid points of log mu and eta values
mu_mesh, eta_mesh = np.meshgrid(
        np.linspace(mu_exp_low, mu_exp_high, step_num).astype(float),
        np.linspace(eta_exp_low, eta_exp_high, step_num).astype(float))

# calculate DF and DD values across mu eta mesh
df_surf, dd_surf = kevin_voigt(10**mu_mesh,
                               10**eta_mesh,
                               rho_f=film_density,
                               h_f=film_thickness)


df_exp = -100
dd_exp = 5

# find contours corresponding to measured DF and DD values


# plot delta F heatmap
cs = plt.contourf(mu_mesh, eta_mesh, df_surf, 50, [100], cmap='jet')


plt.contourf(mu_mesh, eta_mesh, df_surf, 50, cmap='jet')
plot_setup(title='Delta F', labels=['Log (mu)', 'Log (eta)'], colorbar=True)
plt.show()

# plot delta D heatmap
plt.contourf(mu_mesh, eta_mesh, dd_surf, 50, cmap='jet')
plot_setup(title='Delta D', labels=['Log (mu)', 'Log (eta)'], colorbar=True)
plt.show()





















