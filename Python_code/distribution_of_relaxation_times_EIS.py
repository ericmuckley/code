# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:54:16 2020
@author: ericmuckley@gmail.com

Gaussian Process Distribution of Relaxation Times, adapted from:
https://github.com/ciuccislab/GP-DRT/blob/master/tutorial/ex4_experimental_data.ipynb

The input file should have three columns named like this (all lowercase):
    1. f
    2. z
    3. phi
Frequency in Hz, Z in Ohms, and Phase in degrees.

"""
import time
import GP_DRT
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

start_time = time.time()

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
    fig.set_size_inches(6, 4)
    if legend:
        plt.legend(fontsize=fsize-4)
    if setlimits:
        plt.xlim((limits[0], limits[1]))
        plt.ylim((limits[2], limits[3]))
    if save:
        fig.savefig(filename, dpi=120, bbox_inches='tight')
        plt.tight_layout()


def import_eis_data(input_filename):
    """Import and format impedance data with three columns: [f, z, phi]."""
    # import the data
    df = pd.read_csv(input_filename)
    # sort dataframe by ascending frequencies
    df.sort_values('f', inplace=True)
    # create columns for real Z, imaginary Z, and log frequencies
    df['rez'] = df['z'] * np.cos(df['phi'] / 180 * np.pi)
    df['imz'] = df['z'] * np.sin(df['phi'] / 180 * np.pi)
    df['z_complex'] = df['rez'].values + 1j*df['imz'].values
    df['log_f'] = np.log(df['f'])
    return df


def core_gp_drt(df, sigma_n, sigma_f, ell):
    """Perform core GP-DRT calculations using dataframe of impedance data
    with optimized hyperparameters sigma_n, sigma_f, and ell."""
    # calculate the matrices shown in eq (18)
    K = GP_DRT.matrix_K(df['log_f'], df['log_f'], sigma_f, ell)
    L_im_K = GP_DRT.matrix_L_im_K(df['log_f'], df['log_f'], sigma_f, ell)
    L2_im_K = GP_DRT.matrix_L2_im_K(df['log_f'], df['log_f'], sigma_f, ell)
    # factorize the matrices and solve linear equations
    # the matrix L^2_{im} K + sigma_n^2 I whose inverse is needed
    K_im_full = L2_im_K + np.square(sigma_n) * np.eye(len(df))
    # Cholesky factorization, L is a lower-triangular matrix
    L = np.linalg.cholesky(K_im_full)
    # covariance matrix
    inv_L = np.linalg.inv(L)
    inv_K_im_full = np.dot(inv_L.T, inv_L)
    # Predict the imaginary part of the GP-DRT and impedance
    drt = {'imz': np.empty(len(df)), 'imz_sigma': np.empty(len(df)),
           'gamma': np.empty(len(df)), 'gamma_sigma': np.empty(len(df))}

    # calculate the imaginary part of impedance at each point
    for index, val in enumerate(df['log_f'].values):
        xi_star = np.array([val])
        L_im_k_star = L_im_K[:, index]
        L2_im_k_star = L2_im_K[:, index]
        k_star_star = K[index, index]
        L2_im_k_star_star = GP_DRT.matrix_L2_im_K(
                xi_star, xi_star, sigma_f, ell)
        # compute Z_im_star mean and standard deviation using eq (26)
        drt['imz'][index] = np.dot(L2_im_k_star.T,
                     np.dot(inv_K_im_full, df['imz']))
        drt['imz_sigma'][index] = L2_im_k_star_star-np.dot(L2_im_k_star.T,
                           np.dot(inv_K_im_full, L2_im_k_star))
        drt['gamma'][index] = -np.dot(L_im_k_star.T,
                      np.dot(inv_K_im_full, df['imz']))
        drt['gamma_sigma'][index] = k_star_star-np.dot(
                L_im_k_star.T, np.dot(inv_K_im_full, L_im_k_star))
    
    # get confidence ranges in both Z and gamma
    drt['gamma_range'] = [drt['gamma'] - 3*np.sqrt(abs(drt['gamma_sigma'])),
        drt['gamma'] + 3*np.sqrt(abs(drt['gamma_sigma']))]
    drt['imz_range'] = [-drt['imz'] - 3*np.sqrt(abs(drt['imz_sigma'])),
        -drt['imz'] + 3*np.sqrt(abs(drt['imz_sigma']))]

    return drt



# input data file and format
input_filename = 'C:\\Users\\a6q\\exp_data\\EIS_for_GP-DRT.csv'

# import the data
df = import_eis_data(input_filename)

# create the nyquist plot
plt.plot(df['rez'], df['imz'], marker='o')
plot_setup(labels=['Re(Z) (Ohm)', 'Im(Z) (Ohm)'], title='Nyquist plot')
plt.show()


# compute optimal hyperparameters for marginal log-likelihood in eq (31)
sigma_n = 1.0e3
sigma_f = 1.0e5
ell = 4
theta_0 = np.array([sigma_n, sigma_f, ell])
seq_theta = np.copy(theta_0)
def print_results(theta):
    global seq_theta
    seq_theta = np.vstack((seq_theta, theta))
    print('{0:.7f}  {1:.7f}  {2:.7f}'.format(theta[0], theta[1], theta[2]))
print('sigma_n,   sigma_f,   ell')

# minimize the NMLL using optimization in scipy
res = minimize(GP_DRT.NMLL_fct, theta_0,
               args=(df['z_complex'].values, df['log_f'].values),
               #method='BFGS',
               method='L-BFGS-B',
               bounds=[(0, None), (0, None), (0, None)],
               jac=GP_DRT.grad_NMLL_fct,
               callback=print_results,
               options={'disp': True})

print('OPTIMIZATION RESULTS:')
print(res)
print('\nOriginal parameters: %.5E, %0.5E, %.5E' %(sigma_n, sigma_f, ell))
sigma_n, sigma_f, ell = res.x
print('Optimized parameters: %.5E, %0.5E, %.5E' %(sigma_n, sigma_f, ell))
# get core DRT results using optimized hyperparameters
drt = core_gp_drt(df, sigma_n, sigma_f, ell)


# plot the DRT and its confidence region
plt.semilogx(df['f'], drt['gamma'], linewidth=2, color='r', label='GP-DRT')
plt.fill_between(df['f'], drt['gamma_range'][0], drt['gamma_range'][1],
                 color='k', alpha=0.3, label='confidence region')

plot_setup(labels=['Frequency (Hz)', '$\gamma$ ($\Omega$)'], legend=True)
plt.show()



# plot imaginary part of the GP-DRT impedance with the experimental data
plt.semilogx(df['f'], -df['imz'].values, "o",
             markersize=10, color="black", label='experiment')
plt.semilogx(df['f'], -drt['imz'], linewidth=2, color='r', label='GP-DRT')
plt.fill_between(df['f'], drt['imz_range'][0], drt['imz_range'][1],
                 color='k', alpha=0.3, label='confidence region')
plot_setup(labels=['Frequency (Hz)', '-Im(Z) (Ohm)'], legend=True)
plt.show()


# save final results to pandas dataframe
results = pd.DataFrame(
        columns=['freq', 'gamma', 'gamma_low', 'gamma_high',
                 '-exp_imz', 'GP-DRT_fit', 'drt_low', 'drt_high'],
                 data=np.column_stack((df['f'], drt['gamma'],
                                       drt['gamma_range'][0],
                                       drt['gamma_range'][1],
                                       -df['imz'], -drt['imz'],
                                       drt['imz_range'][0],
                                       drt['imz_range'][1])))

print('total runtime: %0.2f seconds' %(time.time() - start_time))













