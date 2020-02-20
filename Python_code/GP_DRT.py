# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 2019
@author: Jiapeng Liu, Francesco Ciucci (francesco.ciucci@ust.hk)

This module includes all necessary functions for GP-DRT model implemented in
the paper "Liu, J., & Ciucci, F. (2019). The Gaussian process distribution of
relaxation times: A machine learning tool for the analysis and prediction of
electrochemical impedance spectroscopy data. Electrochimica Acta, 135316."
"""

from numpy import exp
from numpy import pi
from numpy import log
from scipy import integrate
import numpy as np


def kernel(xi, xi_prime, sigma_f, ell):
    """Define squared exponential kernel."""
    return (sigma_f**2)*exp(-0.5/(ell**2)*((xi-xi_prime)**2))


def integrand_L_im(x, delta_xi, sigma_f, ell):
    """Function to be integrated in eq (65) w/o constant and minus sign."""
    sqr_exp = exp(-0.5/np.square(ell)*np.square(x))
    a = delta_xi - x    
    kernel_part = exp(-a)/(1.+exp(-2*a)) if a > 0 else exp(a)/(1.+exp(2*a))
    return kernel_part * sqr_exp


def integrand_L2_im(x, xi, xi_prime, sigma_f, ell):
    """Function to be integrated in eq (76), omitting the constant part."""
    if x < 0:
        numerator = exp(x-0.5/(ell**2)*(x**2))*(x+xi_prime-xi)
        denominator = (-1+((exp(xi_prime)/exp(xi))**2)*exp(2*x))
    else:
        numerator = exp(-x-0.5/(ell**2)*(x**2))*(x+xi_prime-xi)
        denominator = (-exp(-2*x)+((exp(xi_prime)/exp(xi))**2))
    return numerator / denominator


def integrand_der_ell_L2_im(x, xi, xi_prime, sigma_f, ell):
    """Derivative of the integrand in eq (76)."""
    if x < 0:
        numerator = (x**2)*exp(x-0.5/(ell**2)*(x**2))*(x+xi_prime-xi)
        denominator = (-1+((exp(xi_prime)/exp(xi))**2)*exp(2*x))
    else:
        numerator = (x**2)*exp(-x-0.5/(ell**2)*(x**2))*(x+xi_prime-xi)
        denominator = (-exp(-2*x)+((exp(xi_prime)/exp(xi))**2))
    return numerator / denominator


def matrix_K(xi_n_vec, xi_m_vec, sigma_f, ell):
    """Assemble the covariance matrix K as shown in eq (18a)."""
    K = np.zeros((len(xi_n_vec), len(xi_m_vec)))
    for n in range(len(xi_n_vec)):
        for m in range(len(xi_m_vec)):
            K[n,m] = kernel(xi_n_vec[n], xi_m_vec[m], sigma_f, ell)
    return K


def matrix_L_im_K(xi_n_vec, xi_m_vec, sigma_f, ell):
    """Assemble the matrix of eq (18b)."""
    if np.array_equal(xi_n_vec, xi_m_vec):
        # considering the matrices are symmetric
        xi_vec = xi_n_vec
        L_im_K = np.zeros((len(xi_vec), len(xi_vec)))
        for n in range(0, len(xi_vec)):
            delta_xi = xi_vec[n]-xi_vec[0] + log(2*pi)
            integral, tol = integrate.quad(
                    integrand_L_im,
                    -np.inf, np.inf,
                    epsabs=1E-12, epsrel=1E-12,
                    args=(delta_xi, sigma_f, ell))
            np.fill_diagonal(L_im_K[n:, :], (sigma_f**2)*(integral))
            
            delta_xi = xi_vec[0]-xi_vec[n] + log(2*pi)
            integral, tol = integrate.quad(
                    integrand_L_im,
                    -np.inf, np.inf,
                    epsabs=1E-12, epsrel=1E-12,
                    args=(delta_xi, sigma_f, ell))
            np.fill_diagonal(L_im_K[:, n:], (sigma_f**2)*(integral))
    else:
        N_n_freqs = xi_n_vec.size
        N_m_freqs = xi_m_vec.size
        L_im_K = np.zeros([N_n_freqs, N_m_freqs])

        for n in range(0, N_n_freqs):
            for m in range(0, N_m_freqs):
                delta_xi = xi_n_vec[n]-xi_m_vec[m] + log(2*pi)
                integral, tol = integrate.quad(
                        integrand_L_im,
                        -np.inf, np.inf,
                        epsabs=1E-12, epsrel=1E-12,
                        args=(delta_xi, sigma_f, ell))
                L_im_K[n,m] =  (sigma_f**2)*(integral);
    return L_im_K


def matrix_L2_im_K(xi_n_vec, xi_m_vec, sigma_f, ell):
    """Assemble the matrix of eq (18d)."""
    if np.array_equal(xi_n_vec, xi_m_vec):
        # considering the matrices are symmetrical
        xi_vec = xi_n_vec
        L2_im_K = np.zeros((len(xi_vec), len(xi_vec)))
        for n in range(0, (len(xi_vec))):
            integral, tol = integrate.quad(
                    integrand_L2_im,
                    -np.inf, np.inf,
                    epsabs=1E-12, epsrel=1E-12,
                    args=(xi_vec[n], xi_vec[0], sigma_f, ell))
            
            np.fill_diagonal(L2_im_K[n:, :],
                             exp(xi_vec[0]-xi_vec[n])*(sigma_f**2)*integral)
            np.fill_diagonal(L2_im_K[:, n:],
                             exp(xi_vec[0]-xi_vec[n])*(sigma_f**2)*integral)
    else:
        N_n_freqs = xi_n_vec.size
        N_m_freqs = xi_m_vec.size
        L2_im_K = np.zeros([N_n_freqs, N_m_freqs])
        for n in range(0, N_n_freqs):
            for m in range(0, N_m_freqs):
                integral, tol = integrate.quad(
                        integrand_L2_im,
                        -np.inf, np.inf,
                        epsabs=1E-12, epsrel=1E-12,
                        args=(xi_n_vec[n], xi_m_vec[m], sigma_f, ell))
                L2_im_K[n,m] = exp(
                        xi_m_vec[m]-xi_n_vec[n])*(sigma_f**2)*integral
    return L2_im_K


def der_ell_matrix_L2_im_K(xi_vec, sigma_f, ell):
    """Assemble the matrix corresponding to the derivative of eq (18d)."""
    der_ell_L2_im_K = np.zeros((len(xi_vec), len(xi_vec)))
    for n in range(len(xi_vec)):
        integral, tol = integrate.quad(
                integrand_der_ell_L2_im,
                -np.inf, np.inf,
                epsabs=1E-12,
                epsrel=1E-12,
                args=(xi_vec[n], xi_vec[0], sigma_f, ell))
        np.fill_diagonal(der_ell_L2_im_K[n:, :],
                         exp(xi_vec[0]-xi_vec[n])*(sigma_f**2)/(ell**3)*integral)
        np.fill_diagonal(der_ell_L2_im_K[:, n:],
                         exp(xi_vec[0]-xi_vec[n])*(sigma_f**2)/(ell**3)*integral)
    return der_ell_L2_im_K


def NMLL_fct(theta, Z_exp, xi_vec):
    """Calculate the negative marginal log-likelihood (NMLL) of eq (31)."""
    # load the initial value for parameters needed to optimize
    sigma_n, sigma_f, ell = theta
    Sigma = np.square(sigma_n)*np.eye(len(xi_vec))                    
    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell) 
    K_im_full = L2_im_K + Sigma
    L = np.linalg.cholesky(K_im_full)
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)
    # return the final result of eq (32).
    # note that $\frac{N}{2} \log 2\pi$ is not included as it is constant.
    # the determinant of $\mathbf K_{\rm im}^{\rm full}$ is calculated
    # as the product of the diagonal element of L
    return 0.5 * np.dot(Z_exp.imag, alpha) + np.sum(np.log(np.diag(L)))


def grad_NMLL_fct(theta, Z_exp, xi_vec):
    """Gradient of the negative marginal log-likelihhod (NMLL)."""
    # load the initial value for parameters needed to optimize
    sigma_n, sigma_f, ell = theta
    Sigma = np.square(sigma_n)*np.eye(len(xi_vec))
    L2_im_K = matrix_L2_im_K(xi_vec, xi_vec, sigma_f, ell)
    K_im_full = L2_im_K + Sigma
    L = np.linalg.cholesky(K_im_full)
    # solve for alpha
    alpha = np.linalg.solve(L, Z_exp.imag)
    alpha = np.linalg.solve(L.T, alpha)
    # compute inverse of K_im_full
    inv_L = np.linalg.inv(L)
    inv_K_im_full = np.dot(inv_L.T, inv_L)
    # calculate the derivative of matrices
    der_mat_sigma_n = (2.*sigma_n)*np.eye(len(xi_vec))
    der_mat_sigma_f = (2./sigma_f)*L2_im_K
    der_mat_ell = der_ell_matrix_L2_im_K(xi_vec, sigma_f, ell)
    # calculate the derivative according to eq (78)
    d_K_im_full_d_sigma_n = - 0.5*np.dot(
            alpha.T, np.dot(der_mat_sigma_n, alpha)) + 0.5*np.trace(np.dot(
                    inv_K_im_full, der_mat_sigma_n))    
    d_K_im_full_d_sigma_f = - 0.5*np.dot(
            alpha.T, np.dot(der_mat_sigma_f, alpha)) + 0.5*np.trace(
                    np.dot(inv_K_im_full, der_mat_sigma_f))
    d_K_im_full_d_ell = - 0.5*np.dot(
            alpha.T, np.dot(der_mat_ell, alpha)) + 0.5*np.trace(
                    np.dot(inv_K_im_full, der_mat_ell))
    grad = np.array([
            d_K_im_full_d_sigma_n, d_K_im_full_d_sigma_f, d_K_im_full_d_ell])
    return grad