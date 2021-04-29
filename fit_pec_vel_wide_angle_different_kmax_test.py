# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:19:38 2021

@author:Yanxiang Lai
"""

import numpy as np
import scipy as sp
import sys
import time
import copy
from numpy import linalg
from scipy import integrate
from scipy import interpolate
from scipy import optimize
import emcee
from emcee import backends
from scipy.linalg import lapack
import pandas as pd
import math
from multiprocessing import Pool
import time

# Speed of light in km/s
LightSpeed = 299792.458

def get_dL(params_fid):

    ffid, sigma_v_fid, bfid, baddfid, sigmag_fid = params_fid
    # sigmag_fid = sigma_g

    # Compute derivatives about fiducial values
    sigmag_array = np.array([sigmag_fid**(2*n) for n in range(0,7)])
    dsigmag_array = np.array([2.0*n*sigmag_fid**(2*n-1) for n in range(0,7)])
    d2sigmag_array = np.array([2.0*n*(2.0*n-1.0)*sigmag_fid**(2*n-2) for n in range(0,7)])
    Cfid = np.sum(sigmag_array*(bfid**2*conv_b + bfid*ffid*conv_bf + ffid**2*conv_f + baddfid**2*conv_badd), axis=-1) + sigma_v_fid**2*conv_sigma_v + conv_noise

    pivots = np.zeros(int(ncomp_velocity + ncomp_galaxy), np.intc)
    conv_pk_copy, pivots, info = lapack.dgetrf(Cfid)
    abs_element = np.fabs(np.diagonal(conv_pk_copy))
    detCfid = np.sum(np.log(abs_element))
    # Cfid_inv = lapack.dgesv(Cfid, np.eye(int(ncomp_velocity + ncomp_galaxy)))[2]
    Cfid_inv = np.linalg.solve(Cfid, np.eye(int(ncomp_velocity + ncomp_galaxy)))
    chisquared_fid = datagrid_comp @ Cfid_inv @ datagrid_comp
    if (chisquared_fid < 0):
        raise ValueError('Negative chi-squared')
    loglikefid = -0.5*(chisquared_fid + detCfid)

    # Derivatices of model
    dCdb = np.sum(sigmag_array*(2.0*bfid*conv_b + ffid*conv_bf), axis=-1)
    dCdbadd = np.sum(sigmag_array*(2.0*baddfid*conv_badd), axis=-1)
    dCdf = np.sum(sigmag_array*(bfid*conv_bf + 2.0*ffid*conv_f), axis=-1)
    dCdsigmag = np.sum(dsigmag_array*(bfid**2*conv_b + bfid*ffid*conv_bf + ffid**2*conv_f + baddfid**2*conv_badd), axis=-1)
    dCdsigmav = 2.0*sigma_v_fid*conv_sigma_v

    #d2Cdb2 = 2.0 * np.sum(sigmag_array*conv_b, axis=-1)
    #d2Cdbadd2 = d2Cdb2
    #d2Cdf2 = 2.0 * np.sum(sigmag_array*conv_f, axis=-1)
    #d2Cdsigmag2 = np.sum(d2sigmag_array*(bfid**2*conv_b + bfid*ffid*conv_bf + ffid**2*conv_f), axis=-1)
    #d2Cdsigmav2 = 2.0*conv_sigma_v

    # Derivatives of loglikelihood function
    weirdbit = np.matmul(np.outer((Cfid_inv @ datagrid_comp), datagrid_comp), Cfid_inv)
    #dLdC = -0.5*(2.0 * Cfid_inv - (Cfid_inv * np.eye(len(datagrid_comp))) - 2.0 * weirdbit + (weirdbit * np.eye(len(datagrid_comp))))
    dLdC = -0.5*(Cfid_inv - weirdbit)
    dLdb = np.sum(np.diag(np.matmul(dLdC, dCdb)))
    dLdbadd = np.sum(np.diag(np.matmul(dLdC, dCdbadd)))
    dLdf = np.sum(np.diag(np.matmul(dLdC, dCdf)))
    dLdsigmag = np.sum(np.diag(np.matmul(dLdC, dCdsigmag)))
    dLdsigmav = np.sum(np.diag(np.matmul(dLdC, dCdsigmav)))
    Fisher_dCdb = np.matmul(np.matmul(Cfid_inv, dCdb), Cfid_inv)
    Fisher_dCdbadd = np.matmul(np.matmul(Cfid_inv, dCdbadd), Cfid_inv)
    Fisher_dCdf = np.matmul(np.matmul(Cfid_inv, dCdf), Cfid_inv)
    Fisher_dCdsigmav = np.matmul(np.matmul(Cfid_inv, dCdsigmav), Cfid_inv)
    Fisher_dCdsigmag = np.matmul(np.matmul(Cfid_inv, dCdsigmag), Cfid_inv)
    
    d2Ldb2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdb, dCdb))
    d2Ldbdbadd = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdb, dCdbadd))
    d2Ldbdf = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdb, dCdf))
    d2Ldbdsigmav = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdb, dCdsigmav))
    d2Ldbdsigmag = 0.5*np.sum(np.einsum('ij, ji->i', Fisher_dCdb, dCdsigmag))
    d2Ldf2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdf, dCdf))
    d2Ldfdbadd = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdf, dCdbadd))
    d2Ldfdsigmav = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdf, dCdsigmav))
    d2Ldfdsigmag = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdf, dCdsigmag))
    d2Ldbadd2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdbadd, dCdbadd))
    d2Ldbadddsigmav = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdbadd, dCdsigmav))
    d2Ldbadddsigmag = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdbadd, dCdsigmag))
    d2Ldsigmav2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdsigmav, dCdsigmav))
    d2Ldsigmavdsigmag = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdsigmav, dCdsigmag))
    d2Ldsigmag2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdsigmag, dCdsigmag))
    
    # d2Ldb2 = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdb), Cfid_inv), dCdb)))
    # d2Ldbdbadd = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdb), Cfid_inv), dCdbadd)))
    # d2Ldbadddb = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdbadd), Cfid_inv), dCdb)))
    # d2Ldbdf = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdb), Cfid_inv), dCdf)))
    # d2Ldfdb = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdf), Cfid_inv), dCdb)))
    # d2Ldbdsigmag = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdb), Cfid_inv), dCdsigmag)))
    # d2Ldsigmagdb = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmag), Cfid_inv), dCdb)))
    # d2Ldbdsigmav = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdb), Cfid_inv), dCdsigmav)))
    # d2Ldsigmavdb = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmav), Cfid_inv), dCdb)))
    # d2Ldbadd2 = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdbadd), Cfid_inv), dCdbadd)))
    # d2Ldbadddf = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdbadd), Cfid_inv), dCdf)))
    # d2Ldfdbadd = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdf), Cfid_inv), dCdbadd)))
    # d2Ldbadddsigmag = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdbadd), Cfid_inv), dCdsigmag)))
    # d2Ldsigmagdbadd = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmag), Cfid_inv), dCdbadd)))
    # d2Ldbadddsigmav = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdbadd), Cfid_inv), dCdsigmav)))
    # d2Ldsigmavdbadd = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmav), Cfid_inv), dCdbadd)))
    # d2Ldf2 = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdf), Cfid_inv), dCdf)))
    # d2Ldfdsigmag = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdf), Cfid_inv), dCdsigmag)))
    # d2Ldsigmagdf = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmag), Cfid_inv), dCdf)))
    # d2Ldfdsigmav = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdf), Cfid_inv), dCdsigmav)))
    # d2Ldsigmavdf = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmav), Cfid_inv), dCdf)))
    # d2Ldsigmag2 = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmag), Cfid_inv), dCdsigmag)))
    # d2Ldsigmagdsigmav = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmag), Cfid_inv), dCdsigmav)))
    # d2Ldsigmavdsigmag = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmav), Cfid_inv), dCdsigmag)))
    # d2Ldsigmav2 = 0.5*np.sum(np.diag(np.matmul(np.matmul(np.matmul(Cfid_inv, dCdsigmav), Cfid_inv), dCdsigmav)))

    # dL = np.array([dLdf, dLdsigmav, dLdb, dLdbadd])
    
    dL = np.array([dLdf, dLdsigmav, dLdb, dLdbadd, dLdsigmag])
    # d2L = np.array([[d2Ldf2, d2Ldfdsigmav, d2Ldbdf, d2Ldbadddf], 
    #                 [d2Ldfdsigmav, d2Ldsigmav2, d2Ldbdsigmav, d2Ldbadddsigmav], 
    #                 [d2Ldbdf, d2Ldbdsigmav, d2Ldb2, d2Ldbdbadd], 
    #                 [d2Ldbadddf, d2Ldbadddsigmav, d2Ldbdbadd,  d2Ldbadd2]])
    
    # d2L = np.array([[d2Ldf2, d2Ldfdsigmav, d2Ldfdb, d2Ldfdbadd], 
    #             [d2Ldsigmavdf, d2Ldsigmav2, d2Ldsigmavdb, d2Ldsigmavdbadd], 
    #             [d2Ldbdf, d2Ldbdsigmav, d2Ldb2, d2Ldbdbadd], 
    #             [d2Ldbadddf, d2Ldbadddsigmav, d2Ldbadddb,  d2Ldbadd2]])
    
    # d2L = np.array([[d2Ldf2, d2Ldfdsigmav, d2Ldfdb, d2Ldfdbadd, d2Ldfdsigmag], 
    #             [d2Ldsigmavdf, d2Ldsigmav2, d2Ldsigmavdb, d2Ldsigmavdbadd, d2Ldsigmavdsigmag], 
    #             [d2Ldbdf, d2Ldbdsigmav, d2Ldb2, d2Ldbdbadd, d2Ldbdsigmag], 
    #             [d2Ldbadddf, d2Ldbadddsigmav, d2Ldbadddb,  d2Ldbadd2, d2Ldbadddsigmag],
    #             [d2Ldsigmagdf, d2Ldsigmagdsigmav, d2Ldsigmagdb, d2Ldsigmagdbadd, d2Ldsigmag2]])
    
    d2L = np.array([[d2Ldf2, d2Ldfdsigmav, d2Ldbdf, d2Ldfdbadd, d2Ldfdsigmag], 
            [d2Ldfdsigmav, d2Ldsigmav2, d2Ldbdsigmav, d2Ldbadddsigmav, d2Ldsigmavdsigmag], 
            [d2Ldbdf, d2Ldbdsigmav, d2Ldb2, d2Ldbdbadd, d2Ldbdsigmag], 
            [d2Ldfdbadd, d2Ldbadddsigmav, d2Ldbdbadd,  d2Ldbadd2, d2Ldbadddsigmag],
            [d2Ldfdsigmag, d2Ldsigmavdsigmag, d2Ldbdsigmag, d2Ldbadddsigmag, d2Ldsigmag2]])


    return loglikefid, dL, d2L

def read_chain_backend(chainfile):

    reader = backends.HDFBackend(chainfile)

    tau = reader.get_autocorr_time()
    burnin = int(2 * np.max(tau))
    samples = reader.get_chain(discard=burnin, flat=True)
    log_prob_samples = reader.get_log_prob(discard=burnin, flat=True)
    bestid = np.argmax(log_prob_samples)

    return samples, copy.copy(samples[bestid]), log_prob_samples

# Calculates H(z)/H0
def Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    fz = ((1.0+redshift)**(3*(1.0+w0+wa*ap)))*np.exp(-3*wa*(redshift/(1.0+redshift)))
    omega_k = 1.0-omega_m-omega_lambda-omega_rad
    return np.sqrt(omega_rad*(1.0+redshift)**4+omega_m*(1.0+redshift)**3+omega_k*(1.0+redshift)**2+omega_lambda*fz)

# The Comoving Distance Integrand
def DistDcIntegrand(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    return 1.0/Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)

# The Comoving Distance in Mpc
def DistDc(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    return (LightSpeed/Hubble_Constant)*integrate.quad(DistDcIntegrand, 0.0, redshift, args=(omega_m, omega_lambda, omega_rad, w0, wa, ap))[0]

#def linear_interpolation(conv_matrix_vg, conv_matrix_vv_g, conv_matrix_vv_ng, sigma_u, ngrid_comp, ncomp):
#    #This function linearly interpolates the covariance matrices. 
#    
#    # Modify the covariance matrix based on the value of sigma_u
#    job_number_up = int(math.ceil(sigma_u/0.2 - 1.0))
#    job_number_lo = int(math.floor(sigma_u/0.2 - 1.0))
#    
#    if (job_number_lo < 0):
#        job_number_lo = 0
#        job_number_up = 0
#        
#    if (job_number_up > 124):
#        job_number_up = 124
#        job_number_lo = 124
#        
#    if (job_number_up == job_number_lo):
#        conv_pk_sigma_u = conv_matrix_vv_g[job_number_lo]
#        conv_pk_sigma_u_ng = conv_matrix_vv_ng[job_number_lo]
#        conv_vg = conv_matrix_vg[job_number_lo]
#        for i in range(ncomp):
#            conv_pk_sigma_u[i][i] += (conv_pk_sigma_u_ng[i][i]-conv_pk_sigma_u[i][i])/ngrid_comp[i]
#            #"Shot noise" correction for the velocity auto-covariance matrix
#        
#    else:
#        sigma_u_lo = 0.2*(job_number_lo+1.0)
#        conv_pk_sigma_lo = conv_matrix_vv_g[job_number_lo]
#        conv_pk_sigma_lo_ng = conv_matrix_vv_ng[job_number_lo]
#        conv_vg_lo = conv_matrix_vg[job_number_lo]
#        
#        conv_pk_sigma_up = conv_matrix_vv_g[job_number_up]
#        conv_pk_sigma_up_ng = conv_matrix_vv_ng[job_number_up]
#        conv_vg_up = conv_matrix_vg[job_number_up]
#        
#        correction = (sigma_u - sigma_u_lo)/0.2
#        conv_pk_sigma_u = (1.0-correction)*conv_pk_sigma_lo + (correction)*conv_pk_sigma_up 
#        conv_pk_sigma_u_ng = (1.0-correction)*conv_pk_sigma_lo_ng + (correction)*conv_pk_sigma_up_ng
#        conv_vg = (1.0-correction)*conv_vg_lo + correction*conv_vg_up
#        for i in range(ncomp):
#            conv_pk_sigma_u[i][i] += (conv_pk_sigma_u_ng[i][i]-conv_pk_sigma_u[i][i])/ngrid_comp[i]
#            #"Shot noise" correction for the velocity auto-covariance matrix
#        
#    return conv_pk_sigma_u, conv_vg
    
def linear_interpolation(conv_pk_array, sigma_u, num):
    output = []
    for i in range(num):
        index_low = int(math.floor(sigma_u))
        index_high = index_low + 1
        
        correction = sigma_u - index_low
        
        output.append((1.0-correction)*conv_pk_array[index_low][i] + correction*conv_pk_array[index_high][i])
        
    return output
        

def lnpost(params):

    # This returns the log posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params)
    return prior + like

# def lnpost_mcmc(params):

#     # This returns the log posterior distribution which is given by the log prior plus the log likelihood
#     prior = lnprior_mcmc(params)
#     if not np.isfinite(prior):
#         return -np.inf
#     like = lnlike_mcmc(params)
#     return prior + like

def lnprior(params):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
#    fsigma8, sigma_v, bsigma8, b_add_sigma8, sigma_u = params
    # fsigma8, sigma_v, bsigma8, sigma_u, sigma_g = params
    # fsigma8, sigma_v, bsigma8, sigma_g = params
    # fsigma8, sigma_v, bsigma8, b_add_sigma8 = params
    fsigma8, sigma_v, bsigma8, b_add_sigma8, sigma_g = params

    
    # sigma_v = params
    
    # Don't allow fsigma8 to be less that 0.0
    if (fsigma8 >= 0.0):
        fsigma8prior = 1.0 
    else:
        return -np.inf

    # Flat prior for sigma_v between 0.0 and 1000.0 km/s (from Andrews paper)
    if (1.0 < sigma_v <= 1000.0):
        sigmavprior = 1.0/sigma_v
    else:
        return -np.inf 
    
    # Flat prior for bsigma8 between 0.0 and 3.0 
    if (0.0 < bsigma8 < 3.0):
        bsigma8prior = 1.0/3.0
    else:
        return -np.inf
    
    # Flat prior for b_add_sigma8 between 0.0 and 10.0
    if (0.0 < b_add_sigma8 <= 10.0):
        b_add_sigma8prior = 1.0/10.0
    else:
        return -np.inf
    
    # # Flat prior for sigma_u between 0.2 and 25.0
    # if (0.0 <= sigma_u < 25.0):
    #     sigma_uprior = 1.0/25.0
    # else:
    #     return -np.inf
    
    if (0.0 < sigma_g < 10.0):
        sigma_gprior = 1.0/10.0
    else:
        return -np.inf

    return 0.0 

# def lnprior_mcmc(params):

#     # Here we define the prior for all the parameters. We'll ignore the constants as they
#     # cancel out when subtracting the log posteriors
# #    fsigma8, sigma_v, bsigma8, b_add_sigma8, sigma_u = params
#     # fsigma8, sigma_v, bsigma8, sigma_u, sigma_g = params
#     # fsigma8, sigma_v, bsigma8, sigma_g = params
#     # fsigma8, sigma_v, bsigma8, b_add_sigma8 = params
    
#     sigma_v = params
    
#     # # Don't allow fsigma8 to be less that 0.0
#     # if (fsigma8 >= 0.0):
#     #     fsigma8prior = 1.0 
#     # else:
#     #     return -np.inf

#     # Flat prior for sigma_v between 0.0 and 1000.0 km/s (from Andrews paper)
#     if (0.0 <= sigma_v < 1000.0):
#         sigmavprior = 1.0/1000.0
#     else:
#         return -np.inf 
    
#     # # Flat prior for bsigma8 between 0.0 and 3.0 
#     # if (0.0 < bsigma8 < 3.0):
#     #     bsigma8prior = 1.0/3.0
#     # else:
#     #     return -np.inf
    
#     # # Flat prior for b_add_sigma8 between 0.0 and 10.0
#     # if (0.0 <= b_add_sigma8 <= 10.0):
#     #     b_add_sigma8prior = 1.0/10.0
#     # else:
#     #     return -np.inf
    
#     # # Flat prior for sigma_u between 0.2 and 25.0
#     # if (0.0 <= sigma_u < 25.0):
#     #     sigma_uprior = 1.0/25.0
#     # else:
#     #     return -np.inf
    
#     # if (1.0 <= sigma_g < 8.0):
#     #     sigma_gprior = 1.0/7.0
#     # else:
#     #     return -np.inf

#     return 0.0

def lnlike(params): 

    # Return the log likelihood for a model. Here are the parameters we want to fit for.
    # fsigma8, sigma_v, bsigma8, b_add_sigma8 = params
    fsigma8, sigma_v, bsigma8, b_add_sigma8, sigma_g = params

    # sigma_v = params
    
    """conv_pk_gal_gal = (bsigma8/bsigma8_old)**2*(conv_pk_gal_gal_0) + (bsigma8*fsigma8)/(bsigma8_old*fsigma8_old)*conv_pk_gal_gal_1 + (fsigma8/fsigma8_old)**2*conv_pk_gal_gal_2 + (b_add_sigma8/bsigma8_old)**2*conv_pk_gal_gal_badd
    conv_pk_vel_gal = (bsigma8/bsigma8_old)*(fsigma8/fsigma8_old)*conv_pk_vel_gal_0 + (fsigma8/fsigma8_old)**2*conv_pk_vel_gal_1
    conv_pk_gal_vel = conv_pk_vel_gal.T
    conv_pk_vel_vel = (fsigma8/fsigma8_old)**2*conv_pk_vel_vel_0
    
    # Modify the velocity covariance matrix based on the data and random uncertainties.
    sigma_varr = np.zeros(ncomp_velocity)+sigma_v #The error from the nonlinear velocity dispersion sigmav
    err_rand = datagrid_vec[0:,4]*sigma_varr/np.sqrt(ngrid_SDSS_vel)
    conv_pk_diag = conv_pk_vel_vel[np.diag_indices(ncomp_velocity)]
    np.fill_diagonal(conv_pk_vel_vel, conv_pk_diag + errgrid_SDSS + err_rand**2) #Add the error to the diagonal of the
            
    #Modify the galaxy covariance matrix based on shot noise
    shot_noise = 1.0/data_expect
    conv_gg_diag = conv_pk_gal_gal[np.diag_indices(ncomp_galaxy)]
    np.fill_diagonal(conv_pk_gal_gal, conv_gg_diag + shot_noise)

    conv_pk_new_block_1 = np.concatenate((conv_pk_gal_gal, conv_pk_gal_vel), axis = 1)
    conv_pk_new_block_2 = np.concatenate((conv_pk_vel_gal, conv_pk_vel_vel), axis = 1)
    conv_pk_new = np.concatenate((conv_pk_new_block_1, conv_pk_new_block_2), axis = 0)

    pivots = np.zeros(int(ncomp_velocity + ncomp_galaxy), np.intc)
    conv_pk_copy, pivots, info = lapack.dgetrf(conv_pk_new)
    abs_element = np.fabs(np.diagonal(conv_pk_copy))
    det = np.sum(np.log(abs_element))

    #Calculates the chi-squared with matrix multiplication. \
    # chi_squared = np.matmul(datagrid_comp.T, np.matmul(cov_inv, datagrid_comp))
    chi_squared = np.matmul(datagrid_comp.T, np.linalg.solve(conv_pk_new, datagrid_comp))"""

    # Calculate the log likelihood using a taylor expansion
    diffs = np.array([fsigma8/fsigma8_old, sigma_v, bsigma8/bsigma8_old, b_add_sigma8/bsigma8_old, sigma_g]) - params_fid
    loglike = loglikefid + np.sum(dL*diffs) - 0.5 * diffs @ d2L @ diffs
    # loglike = loglikefid + np.sum(dL*diffs) - 0.5 * np.matmul(diffs, np.matmul(d2L, diffs.T))
    #print(params, loglike)

    #print(loglike, -0.5*(det+chi_squared))

    #while (chi_squared < 0):
    #    w, v = np.linalg.eig(conv_pk_new)
    #    print(chi_squared, fsigma8, bsigma8, np.min(w), np.max(w))
    #    raise ValueError('Negative chi-squared')
    
    #return -0.5*(det+chi_squared)
    return loglike

# def lnlike_mcmc(params): 

#     # Return the log likelihood for a model. Here are the parameters we want to fit for.
#     # fsigma8, sigma_v, bsigma8, b_add_sigma8 = params
#     sigma_v = params
    
#     """conv_pk_gal_gal = (bsigma8/bsigma8_old)**2*(conv_pk_gal_gal_0) + (bsigma8*fsigma8)/(bsigma8_old*fsigma8_old)*conv_pk_gal_gal_1 + (fsigma8/fsigma8_old)**2*conv_pk_gal_gal_2 + (b_add_sigma8/bsigma8_old)**2*conv_pk_gal_gal_badd
#     conv_pk_vel_gal = (bsigma8/bsigma8_old)*(fsigma8/fsigma8_old)*conv_pk_vel_gal_0 + (fsigma8/fsigma8_old)**2*conv_pk_vel_gal_1
#     conv_pk_gal_vel = conv_pk_vel_gal.T
#     conv_pk_vel_vel = (fsigma8/fsigma8_old)**2*conv_pk_vel_vel_0
    
#     # Modify the velocity covariance matrix based on the data and random uncertainties.
#     sigma_varr = np.zeros(ncomp_velocity)+sigma_v #The error from the nonlinear velocity dispersion sigmav
#     err_rand = datagrid_vec[0:,4]*sigma_varr/np.sqrt(ngrid_SDSS_vel)
#     conv_pk_diag = conv_pk_vel_vel[np.diag_indices(ncomp_velocity)]
#     np.fill_diagonal(conv_pk_vel_vel, conv_pk_diag + errgrid_SDSS + err_rand**2) #Add the error to the diagonal of the
            
#     #Modify the galaxy covariance matrix based on shot noise
#     shot_noise = 1.0/data_expect
#     conv_gg_diag = conv_pk_gal_gal[np.diag_indices(ncomp_galaxy)]
#     np.fill_diagonal(conv_pk_gal_gal, conv_gg_diag + shot_noise)

#     conv_pk_new_block_1 = np.concatenate((conv_pk_gal_gal, conv_pk_gal_vel), axis = 1)
#     conv_pk_new_block_2 = np.concatenate((conv_pk_vel_gal, conv_pk_vel_vel), axis = 1)
#     conv_pk_new = np.concatenate((conv_pk_new_block_1, conv_pk_new_block_2), axis = 0)

#     pivots = np.zeros(int(ncomp_velocity + ncomp_galaxy), np.intc)
#     conv_pk_copy, pivots, info = lapack.dgetrf(conv_pk_new)
#     abs_element = np.fabs(np.diagonal(conv_pk_copy))
#     det = np.sum(np.log(abs_element))

#     #Calculates the chi-squared with matrix multiplication. \
#     # chi_squared = np.matmul(datagrid_comp.T, np.matmul(cov_inv, datagrid_comp))
#     chi_squared = np.matmul(datagrid_comp.T, np.linalg.solve(conv_pk_new, datagrid_comp))"""

#     # Calculate the log likelihood using a taylor expansion
#     diffs = np.array([fsigma8/fsigma8_old, sigma_v, bsigma8/bsigma8_old, b_add_sigma8/bsigma8_old]) - params_fid
#     loglike = loglikefid + np.sum(dL*diffs) - 0.5 * diffs @ d2L @ diffs
#     # loglike = loglikefid + np.sum(dL*diffs) - 0.5 * np.matmul(diffs, np.matmul(d2L, diffs.T))
#     #print(params, loglike)

#     #print(loglike, -0.5*(det+chi_squared))

#     #while (chi_squared < 0):
#     #    w, v = np.linalg.eig(conv_pk_new)
#     #    print(chi_squared, fsigma8, bsigma8, np.min(w), np.max(w))
#     #    raise ValueError('Negative chi-squared')
    
#     #return -0.5*(det+chi_squared)
#     return loglike

fsigma8_old = 0.8150         # The value of fsigma8 used to compute the covariance matrix. 
sigma_y = 0.2**2             # The value of sigma_y used for Eq. 24.
bsigma8_old = 0.8150         # The value of bsigma8 used to compute the covariance matrix. 
r_g = 1.0
omega_m = 0.3121

# Make sure these match the values used to estimate the covariance matrix
kmin = float(sys.argv[1])    # What kmin to use
kmax_galaxy = float(sys.argv[2])    # What kmax to use
gridsize = int(sys.argv[3])  # What gridsize to use
progress = bool(sys.argv[4])  # Whether or not to print out progress
mock_num = int(sys.argv[5]) #Which mock to use. 
sigma_u = int(sys.argv[6]) #Total number of different values of sigmau.
correction = int(sys.argv[7]) #Enter 0 for non-corrected log-distance ratio, 1 for corrected log-distance ratio. 
process_num = int(sys.argv[8]) #This is the number of processors you want to use. 
sigma_g = int(sys.argv[9])  #This is the value of sigma_g to use. 
kmax_velocity = float(sys.argv[10])

file_num, dot_num = divmod(mock_num, 8)
# expect_file = str('/Volumes/Work/UQ/SDSS_dists/data/SDSS_randoms.csv')


# datafile = str("/Volumes/Work/UQ/SDSS_dists/mocks/v4/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr_again")

# expect_file = str('SDSS_randoms.csv')
# datafile = str("./SDSS_mocks/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr_again")

# if (correction == 0):
#     chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_no_correction_sigmau_%d_sigmag_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u, sigma_g))
# elif (correction == 1):
#     chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_correction_sigmau_%d_sigmag_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u, sigma_g))

# expect_file = str('SDSS_randoms.csv')
# datafile = str("./SDSS_mocks/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr_again")

# if (correction == 0):
#     chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_no_correction_sigmau_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u))
# elif (correction == 1):
#     chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_correction_sigmau_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u))

expect_file = str('/data/s4479813/SDSS_randoms.csv')


datafile = str("/data/s4479813/SDSS_mocks/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr_again")

# if (correction == 0):
#     chainfile = str('/data/s4479813/SDSS_data/fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_no_correction_sigmau_%d_sigmag_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u, sigma_g))
# elif (correction == 1):
#     chainfile = str('/data/s4479813/SDSS_data/fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_correction_sigmau_%d_sigmag_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u, sigma_g))

if (correction == 0):
    chainfile = str('/data/s4479813/SDSS_data/fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_no_correction_sigmau_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u))
elif (correction == 1):
    chainfile = str('/data/s4479813/SDSS_data/fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_correction_sigmau_%d_Taylor.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u))


# Generate the grid used to compute the covariance matrix. Again make sure these match the covariance matrix code.
# xmin, xmax = -175.0, 215.0
xmin, xmax = -170.0, 210.0
ymin, ymax = -260.0, 280.0
zmin, zmax = -300.0, 0.0
nx = int(np.ceil((xmax-xmin)/gridsize))
ny = int(np.ceil((ymax-ymin)/gridsize))
nz = int(np.ceil((zmax-zmin)/gridsize))
nelements = nx*ny*nz
print(nx, ny, nz)

# Compute some useful quantities on the grid. For this we need a redshift-dist lookup table
nbins = 5000
redmax = 0.5
red = np.empty(nbins)
dist = np.empty(nbins)
for i in range(nbins):
    red[i] = i*redmax/nbins
    dist[i] = DistDc(red[i], omega_m, 1.0-omega_m, 0.0, 100.0, -1.0, 0.0, 0.0)
red_spline = sp.interpolate.splrep(dist, red, s=0) 
radial_spline=sp.interpolate.splrep(red, dist, s=0) #Interpolate the comoving distance with redshift.


datagrid_vec = np.empty((nelements,5))
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            ind = (i*ny+j)*nz+k
            x = (i+0.5)*gridsize+xmin
            y = (j+0.5)*gridsize+ymin
            z = (k+0.5)*gridsize+zmin
            r = np.sqrt(x**2+y**2+z**2)
            red = sp.interpolate.splev(r, red_spline, der=0)
            ez = Ez(red, omega_m, 1.0-omega_m, 0.0, -1.0, 0.0, 0.0)
            datagrid_vec[ind,0] = x
            datagrid_vec[ind,1] = y
            datagrid_vec[ind,2] = z
            datagrid_vec[ind,3] = r
            datagrid_vec[ind,4] = (1.0/np.log(10))*(1.0+red)/(100.0*ez*r)
            
# Read in the random file
data_expect_all = np.array(pd.read_csv(expect_file, header=None, skiprows=1))

RA_expect = data_expect_all[:, 0]/180.0*np.pi
Dec_expect = data_expect_all[:, 1]/180.0*np.pi
redshift_expect = data_expect_all[:, 2]/LightSpeed
rd_expect = sp.interpolate.splev(redshift_expect, radial_spline)
phi = 241.0
data_z_expect = np.sin(Dec_expect)
data_y_expect = np.cos(Dec_expect)*np.sin(RA_expect - np.pi)
data_x_expect = np.cos(Dec_expect)*np.cos(RA_expect - np.pi)
xnew_expect = data_x_expect*np.cos(phi*np.pi/180.0) - data_z_expect*np.sin(phi*np.pi/180.0)
znew_expect = data_x_expect*np.sin(phi*np.pi/180.0) + data_z_expect*np.cos(phi*np.pi/180.0)
data_x_expect = xnew_expect
data_z_expect = znew_expect

x_expect = data_x_expect*rd_expect
y_expect = data_y_expect*rd_expect
z_expect = data_z_expect*rd_expect

data_expect = np.zeros(nelements)
for i in range(len(data_expect_all)):
    ix = int(np.floor((x_expect[i]-xmin)/gridsize))
    iy = int(np.floor((y_expect[i]-ymin)/gridsize))
    iz = int(np.floor((z_expect[i]-zmin)/gridsize))
    if (ix == nx):
        ix = nx-1
    if (iy == ny):
        iy = ny-1
    if (iz == nz):
        iz = nz-1
    ind = int((ix*ny+iy)*nz+iz)
    data_expect[ind] += 1.0
print(np.sum(data_expect))


data = np.array(pd.read_csv(datafile, header=None, skiprows=1))

RA = data[:, 1]/180.0*np.pi
Dec = data[:, 2]/180.0*np.pi
redshift = data[:, 4]/LightSpeed

rd = sp.interpolate.splev(redshift, radial_spline)




# x = rd*np.cos(Dec)*np.cos(RA)
# y = rd*np.cos(Dec)*np.sin(RA)
# z = rd*np.sin(Dec)

# x = -rd*np.sin(Dec)
# y = rd*np.cos(Dec)*np.sin(RA - np.pi)
# z = rd*np.cos(Dec)*np.cos(RA - np.pi)
phi = 241.0
data_z = np.sin(Dec)
data_y = np.cos(Dec)*np.sin(RA - np.pi)
data_x = np.cos(Dec)*np.cos(RA - np.pi)
xnew = data_x*np.cos(phi*np.pi/180.0) - data_z*np.sin(phi*np.pi/180.0)
znew = data_x*np.sin(phi*np.pi/180.0) + data_z*np.cos(phi*np.pi/180.0)
data_x = xnew
data_z = znew

x = data_x*rd
y = data_y*rd
z = data_z*rd

if (correction == 0):
    log_dist = data[:, 33]
    log_dist_err = data[:, 34]
elif (correction == 1):
    log_dist = data[:, 36]
    log_dist_err = data[:, 37]

data_count = len(x)

# log_dist = data[:,32]
# log_dist_err = np.zeros(data_count)


x = np.reshape(x, (data_count, 1))
y = np.reshape(y, (data_count, 1))
z = np.reshape(z, (data_count, 1))

log_dist = np.reshape(log_dist, (data_count, 1))
log_dist_err = np.reshape(log_dist_err, (data_count, 1))

data_SDSS = np.concatenate((x,y,z,log_dist,log_dist_err), axis=1)

print(np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z))

# Grid the data and find out which grid cells are
# empty and as such don't need including from our theoretical covariance matrix
clipflag = np.zeros(nelements)
ngrid_SDSS = np.zeros(nelements)
datagrid_SDSS = np.zeros(nelements)
errgrid_SDSS = np.zeros(nelements)
for i in range(len(data_SDSS)):
    ix = int(np.floor((data_SDSS[i,0]-xmin)/gridsize))
    iy = int(np.floor((data_SDSS[i,1]-ymin)/gridsize))
    iz = int(np.floor((data_SDSS[i,2]-zmin)/gridsize))
    if (ix == nx):
        ix = nx-1
    if (iy == ny):
        iy = ny-1
    if (iz == nz):
        iz = nz-1
    ind = int((ix*ny+iy)*nz+iz)
    ngrid_SDSS[ind] += 1.0 
    datagrid_SDSS[ind] += data_SDSS[i,3]
    errgrid_SDSS[ind] += data_SDSS[i,4]**2
print(np.sum(ngrid_SDSS))
norm = np.sum(ngrid_SDSS)/np.sum(data_expect)
data_expect = norm*data_expect

comp_galaxy = np.where(data_expect > 0)[0]
ncomp_galaxy = len(comp_galaxy)
ngrid_SDSS_gal = ngrid_SDSS[comp_galaxy]
remove_galaxy = np.where(data_expect == 0)[0]
data_expect = data_expect[comp_galaxy]

comp_velocity = np.where(ngrid_SDSS > 0)[0]
ncomp_velocity = len(comp_velocity)
ngrid_SDSS_vel = ngrid_SDSS[comp_velocity]
data_vel = datagrid_SDSS[comp_velocity]
errgrid_SDSS = errgrid_SDSS[comp_velocity]
remove_velocity = np.where(ngrid_SDSS == 0)[0]

# Correct the data and covariance matrix for the gridding. Eqs. 19 and 22.
data_vel /= ngrid_SDSS_vel        # We summed the velocities in each cell, now get the mean
errgrid_SDSS /= ngrid_SDSS_vel**2.0    # This is the standard error on the mean.

data_gal= (ngrid_SDSS_gal - data_expect)/data_expect

datagrid_comp = np.concatenate((data_gal, data_vel))

conv_vg = []
conv_vv = []
conv_gg = []
conv_gg_badd = []
# for i in range(N_sigmau):
#     c = 1
#     d = 0
#     conv_vg_sigma_u = []
#     for j in range(8):
        
#         data_file_conv_vg = str('/Volumes/Work/UQ/SDSS_dists/data/wide_angle_covariance_k0p002_0p150_gridcorr30_dv_%d_%d_sigmau%03d.dat' %(c, d, int(10.0*i)))
#         # data_file_conv_vg = str('wide_angle_covariance_k0p002_0p150_gridcorr30_dv_%d_%d_sigmau%03d.dat' %(c, d, int(10.0*i)))
#         print(data_file_conv_vg)
#         conv_vg_element = np.array(pd.read_csv(data_file_conv_vg, delim_whitespace=True, header=None, skiprows=1))/1.0e6
#         conv_vg_final = np.delete(np.delete(conv_vg_element, remove_velocity, axis = 0), remove_galaxy, axis = 1)
#         conv_vg_sigma_u.append(conv_vg_final)
#         d += 2
#         if (d > 7):
#             c += 1
#             d = 0
#     conv_vg.append(conv_vg_sigma_u)
#     data_file_conv_vv = str('/Volumes/Work/UQ/SDSS_dists/data/wide_angle_covariance_k0p002_0p150_gridcorr30_vv_sigmau%03d.dat' %(int(10.0*i)))
#     # data_file_conv_vv = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_sigmau%03d.dat' %(int(10.0*i)))
#     conv_vv_element = np.array(pd.read_csv(data_file_conv_vv, delim_whitespace=True, header=None, skiprows=1))/1.0e6
#     conv_vv_final = np.delete(np.delete(conv_vv_element, remove_velocity, axis = 0), remove_velocity, axis = 1)
   
#     data_file_conv_vv_ng = str('/Volumes/Work/UQ/SDSS_dists/data/wide_angle_covariance_k0p002_0p150_gridcorr30_vv_ng_sigmau%03d.dat' %(int(10.0*i)))
#     # data_file_conv_vv_ng = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_ng_sigmau%03d.dat' %(int(10.0*i)))
#     conv_vv_ng_element = np.array(pd.read_csv(data_file_conv_vv_ng, delim_whitespace=True, header=None, skiprows=1))/1.0e6
#     conv_vv_ng_final = np.delete(np.delete(conv_vv_ng_element, remove_velocity, axis = 0), remove_velocity, axis = 1)
    
#     for k in range(ncomp_velocity):
#         conv_vv_final[k][k] += (conv_vv_ng_final[k][k] - conv_vv_final[k][k])/ngrid_SDSS_vel[k]
#     conv_vv.append([conv_vv_final])
#     print(i)

conv_vg_sigma_u = []
c = 1
d = 0
for j in range(8):
    # data_file_conv_vg = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_dv_%d_%d_sigmau%03d.dat' %(int(1000.0*kmax_velocity), c, d, int(10.0*sigma_u)))
    # data_file_conv_vg = str('wide_angle_covariance_k0p002_0p150_gridcorr30_dv_%d_%d_sigmau%03d.dat' %(c, d, int(10.0*sigma_u)))
    data_file_conv_vg = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr20_dv_%d_%d_sigmau%03d.dat' %(int(1000.0*kmax_velocity), c, d, int(10.0*sigma_u)))
    print(data_file_conv_vg)
    conv_vg_element = np.array(pd.read_csv(data_file_conv_vg, delim_whitespace=True, header=None, skiprows=1))/10**(8+d)
    conv_vg_final = np.delete(np.delete(conv_vg_element, remove_velocity, axis = 0), remove_galaxy, axis = 1)
    conv_vg_sigma_u.append(conv_vg_final)
    d += 2
    if (d > 7):
        c += 1
        d = 0
        
conv_vg.append(conv_vg_sigma_u)

# data_file_conv_vv = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_vv_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
# data_file_conv_vv = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_sigmau%03d.dat' %(int(10.0*sigma_u)))
data_file_conv_vv = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr20_vv_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
conv_vv_element = np.array(pd.read_csv(data_file_conv_vv, delim_whitespace=True, header=None, skiprows=1))/1.0e6
conv_vv_final = np.delete(np.delete(conv_vv_element, remove_velocity, axis = 0), remove_velocity, axis = 1)
   
# data_file_conv_vv_ng = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_vv_ng_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
# data_file_conv_vv_ng = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_ng_sigmau%03d.dat' %(int(10.0*sigma_u)))
data_file_conv_vv_ng = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr20_vv_ng_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
conv_vv_ng_element = np.array(pd.read_csv(data_file_conv_vv_ng, delim_whitespace=True, header=None, skiprows=1))/1.0e6
conv_vv_ng_final = np.delete(np.delete(conv_vv_ng_element, remove_velocity, axis = 0), remove_velocity, axis = 1)

for k in range(ncomp_velocity):
    conv_vv_final[k][k] += (conv_vv_ng_final[k][k] - conv_vv_final[k][k])/ngrid_SDSS_vel[k]
conv_vv.append([conv_vv_final])


a = 0
b = 0
for k in range(21):
    # data_file_conv_gg = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_dd_%d_%d.dat' %(int(1000.0*kmax_galaxy), b, a))
    # data_file_conv_gg = str('wide_angle_covariance_k0p002_0p150_gridcorr30_dd_'+str(b)+'_'+str(a)+'.dat')
    data_file_conv_gg = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr20_dd_%d_%d.dat' %(int(1000.0*kmax_galaxy), b, a))
    print(data_file_conv_gg)
    conv_gg_element = np.array(pd.read_csv(data_file_conv_gg, delim_whitespace=True, header=None, skiprows=1))/10**(8+a)
    conv_gg_final = (np.delete(np.delete(conv_gg_element, remove_galaxy, axis = 0), remove_galaxy, axis = 1)).astype('float64')
    conv_gg.append(conv_gg_final)
    if (k < 7):
        #data_file_conv_gg_badd = str('wide_angle_covariance_k0p150_0p999_gridcorr30_dd_'+str(b)+'_'+str(a)+'.dat')
        #data_file_conv_gg_badd = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_dd_%d_%d.dat' %(int(1000.0*kmax_galaxy), b, a))
        # data_file_conv_gg_badd = str('wide_angle_covariance_k0p%03d_0p%03d_gridcorr%d_dd_%d_%d.dat' % (int(1000.0*kmax_galaxy), int(1000.0*0.999), gridsize, b, a)) 
        data_file_conv_gg_badd = str('/data/s4479813/wide_angle_covariance_k0p%03d_0p%03d_gridcorr%d_dd_%d_%d.dat' % (int(1000.0*kmax_galaxy), int(1000.0*0.999), gridsize, b, a))         
        print(data_file_conv_gg_badd)
        conv_gg_badd_element = np.array(pd.read_csv(data_file_conv_gg_badd, delim_whitespace=True, header=None, skiprows=1))/10**(8+a)
        conv_gg_badd_final = (np.delete(np.delete(conv_gg_badd_element, remove_galaxy, axis = 0), remove_galaxy, axis = 1)).astype('float64')
        conv_gg_badd.append(conv_gg_badd_final)
        
    a += 2
    if (a > 13):
        a = 0
        b += 1
    
datagrid_vec = datagrid_vec[comp_velocity,:]

conv_pk_gal_gal_0 = 0
conv_pk_gal_gal_1 = 0
conv_pk_gal_gal_2 = 0

conv_pk_gal_gal_badd = 0 

n = 0
for i in range(7):
    conv_pk_gal_gal_0 += conv_gg[i]*sigma_g**n
    conv_pk_gal_gal_1 += conv_gg[i+7]*sigma_g**n
    conv_pk_gal_gal_2 += conv_gg[i+14]*sigma_g**n
    conv_pk_gal_gal_badd += conv_gg_badd[i]*sigma_g**n
    n += 2


conv_pk_vel_gal_0 = 0
conv_pk_vel_gal_1 = 0
n = 0

for j in range(4):
    conv_pk_vel_gal_0 += conv_vg[0][j]*sigma_g**n
    conv_pk_vel_gal_1 += conv_vg[0][j+4]*sigma_g**n
    n += 2

# conv_pk_vel_gal_0 = conv_pk_vel_gal_0
# conv_pk_vel_gal_1 = conv_pk_vel_gal_1
conv_pk_vel_vel_0 = conv_vv[0][0]

# conv_pk_gal_vel_0 = conv_pk_vel_gal_0.T
# conv_pk_gal_vel_1 = conv_pk_vel_gal_1.T

# conv_pk_gal_gal = bsigma8_old**2*conv_pk_gal_gal_0 + bsigma8_old*fsigma8_old*conv_pk_gal_gal_1 + fsigma8_old**2*conv_pk_gal_gal_2
# conv_pk_vel_gal = bsigma8_old*fsigma8_old*conv_pk_vel_gal_0 + fsigma8_old**2*conv_pk_vel_gal_1
# conv_pk_gal_vel = bsigma8_old*fsigma8_old*conv_pk_gal_vel_0 + fsigma8_old**2*conv_pk_gal_vel_1
# conv_pk_vel_vel = fsigma8_old**2*conv_pk_vel_vel_0

# # conv_pk_gal_gal = bsigma8_old**2*conv_pk_gal_gal_0 + 0.0*conv_pk_gal_gal_1 + 0.0*conv_pk_gal_gal_2
# # conv_pk_vel_gal = bsigma8_old*fsigma8_old*conv_pk_vel_gal_0 + 0.0*conv_pk_vel_gal_1
# # conv_pk_gal_vel = bsigma8_old*fsigma8_old*conv_pk_gal_vel_0 + 0.0*conv_pk_gal_vel_1
# # conv_pk_vel_vel = fsigma8_old**2*conv_pk_vel_vel_0

# conv_pk_new_block_1 = np.concatenate((conv_pk_gal_gal, conv_pk_gal_vel), axis = 1)
# conv_pk_new_block_2 = np.concatenate((conv_pk_vel_gal, conv_pk_vel_vel), axis = 1)
# conv_pk_new = np.concatenate((conv_pk_new_block_1, conv_pk_new_block_2), axis = 0)

# Convert these to full matrices so that we can write model as a single sum
conv_vv = conv_vv[0]
conv_vg = conv_vg[0]
print(np.shape(conv_vv), np.shape(conv_vg), np.shape(conv_gg))

zeros_gg = np.zeros(np.shape(conv_gg[0]))
zeros_vg = np.zeros(np.shape(conv_vg[0]))
zeros_vv = np.zeros(np.shape(conv_vv[0]))
eye_vv = np.diag((datagrid_vec[0:,4]/np.sqrt(ngrid_SDSS_vel))**2)
conv_b = np.empty((len(datagrid_comp), len(datagrid_comp), 7))
conv_badd = np.empty((len(datagrid_comp), len(datagrid_comp), 7))
conv_bf = np.empty((len(datagrid_comp), len(datagrid_comp), 7))
conv_f = np.empty((len(datagrid_comp), len(datagrid_comp), 7))
for i in range(7):
    conv_b[:,:,i] = np.concatenate((np.concatenate((conv_gg[i], zeros_vg.T), axis=1), np.concatenate((zeros_vg, zeros_vv), axis=1)), axis = 0)
    conv_badd[:,:,i] = np.concatenate((np.concatenate((conv_gg_badd[i], zeros_vg.T), axis=1), np.concatenate((zeros_vg, zeros_vv), axis=1)), axis = 0)
conv_bf[:,:,0] = np.concatenate((np.concatenate((conv_gg[7], conv_vg[0].T), axis=1), np.concatenate((conv_vg[0], zeros_vv), axis=1)), axis = 0)
conv_bf[:,:,1] = np.concatenate((np.concatenate((conv_gg[8], conv_vg[1].T), axis=1), np.concatenate((conv_vg[1], zeros_vv), axis=1)), axis = 0)
conv_bf[:,:,2] = np.concatenate((np.concatenate((conv_gg[9], conv_vg[2].T), axis=1), np.concatenate((conv_vg[2], zeros_vv), axis=1)), axis = 0)
conv_bf[:,:,3] = np.concatenate((np.concatenate((conv_gg[10], conv_vg[3].T), axis=1), np.concatenate((conv_vg[3], zeros_vv), axis=1)), axis = 0)
conv_bf[:,:,4] = np.concatenate((np.concatenate((conv_gg[11], zeros_vg.T), axis=1), np.concatenate((zeros_vg, zeros_vv), axis=1)), axis = 0)
conv_bf[:,:,5] = np.concatenate((np.concatenate((conv_gg[12], zeros_vg.T), axis=1), np.concatenate((zeros_vg, zeros_vv), axis=1)), axis = 0)
conv_bf[:,:,6] = np.concatenate((np.concatenate((conv_gg[13], zeros_vg.T), axis=1), np.concatenate((zeros_vg, zeros_vv), axis=1)), axis = 0)
conv_f[:,:,0] = np.concatenate((np.concatenate((conv_gg[14], conv_vg[4].T), axis=1), np.concatenate((conv_vg[4], conv_vv[0]), axis=1)), axis = 0)
conv_f[:,:,1] = np.concatenate((np.concatenate((conv_gg[15], conv_vg[5].T), axis=1), np.concatenate((conv_vg[5], zeros_vv), axis=1)), axis = 0)
conv_f[:,:,2] = np.concatenate((np.concatenate((conv_gg[16], conv_vg[6].T), axis=1), np.concatenate((conv_vg[6], zeros_vv), axis=1)), axis = 0)
conv_f[:,:,3] = np.concatenate((np.concatenate((conv_gg[17], conv_vg[7].T), axis=1), np.concatenate((conv_vg[7], zeros_vv), axis=1)), axis = 0)
conv_f[:,:,4] = np.concatenate((np.concatenate((conv_gg[18], zeros_vg.T), axis=1), np.concatenate((zeros_vg, zeros_vv), axis=1)), axis = 0)
conv_f[:,:,5] = np.concatenate((np.concatenate((conv_gg[19], zeros_vg.T), axis=1), np.concatenate((zeros_vg, zeros_vv), axis=1)), axis = 0)
conv_f[:,:,6] = np.concatenate((np.concatenate((conv_gg[20], zeros_vg.T), axis=1), np.concatenate((zeros_vg, zeros_vv), axis=1)), axis = 0)
conv_sigma_v = np.concatenate((np.concatenate((zeros_gg, zeros_vg.T), axis=1), np.concatenate((zeros_vg, eye_vv), axis=1)), axis = 0)
conv_noise = np.concatenate((np.concatenate((np.diag(1.0/data_expect), zeros_vg.T), axis=1), np.concatenate((zeros_vg, np.diag(errgrid_SDSS)), axis=1)), axis = 0)

if __name__ == "__main__":

    from scipy.optimize import differential_evolution

    bfid = 1.7/bsigma8_old
    ffid = 0.2/fsigma8_old
    sigmag_fid = sigma_g
    baddfid = 0.5/bsigma8_old
    sigma_v_fid = 300.0

    params_fid = np.array([ffid, sigma_v_fid, bfid, baddfid, sigmag_fid])
    converged = 1
    while(~converged):

        # Compute derivatives about fiducial values
        loglikefid, dL, d2L = get_dL(params_fid)
        # print(dL, np.sqrt(np.diag(np.linalg.inv(d2L))))

        """result = basinhopping(
            lambda *args: -lnpost(*args),
            [ffid, sigma_v_fid, bfid, baddfid],
            niter_success=10,
            niter=100,
            stepsize=0.01,
            minimizer_kwargs={
                "method": "Nelder-Mead",
                "tol": 1.0e-4,
                "options": {"maxiter": 40000, "xatol": 1.0e-4, "fatol": 1.0e-4},
            },
        )"""


        fvals, sigmavvals, bvals, baddvals, sigmagvals= (0.0, 1.0), (1.0, 1000.0), (0.0, 3.0), (0.0, 10.0), (0.0, 10.0)
        result = differential_evolution(lambda *args: -lnpost(*args), bounds=(fvals, sigmavvals, bvals, baddvals, sigmagvals), maxiter=10000, tol=1.0e-6)
        print("#-------------- Best-fit----------------")
        print(result["x"])

        params_fid_new = np.array([result["x"][0]/fsigma8_old, result["x"][1], result["x"][2]/bsigma8_old, result["x"][3]/bsigma8_old, result["x"][4]])
        converged = np.all(params_fid_new/params_fid - 1.0 < 1.0e-3)
        # converged = np.all(np.less(params_fid_new/params_fid - 1.0,  [1.0e-5, 1.0e-3, 1.0e-5, 1.0e-5]))

        params_fid = params_fid_new

    loglikefid, dL, d2L = get_dL(params_fid)
    
    # fsigma8 = params_fid[0]
    # bsigma8 = params_fid[2]
    # b_add_sigma8 = params_fid[3]

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    ndim, nwalkers = 5, 40
    
    # Set up the first points for the chain (for emcee we need to give each walker a random value about this point so we just sample the prior or a reasonable region)
    # begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 3.0*np.random.rand(), 25.0*np.random.rand(), 7.0*np.random.rand()+1.0] for i in range(nwalkers)]
    # begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 3.0*np.random.rand(), 7.0*np.random.rand()+1.0] for i in range(nwalkers)]
    #The possible values are 0<=fsigma8<=1, 0<=sigmav<=1000, 0<=bsigma8<=3.
    #begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 0.2] for i in range(nwalkers)]
    # begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 3.0*np.random.rand(), 10.0*np.random.rand()] for i in range(nwalkers)]

    begin = [[1.0*np.random.rand(), 999.0*np.random.rand()+1.0, 3.0*np.random.rand(), 10.0*np.random.rand(), 10.0*np.random.rand()] for i in range(nwalkers)]

    # begin = [[(0.01 * np.random.rand() + 0.995) * j for j in params_fid] for _ in range(nwalkers)]
    
    # begin = [[1000.0*np.random.rand()] for i in range(nwalkers)]
    
    # Set up the output file
    backend = backends.HDFBackend(chainfile)
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, backend=backend)
        
    # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
    # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
    # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
    max_iter = 20000
    index = 0
    old_tau = np.inf
    autocorr = np.empty(max_iter)
    counter = 0

    for sample in sampler.sample(begin, iterations=max_iter, progress=progress):
    
        # Only check convergence every 100 steps
        if sampler.iteration % 100:
            continue
    
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        counter += 100
        print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
        print("Mean Auto-Correlation time: {0:.3f}".format(autocorr[index]))
    
        # Check convergence
        converged = np.all(tau * 100 < sampler.iteration)
        converged &= np.all(np.abs(old_tau - tau) / tau < 1.0/100.0)
        if converged:
            print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
            break
        old_tau = tau
        index += 1

    samples = read_chain_backend(chainfile)[0]
    print(np.mean(samples, axis=0), np.std(samples, axis=0))
    
    # with Pool(processes=process_num) as pool:
    
    #     sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, pool=pool, backend=backend)
        
    #     # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
    #     # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
    #     # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
    #     max_iter = 20000
    #     index = 0
    #     old_tau = np.inf
    #     autocorr = np.empty(max_iter)
    #     counter = 0
    #     for sample in sampler.sample(begin, iterations=max_iter, progress=progress):
        
    #         # Only check convergence every 100 steps
    #         if sampler.iteration % 100:
    #             continue
        
    #         # Compute the autocorrelation time so far
    #         # Using tol=0 means that we'll always get an estimate even
    #         # if it isn't trustworthy
    #         tau = sampler.get_autocorr_time(tol=0)
    #         autocorr[index] = np.mean(tau)
    #         counter += 100
    #         print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
    #         print("Mean Auto-Correlation time: {0:.3f}".format(autocorr[index]))
        
    #         # Check convergence
    #         converged = np.all(tau * 100 < sampler.iteration)
    #         converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    #         if converged:
    #             print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
    #             break
    #         old_tau = tau
    #         index += 1

    # print(read_chain_backend(chainfile))