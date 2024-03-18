# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 09:43:46 2022

This code uses the new maximum likelihood fields method in ... to constrain the growth rate of structure. 

@author: Yan Xiang Lai
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
from configobj import ConfigObj

# Speed of light in km/s
LightSpeed = 299792.458

def get_max_dL(params_fid):
    """
    This function finds the loglikelihood given fiducial values. 

    Parameters
    ----------
    params_fid : Numpy array.
        This array contains the fiducial value for fsigma8, sigma_v, bsigma8, b_addsigma8 and sigma_g.

    Raises
    ------
    ValueError
        If the input covariance matrix gives negative chi-squared value, a value error will be raised.

    Returns
    -------
    loglikefid : float
        The fiducial log likelihood value.

    """
    
    #Read in the fiducial values. 
    ffid, sigma_v_fid, bfid, baddfid, sigmag_fid = params_fid
    
    #Construct the analytical covariance matrix with the fiducial free parameters. 
    
    #Compute the power of sigma_g that will be used to calculate the covariance matrix. Cross-reference equation (15) in the paper. Notice here 
    #we only calculate up to n = 6 because we only use the first three order of the Taylor expansion. So if you are using first m order of 
    #Taylor expansion, you should set the maximum n to 2m. 
    sigmag_array = np.array([sigmag_fid**(2*n) for n in range(0,7)])
    
    #Construct the covariance matrix according to equation (37) in the paper. 
    Cfid = np.sum(sigmag_array*(bfid**2*conv_b + bfid*ffid*conv_bf + ffid**2*conv_f + baddfid**2*conv_badd), axis=-1) + sigma_v_fid**2*conv_sigma_v + conv_noise
    
    #Find the log of the determinant of the fiducial covariance matrix.
    detCfid = np.linalg.slogdet(Cfid)[1]
    
    #Find the extra factors due to the zero-point correction. Equation (41) in the paper. x2 here is equivalent to the vector x in the paper.  
    x2_Cinv = np.linalg.solve(Cfid, x_2) 
    
    chi_sq_x2 = np.matmul(x_2.T, x2_Cinv)[0][0]
    #factor_u is Nx*sigma_y in the paper. 
    factor_u = 1.0 + chi_sq_x2*sigmab_square
    factor_1 = np.log(factor_u)
    
    #This is equivalent to Ny in the paper. 
    factor_a = np.matmul(datagrid_comp_new.T, x2_Cinv)[0][0]
    #This is equivalent to Nx^2 in the paper. 
    factor_b = chi_sq_x2 + sigmab_square**(-1)
    factor_3 = -factor_a**2/factor_b
    
    
    #Find the fiducial chi-squared and the fiducial log likelihood.
    chisquared_fid = np.matmul(datagrid_comp.T, np.linalg.solve(Cfid, datagrid_comp))
    if (chisquared_fid < 0):
        raise ValueError('Negative chi-squared. Check whether your covariance matrix is positive semi-definite')
        
    #This is the loglikelihood after marginalizing over the zero-point.  
    loglikefid = -0.5*(chisquared_fid + detCfid + factor_1 + factor_3)
    
    return loglikefid

def get_dL(params_fid):
    """
    This function finds the derivative and the second derivative of the likelihood function with respect to fsigma8, sigma_v, bsigma8, b_addsigma8, and sigma_g at their 
    given fiducial values. 

    Parameters
    ----------
    params_fid : Numpy array.
        This array contains the fiducial value for fsigma8, sigma_v, bsigma8, b_addsigma8 and sigma_g.

    Raises
    ------
    ValueError
        If the input covariance matrix gives negative chi-squared value, a value error will be raised.

    Returns
    -------
    loglikefid : float
        The fiducial log likelihood value.
    dL : list
        The derivative of the analytical covariance matrix with respect to the free parameters.
    d2L : list
        The second derivative of the analytical covariance matrix with respect to the free parameters.

    """
    
    #The first half of this function (up to log likelihood calculation) is the same as the previous function. Check the comment for more detail. 
    
    #Read in the fiducial values. 
    ffid, sigma_v_fid, bfid, baddfid, sigmag_fid = params_fid
    
    #Construct the analytical covariance matrix with the fiducial free parameters. 
    sigmag_array = np.array([sigmag_fid**(2*n) for n in range(0,7)])
    #The first derivative with respect to sigmag. 
    dsigmag_array = np.array([2.0*n*sigmag_fid**(2*n-1) for n in range(0,7)])
    
    Cfid = np.sum(sigmag_array*(bfid**2*conv_b + bfid*ffid*conv_bf + ffid**2*conv_f + baddfid**2*conv_badd), axis=-1) + sigma_v_fid**2*conv_sigma_v + conv_noise
    
    #Find the log determinant of the fiducial covariance matrix.
    detCfid = np.linalg.slogdet(Cfid)[1]
    
    #Find the inverse of the fiducial covariance matrix.
    Cfid_inv = np.linalg.solve(Cfid, np.eye(int(ncomp_velocity + ncomp_galaxy)))
    
    #Find the extra factors due to the zero-point correction. 
    x2_Cinv = np.matmul(Cfid_inv, x_2)
    
    
    chi_sq_x2 = np.matmul(x_2.T, x2_Cinv)[0][0]
    factor_u = 1.0 + chi_sq_x2*sigmab_square
    
    factor_x2 = -np.matmul(x2_Cinv, x2_Cinv.T)
    factor_1 = np.log(factor_u)
    factor_1_derivative = 1.0/(factor_u)*sigmab_square*factor_x2
    
    factor_a = np.matmul(datagrid_comp_new.T, x2_Cinv)[0][0]
    factor_b = chi_sq_x2 + sigmab_square**(-1)
    da_dC = np.matmul(np.matmul(-Cfid_inv, datagrid_comp_new), x2_Cinv.T)
    db_dC = factor_x2
    #This is -Ny^2/Nx^2
    factor_3 = -factor_a**2/factor_b
    factor_3_derivative = (-2.0*factor_a*da_dC*factor_b + db_dC*factor_a**2)/factor_b**2
    
    #Find the fiducial chi-squared and the fiducial log likelihood.
    chisquared_fid = datagrid_comp @ Cfid_inv @ datagrid_comp
    # chisquared_fid = np.matmul(datagrid_comp.T, np.linalg.solve(Cfid, datagrid_comp))
    if (chisquared_fid < 0):
        raise ValueError('Negative chi-squared')
    # loglikefid = -0.5*(chisquared_fid + detCfid + factor_1[0][0] + factor_2[0][0] + factor_3[0][0])
    loglikefid = -0.5*(chisquared_fid + detCfid + factor_1 + factor_3)
    
    # Derivatices of the analytical covariance matrix with respect to each free parameter. 
    dCdb = np.sum(sigmag_array*(2.0*bfid*conv_b + ffid*conv_bf), axis=-1)
    dCdbadd = np.sum(sigmag_array*(2.0*baddfid*conv_badd), axis=-1)
    dCdf = np.sum(sigmag_array*(bfid*conv_bf + 2.0*ffid*conv_f), axis=-1)
    dCdsigmag = np.sum(dsigmag_array*(bfid**2*conv_b + bfid*ffid*conv_bf + ffid**2*conv_f + baddfid**2*conv_badd), axis=-1)
    dCdsigmav = 2.0*sigma_v_fid*conv_sigma_v
    
    #Here are the components required to construct the first and second derivative of the log likelihood function. Check appendix D of the paper 
    #for more detail. 
    dCdb_x2_Cinv = np.matmul(dCdb, x2_Cinv)
    dCdb_Cfid_inv = np.matmul(dCdb, Cfid_inv)
    # dCdb_Cfid_inv = np.linalg.solve(Cfid, dCdb)
    dCdbadd_x2_Cinv = np.matmul(dCdbadd, x2_Cinv)
    dCdbadd_Cfid_inv = np.matmul(dCdbadd, Cfid_inv)
    # dCdbadd_Cfid_inv = np.linalg.solve(Cfid, dCdbadd)
    dCdf_x2_Cinv = np.matmul(dCdf, x2_Cinv)
    dCdf_Cfid_inv = np.matmul(dCdf, Cfid_inv)
    # dCdf_Cfid_inv = np.linalg.solve(Cfid, dCdf)
    dCdsigmag_x2_Cinv = np.matmul(dCdsigmag, x2_Cinv)
    dCdsigmag_Cfid_inv = np.matmul(dCdsigmag, Cfid_inv)
    # dCdsigmag_Cfid_inv = np.linalg.solve(Cfid, dCdsigmag)
    dCdsigmav_x2_Cinv = np.matmul(dCdsigmav, x2_Cinv)
    dCdsigmav_Cfid_inv = np.matmul(dCdsigmav, Cfid_inv)
    # dCdsigmav_Cfid_inv = np.linalg.solve(Cfid, dCdsigmav)
    
    dCdb_t6 = np.matmul(Cfid_inv, dCdb_x2_Cinv)
    dCdb_t7 = dCdb_t6.T
    dCdbadd_t6 = np.matmul(Cfid_inv, dCdbadd_x2_Cinv)
    dCdbadd_t7 = dCdbadd_t6.T
    dCdf_t6 = np.matmul(Cfid_inv, dCdf_x2_Cinv)
    dCdf_t7 = dCdf_t6.T
    dCdsigmag_t6 = np.matmul(Cfid_inv, dCdsigmag_x2_Cinv)
    dCdsigmag_t7 = dCdsigmag_t6.T
    dCdsigmav_t6 = np.matmul(Cfid_inv, dCdsigmav_x2_Cinv)
    dCdsigmav_t7 = dCdsigmav_t6.T
    
    t5 = sigmab_square/factor_u
    
    # print(np.shape(x_2.T), np.shape(dCdb_t6), np.shape(factor_x2), np.shape(x2_Cinv), np.shape(dCdb_t7))
    
    t8_b = np.matmul(x_2.T, dCdb_t6)[0][0]
    t8_badd = np.matmul(x_2.T, dCdbadd_t6)[0][0]
    t8_f = np.matmul(x_2.T, dCdf_t6)[0][0]
    t8_sigmag = np.matmul(x_2.T, dCdsigmag_t6)[0][0]
    t8_sigmav = np.matmul(x_2.T, dCdsigmav_t6)[0][0]
    
    SD_dCdb_1 = -0.5*(-(sigmab_square**2/factor_u**2*t8_b*(-factor_x2)-(t5*np.matmul(x2_Cinv, dCdb_t7) + t5*np.matmul(dCdb_t6, x2_Cinv.T))))
    SD_dCdbadd_1 = -0.5*(-(sigmab_square**2/factor_u**2*t8_badd*(-factor_x2)-(t5*np.matmul(x2_Cinv, dCdbadd_t7) + t5*np.matmul(dCdbadd_t6, x2_Cinv.T))))
    SD_dCdf_1 = -0.5*(-(sigmab_square**2/factor_u**2*t8_f*(-factor_x2)-(t5*np.matmul(x2_Cinv, dCdf_t7) + t5*np.matmul(dCdf_t6, x2_Cinv.T))))
    SD_dCdsigmag_1 = -0.5*(-(sigmab_square**2/factor_u**2*t8_sigmag*(-factor_x2)-(t5*np.matmul(x2_Cinv, dCdsigmag_t7) + t5*np.matmul(dCdsigmag_t6, x2_Cinv.T))))
    SD_dCdsigmav_1 = -0.5*(-(sigmab_square**2/factor_u**2*t8_sigmav*(-factor_x2)-(t5*np.matmul(x2_Cinv, dCdsigmav_t7) + t5*np.matmul(dCdsigmav_t6, x2_Cinv.T))))
    
    # SD_dCdb_1 = -0.5*(-(sigmab_square**2/factor_u**2*np.matmul(x_2.T, np.matmul(dCdb_t6, -factor_x2))-(t5*np.matmul(x2_Cinv, dCdb_t7) + t5*np.matmul(dCdb_t6, x2_Cinv.T))))
    # SD_dCdbadd_1 = -0.5*(-(sigmab_square**2/factor_u**2*np.matmul(x_2.T, np.matmul(dCdbadd_t6, -factor_x2))-(t5*np.matmul(x2_Cinv, dCdbadd_t7) + t5*np.matmul(dCdbadd_t6, x2_Cinv.T))))
    # SD_dCdf_1 = -0.5*(-(sigmab_square**2/factor_u**2*np.matmul(x_2.T, np.matmul(dCdf_t6, -factor_x2))-(t5*np.matmul(x2_Cinv, dCdf_t7) + t5*np.matmul(dCdf_t6, x2_Cinv.T))))
    # SD_dCdsigmag_1 = -0.5*(-(sigmab_square**2/factor_u**2*np.matmul(x_2.T, np.matmul(dCdsigmag_t6, -factor_x2))-(t5*np.matmul(x2_Cinv, dCdsigmag_t7) + t5*np.matmul(dCdsigmag_t6, x2_Cinv.T))))
    # SD_dCdsigmav_1 = -0.5*(-(sigmab_square**2/factor_u**2*np.matmul(x_2.T, np.matmul(dCdsigmav_t6, -factor_x2))-(t5*np.matmul(x2_Cinv, dCdsigmav_t7) + t5*np.matmul(dCdsigmav_t6, x2_Cinv.T))))
    
    t13 = np.matmul(datagrid_comp_new.T, Cfid_inv)
    dCdb_t4 = np.matmul(Cfid_inv, dCdb_x2_Cinv)
    dCdb_t5 = np.matmul(x_2.T, dCdb_t4)[0][0]
    dCdb_t12 = np.matmul(datagrid_comp_new.T, dCdb_t4)[0][0]
    dCdb_t17 = dCdb_t4.T
    dCdb_t18 = np.matmul(t13, dCdb_Cfid_inv)
    dCdbadd_t4 = np.matmul(Cfid_inv, dCdbadd_x2_Cinv)
    dCdbadd_t5 = np.matmul(x_2.T, dCdbadd_t4)[0][0]
    dCdbadd_t12 = np.matmul(datagrid_comp_new.T, dCdbadd_t4)[0][0]
    dCdbadd_t17 = dCdbadd_t4.T
    dCdbadd_t18 = np.matmul(t13, dCdbadd_Cfid_inv)
    dCdf_t4 = np.matmul(Cfid_inv, dCdf_x2_Cinv)
    dCdf_t5 = np.matmul(x_2.T, dCdf_t4)[0][0]
    dCdf_t12 = np.matmul(datagrid_comp_new.T, dCdf_t4)[0][0]
    dCdf_t17 = dCdf_t4.T
    dCdf_t18 = np.matmul(t13, dCdf_Cfid_inv)
    dCdsigmag_t4 = np.matmul(Cfid_inv, dCdsigmag_x2_Cinv)
    dCdsigmag_t5 = np.matmul(x_2.T, dCdsigmag_t4)[0][0]
    dCdsigmag_t12 = np.matmul(datagrid_comp_new.T, dCdsigmag_t4)[0][0]
    dCdsigmag_t17 = dCdsigmag_t4.T
    dCdsigmag_t18 = np.matmul(t13, dCdsigmag_Cfid_inv)
    dCdsigmav_t4 = np.matmul(Cfid_inv, dCdsigmav_x2_Cinv)
    dCdsigmav_t5 = np.matmul(x_2.T, dCdsigmav_t4)[0][0]
    dCdsigmav_t12 = np.matmul(datagrid_comp_new.T, dCdsigmav_t4)[0][0]
    dCdsigmav_t17 = dCdsigmav_t4.T
    dCdsigmav_t18 = np.matmul(t13, dCdsigmav_Cfid_inv)
    # print(np.shape(dCdb_t4), np.shape(dCdb_t5), np.shape(dCdb_t12), np.shape(dCdb_t17), np.shape(dCdb_t18))
    
    t6 = factor_a**2
    t7 = factor_b**2
    t9 = t6/t7
    t10 = 2*factor_a/t7
    t15 = 2.0/factor_b
    t16 = t15*factor_a
    
    SD_dCdb_2 = -0.5*(-((2.0*dCdb_t5*t6)/factor_b**3*(-factor_x2) - t10*dCdb_t5*(-da_dC.T) - t9*np.matmul(dCdb_t4, x2_Cinv.T) - t9*np.matmul(x2_Cinv, dCdb_t17) 
    -(t10*dCdb_t12*(-factor_x2) -dCdb_t12*t15*(-da_dC.T)-t16*np.matmul(dCdb_t4, t13) - t16*np.matmul(x2_Cinv, dCdb_t18))))
    SD_dCdbadd_2 = -0.5*(-((2.0*dCdbadd_t5*t6)/factor_b**3*(-factor_x2) - t10*dCdbadd_t5*(-da_dC.T) - t9*np.matmul(dCdbadd_t4, x2_Cinv.T) - t9*np.matmul(x2_Cinv, dCdbadd_t17) 
    -(t10*dCdbadd_t12*(-factor_x2) -dCdbadd_t12*t15*(-da_dC.T)-t16*np.matmul(dCdbadd_t4, t13) - t16*np.matmul(x2_Cinv, dCdbadd_t18))))
    SD_dCdf_2 = -0.5*(-((2.0*dCdf_t5*t6)/factor_b**3*(-factor_x2) - t10*dCdf_t5*(-da_dC.T) - t9*np.matmul(dCdf_t4, x2_Cinv.T) - t9*np.matmul(x2_Cinv, dCdf_t17) 
    -(t10*dCdf_t12*(-factor_x2) -dCdf_t12*t15*(-da_dC.T)-t16*np.matmul(dCdf_t4, t13) - t16*np.matmul(x2_Cinv, dCdf_t18))))
    SD_dCdsigmag_2 = -0.5*(-((2.0*dCdsigmag_t5*t6)/factor_b**3*(-factor_x2) - t10*dCdsigmag_t5*(-da_dC.T) - t9*np.matmul(dCdsigmag_t4, x2_Cinv.T) - t9*np.matmul(x2_Cinv, dCdsigmag_t17) 
    -(t10*dCdsigmag_t12*(-factor_x2) -dCdsigmag_t12*t15*(-da_dC.T)-t16*np.matmul(dCdsigmag_t4, t13) - t16*np.matmul(x2_Cinv, dCdsigmag_t18))))
    SD_dCdsigmav_2 = -0.5*(-((2.0*dCdsigmav_t5*t6)/factor_b**3*(-factor_x2) - t10*dCdsigmav_t5*(-da_dC.T) - t9*np.matmul(dCdsigmav_t4, x2_Cinv.T) - t9*np.matmul(x2_Cinv, dCdsigmav_t17) 
    -(t10*dCdsigmav_t12*(-factor_x2) -dCdsigmav_t12*t15*(-da_dC.T)-t16*np.matmul(dCdsigmav_t4, t13) - t16*np.matmul(x2_Cinv, dCdsigmav_t18))))
    
    SD_dC_db2 = np.trace(np.matmul(SD_dCdb_1, dCdb)) + np.trace(np.matmul(SD_dCdb_2, dCdb))
    SD_dC_dbadd2 = np.trace(np.matmul(SD_dCdbadd_1, dCdbadd)) + np.trace(np.matmul(SD_dCdbadd_2, dCdbadd))
    SD_dC_df2 = np.trace(np.matmul(SD_dCdf_1, dCdf)) + np.trace(np.matmul(SD_dCdf_2, dCdf))
    SD_dC_dsigmag2 = np.trace(np.matmul(SD_dCdsigmag_1, dCdb)) + np.trace(np.matmul(SD_dCdsigmag_2, dCdb))
    SD_dC_dsigmav2 = np.trace(np.matmul(SD_dCdsigmav_1, dCdsigmav)) + np.trace(np.matmul(SD_dCdsigmav_2, dCdsigmav))
    
    SD_dC_db_dbadd = np.trace(np.matmul(SD_dCdb_1, dCdbadd)) + np.trace(np.matmul(SD_dCdb_2, dCdbadd))
    SD_dC_db_df = np.trace(np.matmul(SD_dCdb_1, dCdf)) + np.trace(np.matmul(SD_dCdb_2, dCdf))
    SD_dC_db_dsigmag = np.trace(np.matmul(SD_dCdb_1, dCdsigmag)) + np.trace(np.matmul(SD_dCdb_2, dCdsigmag))
    SD_dC_db_dsigmav = np.trace(np.matmul(SD_dCdb_1, dCdsigmav)) + np.trace(np.matmul(SD_dCdb_2, dCdsigmav))
    SD_dC_df_dbadd = np.trace(np.matmul(SD_dCdf_1, dCdbadd)) + np.trace(np.matmul(SD_dCdf_2, dCdbadd))
    SD_dC_df_dsigmag = np.trace(np.matmul(SD_dCdf_1, dCdsigmag)) + np.trace(np.matmul(SD_dCdf_2, dCdsigmag))
    SD_dC_df_dsigmav = np.trace(np.matmul(SD_dCdf_1, dCdsigmav)) + np.trace(np.matmul(SD_dCdf_2, dCdsigmav))
    SD_dC_dbadd_dsigmav = np.trace(np.matmul(SD_dCdbadd_1, dCdsigmav)) + np.trace(np.matmul(SD_dCdbadd_2, dCdsigmav))
    SD_dC_dbadd_dsigmag = np.trace(np.matmul(SD_dCdbadd_1, dCdsigmag)) + np.trace(np.matmul(SD_dCdbadd_2, dCdsigmag))
    SD_dC_dsigmav_dsigmag = np.trace(np.matmul(SD_dCdsigmav_1, dCdsigmag)) + np.trace(np.matmul(SD_dCdsigmav_2, dCdsigmag))
    
    
    # Derivatives of loglikelihood function with respect to the analytical covariance matrix. The formula is obtained from the Matrix Cookbook. 
    weirdbit = np.matmul(t13.T, t13)
    
    #The first derivative of the log likelihood with respect to the covariance matrix. 
    dLdC = -0.5*(Cfid_inv - weirdbit + factor_1_derivative + factor_3_derivative)
    
    #The first derivative of the likelihood function with respect to each free parameter. 
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
    
    # The second derivative of the likelihood function with respect to each free parameter. The formula is obtained from the Matrix Cookbook. 
    d2Ldb2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdb, dCdb)) - SD_dC_db2
    d2Ldbdbadd = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdb, dCdbadd)) - SD_dC_db_dbadd
    d2Ldbdf = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdb, dCdf)) - SD_dC_db_df
    d2Ldbdsigmav = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdb, dCdsigmav)) - SD_dC_db_dsigmav
    d2Ldbdsigmag = 0.5*np.sum(np.einsum('ij, ji->i', Fisher_dCdb, dCdsigmag)) - SD_dC_db_dsigmag
    d2Ldf2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdf, dCdf)) - SD_dC_df2
    d2Ldfdbadd = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdf, dCdbadd)) - SD_dC_df_dbadd
    d2Ldfdsigmav = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdf, dCdsigmav)) - SD_dC_df_dsigmav
    d2Ldfdsigmag = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdf, dCdsigmag)) - SD_dC_df_dsigmag
    d2Ldbadd2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdbadd, dCdbadd)) - SD_dC_dbadd2
    d2Ldbadddsigmav = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdbadd, dCdsigmav)) - SD_dC_dbadd_dsigmav
    d2Ldbadddsigmag = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdbadd, dCdsigmag)) - SD_dC_dbadd_dsigmag
    d2Ldsigmav2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdsigmav, dCdsigmav)) - SD_dC_dsigmav2
    d2Ldsigmavdsigmag = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdsigmav, dCdsigmag)) - SD_dC_dsigmav_dsigmag
    d2Ldsigmag2 = 0.5*np.sum(np.einsum('ij,ji->i', Fisher_dCdsigmag, dCdsigmag)) - SD_dC_dsigmag2
    
    
    dL = np.array([dLdf, dLdsigmav, dLdb, dLdbadd, dLdsigmag])
    
    d2L = np.array([[d2Ldf2, d2Ldfdsigmav, d2Ldbdf, d2Ldfdbadd, d2Ldfdsigmag], 
            [d2Ldfdsigmav, d2Ldsigmav2, d2Ldbdsigmav, d2Ldbadddsigmav, d2Ldsigmavdsigmag], 
            [d2Ldbdf, d2Ldbdsigmav, d2Ldb2, d2Ldbdbadd, d2Ldbdsigmag], 
            [d2Ldfdbadd, d2Ldbadddsigmav, d2Ldbdbadd,  d2Ldbadd2, d2Ldbadddsigmag],
            [d2Ldfdsigmag, d2Ldsigmavdsigmag, d2Ldbdsigmag, d2Ldbadddsigmag, d2Ldsigmag2]])


    return loglikefid, dL, d2L

#This function generates a file to store the chain of the MCMC process. 
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

    
# def linear_interpolation(conv_pk_array, sigma_u, num):
#     output = []
#     for i in range(num):
#         index_low = int(math.floor(sigma_u))
#         index_high = index_low + 1
        
#         correction = sigma_u - index_low
        
#         output.append((1.0-correction)*conv_pk_array[index_low][i] + correction*conv_pk_array[index_high][i])
        
#     return output
        
#This function calculates the log of the posterior probability. 
def lnpost(params):

    # This returns the log posterior distribution which is given by the log prior plus the log likelihood
    prior = lnprior(params)
    if not np.isfinite(prior):
        return -np.inf
    like = lnlike(params)
    # print(prior, like)
    return prior + like

#This function sets the prior for each free parameter. 
def lnprior(params):

    # Here we define the prior for all the parameters.
#    fsigma8, sigma_v, bsigma8, b_add_sigma8, sigma_u = params
    # fsigma8, sigma_v, bsigma8, sigma_u, sigma_g = params
    # fsigma8, sigma_v, bsigma8, sigma_g = params
    # fsigma8, sigma_v, bsigma8, b_add_sigma8 = params
    fsigma8, sigma_v, bsigma8, b_add_sigma8, sigma_g = params

    
    #We use flat prior for all the free parameters. You can change this based on your situation. 
    
    # Don't allow fsigma8 to be less that 0.0
    if (0.0 <= fsigma8 <= 1.0):
        fsigma8prior = 1.0 
    else:
        return -np.inf

    # Flat prior for sigma_v between 0.0 and 1000.0 km/s (from Andrews paper)
    if (1.0 < sigma_v <= 5000.0):
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

#This function calculates the log-likehood of the input parameters. 
def lnlike(params): 

    # Return the log likelihood for a model. Here are the parameters we want to fit for.
    # fsigma8, sigma_v, bsigma8, b_add_sigma8 = params
    fsigma8, sigma_v, bsigma8, b_add_sigma8, sigma_g = params
    
    #Find the cloest point where the first and second derivative of the log likelihood function is calculated. 
    index = np.int32(np.floor(fsigma8/step_N + 0.5))
    # print(index)
    params_fiducial = np.array([fsigma8_diff[index], params_fid[1], params_fid[2], params_fid[3], params_fid[4]])
    #Read in the calculated fiducial log likelihood and its first and second derivative. 
    loglikefid = loglikefid_all[index]
    dL = dL_all[index]
    d2L = d2L_all[index]
    # sigma_v = params
    
    # Calculate the log likelihood using a taylor expansion at the maximum likihood point. 
    diffs = np.array([fsigma8/fsigma8_old, sigma_v, bsigma8/bsigma8_old, b_add_sigma8/bsigma8_old, sigma_g]) - params_fiducial
    
    #Calculating the log likelihood with the Taylor expansion. 
    loglike = loglikefid + np.sum(dL*diffs) - 0.5 * diffs @ d2L @ diffs
    
    return loglike


########################################################################################################### The main code below
configfile = sys.argv[1] #input the location of the configuration file 
pardict = ConfigObj(configfile)

expect_file = pardict['expect_file']
datafile = pardict['datafile']
omega_m = float(pardict['omega_m'])
fsigma8_old = float(pardict['fsigma8_old'])
sigmab_square = float(pardict['sigma_b'])**2
bsigma8_old = float(pardict['bsigma8_old'])
r_g = float(pardict['r_g'])
kmin = float(pardict['kmin'])
kmax_galaxy = float(pardict['kmax_galaxy'])
kmax_velocity = float(pardict['kmax_velocity'])
gridsize = int(pardict['gridsize'])
progress = bool(int(pardict['progress']))
sigma_u = float(pardict['sigma_u'])
sigma_g = float(pardict['sigma_g'])
effective_redshift = float(pardict['effective_redshift'])
sigma8_eff = float(pardict['sigma8_eff'])
xmin = float(pardict['xmin'])
xmax = float(pardict['xmax'])
ymin = float(pardict['ymin'])
ymax = float(pardict['ymax'])
zmin = float(pardict['zmin'])
zmax = float(pardict['zmax'])

#The name of the chain file. 
chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_full_data_sigmau_%d_Taylor_local_remove_no_cut.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, sigma_u))


nx = int(np.ceil((xmax-xmin)/gridsize))
ny = int(np.ceil((ymax-ymin)/gridsize))
nz = int(np.ceil((zmax-zmin)/gridsize))
nelements = nx*ny*nz
print(nx, ny, nz)

# Compute some useful quantities on the grid. For this we need a redshift-dist lookup table with scipy interpolation.
nbins = 5000
#The maximum redshift. 
redmax = 0.5
red = np.empty(nbins)
dist = np.empty(nbins)
for i in range(nbins):
    red[i] = i*redmax/nbins
    dist[i] = DistDc(red[i], omega_m, 1.0-omega_m, 0.0, 100.0, -1.0, 0.0, 0.0)
red_spline = sp.interpolate.splrep(dist, red, s=0) #Interpolate distance to get redshift. 
radial_spline=sp.interpolate.splrep(red, dist, s=0) #Interpolate redshift to get comoving radial distance. 

#This array stores the position, radial distance and the conversion factor (equation 23) from peculiar velocity to log-distance ratio for each grid. 
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

#These are the extra factors required to calculate constrain the growth rate at the effective redshift. If at redshift zero, all three factors
#should be 1. 
factor_gg = sigma8_eff**2/0.8150**2
factor_gv = (1.0/(1.0+effective_redshift))*Ez(effective_redshift, omega_m, 1.0-omega_m, 0.0, -1.0, 0.0, 0.0)*factor_gg
factor_vv = ((1.0/(1.0+effective_redshift))*Ez(effective_redshift, omega_m, 1.0-omega_m, 0.0, -1.0, 0.0, 0.0))**2*factor_gg
print(factor_gg, factor_gv, factor_vv)
            
# Read in the random file
data_expect_all = np.array(pd.read_csv(expect_file, header=None, skiprows=1))

#convert from degree to radian. 
#The first three coloums of the random file should be RA, Dec, and redshift. 
RA_expect = data_expect_all[:, 0]/180.0*np.pi
Dec_expect = data_expect_all[:, 1]/180.0*np.pi
redshift_expect = data_expect_all[:, 2]
rd_expect = sp.interpolate.splev(redshift_expect, radial_spline)
#This is the extra rotation for the SDSS PV catalogue. 
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

#Determine which grid cell the galaxy belongs to in the random catalogue. 
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

#Read in the data file. 
data = dict(pd.read_csv(datafile, delim_whitespace = True, skiprows=0))

RA = np.array(data["RA"])/180.0*np.pi
Dec = np.array(data["Dec"])/180.0*np.pi
redshift = np.array(data["zcmb"])

#Convert redshift to distance. 
rd = sp.interpolate.splev(redshift, radial_spline)

#Same rotation as the random data. 
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


#Read in the log-distance ratios and their errors. 
log_dist = np.array(data["logdist_corr"])
log_dist_err = np.array(data["logdist_corr_err"])

data_count = len(x)

#Reshape the data so I can concatenate them together later. 
x = np.reshape(x, (data_count, 1))
y = np.reshape(y, (data_count, 1))
z = np.reshape(z, (data_count, 1))

log_dist = np.reshape(log_dist, (data_count, 1))
log_dist_err = np.reshape(log_dist_err, (data_count, 1))

data_SDSS_all = np.concatenate((x,y,z,log_dist,log_dist_err), axis=1)

#Just checking all the data is within the grid we defined. 
print(np.min(x), np.max(x), np.min(y), np.max(y), np.min(z), np.max(z))

#Cutting out data that are more than 10 sigma away from the mean. 
data_SDSS = []
median_log_dist = np.median(log_dist)
for i in range(len(data_SDSS_all)):
    sigma = np.sqrt((data_SDSS_all[i, 3] - median_log_dist)**2/data_SDSS_all[i, 4]**2)
    if sigma > 10.0:
        continue
    data_SDSS.append(data_SDSS_all[i])
    
data_SDSS = np.array(data_SDSS)
print(len(data_SDSS))

# Grid the data and find out which grid cells are empty and as such don't need including from our theoretical covariance matrix
clipflag = np.zeros(nelements)
#The number of galaxies inside the grid cell. 
ngrid_SDSS = np.zeros(nelements)
#The log-distance ratio of the grid cell
datagrid_SDSS = np.zeros(nelements)
#The error of the log-distance ratio of the grid cell. 
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
#Normalize this with respect to the random catalogue. 
norm = np.sum(ngrid_SDSS)/np.sum(data_expect)
data_expect = norm*data_expect

#Calculate the galaxy overdensity, if the galaxy overdensity in the random catalogue is zero. Automatically returns 100 (which will be cut out later.)
data_gal_all = np.divide((ngrid_SDSS - data_expect), data_expect, out= 100.0*np.ones(len(data_expect)), where=data_expect!=0)

# zero = np.where(data_gal_all < -2.0)[0]

#Cut out all grids with galaxy overdensity over 50 because our model is not able to deal with such high non-linearity. 
remove_galaxy = np.where(data_gal_all > 50.0)[0]

data_gal = np.delete(data_gal_all, remove_galaxy)
ncomp_galaxy = len(data_gal)

data_expect = np.delete(data_expect, remove_galaxy)

#Cut out grids where there is no log-distance ratio measurements. 
comp_velocity = np.where(ngrid_SDSS > 0)[0]
ncomp_velocity = len(comp_velocity)
ngrid_SDSS_vel = ngrid_SDSS[comp_velocity]
data_vel = datagrid_SDSS[comp_velocity]
errgrid_SDSS = errgrid_SDSS[comp_velocity]
remove_velocity = np.where(ngrid_SDSS == 0)[0]

# Correct the data and covariance matrix for the gridding. Eqs. 19 and 22.
data_vel /= ngrid_SDSS_vel        # We summed the velocities in each cell, now get the mean
errgrid_SDSS /= ngrid_SDSS_vel**2.0    # This is the standard error on the mean.

datagrid_comp = np.concatenate((data_gal, data_vel))

datagrid_comp_new = datagrid_comp.reshape((len(datagrid_comp), 1))

length_gal = len(data_gal)
length_vel = len(data_vel)

print('The length of the galaxy data is ' + str(length_gal)+' and the length of the velocity data is '+str(length_vel))

#The x_2 vector is used to calculate the effect of the uncertainty of the zero-point correction on the analytical covariance matrix. 
# x_1 = np.concatenate((np.ones(length_gal), np.zeros(length_vel)))
x_2 = np.concatenate((np.zeros(length_gal), np.ones(length_vel)))

# x_1 = np.reshape(x_1, (len(x_1), 1))
x_2 = np.reshape(x_2, (len(x_2), 1))

conv_vg = []
conv_vv = []
conv_gg = []
conv_gg_badd = []

#read in the pre-computed velocity-galaxy cross-covariance matrix. The galaxy-velocity cross-covariance matrix is just the transpose of the velocity-galaxy cross-covariance
#matrix. 
conv_vg_sigma_u = []
c = 1
d = 0
for j in range(8):
    #The filename of the stored components of the cross-covariance matrix. 
    # data_file_conv_vg = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_dv_%d_%d_sigmau%03d.dat' %(int(1000.0*kmax_velocity), c, d, int(10.0*sigma_u)))
    # data_file_conv_vg = str('wide_angle_covariance_k0p002_0p150_gridcorr30_dv_%d_%d_sigmau%03d.dat' %(c, d, int(10.0*sigma_u)))
    # data_file_conv_vg = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr20_dv_%d_%d_sigmau%03d.dat' %(int(1000.0*kmax_velocity), c, d, int(10.0*sigma_u)))
    data_file_conv_vg = str('wide_angle_covariance_k0p002_0p%03d_gridcorr20_dv_%d_%d_sigmau%03d.dat' %(int(1000.0*kmax_velocity), c, d, int(10.0*sigma_u)))
    print(data_file_conv_vg)
    #The value of the covariance matrix can be extremely small, so we scale it up by 10**(8+d) in the c code. 
    conv_vg_element = np.array(pd.read_csv(data_file_conv_vg, delim_whitespace=True, header=None, skiprows=1))/10**(8+d)
    #Delete the elements where there is no log-distance ratio measurement or overdensity is over 50. 
    conv_vg_final = np.delete(np.delete(conv_vg_element, remove_velocity, axis = 0), remove_galaxy, axis = 1)
    #Multiply the extra factor to convert it to the effective redshift. 
    conv_vg_sigma_u.append(factor_gv*conv_vg_final)
    d += 2
    if (d > 7):
        c += 1
        d = 0
        
conv_vg.append(conv_vg_sigma_u)

#Read in the gridded and non-gridded version of the velocity auto-covariance matrix. Both matrices are being scaled up by 10**6 in the c code. 
# data_file_conv_vv = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_vv_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
# data_file_conv_vv = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_sigmau%03d.dat' %(int(10.0*sigma_u)))
# data_file_conv_vv = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr20_vv_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
data_file_conv_vv = str('wide_angle_covariance_k0p002_0p%03d_gridcorr20_vv_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
#The velocity auto-covariance matrix is being scale up by 10^6 in the c-code, so we dividing the scaling factor here. 
conv_vv_element = np.array(pd.read_csv(data_file_conv_vv, delim_whitespace=True, header=None, skiprows=1))/1.0e6
#Delete the grid cells where there is no log-distance ratio measurements. 
conv_vv_final = np.delete(np.delete(conv_vv_element, remove_velocity, axis = 0), remove_velocity, axis = 1)
   
# data_file_conv_vv_ng = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_vv_ng_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
# data_file_conv_vv_ng = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_ng_sigmau%03d.dat' %(int(10.0*sigma_u)))
# data_file_conv_vv_ng = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr20_vv_ng_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
data_file_conv_vv_ng = str('wide_angle_covariance_k0p002_0p%03d_gridcorr20_vv_ng_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
conv_vv_ng_element = np.array(pd.read_csv(data_file_conv_vv_ng, delim_whitespace=True, header=None, skiprows=1))/1.0e6
conv_vv_ng_final = np.delete(np.delete(conv_vv_ng_element, remove_velocity, axis = 0), remove_velocity, axis = 1)

#Accouting the for the shot-noise of the velocity auto-covariance matrix (equation (31) in the paper). 
for k in range(ncomp_velocity):
    conv_vv_final[k][k] += (conv_vv_ng_final[k][k] - conv_vv_final[k][k])/ngrid_SDSS_vel[k]
conv_vv.append([factor_vv*conv_vv_final])

#Read in the galaxy auto-covariance matrices and the b_add matrices. Both matrices are being scaled up by 10**(8+a) in the c code. 
a = 0
b = 0
for k in range(21):
    #The filename of the components of the galaxy auto-covariance matrix. 
    # data_file_conv_gg = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_dd_%d_%d.dat' %(int(1000.0*kmax_galaxy), b, a))
    # data_file_conv_gg = str('wide_angle_covariance_k0p002_0p150_gridcorr30_dd_'+str(b)+'_'+str(a)+'.dat')
    # data_file_conv_gg = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr20_dd_%d_%d.dat' %(int(1000.0*kmax_galaxy), b, a))
    data_file_conv_gg = str('wide_angle_covariance_k0p002_0p%03d_gridcorr20_dd_%d_%d.dat' %(int(1000.0*kmax_galaxy), b, a))
    print(data_file_conv_gg)
    #divided the scaling factor in the c code. 
    conv_gg_element = np.array(pd.read_csv(data_file_conv_gg, delim_whitespace=True, header=None, skiprows=1))/10**(8+a)
    #Delete the grid cells where the galaxy overdensity is over 50. 
    conv_gg_final = (np.delete(np.delete(conv_gg_element, remove_galaxy, axis = 0), remove_galaxy, axis = 1)).astype('float64')
    conv_gg.append(factor_gg*conv_gg_final)
    if (k < 7):
        #The filename of the components of the b_add matrices and divided off the extra factor and remove the grid cells the same as the galaxy
        #auto-covariance matrix. 
        
        #data_file_conv_gg_badd = str('wide_angle_covariance_k0p150_0p999_gridcorr30_dd_'+str(b)+'_'+str(a)+'.dat')
        #data_file_conv_gg_badd = str('wide_angle_covariance_k0p002_0p%03d_gridcorr30_dd_%d_%d.dat' %(int(1000.0*kmax_galaxy), b, a))
        # data_file_conv_gg_badd = str('wide_angle_covariance_k0p%03d_0p%03d_gridcorr%d_dd_%d_%d.dat' % (int(1000.0*kmax_galaxy), int(1000.0*0.999), gridsize, b, a)) 
        # data_file_conv_gg_badd = str('/data/s4479813/wide_angle_covariance_k0p%03d_0p%03d_gridcorr%d_dd_%d_%d.dat' % (int(1000.0*kmax_galaxy), int(1000.0*0.999), gridsize, b, a))         
        data_file_conv_gg_badd = str('wide_angle_covariance_k0p%03d_0p%03d_gridcorr%d_dd_%d_%d.dat' % (int(1000.0*kmax_galaxy), int(1000.0*0.999), gridsize, b, a))         
        print(data_file_conv_gg_badd)
        conv_gg_badd_element = np.array(pd.read_csv(data_file_conv_gg_badd, delim_whitespace=True, header=None, skiprows=1))/10**(8+a)
        conv_gg_badd_final = (np.delete(np.delete(conv_gg_badd_element, remove_galaxy, axis = 0), remove_galaxy, axis = 1)).astype('float64')
        conv_gg_badd.append(factor_gg*conv_gg_badd_final)
        
    a += 2
    if (a > 13):
        a = 0
        b += 1
    
#Only use grid cells where there is a log-distance ratio measurement. 
datagrid_vec = datagrid_vec[comp_velocity,:]

# Convert these to full matrices so that we can write model as a single sum
conv_vv = conv_vv[0]
conv_vg = conv_vg[0]
print(np.shape(conv_vv), np.shape(conv_vg), np.shape(conv_gg))

#Construct each part of the full covariance matrix based on the analytical formula for the full covariance matrix (equation (C2) in the paper). 
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

    from scipy.optimize import differential_evolution, minimize
    
    #The fiducial value for the free parameters. They are set quite far away from the truth values. 
    bfid = 1.7/bsigma8_old
    ffid = 0.2/fsigma8_old
    sigmag_fid = sigma_g
    baddfid = 0.5/bsigma8_old
    sigma_v_fid = 300.0
    
    
    params_fid = np.array([ffid, sigma_v_fid, bfid, baddfid, sigmag_fid])
    
    #Using Scipy differential_evolution to find the free parameters that give the maximum likelihood. 
    # converged = 1
    # count = 0
    # while(~converged):
    
    #     # Compute derivatives about fiducial values
    #     loglikefid, dL, d2L = get_dL(params_fid)
    #     # print(dL, np.sqrt(np.diag(np.linalg.inv(d2L))))
    
    #     """result = basinhopping(
    #         lambda *args: -lnpost(*args),
    #         [ffid, sigma_v_fid, bfid, baddfid],
    #         niter_success=10,
    #         niter=100,
    #         stepsize=0.01,
    #         minimizer_kwargs={
    #             "method": "Nelder-Mead",
    #             "tol": 1.0e-4,
    #             "options": {"maxiter": 40000, "xatol": 1.0e-4, "fatol": 1.0e-4},
    #         },
    #     )"""
    
    
    #     fvals, sigmavvals, bvals, baddvals, sigmagvals= (0.0, 1.0), (1.0, 5000.0), (0.0, 3.0), (0.0, 10.0), (0.0, 10.0)
    #     result = differential_evolution(lambda *args: -lnpost(*args), bounds=(fvals, sigmavvals, bvals, baddvals, sigmagvals), maxiter=10000, tol=1.0e-6)
    #     print("#-------------- Best-fit----------------")
    #     print(result["x"])
    
    #     params_fid_new = np.array([result["x"][0]/fsigma8_old, result["x"][1], result["x"][2]/bsigma8_old, result["x"][3]/bsigma8_old, result["x"][4]])
        
    #     compare = np.abs(params_fid_new/params_fid - 1.0)
    #     print(compare)
        
    #     #The convergence criteria is that the relative change between two evaluation is less than 0.1% for fsigma8 and 1% for the other free parameters. 
    #     # converged = np.all(np.abs(params_fid_new/params_fid - 1.0) < 1.0e-3)
    #     converged = np.all(np.less_equal(compare, [1e-3, 1e-2, 1e-2, 1e-2, 1e-2]))
    #     # converged = np.all(np.less(params_fid_new/params_fid - 1.0,  [1.0e-5, 1.0e-3, 1.0e-5, 1.0e-5]))
    
    #     params_fid = params_fid_new
        
    #     count += 1
    #     if (count > 50):
    #         break
    
    start = time.time()
    #This should be the same as the prior range for the free parameters. 
    fvals, sigmavvals, bvals, baddvals, sigmagvals= (0.0, 1.0), (1.0, 5000.0), (0.0, 3.0), (0.0, 10.0), (0.0, 10.0)
    # result = differential_evolution(lambda *args: -get_max_dL(*args), bounds=(fvals, sigmavvals, bvals, baddvals, sigmagvals))
    #Find the best-fit parameters. 
    result = minimize(lambda *args: -get_max_dL(*args), params_fid, method='Powell', bounds=(fvals, sigmavvals, bvals, baddvals, sigmagvals))
    #Check whether the minimization is successful. 
    print("#-------------- Best-fit----------------")
    print(result["x"], result["fun"], result["success"])
    # params_fid_new = np.array([result["x"][0]/fsigma8_old, result["x"][1], result["x"][2]/bsigma8_old, result["x"][3]/bsigma8_old, result["x"][4]])
    params_fid_new = np.array([result["x"][0], result["x"][1], result["x"][2], result["x"][3], result["x"][4]])
    params_fid = params_fid_new
    end = time.time()
    print('It takes ' + str(end - start) + ' seconds to finish the optimization.')
    
    # #Find the maximum log-likelihood and the derivative and the second derivative at the maximum likelihood. 
    # loglikefid, dL, d2L = get_dL(params_fid)
    
    # # fsigma8 = params_fid[0]
    # # bsigma8 = params_fid[2]
    # # b_add_sigma8 = params_fid[3]
    
    #Compute the fiducial log-likelihood and its first and second derivative at 21 different points. 
    start = time.time()
    N = 21
    fsigma8_diff = np.linspace(0.0, 1.0, N)/fsigma8_old
    step_N = 1.0/np.float32(N-1)
    loglikefid_all = []
    dL_all = []
    d2L_all = []
    for i in range(len(fsigma8_diff)):
        params_fit = np.array([fsigma8_diff[i], params_fid[1], params_fid[2], params_fid[3], params_fid[4]])
        loglikefid, dL, d2L = get_dL(params_fit)
        loglikefid_all.append(loglikefid)
        dL_all.append(dL)
        d2L_all.append(d2L)
        print('Finish the' + str(i) + 'th fiducial covariance matrix calculation.')
    end = time.time()
    print('It takes ' + str(end - start) + ' seconds to finish calculating the fiducial covariance matrices.')

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    ndim, nwalkers = 5, 40
    
    # Set up the first points for the chain (for emcee we need to give each walker a random value about this point so we just sample the prior or a reasonable region)
    # begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 3.0*np.random.rand(), 25.0*np.random.rand(), 7.0*np.random.rand()+1.0] for i in range(nwalkers)]
    # begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 3.0*np.random.rand(), 7.0*np.random.rand()+1.0] for i in range(nwalkers)]
    #The possible values are 0<=fsigma8<=1, 0<=sigmav<=1000, 0<=bsigma8<=3.
    #begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 0.2] for i in range(nwalkers)]
    # begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 3.0*np.random.rand(), 10.0*np.random.rand()] for i in range(nwalkers)]
    
    #initial guess for the MCMC algorithm. 
    begin = [[1.0*np.random.rand(), 4999.0*np.random.rand()+1.0, 3.0*np.random.rand(), 10.0*np.random.rand(), 10.0*np.random.rand()] for i in range(nwalkers)]
    
    # Set up the output file
    backend = backends.HDFBackend(chainfile)
    backend.reset(nwalkers, ndim)
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, backend=backend)
        
    # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
    # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
    # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
    max_iter = 40000
    index = 0
    old_tau = np.inf
    autocorr = np.empty(max_iter)
    counter = 0

    for sample in sampler.sample(begin, iterations=max_iter, progress=progress):
    
        # Only check convergence every 100 steps
        if sampler.iteration % 1000:
            continue
    
        # Compute the autocorrelation time so far
        # Using tol=0 means that we'll always get an estimate even
        # if it isn't trustworthy
        tau = sampler.get_autocorr_time(tol=0)
        autocorr[index] = np.mean(tau)
        counter += 1000
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
    
    
    #If you want to run mutliple threads on a local machine. Uncomment the following lines and comment out the corresponding lines above. 
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