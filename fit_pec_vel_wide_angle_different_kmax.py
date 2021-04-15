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

def lnprior(params):

    # Here we define the prior for all the parameters. We'll ignore the constants as they
    # cancel out when subtracting the log posteriors
#    fsigma8, sigma_v, bsigma8, b_add_sigma8, sigma_u = params
    # fsigma8, sigma_v, bsigma8, sigma_u, sigma_g = params
    # fsigma8, sigma_v, bsigma8, sigma_g = params
    fsigma8, sigma_v, bsigma8, b_add_sigma8 = params

    # Don't allow fsigma8 to be less that 0.0
    if (fsigma8 >= 0.0):
        fsigma8prior = 1.0 
    else:
        return -np.inf

    # Flat prior for sigma_v between 0.0 and 1000.0 km/s (from Andrews paper)
    if (0.0 <= sigma_v < 1000.0):
        sigmavprior = 1.0/1000.0
    else:
        return -np.inf 
    
    # Flat prior for bsigma8 between 0.0 and 3.0 
    if (0.0 < bsigma8 < 3.0):
        bsigma8prior = 1.0/3.0
    else:
        return -np.inf
    
    # Flat prior for b_add_sigma8 between 0.0 and 10.0
    if (0.0 <= b_add_sigma8 <= 10.0):
        b_add_sigma8prior = 1.0/10.0
    else:
        return -np.inf
    
    # # Flat prior for sigma_u between 0.2 and 25.0
    # if (0.0 <= sigma_u < 25.0):
    #     sigma_uprior = 1.0/25.0
    # else:
    #     return -np.inf
    
    # if (1.0 <= sigma_g < 8.0):
    #     sigma_gprior = 1.0/7.0
    # else:
    #     return -np.inf

    return 0.0

def lnlike(params): 

    # Return the log likelihood for a model. Here are the two parameters we want to fit for.
#  fsigma8, sigma_v, bsigma8, b_add_sigma8, sigma_u = params
    # fsigma8, sigma_v, bsigma8, sigma_u, sigma_g = params
    # fsigma8, sigma_v, bsigma8, sigma_g = params
    # fsigma8, sigma_v, bsigma8 = params
    fsigma8, sigma_v, bsigma8, b_add_sigma8 = params
    
    # fsigma8 = params[:,0]
    # sigma_v = params[:,1]
    # bsigma8 = params[:,2]
    # b_add_sigma8 = params[:,3]
    # sigma_u = 0.0
    # sigma_g = 3.0
    
    # conv_pk_vel_gal_array = linear_interpolation(conv_vg, sigma_u, 8)
    
    # conv_pk_vel_vel_array = linear_interpolation(conv_vv, sigma_u, 1)
    
    
    # conv_pk_gal_gal = 0
    
    # m = (bsigma8/bsigma8_old)**2
    # n = 0 
    # l = 0
    # for i in range(len(conv_gg)):
    #     conv_pk_gal_gal += conv_gg[i]*m*sigma_g**n 
        
    #     n += 2
    #     if ((n > 13) and (l == 0)):
    #         n = 0
    #         m = (bsigma8*fsigma8/(bsigma8_old*fsigma8_old))
    #         l = 1
    #     elif ((n > 13) and (l == 1)):
    #         n = 0
    #         m = (fsigma8/fsigma8_old)**2
    #         l = 2
    
    # conv_pk_vel_gal = 0
    # m = (bsigma8*fsigma8/(bsigma8_old*fsigma8_old))
    # n = 0
    
    # for j in range(len(conv_pk_vel_gal_array)):
    #     conv_pk_vel_gal += conv_pk_vel_gal_array[j]*m*sigma_g**n
    #     n += 2
    #     if (n > 7):
    #         n = 0
    #         m = (fsigma8/fsigma8_old)**2
    
    # conv_pk_vel_gal = conv_pk_vel_gal
    # conv_pk_vel_vel = (fsigma8/fsigma8_old)**2*conv_pk_vel_vel_array[0]
    
    # conv_pk_gal_vel = conv_pk_vel_gal.T
    
    conv_pk_gal_gal = (bsigma8/bsigma8_old)**2*(conv_pk_gal_gal_0) + (bsigma8*fsigma8)/(bsigma8_old*fsigma8_old)*conv_pk_gal_gal_1 + (fsigma8/fsigma8_old)**2*conv_pk_gal_gal_2 + (b_add_sigma8)**2*conv_pk_gal_gal_badd
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
    #Add the shot noise to the diagonal of the galaxy auto-covariance matrices.

    # The covariance matrix for the given cosmology. Given that we are using scale-independent fsigma8 and no RSD damping 
    # we just need to scale this based on the fsigma8 value used in the original of the covariance matrix
   
    
  
#  conv_gg_new_new = conv_gg_new + b_add_sigma8**2*conv_gg_add
#  #Increase the value of the galaxy auto-covariance matrix with the additional matrix. 
  
    #This gives the full covariance matrix. 

    conv_pk_new_block_1 = np.concatenate((conv_pk_gal_gal, conv_pk_gal_vel), axis = 1)
    conv_pk_new_block_2 = np.concatenate((conv_pk_vel_gal, conv_pk_vel_vel), axis = 1)
    conv_pk_new = np.concatenate((conv_pk_new_block_1, conv_pk_new_block_2), axis = 0)

       #Calculate the determinant and inverse of the covariance matrix with LAPACK

    pivots = np.zeros(int(ncomp_velocity + ncomp_galaxy), np.intc)
    conv_pk_copy, pivots, info = lapack.dgetrf(conv_pk_new)
    abs_element = np.fabs(np.diagonal(conv_pk_copy))
    det = np.sum(np.log(abs_element))

#    identity = np.eye(int(ncomp_velocity + ncomp_galaxy))
#    ident = np.copy(identity)
#    conv_pk_lu, pivots2, cov_inv, info2 = lapack.dgesv(conv_pk_new, ident)

    # cov_inv = sp.linalg.inv(conv_pk_new)

    #Calculates the chi-squared with matrix multiplication. \

    # chi_squared = np.matmul(datagrid_comp.T, np.matmul(cov_inv, datagrid_comp))
    chi_squared = np.matmul(datagrid_comp.T, np.linalg.solve(conv_pk_new, datagrid_comp)) 

    
    while (chi_squared < 0):
        w, v = np.linalg.eig(conv_pk_new)
        print(chi_squared, fsigma8, bsigma8, np.min(w), np.max(w))
        
        # para = len(np.where(w < 0))/len(w)
        
        # alpha = 0.01*np.min(np.abs(w))
        
        # w_new = np.where(w > 0.0, w, alpha)
        
        # conv_pk_new_dash = np.matmul(v, np.matmul(np.diag(w_new), sp.linalg.inv(v)))
        
        # chi_squared = np.matmul(datagrid_comp.T, np.linalg.solve(conv_pk_new_dash, datagrid_comp)) 
        
        # diff = np.sum(np.abs(conv_pk_new_dash - conv_pk_new))
        
        # print(len(w), para, diff)
        
        # conv_pk_new = conv_pk_new_dash
        raise ValueError('Negative chi-squared')
    
    return -0.5*(det+chi_squared)



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
expect_file = str('/data/s4479813/SDSS_randoms.csv')


datafile = str("/data/s4479813/SDSS_mocks/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr_again")

if (correction == 0):
    chainfile = str('/data/s4479813/SDSS_data/fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_no_correction_sigmau_%d_sigmag_%d.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u, sigma_g))
elif (correction == 1):
    chainfile = str('/data/s4479813/SDSS_data/fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_correction_sigmau_%d_sigmag_%d.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u, sigma_g))

# expect_file = str('SDSS_randoms.csv')
# datafile = str("./SDSS_mocks/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr_again")

# if (correction == 0):
#     chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_no_correction_sigmau_%d_sigmag_%d.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u, sigma_g))
# elif (correction == 1):
#     chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_full_correction_sigmau_%d_sigmag_%d.hdf5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize, mock_num, sigma_u, sigma_g))


# Generate the grid used to compute the covariance matrix. Again make sure these match the covariance matrix code.
xmin, xmax = -175.0, 215.0
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
        
#         data_file_conv_vg = str('/data/s4479813/wide_angle_covariance_k0p002_0p150_gridcorr30_dv_%d_%d_sigmau%03d.dat' %(c, d, int(10.0*i)))
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
#     data_file_conv_vv = str('/data/s4479813/wide_angle_covariance_k0p002_0p150_gridcorr30_vv_sigmau%03d.dat' %(int(10.0*i)))
#     # data_file_conv_vv = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_sigmau%03d.dat' %(int(10.0*i)))
#     conv_vv_element = np.array(pd.read_csv(data_file_conv_vv, delim_whitespace=True, header=None, skiprows=1))/1.0e6
#     conv_vv_final = np.delete(np.delete(conv_vv_element, remove_velocity, axis = 0), remove_velocity, axis = 1)
   
#     data_file_conv_vv_ng = str('/data/s4479813/wide_angle_covariance_k0p002_0p150_gridcorr30_vv_ng_sigmau%03d.dat' %(int(10.0*i)))
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
    data_file_conv_vg = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr30_dv_%d_%d_sigmau%03d.dat' %(int(1000.0*kmax_velocity), c, d, int(10.0*sigma_u)))
    # data_file_conv_vg = str('wide_angle_covariance_k0p002_0p150_gridcorr30_dv_%d_%d_sigmau%03d.dat' %(c, d, int(10.0*sigma_u)))
    print(data_file_conv_vg)
    conv_vg_element = np.array(pd.read_csv(data_file_conv_vg, delim_whitespace=True, header=None, skiprows=1))/1.0e6
    conv_vg_final = np.delete(np.delete(conv_vg_element, remove_velocity, axis = 0), remove_galaxy, axis = 1)
    conv_vg_sigma_u.append(conv_vg_final)
    d += 2
    if (d > 7):
        c += 1
        d = 0
        
conv_vg.append(conv_vg_sigma_u)

data_file_conv_vv = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr30_vv_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
# data_file_conv_vv = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_sigmau%03d.dat' %(int(10.0*sigma_u)))
conv_vv_element = np.array(pd.read_csv(data_file_conv_vv, delim_whitespace=True, header=None, skiprows=1))/1.0e6
conv_vv_final = np.delete(np.delete(conv_vv_element, remove_velocity, axis = 0), remove_velocity, axis = 1)
   
data_file_conv_vv_ng = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr30_vv_ng_sigmau%03d.dat' %(int(1000.0*kmax_velocity), int(10.0*sigma_u)))
# data_file_conv_vv_ng = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_ng_sigmau%03d.dat' %(int(10.0*sigma_u)))
conv_vv_ng_element = np.array(pd.read_csv(data_file_conv_vv_ng, delim_whitespace=True, header=None, skiprows=1))/1.0e6
conv_vv_ng_final = np.delete(np.delete(conv_vv_ng_element, remove_velocity, axis = 0), remove_velocity, axis = 1)

for k in range(ncomp_velocity):
    conv_vv_final[k][k] += (conv_vv_ng_final[k][k] - conv_vv_final[k][k])/ngrid_SDSS_vel[k]
conv_vv.append([conv_vv_final])


a = 0
b = 0
for k in range(21):
    data_file_conv_gg = str('/data/s4479813/wide_angle_covariance_k0p002_0p%03d_gridcorr30_dd_%d_%d.dat' %(int(1000.0*kmax_galaxy), b, a))
    # data_file_conv_gg = str('wide_angle_covariance_k0p002_0p150_gridcorr30_dd_'+str(b)+'_'+str(a)+'.dat')
    print(data_file_conv_gg)
    conv_gg_element = np.array(pd.read_csv(data_file_conv_gg, delim_whitespace=True, header=None, skiprows=1))/1.0e6
    conv_gg_final = np.delete(np.delete(conv_gg_element, remove_galaxy, axis = 0), remove_galaxy, axis = 1)
    conv_gg.append(conv_gg_final)
    if (k < 7):
        # data_file_conv_gg_badd = str('wide_angle_covariance_k0p150_0p999_gridcorr30_dd_'+str(b)+'_'+str(a)+'.dat')
        data_file_conv_gg_badd = str('/data/s4479813/wide_angle_covariance_k0p%03d_0p%03d_gridcorr%d_dd_%d_%d.dat' % (int(1000.0*kmax_galaxy), int(1000.0*0.999), gridsize, b, a)) 
        print(data_file_conv_gg_badd)
        conv_gg_badd_element = np.array(pd.read_csv(data_file_conv_gg_badd, delim_whitespace=True, header=None, skiprows=1))/1.0e6
        conv_gg_badd_final = np.delete(np.delete(conv_gg_badd_element, remove_galaxy, axis = 0), remove_galaxy, axis = 1)
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

if __name__ == "__main__":

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    ndim, nwalkers = 4, 32
    
    # Set up the first points for the chain (for emcee we need to give each walker a random value about this point so we just sample the prior or a reasonable region)
    # begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 3.0*np.random.rand(), 25.0*np.random.rand(), 7.0*np.random.rand()+1.0] for i in range(nwalkers)]
    # begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 3.0*np.random.rand(), 7.0*np.random.rand()+1.0] for i in range(nwalkers)]
    #The possible values are 0<=fsigma8<=1, 0<=sigmav<=1000, 0<=bsigma8<=3.
    #begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 0.2] for i in range(nwalkers)]
    begin = [[1.0*np.random.rand(), 1000.0*np.random.rand(), 3.0*np.random.rand(), 10.0*np.random.rand()] for i in range(nwalkers)]

    
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
        converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
        if converged:
            print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
            break
        old_tau = tau
        index += 1

    print(read_chain_backend(chainfile))
    
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