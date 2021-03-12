# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:35:40 2021

@author: Yanxiang Lai
"""

# A python code to MCMC fit the peculiar velocity datasets of 2MTF and 6dFGSv separately and in combination

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
from scipy.linalg import lapack, inv
import pandas as pd
from chainconsumer import ChainConsumer
from multiprocessing import Pool

# Speed of light in km/s
LightSpeed = 299792.458

def read_chain_backend(chainfile):

    reader = emcee.backends.HDFBackend(chainfile)

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
    fsigma8, sigma_v = params

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

    return 0.0

def lnlike(params): 

    # Return the log likelihood for a model. Here are the two parameters we want to fit for.
    fsigma8, sigma_v = params
    

    # The covariance matrix for the given cosmology. Given that we are using scale-independent fsigma8 and no RSD damping 
    # we just need to scale this based on the fsigma8 value used in the original calculation of the covariance matrix
    factor = (fsigma8/fsigma8_old)**2
    conv_pk_new = conv_pk*factor

    # Modify the covariance matrix based on the data and random uncertainties.
    sigma_varr = np.zeros(ncomp)+sigma_v
    err_rand = datagrid_vec[0:,4]*sigma_varr/np.sqrt(ngrid_SDSS)
    conv_pk_diag = conv_pk_new[np.diag_indices(ncomp)]
    np.fill_diagonal(conv_pk_new, conv_pk_diag + errgrid_SDSS + err_rand**2)

    # Calculate the determinant and inverse of the covariance matrix 
    pivots = np.zeros(ncomp, np.intc)
    conv_pk_copy, pivots, info = lapack.dgetrf(conv_pk_new)
    abs_element = np.fabs(np.diagonal(conv_pk_copy))
    det = np.sum(np.log(abs_element))

    # identity = np.eye(ncomp)
    # ident = np.copy(identity)
    # conv_pk_lu, pivots2, cov_inv, info2 = lapack.dgesv(conv_pk_new, ident)
    
    # cov_inv = inv(conv_pk_new)
    # # Modify the inverse covariance matrix based on the zero-point uncertainty and calculate the new factor required for the log likelihood
    # # This automatically marginalises over this uncertainty so we don't have to fit it as a free parameter according to Bridle et. al., 2002. Neat!
    # chi_squared = 0.0
    # cov_fac1 = np.sum(cov_inv)
    # cov_fac2 = np.empty(ncomp)
    # for i in range(ncomp):
    #     cov_fac2[i] = np.sum(cov_inv[0:,i])

    # # Compute the chi-squared (part in exponential in Eq. 23)
    # for i in range(ncomp):
    #     chi_squared += datagrid_comp[i]*np.sum((cov_inv[i,0:] - ((cov_fac2[i]*cov_fac2[0:])/(cov_fac1 + 1.0/sigma_y)))*datagrid_comp[0:])

    # # print(fsigma8, sigma_v, det, np.log(1.0+cov_fac1*sigma_y), chi_squared, -0.5*(det+np.log(1.0+cov_fac1*sigma_y)+chi_squared))
    
    # det = np.linalg.det(conv_pk_new)
    # cov_inv = np.linalg.inv(conv_pk_new)
  
  
    # chi_squared = np.matmul(datagrid_SDSS.T, np.matmul(cov_inv, datagrid_SDSS))
    chi_squared = np.matmul(datagrid_SDSS.T, np.linalg.solve(conv_pk_new, datagrid_SDSS))  #This calculate the 
    #chi-squared without doing the inverse. It is a lot faster. 
    #print(chi_squared)
    # return the log likelihood (Log of Eq. 23)
    
    # return -0.5*(det+np.log(1.0+cov_fac1*sigma_y)+chi_squared)
    return -0.5*(det+chi_squared)

# Run parameters
start_time = time.time()
omega_m = 0.3121      # Used to convert the galaxy redshifts to distances. Should match that used to compute the covariance matrix
fsigma8_old = 0.8150         # The value of fsigma8 used to compute the covariance matrix. 
sigma_y = 0.2**2             # The value of sigma_y used for Eq. 24.

# Make sure these match the values used to estimate the covariance matrix
kmin = float(sys.argv[1])    # What kmin to use
kmax = float(sys.argv[2])    # What kmax to use
gridsize = int(sys.argv[3])  # What gridsize to use
progress = bool(sys.argv[4])  # Whether or not to print out progress
mock_num = int(sys.argv[5]) #The number of the mock
correction = int(sys.argv[6]) #Whether to use the corrected log-distance ratio. Enter 0 for no correction and 1 for correction
num_processor = int(sys.argv[7]) #This is the number of processors to use.  

file_num, dot_num = divmod(mock_num, 8)

# Read in the data
datafile = str("/data/s4479813/SDSS_mocks/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr")

# covfile = str('/data/s4479813/conv_pk_vel_SDSS_k0p002_0p150_vel_vel_gridcorr20.dat')
# covfile_ng = str('/data/s4479813/conv_pk_vel_SDSS_k0p002_0p150_vel_vel_gridcorr20_ng.dat')

covfile = str('/data/s4479813/wide_angle_covariance_k0p002_0p150_gridcorr20_vv_sigmau000.dat')
covfile_ng = str('/data/s4479813/wide_angle_covariance_k0p002_0p150_gridcorr20_vv_ng_sigmau000.dat')

if (correction == 0):
    chainfile = str('/data/s4479813/SDSS_data/fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_err_no_correction.hdf5' % (int(1000.0*kmin), int(1000.0*kmax), gridsize, mock_num))
elif (correction == 1):
    chainfile = str('/data/s4479813/SDSS_data/fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_err_correction.hdf5' % (int(1000.0*kmin), int(1000.0*kmax), gridsize, mock_num))

# chainfile = str('/data/s4479813/SDSS_data/fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_err_true.hdf5' % (int(1000.0*kmin), int(1000.0*kmax), gridsize, mock_num))

# datafile = str("./SDSS_mocks/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr")

# # covfile = str('conv_pk_vel_SDSS_k0p002_0p150_vel_vel_gridcorr20.dat')
# # covfile_ng = str('conv_pk_vel_SDSS_k0p002_0p150_vel_vel_gridcorr20_ng.dat')

# covfile = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_sigmau000.dat')
# covfile_ng = str('wide_angle_covariance_k0p002_0p150_gridcorr30_vv_ng_sigmau000.dat')

# if (correction == 0):
#     chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_err_no_correction.hdf5' % (int(1000.0*kmin), int(1000.0*kmax), gridsize, mock_num))
# elif (correction == 1):
#     chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_err_correction.hdf5' % (int(1000.0*kmin), int(1000.0*kmax), gridsize, mock_num))

# chainfile = str('fit_pec_vel_SDSS_k0p%03d_0p%03d_gridcorr_%d_%d_err_true.hdf5' % (int(1000.0*kmin), int(1000.0*kmax), gridsize, mock_num))

print(datafile)
print(covfile)
print(covfile_ng)
print(chainfile)

gridsize = float(gridsize)

# Generate the grid used to compute the covariance matrix. Again make sure these match the covariance matrix code.
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
            datagrid_vec[ind,4] = (1.0/np.log(10))*(1.0+red)/(100.0*ez*r) #The prefactor of the log-distance ratio

data = np.array(pd.read_csv(datafile, header=None, skiprows=1)) #Read in the data files. 

RA = data[:, 1]/180.0*np.pi
Dec = data[:, 2]/180.0*np.pi
redshift = data[:, 4]/LightSpeed

rd = sp.interpolate.splev(redshift, radial_spline) #Convert redshift into comoving distance

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
#Converts redshift, RA, Dec into carteisian coordinates. 

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
ngrid_SDSS = np.zeros(nelements) #The number of galaxies in each grid.
datagrid_SDSS = np.zeros(nelements) #The log-distance ratio of each grid.
errgrid_SDSS = np.zeros(nelements) #The uncertainty of each grid. 
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

# Read in the covariance matrices
conv_pk = np.array(pd.read_csv(covfile, delim_whitespace=True, header=None, skiprows=1))/1.0e6          # The C code multiplies by a factor of 1x10^6 so we don't lose precision when writing to file
conv_pk_ng = np.array(pd.read_csv(covfile_ng, delim_whitespace=True, header=None, skiprows=1))/1.0e6    # The C code multiplies by a factor of 1x10^6 so we don't lose precision when writing to file

# Compress the datagrid based on whether or not there are any galaxies in that cell. Do the same for the covariance matrix.
# We are removing all elements from our matrices that have no data in them, as these should not count towards the likelihood.
# comp_index = np.where(ngrid_SDSS > 0)[0]
# ncomp = len(comp_index)
# datagrid_SDSS = datagrid_SDSS[comp_index]
# errgrid_SDSS = errgrid_SDSS[comp_index]
# datagrid_vec = datagrid_vec[comp_index,:]
# remove_index = np.where(ngrid_SDSS == 0)[0]
# ngrid_SDSS = ngrid_SDSS[comp_index]

comp_index = np.where(ngrid_SDSS > 0)[0] #We only use grid cells with more than 1 galaxies in it.
distance_limit = sp.interpolate.splev(0.075, radial_spline, der=0) #Convert the redshift limit to a radial distance limit. 
distance = datagrid_vec[:,3] #The distance to the center of each grid cell
comp_index_delete = np.intersect1d(np.where(distance>distance_limit)[0], comp_index) #This gives the grid cells which have more than
#one galaxies in it and also at a distance greater than z = 0.075.

delete_index = []
for i in range(len(comp_index)):
    for j in range(len(comp_index_delete)):
        if (comp_index[i] == comp_index_delete[j]):
            delete_index.append(i)
delete_index = np.array(delete_index)
#It may be possible to do this without using two for loops but it is still pretty fast. 
comp_index = np.delete(comp_index, delete_index) #Delete the galaxies that is over the distance limit
ncomp = len(comp_index)
datagrid_SDSS = datagrid_SDSS[comp_index]
errgrid_SDSS = errgrid_SDSS[comp_index]
datagrid_vec = datagrid_vec[comp_index,:]
#only uses the grid that has more than one galaxies and within the distance limit.
remove_index_empty = np.where(ngrid_SDSS == 0)[0] #These are the grids with 0 galaxies in them. 
remove_index = np.union1d(remove_index_empty, comp_index_delete) #Delete the component of the covariance matrix if it has
#no galaxies in it or over the distance limit. 
ngrid_SDSS = ngrid_SDSS[comp_index]

conv_pk = np.delete(np.delete(conv_pk, remove_index, axis=0), remove_index, axis=1)
conv_pk_ng = np.delete(np.delete(conv_pk_ng, remove_index, axis=0), remove_index, axis=1)
print(np.shape(conv_pk), np.shape(conv_pk_ng))


# Correct the data and covariance matrix for the gridding. Eqs. 19 and 22.
datagrid_SDSS /= ngrid_SDSS        # We summed the velocities in each cell, now get the mean
errgrid_SDSS /= ngrid_SDSS**2.0    # This is the standard error on the mean.
for i in range(ncomp):
    conv_pk[i][i] += (conv_pk_ng[i][i] - conv_pk[i][i])/ngrid_SDSS[i]

if __name__ == "__main__":
    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    ndim, nwalkers = 2, 16
    
    # Set up the first points for the chain (for emcee we need to give each walker a random value about this point so we just sample the prior or a reasonable region)
    begin = [[1.0*np.random.rand(), 1000.0*np.random.rand()] for i in range(nwalkers)]
    
    # Set up the output file
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, ndim)
    
    with Pool(processes=num_processor) as pool:
    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, pool=pool, backend=backend)
        
        # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
        # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
        # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
        max_iter = 10000
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
            # Set up the MCMC
# # How many free parameters and walkers (this is for emcee's method)
# ndim, nwalkers = 2, 16

# # Set up the first points for the chain (for emcee we need to give each walker a random value about this point so we just sample the prior or a reasonable region)
# begin = [[1.0*np.random.rand(), 1000.0*np.random.rand()] for i in range(nwalkers)]

# # Set up the output file
# backend = emcee.backends.HDFBackend(chainfile)
# backend.reset(nwalkers, ndim)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, backend=backend)

# # Run the sampler for a max of 20000 iterations. We check convergence every 100 steps and stop if
# # the chain is longer than 100 times the estimated autocorrelation time and if this estimate
# # changed by less than 1%. I copied this from the emcee site as it seemed reasonable.
# max_iter = 10000
# index = 0
# old_tau = np.inf
# autocorr = np.empty(max_iter)
# counter = 0
# for sample in sampler.sample(begin, iterations=max_iter, progress=progress):

#     # Only check convergence every 100 steps
#     if sampler.iteration % 100:
#         continue

#     # Compute the autocorrelation time so far
#     # Using tol=0 means that we'll always get an estimate even
#     # if it isn't trustworthy
#     tau = sampler.get_autocorr_time(tol=0)
#     autocorr[index] = np.mean(tau)
#     counter += 100
#     print("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
#     print("Mean Auto-Correlation time: {0:.3f}".format(autocorr[index]))

#     # Check convergence
#     converged = np.all(tau * 100 < sampler.iteration)
#     converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
#     if converged:
#         print("Reached Auto-Correlation time goal: %d > 100 x %.3f" % (counter, autocorr[index]))
#         break
#     old_tau = tau
#     index += 1

# print(read_chain_backend(chainfile))