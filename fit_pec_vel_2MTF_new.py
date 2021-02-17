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
from scipy.linalg import lapack
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
    err_rand = datagrid_vec[0:,4]*sigma_varr/np.sqrt(ngrid_2MTF)
    conv_pk_diag = conv_pk_new[np.diag_indices(ncomp)]
    np.fill_diagonal(conv_pk_new, conv_pk_diag + errgrid_2MTF + err_rand**2)

    # Calculate the determinant and inverse of the covariance matrix 
    pivots = np.zeros(ncomp, np.intc)
    conv_pk_copy, pivots, info = lapack.dgetrf(conv_pk_new)
    abs_element = np.fabs(np.diagonal(conv_pk_copy))
    det = np.sum(np.log(abs_element))

#    identity = np.eye(ncomp)
#    ident = np.copy(identity)
#    conv_pk_lu, pivots2, cov_inv, info2 = lapack.dgesv(conv_pk_new, ident)
    
    cov_inv = sp.linalg.inv(conv_pk_new)

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
  
  
    chi_squared = np.matmul(datagrid_2MTF.T, np.matmul(cov_inv, datagrid_2MTF))
    #print(chi_squared)
    # return the log likelihood (Log of Eq. 23)
    
    # return -0.5*(det+np.log(1.0+cov_fac1*sigma_y)+chi_squared)
    return -0.5*(det+chi_squared)


# Run parameters
global fsigma8_old, conv_pk, ncomp, ngrid_2MTF, errgrid_2MTF, datagrid_vec, datagrid_2MTF
start_time = time.time()
omega_m = 0.3121             # Used to convert the galaxy redshifts to distances. Should match that used to compute the covariance matrix
fsigma8_old = 0.8150         # The value of fsigma8 used to compute the covariance matrix. 
sigma_y = 0.2**2             # The value of sigma_y used for Eq. 24.

# Make sure these match the values used to estimate the covariance matrix
kmin = float(sys.argv[1])    # What kmin to use
kmax = float(sys.argv[2])    # What kmax to use
gridsize = int(sys.argv[3])  # What gridsize to use
progress = bool(sys.argv[4])  # Whether or not to print out progress

# Read in the data
# datafile = str("./2MTF_mocks/MOCK_HAMHOD_2MTF_v4_R19000.0_err")
# covfile = str("./conv_pk_vel_2MTF_k0p%03d_0p%03d_gridcorr%02d.dat" % ((int)(1000.0*kmin), (int)(1000.0*kmax), gridsize))
# covfile_ng = str("./conv_pk_vel_2MTF_k0p%03d_0p%03d_gridcorr%02d_ng.dat" % ((int)(1000.0*kmin), (int)(1000.0*kmax), gridsize))
datafile = str("2MTF_pkin.dat")

covfile = str('conv_pk_2MTF_k0p007_0p150_vel_vel_gridcorr20.dat')
covfile_ng = str('conv_pk_2MTF_k0p007_0p150_vel_vel_gridcorr20_ng.dat')

chainfile = str('./fit_pec_vel_2MTF_k0p%03d_0p%03d_gridcorr_%d_0.hdf5' % (int(1000.0*kmin), int(1000.0*kmax), gridsize))

print(datafile)
print(covfile)
print(covfile_ng)
print(chainfile)

gridsize = float(gridsize)

# Generate the grid used to compute the covariance matrix. Again make sure these match the covariance matrix code.
xmin, xmax = -120.0, 120.0
ymin, ymax = -120.0, 120.0
zmin, zmax = -120.0, 120.0
nx = int(np.ceil((xmax-xmin)/gridsize))
ny = int(np.ceil((ymax-ymin)/gridsize))
nz = int(np.ceil((zmax-zmin)/gridsize))
nelements = nx*ny*nz
print(nx, ny, nz)

# Compute some useful quantities on the grid. For this we need a redshift-dist lookup table
nbins = 1000
redmax = 0.5
red = np.empty(nbins)
dist = np.empty(nbins)
for i in range(nbins):
    red[i] = i*redmax/nbins
    dist[i] = DistDc(red[i], omega_m, 1.0-omega_m, 0.0, 100.0, -1.0, 0.0, 0.0)
red_spline = sp.interpolate.splrep(dist, red, s=0) 

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

# # Read in the 2MTF data
# data_2MTF = []
# infile = open(datafile, 'r')
# for line in infile:
#     ln = line.split()
#     if (ln[0][0] == '#'):
#         continue
#     ra = float(ln[1])
#     dec = float(ln[2])
#     cz = float(ln[3])
#     logdist = float(ln[5])
#     logdist_err = float(ln[6])
#     r = DistDc(cz/LightSpeed, omega_m, 1.0-omega_m, 0.0, 100.0, -1.0, 0.0, 0.0)
#     x = -r*np.sin(np.pi/180.0*dec)
#     y = r*np.cos(np.pi/180.0*dec)*np.sin(np.pi/180.0*ra - np.pi)
#     z = r*np.cos(np.pi/180.0*dec)*np.cos(np.pi/180.0*ra - np.pi)
#     data_2MTF.append((x,y,z,logdist,logdist_err))
# data_2MTF = np.array(data_2MTF)
# infile.close()

data_2MTF_full = []
distance = []
infile = open(datafile, 'r')
for line in infile:
    ln = line.split()
    if (ln[0][0] == '#'):
        continue
    ra = float(ln[0])
    dec = float(ln[1])
    cz = float(ln[2])
    logdist = float(ln[3])
    logdist_err = float(ln[4])
    r = DistDc(cz/LightSpeed, omega_m, 1.0-omega_m, 0.0, 100.0, -1.0, 0.0, 0.0)
    x = -r*np.sin(np.pi/180.0*dec)
    y = r*np.cos(np.pi/180.0*dec)*np.sin(np.pi/180.0*ra - np.pi)
    z = r*np.cos(np.pi/180.0*dec)*np.cos(np.pi/180.0*ra - np.pi)
    data_2MTF_full.append((x,y,z,logdist,logdist_err))
    distance.append(logdist)
data_2MTF_full = np.array(data_2MTF_full)
infile.close()
print(len(data_2MTF_full))

data_2MTF = []
data_2MTF_mean = np.mean(distance)
for i in range(len(data_2MTF_full)):
    clipcrit = np.sqrt((data_2MTF_full[i][3]-data_2MTF_mean)**2/data_2MTF_full[i][4]**2)
    if clipcrit >= 4.0:
        continue
    data_2MTF.append(data_2MTF_full[i])
data_2MTF = np.array(data_2MTF)
print(len(data_2MTF))


# Grid the data and find out which grid cells are
# empty and as such don't need including from our theoretical covariance matrix
clipflag = np.zeros(nelements)
ngrid_2MTF = np.zeros(nelements)
datagrid_2MTF = np.zeros(nelements)
errgrid_2MTF = np.zeros(nelements)
for i in range(len(data_2MTF)):
    ix = int(np.floor((data_2MTF[i,0]-xmin)/gridsize))
    iy = int(np.floor((data_2MTF[i,1]-ymin)/gridsize))
    iz = int(np.floor((data_2MTF[i,2]-zmin)/gridsize))
    if (ix == nx):
        ix = nx-1
    if (iy == ny):
        iy = ny-1
    if (iz == nz):
        iz = nz-1
    ind = int((ix*ny+iy)*nz+iz)
    ngrid_2MTF[ind] += 1.0 
    datagrid_2MTF[ind] += data_2MTF[i,3]
    errgrid_2MTF[ind] += data_2MTF[i,4]**2
print(np.sum(ngrid_2MTF))

# Read in the covariance matrices
conv_pk = np.array(pd.read_csv(covfile, delim_whitespace=True, header=None, skiprows=1))/1.0e6          # The C code multiplies by a factor of 1x10^6 so we don't lose precision when writing to file
conv_pk_ng = np.array(pd.read_csv(covfile_ng, delim_whitespace=True, header=None, skiprows=1))/1.0e6    # The C code multiplies by a factor of 1x10^6 so we don't lose precision when writing to file
print(np.shape(conv_pk), np.shape(conv_pk_ng))

# Compress the datagrid based on whether or not there are any galaxies in that cell. Do the same for the covariance matrix.
# We are removing all elements from our matrices that have no data in them, as these should not count towards the likelihood.
comp_index = np.where(ngrid_2MTF > 0)[0]
ncomp = len(comp_index)
datagrid_2MTF = datagrid_2MTF[comp_index]
errgrid_2MTF = errgrid_2MTF[comp_index]
datagrid_vec = datagrid_vec[comp_index,:]
remove_index = np.where(ngrid_2MTF == 0)[0]
conv_pk = np.delete(np.delete(conv_pk, remove_index, axis=0), remove_index, axis=1)
conv_pk_ng = np.delete(np.delete(conv_pk_ng, remove_index, axis=0), remove_index, axis=1)
ngrid_2MTF = ngrid_2MTF[comp_index]

# Correct the data and covariance matrix for the gridding. Eqs. 19 and 22.
datagrid_2MTF /= ngrid_2MTF        # We summed the velocities in each cell, now get the mean
errgrid_2MTF /= ngrid_2MTF**2.0    # This is the standard error on the mean.
for i in range(ncomp):
    conv_pk[i][i] += (conv_pk_ng[i][i] - conv_pk[i][i])/ngrid_2MTF[i]
    
if __name__ == "__main__":

    # Set up the MCMC
    # How many free parameters and walkers (this is for emcee's method)
    ndim, nwalkers = 2, 16
    
    # Set up the first points for the chain (for emcee we need to give each walker a random value about this point so we just sample the prior or a reasonable region)
    begin = [[1.0*np.random.rand(), 1000.0*np.random.rand()] for i in range(nwalkers)]
    
    # Set up the output file
    backend = emcee.backends.HDFBackend(chainfile)
    backend.reset(nwalkers, ndim)
    
    with Pool() as pool:
    
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnpost, pool=pool, backend=backend)
        # sampler.run_mcmc(begin, 10000, progress=True)
    
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
    
    # print(read_chain_backend(chainfile))
    
    # samples, log_likelihood, max_log_likelihood = read_chain_backend(chainfile)
    
    # c = ChainConsumer()
    # real = [0.432,254]
    # c.add_chain(samples, parameters = [ "$f\sigma_8$", "$\sigma_v$"], name = "mock2")
    
    # c.configure(legend_artists=True)
    # c.plotter.plot(filename = "new_mock_0.png", figsize = "column", parameters = ["$f\sigma_8$", "$\sigma_v$"], truth = real)
    
