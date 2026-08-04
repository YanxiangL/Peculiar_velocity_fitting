# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 09:37:43 2025

@author: s4479813
"""

import numpy as np
import scipy as sp
import sys
import time
import copy
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from emcee import backends
import pandas as pd
from configobj import ConfigObj
from astropy.io import fits
import struct
import jax.numpy as jnp
import jax
from jax.scipy.linalg import cho_factor, cho_solve

# from jax.scipy.optimize import minimize
from jaxopt import ScipyBoundedMinimize
import warnings

# import numpyro
# from numpyro.infer import MCMC, SA
# import numpyro.distributions as dist
# from functools import partial
# from jax import random
# import h5py
from scipy.stats import skew, kurtosis


# Speed of light in km/s
LightSpeed = 299792.458


def get_max_dL_cholesky(
    params_fid,
    conv_f,
    conv_sigma_v,
    conv_noise,
    x_2,
    sigmab_square,
    datagrid_comp,
    fsigma8_old,
    flag=0,
):
    ffid, sigma_v_fid = params_fid

    ffid /= fsigma8_old

    # Use in-place multiplications instead of separate memory allocations
    Cfid = (
        sigma_v_fid**2 * conv_sigma_v + conv_noise
    )  # Start with the base covariance matrix
    Cfid += ffid**2 * conv_f

    # Cholesky decomposition for efficient inversion
    L, lower = cho_factor(Cfid)

    # Solve linear systems using Cholesky factor
    x2_Cinv = cho_solve((L, lower), x_2)
    chi_sq_x2 = jnp.dot(x_2.T, x2_Cinv)

    factor_u = 1.0 + chi_sq_x2 * sigmab_square
    factor_1 = jnp.log(factor_u)

    factor_a = jnp.matmul(datagrid_comp, x2_Cinv)
    factor_b = chi_sq_x2 + sigmab_square ** (-1)
    factor_3 = -(factor_a**2) / factor_b

    d_Cinv = cho_solve((L, lower), datagrid_comp)
    chisquared_fid = jnp.matmul(datagrid_comp.T, d_Cinv)

    # Compute determinant efficiently without explicit matrix operations
    detCfid = 2 * jnp.sum(jnp.log(jnp.diag(L)))

    # Compute final log likelihood
    loglikefid = -0.5 * (chisquared_fid + detCfid + factor_1 + factor_3)

    # if deri_cal == False:

    #     return loglikefid

    # else:
    #     return loglikefid, sigmag_array, Cfid, x2_Cinv, factor_u, factor_a, factor_b, d_Cinv

    if flag == 0:
        return loglikefid
    elif flag == 1:
        return chisquared_fid
    else:
        return loglikefid, Cfid, x2_Cinv, factor_u, factor_a, factor_b, d_Cinv


def grad_dL(
    params_fid,
    conv_f,
    conv_sigma_v,
    conv_noise,
    x_2,
    sigmab_square,
    datagrid_comp,
    fsigma8_old,
    func_dL,
    func_d2L,
):
    loglikefid, Cfid, x2_Cinv, factor_u, factor_a, factor_b, d_Cinv = func_dL(
        params_fid,
        conv_f,
        conv_sigma_v,
        conv_noise,
        x_2,
        sigmab_square,
        datagrid_comp,
        fsigma8_old,
        flag=2,
    )

    ffid, sigma_v_fid = params_fid

    # Find the inverse of the fiducial covariance matrix.
    Cfid_inv = jnp.linalg.inv(Cfid)

    factor_x2 = -jnp.outer(x2_Cinv, x2_Cinv)
    factor_1_derivative = 1.0 / (factor_u) * sigmab_square * factor_x2

    # da_dC = jnp.outer(jnp.matmul(-Cfid_inv, datagrid_comp), x2_Cinv)
    da_dC = jnp.outer(-d_Cinv, x2_Cinv)
    db_dC = factor_x2
    factor_3_derivative = (
        -2.0 * factor_a * da_dC * factor_b + db_dC * factor_a**2
    ) / factor_b**2

    # Derivatices of the analytical covariance matrix with respect to each free parameter.
    dCdf = 2.0 * ffid * conv_f / fsigma8_old**2
    dCdsigmav = 2.0 * sigma_v_fid * conv_sigma_v

    # t13 = jnp.matmul(datagrid_comp, Cfid_inv)

    # # Derivatives of loglikelihood function with respect to the analytical covariance matrix. The formula is obtained from the Matrix Cookbook.
    # weirdbit = jnp.outer(t13, t13)
    weirdbit = jnp.outer(d_Cinv, d_Cinv.T)

    # The first derivative of the log likelihood with respect to the covariance matrix.
    dLdC = -0.5 * (Cfid_inv - weirdbit + factor_1_derivative + factor_3_derivative)

    # The first derivative of the likelihood function with respect to each free parameter.
    # dLdb = jnp.trace(jnp.matmul(dLdC, dCdb))
    # dLdbadd = jnp.trace(jnp.matmul(dLdC, dCdbadd))
    # dLdf = jnp.trace(jnp.matmul(dLdC, dCdf))
    # dLdsigmag = jnp.trace(jnp.matmul(dLdC, dCdsigmag))
    # dLdsigmav = jnp.trace(jnp.matmul(dLdC, dCdsigmav))

    # Using np.einsum to calculate the trace. Much faster than calculating the matrix product first and then calculate the trace.
    dLdf = jnp.einsum("ij, ji->", dLdC, dCdf)
    dLdsigmav = jnp.einsum("ij, ji->", dLdC, dCdsigmav)

    dL = jnp.array([dLdf, dLdsigmav])

    t_1 = 0.5
    t_11 = factor_u / sigmab_square
    t_13 = 1 / (2 * t_11)
    t_15 = jnp.dot(x_2, d_Cinv.T)
    t_16 = t_11**2
    t_18 = 2 * (t_15**2)
    t_20 = t_18 / (4 * t_16)
    t_22 = (1 / t_16) * t_15
    t_25 = 1 / t_11
    t_26 = t_25 * t_15

    # d2L_dC_db = func_d2L(Cfid_inv, dCdb, x_2, datagrid_comp, jnp.sqrt(sigmab_square))
    # d2L_dC_dbadd = func_d2L(Cfid_inv, dCdbadd, x_2, datagrid_comp, jnp.sqrt(sigmab_square))
    # d2L_dC_df = func_d2L(Cfid_inv, dCdf, x_2, datagrid_comp, jnp.sqrt(sigmab_square))
    # d2L_dC_dsigmag = func_d2L(Cfid_inv, dCdsigmag, x_2, datagrid_comp, jnp.sqrt(sigmab_square))
    # d2L_dC_dsigmav = func_d2L(Cfid_inv, dCdsigmav, x_2, datagrid_comp, jnp.sqrt(sigmab_square))

    # functionValue = np.trace(np.dot(dCdb, (((((t_1 * np.outer(d_Cinv, d_Cinv.T)) - (t_1 * Cfid_inv)) + ((1 / (2 * factor_u/sigmab_square)) * -factor_x2)) + ((1 / (4 * ((factor_u/sigmab_square) ** 2))) * (((2 * (factor_a ** 2)) * -factor_x2) - (((4 * factor_u/sigmab_square) * factor_a) * -da_dC)))))))

    d2L_dC_df = func_d2L(
        Cfid_inv,
        dCdf,
        x_2,
        datagrid_comp,
        t_1,
        x2_Cinv,
        d_Cinv,
        d_Cinv.T,
        x2_Cinv.T,
        t_11,
        t_13,
        -factor_x2,
        t_15,
        t_16,
        t_18,
        t_20,
        t_22,
        -da_dC,
        t_25,
        t_26,
    )
    d2L_dC_dsigmav = func_d2L(
        Cfid_inv,
        dCdsigmav,
        x_2,
        datagrid_comp,
        t_1,
        x2_Cinv,
        d_Cinv,
        d_Cinv.T,
        x2_Cinv.T,
        t_11,
        t_13,
        -factor_x2,
        t_15,
        t_16,
        t_18,
        t_20,
        t_22,
        -da_dC,
        t_25,
        t_26,
    )

    # d2L_db_db = jnp.trace(jnp.matmul(d2L_dC_db, dCdb))
    # d2L_db_dbadd = jnp.trace(jnp.matmul(d2L_dC_db, dCdbadd))
    # d2L_db_df = jnp.trace(jnp.matmul(d2L_dC_db, dCdf))
    # d2L_db_dsigmag = jnp.trace(jnp.matmul(d2L_dC_db, dCdsigmag))
    # d2L_db_dsigmav = jnp.trace(jnp.matmul(d2L_dC_db, dCdsigmav))

    # d2L_dbadd_dbadd = jnp.trace(jnp.matmul(d2L_dC_dbadd, dCdbadd))
    # d2L_dbadd_df = jnp.trace(jnp.matmul(d2L_dC_dbadd, dCdf))
    # d2L_dbadd_dsigmag = jnp.trace(jnp.matmul(d2L_dC_dbadd, dCdsigmag))
    # d2L_dbadd_dsigmav = jnp.trace(jnp.matmul(d2L_dC_dbadd, dCdsigmav))

    # d2L_df_df = jnp.trace(jnp.matmul(d2L_dC_df, dCdf))
    # d2L_df_dsigmag = jnp.trace(jnp.matmul(d2L_dC_df, dCdsigmag))
    # d2L_df_dsigmav = jnp.trace(jnp.matmul(d2L_dC_df, dCdsigmav))

    # d2L_dsigmag_dsigmag = jnp.trace(jnp.matmul(d2L_dC_dsigmag, dCdsigmag))
    # d2L_dsigmag_dsigmav = jnp.trace(jnp.matmul(d2L_dC_dsigmag, dCdsigmav))

    # d2L_dsigmav_dsigmav = jnp.trace(jnp.matmul(d2L_dC_dsigmav, dCdsigmav))

    d2L_df_df = jnp.einsum("ij, ji->", d2L_dC_df, dCdf)
    d2L_df_dsigmav = jnp.einsum("ij, ji->", d2L_dC_df, dCdsigmav)

    d2L_dsigmav_dsigmav = jnp.einsum("ij, ji->", d2L_dC_dsigmav, dCdsigmav)

    d2L = jnp.array(
        [[d2L_df_df, d2L_df_dsigmav], [d2L_df_dsigmav, d2L_dsigmav_dsigmav]]
    )

    return loglikefid, dL, d2L


def d_dL_dm_dC(
    T_0,
    M,
    x,
    s,
    t_1,
    t_2,
    t_8,
    t_9,
    t_10,
    t_11,
    t_13,
    T_14,
    t_15,
    t_16,
    t_18,
    t_20,
    t_22,
    T_24,
    t_25,
    t_26,
):
    # T_0 is the inverse of the covariance matrix, M is the first derivative of the covariance matrix with respect to the parameters, x is the zero-point
    # correction vector, s is the data vector and y is sigmab, the uncertainty on the zero point.

    t_12 = jnp.dot(T_0, jnp.dot(M, t_10))

    t_17 = jnp.dot(x, t_12)
    T_19 = jnp.outer(t_12, t_2)
    T_21 = jnp.outer(t_10, jnp.dot(T_0, jnp.dot(M, t_2)))
    t_23 = jnp.dot(s, t_12)
    t_27 = jnp.dot(T_0, jnp.dot(M, t_8))

    gradient = (
        (
            (
                (
                    (
                        (
                            (
                                (
                                    (t_1 * jnp.linalg.multi_dot([T_0, M, T_0]))
                                    - (
                                        (
                                            t_1
                                            * jnp.outer(
                                                jnp.dot(T_0, jnp.dot(M, t_9)), t_8
                                            )
                                        )
                                        + (t_1 * jnp.outer(t_9, t_27))
                                    )
                                )
                                + (((1 / (t_16 * 2)) * t_17) * T_14)
                            )
                            - (t_13 * T_19)
                        )
                        - (t_13 * T_21)
                    )
                    + (
                        (
                            (8 / ((t_11) * 16))
                            * jnp.einsum(
                                "ij, ji ->",
                                (
                                    (t_18 * T_14)
                                    - (((4 * t_11) * t_15) * jnp.outer(t_10, t_8))
                                )
                                / t_11,
                                M / t_11,
                            )
                        )
                        * T_14
                    )
                )
                - (t_22 * (t_17 * T_24))
            )
            - (t_20 * T_19)
        )
        - (t_20 * T_21)
    ) + (
        (
            ((t_22 * (t_23 * T_14)) + ((t_25 * t_23) * T_24))
            + (t_26 * jnp.outer(t_12, t_8))
        )
        + (t_26 * jnp.outer(t_10, t_27))
    )

    return gradient


def Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    """
    Calculates H(z)/H0

    Parameters
    ----------
    redshift : float
        Redshift of the galaxy.
    omega_m : float
        Total matter density.
    omega_lambda : float
        Dark energy density.
    omega_rad : float
        Radiation density.
    w0 : float
        The dark energy equation of state at redshift zero.
    wa : float
        The time evolution of the dark energy equation of state.
    ap : float
        The pivot redshift.

    Returns
    -------
    float
        H(z)/H0.

    """
    fz = ((1.0 + redshift) ** (3 * (1.0 + w0 + wa * ap))) * jnp.exp(
        -3 * wa * (redshift / (1.0 + redshift))
    )
    omega_k = 1.0 - omega_m - omega_lambda - omega_rad
    return jnp.sqrt(
        omega_rad * (1.0 + redshift) ** 4
        + omega_m * (1.0 + redshift) ** 3
        + omega_k * (1.0 + redshift) ** 2
        + omega_lambda * fz
    )


# The Comoving Distance Integrand
def DistDcIntegrand(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap):
    """
    The integrand of the comoving distance

    Parameters
    ----------
    redshift : float
        Redshift of the galaxy.
    omega_m : float
        Total matter density.
    omega_lambda : float
        Dark energy density.
    omega_rad : float
        Radiation density.
    w0 : float
        The dark energy equation of state at redshift zero.
    wa : float
        The time evolution of the dark energy equation of state.
    ap : float
        The pivot redshift.

    Returns
    -------
    float
        The integrand of the comoving distance.

    """
    return 1.0 / Ez(redshift, omega_m, omega_lambda, omega_rad, w0, wa, ap)


# The Comoving Distance in Mpc
def DistDc(redshift, omega_m, omega_lambda, omega_rad, Hubble_Constant, w0, wa, ap):
    """
    Calculating the comoving distance in Mpc.

    Parameters
    ----------
    Parameters
    ----------
    redshift : float
        Redshift of the galaxy.
    omega_m : float
        Total matter density.
    omega_lambda : float
        Dark energy density.
    omega_rad : float
        Radiation density.
    w0 : float
        The dark energy equation of state at redshift zero.
    wa : float
        The time evolution of the dark energy equation of state.
    ap : float
        The pivot redshift.

    Returns
    -------
    float
        The comoving distance in Mpc.

    """
    return (LightSpeed / Hubble_Constant) * integrate.quad(
        DistDcIntegrand,
        0.0,
        redshift,
        args=(omega_m, omega_lambda, omega_rad, w0, wa, ap),
    )[0]


def grid_corr(theta, phi, k, L):
    """
    Calculating the grid correction.

    Parameters
    ----------
    theta : Numpy array
        The polar angle.
    phi : Numpy array
        The azimuthal angle.
    k : Numpy array
        The wavevector.
    L : float
        The size of the grid.

    Returns
    -------
    Numpy array
        The grid correction with respect to k.

    """
    k_x = (k * L / 2.0) * jnp.sin(phi) * jnp.cos(theta)
    k_y = (k * L / 2.0) * jnp.sin(phi) * jnp.sin(theta)
    k_z = (k * L / 2.0) * jnp.cos(phi)

    return (
        jnp.sinc(k_x / jnp.pi)
        * jnp.sinc(k_y / jnp.pi)
        * jnp.sinc(k_z / jnp.pi)
        * jnp.sin(phi)
        / (4.0 * jnp.pi)
    )


def tri_2_full(array):
    """
    Converting the upper triangular matrix to the full covariance matrix

    Parameters
    ----------
    array : Numpy array
        The 1D array containing the upper triangular part of the covariance matrix.

    Returns
    -------
    output : Numpy array
        The full covariance matrix.

    """

    if len(np.shape(array)) > 1:
        array = array[:, 0]

    n = int(-1 + np.sqrt(1 + 8 * len(array))) // 2
    iu1 = np.triu_indices(n)
    ret = np.empty((n, n))
    ret[iu1] = array
    ret.T[iu1] = array

    return ret


def convert_bin_2_float(filename, size=8, order=0.0, cross=False):
    """
    Convert the binary covariance matrix to float

    Parameters
    ----------
    filename : str
        Location of the covariance matrix file.
    size : int, optional
        The size of the data type. The default is 8 for double.

    Returns
    -------
    output : Numpy array
        The covariance matrix in float64.

    """

    # output = []
    # with open(filename, 'rb') as f:
    #     while True:
    #         chunk = f.read(size)
    #         if not chunk:
    #             break
    #         # convert each chunk to float
    #         output.append(struct.unpack('d', chunk))

    with open(filename, mode="rb") as file:  # b is important -> binary
        fileContent = file.read()
    output = (
        np.array(struct.unpack("d" * (len(fileContent) // 8), fileContent))
        / 10.0**order
    )

    if cross == False:
        return tri_2_full(output)
    else:
        nlength = np.int32(jnp.sqrt(len(output)))
        return np.reshape(output, (nlength, nlength))


def D_integrand(a, omega_m):
    E = Ez(1.0 / a - 1.0, omega_m, 1.0 - omega_m, 0.0, -1.0, 0.0, 0.0)
    return 1.0 / (a**3 * E**3)


# def model(wrapped_func):
#     fsigma8 = numpyro.sample(r"$f\sigma_8$", dist.Uniform(prior_low_new[0], prior_high_new[0]))
#     sigmav = numpyro.sample(r"$\sigma_v$", dist.Uniform(prior_low_new[1], prior_high_new[1]))

#     params = jnp.array([fsigma8, sigmav])
#     # Inject your custom log-likelihood using numpyro.factor
#     loglike = wrapped_func(params)
#     numpyro.factor("custom_loglike", loglike)


configfile = sys.argv[1]  # input the location of the configuration file
mock_num = int(sys.argv[2])

rea_num, sub_num = divmod(mock_num, 27)

pardict = ConfigObj(configfile)

# Reading in the input parameters from the config file.
expect_file = pardict["expect_file"]
# datafile = pardict['datafile']
omega_m = float(pardict["omega_m"])
fsigma8_old = float(pardict["fsigma8_old"])
sigmab_square = float(pardict["sigma_b"]) ** 2
bsigma8_old = float(pardict["bsigma8_old"])
r_g = float(pardict["r_g"])
kmin = float(pardict["kmin"])
kmax_galaxy = float(pardict["kmax_galaxy"])
kmax_velocity = float(pardict["kmax_velocity"])
gridsize = int(pardict["gridsize"])
progress = bool(int(pardict["progress"]))
sigma_u = float(pardict["sigma_u"])
sigma_g = float(pardict["sigma_g"])
effective_redshift = float(pardict["effective_redshift"])
# sigma8_eff = float(pardict['sigma8_eff'])
xmin = float(pardict["xmin"])
xmax = float(pardict["xmax"])
ymin = float(pardict["ymin"])
ymax = float(pardict["ymax"])
zmin = float(pardict["zmin"])
zmax = float(pardict["zmax"])
P0_dens = float(pardict["P0_dens"])
P0_vel = float(pardict["P0_vel"])
sigma_v_fid = float(pardict["sigma_v_fid"])
mock_data = bool(int(pardict["mock_data"]))

# Number of grids in each direction.
nx = jnp.int32(jnp.ceil((xmax - xmin) / gridsize))
ny = jnp.int32(jnp.ceil((ymax - ymin) / gridsize))
nz = jnp.int32(jnp.ceil((zmax - zmin) / gridsize))
nelements = nx * ny * nz
print(nx, ny, nz)


# Compute some useful quantities on the grid. For this we need a redshift-dist lookup table with scipy interpolation.
nbins = 5000
# The maximum redshift.
redmax = 0.5
red = np.empty(nbins)
distance = np.empty(nbins)
for i in range(nbins):
    red[i] = i * redmax / nbins
    distance[i] = DistDc(red[i], omega_m, 1.0 - omega_m, 0.0, 100.0, -1.0, 0.0, 0.0)
red_spline = sp.interpolate.splrep(
    distance, red, s=0
)  # Interpolate distance to get redshift.
radial_spline = sp.interpolate.splrep(
    red, distance, s=0
)  # Interpolate redshift to get comoving radial distance.

# This array stores the position, radial distance and the conversion factor (equation 23) from peculiar velocity to log-distance ratio for each grid.
# datagrid_vec = np.empty((nelements,5))
# for i in range(nx):
#     for j in range(ny):
#         for k in range(nz):
#             ind = (i*ny+j)*nz+k
#             x = (i+0.5)*gridsize+xmin
#             y = (j+0.5)*gridsize+ymin
#             z = (k+0.5)*gridsize+zmin
#             r = jnp.sqrt(x**2+y**2+z**2)
#             red = sp.interpolate.splev(r, red_spline, der=0)
#             ez = Ez(red, omega_m, 1.0-omega_m, 0.0, -1.0, 0.0, 0.0)
#             #Find the comoving cartesian coordinate of the center of each grid and the radial distance to the grid.
#             datagrid_vec[ind,0] = x
#             datagrid_vec[ind,1] = y
#             datagrid_vec[ind,2] = z
#             datagrid_vec[ind,3] = r
#             datagrid_vec[ind,4] = (1.0/jnp.log(10))*(1.0+red)/(100.0*ez*r)

index_all = np.arange(nelements)

x_index = index_all // (ny * nz)

y_index = (index_all // nz) % ny

z_index = index_all % nz

x_array = (x_index + 0.5) * gridsize + xmin

y_array = (y_index + 0.5) * gridsize + ymin

z_array = (z_index + 0.5) * gridsize + zmin

r_array = np.sqrt(x_array**2 + y_array**2 + z_array**2)

red = sp.interpolate.splev(r_array, red_spline, der=0)

ez = Ez(red, omega_m, 1.0 - omega_m, 0.0, -1.0, 0.0, 0.0)

logdist_array = (1.0 / np.log(10)) * (1.0 + red) / (100.0 * ez * r_array)

datagrid_vec = np.vstack((x_array, y_array, z_array, r_array, logdist_array)).T

datagrid_vec = jnp.array(datagrid_vec)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------

if rea_num < 10:
    index_rea = "00" + str(rea_num)
else:
    index_rea = "0" + str(rea_num)

if sub_num < 10:
    index_sub = "00" + str(sub_num)
else:
    index_sub = "0" + str(sub_num)

# datafile_name = pardict['datafile'] + '_ph' + index_rea + '_r' + index_sub + '_v2.fits'
if mock_data == True:
    datafile_pv_name = (
        pardict["datafile_pv"] + "_ph" + index_rea + "_r" + index_sub + ".fits"
    )
else:
    datafile_pv_name = pardict["datafile_pv"]

if pardict["filetype"] == "csv":
    data_pv = pd.read.csv(datafile_pv_name, sep="\s+")

elif pardict["filetype"] == "fits":
    try:
        hdul_data_pv = fits.open(datafile_pv_name)
    except:
        datafile_pv_name = (
            pardict["datafile_pv"] + "_ph" + index_rea + "_r" + index_sub + ".fits"
        )
        hdul_data_pv = fits.open(datafile_pv_name)
    data_pv = hdul_data_pv[1].data
else:
    raise ValueError(
        "Input file type is not supported. It can only read in csv or fits files"
    )

print(datafile_pv_name)

if pardict["weight"] == "full":
    weight_pv = data_pv["WEIGHT"]
    npv = data_pv["NPV"]

RA_pv = jnp.array(data_pv[pardict["RA_header"]])
Dec_pv = jnp.array(data_pv[pardict["Dec_header"]])
redshift_pv = jnp.array(data_pv[pardict["redshift_header"]])
if jnp.int16(pardict["deg"]) == 1:
    RA_pv = RA_pv / 180.0 * jnp.pi
    Dec_pv = Dec_pv / 180.0 * jnp.pi


# #Read in the data file.
# data = dict(pd.read_csv(datafile, delim_whitespace = True, skiprows=0))
rd_pv = sp.interpolate.splev(redshift_pv, radial_spline)
# RA = np.array(data["RA"])/180.0*np.pi
# Dec = np.array(data["Dec"])/180.0*np.pi
# redshift = np.array(data["zcmb"])

# Convert redshift to distance.
data_z_pv = jnp.sin(Dec_pv)
data_y_pv = jnp.cos(Dec_pv) * jnp.sin(RA_pv)
data_x_pv = jnp.cos(Dec_pv) * jnp.cos(RA_pv)

x_pv = data_x_pv * rd_pv
y_pv = data_y_pv * rd_pv
z_pv = data_z_pv * rd_pv


# Read in the log-distance ratios and their errors.
# log_dist = np.array(data["logdist_corr"])
# log_dist_err = np.array(data["logdist_corr_err"])

log_dist = jnp.array(data_pv[pardict["logdist_header"]])
log_dist_err = jnp.array(data_pv[pardict["logdist_err_header"]])

if pardict["weight"] == "full":
    # convert_pv_to_eta = jnp.array((1.0/np.log(10.))*(1.0+redshift_pv)/(100.0*Ez(redshift_pv, omega_m, 1.-omega_m, 0.0, -1.0, 0.0, 0.0)*rd_pv))

    # weight_pv = jnp.array(weight_pv)*(1.0/(sigma_v_fid**2 + (log_dist_err/convert_pv_to_eta)**2 + npv*P0_vel))*convert_pv_to_eta**2
    # weight_pv = jnp.ones_like(weight_pv)

    convert_pv_to_eta = np.array(
        (1.0 / np.log(10.0))
        * (1.0 + redshift_pv)
        / (100.0 * Ez(redshift_pv, omega_m, 1.0 - omega_m, 0.0, -1.0, 0.0, 0.0) * rd_pv)
    )

    # weight_pv = jnp.array(weight_pv)*(1.0/(sigma_v_fid**2 + (log_dist_err/convert_pv_to_eta)**2 + npv*P0_vel))*convert_pv_to_eta**2
    weight_pv = 1.0 / (
        log_dist_err**2 / convert_pv_to_eta + convert_pv_to_eta * npv * P0_vel
    )

data_count_pv = len(x_pv)

redshift_cut_0 = np.where(redshift_pv > np.float32(pardict["red_cut_high"]))[0]
redshift_cut_1 = np.where(redshift_pv <= np.float32(pardict["red_cut_low"]))[0]

redshift_cut = np.union1d(redshift_cut_0, redshift_cut_1)

print(
    "There are "
    + str(len(redshift_cut))
    + " galaxies above the redshift cut of "
    + str(pardict["red_cut_high"])
    + " and below redshift of "
    + str(pardict["red_cut_low"])
)
print(np.min(redshift_pv), np.max(redshift_pv))
log_dist = np.delete(log_dist, redshift_cut)
data_count_pv = data_count_pv - len(redshift_cut)
x_pv = np.delete(x_pv, redshift_cut)
y_pv = np.delete(y_pv, redshift_cut)
z_pv = np.delete(z_pv, redshift_cut)
log_dist_err = np.delete(log_dist_err, redshift_cut)

print(np.median(log_dist), np.std(log_dist), skew(log_dist), kurtosis(log_dist))

del_logdist = np.where(
    np.abs(log_dist - np.median(log_dist)) / np.std(log_dist)
    > np.float32(pardict["sigma_cut"])
)[0]
log_dist = np.delete(log_dist, del_logdist)
print(
    "The distribution of the log-distance ratio in the order of median, standard deviation, skewness, and kurtosis: ",
    np.median(log_dist),
    np.std(log_dist),
    skew(log_dist),
    kurtosis(log_dist),
)

# ------------------------------------------------------------------------------------------------------------------------------------------------
data_count_pv = data_count_pv - len(del_logdist)
x_pv = np.delete(x_pv, del_logdist)
y_pv = np.delete(y_pv, del_logdist)
z_pv = np.delete(z_pv, del_logdist)
log_dist_err = np.delete(log_dist_err, del_logdist)

if pardict["filetype"] == "fits":
    hdul_data_pv.close()
data_count_pv = len(x_pv)

# effective_redshift = np.mean(data['Z'])
# effective_redshift = 0.2

sigma8_eff = (
    sp.integrate.quad(
        D_integrand, 0.0, 1.0 / (1.0 + effective_redshift), args=(omega_m)
    )[0]
    / sp.integrate.quad(D_integrand, 0.0, 1.0, args=(omega_m))[0]
    * fsigma8_old
    * Ez(effective_redshift, omega_m, 1.0 - omega_m, 0.0, -1.0, 0.0, 0.0)
)

if pardict["PT_method"] == "SPT":
    factor_gg = 1.0
else:
    # These are the extra factors required to calculate constrain the growth rate at the effective redshift. If at redshift zero, all three factors
    # should be 1.
    factor_gg = sigma8_eff**2 / fsigma8_old**2
    warnings.warn(
        "Recommend using the SPT model. You must generate the RPT power spectrum model at redshift zero."
    )

factor_gv = (
    (1.0 / (1.0 + effective_redshift))
    * Ez(effective_redshift, omega_m, 1.0 - omega_m, 0.0, -1.0, 0.0, 0.0)
    * factor_gg
)
factor_vv = (
    (1.0 / (1.0 + effective_redshift))
    * Ez(effective_redshift, omega_m, 1.0 - omega_m, 0.0, -1.0, 0.0, 0.0)
) ** 2 * factor_gg
print(
    pardict["PT_method"],
    factor_gg,
    factor_gv,
    factor_vv,
    effective_redshift,
    sigma8_eff,
)

fsigma8_old = sigma8_eff

# #------------------------------------------------------------------------------------------------------------------------------------------------------------
# del_logdist = jnp.where(jnp.abs(log_dist - jnp.median(log_dist))/jnp.std(log_dist) > np.float32(pardict['sigma_cut']))[0]
# log_dist = jnp.delete(log_dist, del_logdist)
# print(jnp.median(log_dist), jnp.std(log_dist), skew(log_dist), kurtosis(log_dist))
# data_count_pv = data_count_pv - len(del_logdist)
# x_pv = jnp.delete(x_pv, del_logdist)
# y_pv = jnp.delete(y_pv, del_logdist)
# z_pv = jnp.delete(z_pv, del_logdist)
# log_dist_err = jnp.delete(log_dist_err, del_logdist)
# #-------------------------------------------------------------------------------------------------------------------------------------------------------------

x_pv = jnp.reshape(x_pv, (data_count_pv, 1))
y_pv = jnp.reshape(y_pv, (data_count_pv, 1))
z_pv = jnp.reshape(z_pv, (data_count_pv, 1))

log_dist = jnp.reshape(log_dist, (data_count_pv, 1))
log_dist_err = jnp.reshape(log_dist_err, (data_count_pv, 1))

data_DESI_pv = jnp.concatenate((x_pv, y_pv, z_pv, log_dist, log_dist_err), axis=1)

# Just checking all the data is within the grid we defined.
print(
    jnp.min(x_pv),
    jnp.max(x_pv),
    jnp.min(y_pv),
    jnp.max(y_pv),
    jnp.min(z_pv),
    jnp.max(z_pv),
)

ngrid_SDSS_pv = jnp.zeros(nelements)
if pardict["weight"] == "none":
    datagrid_SDSS = jnp.zeros(nelements)
    errgrid_SDSS = jnp.zeros(nelements)
elif pardict["weight"] == "invar":
    datagrid_SDSS_weight = jnp.zeros(nelements)
    errgrid_SDSS_weight = jnp.zeros(nelements)
elif pardict["weight"] == "full":
    datagrid_SDSS_weight_FKP = jnp.zeros(nelements)
    errgrid_SDSS_weight_FKP = jnp.zeros(nelements)
    weight_sum_SDSS = jnp.zeros(nelements)
else:
    raise ValueError(
        'Only support three types of weight, enter "None" for no weight, enter "invar" for the inverse variance weight, enter "full" for WKP + incompleteness weight'
    )

# Compute indices for PV data (only up to its length)
valid_pv = jnp.arange(len(data_DESI_pv))
ix_pv = jnp.floor((data_DESI_pv[valid_pv, 0] - xmin) / gridsize).astype(int)
iy_pv = jnp.floor((data_DESI_pv[valid_pv, 1] - ymin) / gridsize).astype(int)
iz_pv = jnp.floor((data_DESI_pv[valid_pv, 2] - zmin) / gridsize).astype(int)

# Ensure indices remain within bounds
ix_pv = jnp.where(ix_pv == nx, nx - 1, ix_pv)
iy_pv = jnp.where(iy_pv == ny, ny - 1, iy_pv)
iz_pv = jnp.where(iz_pv == nz, nz - 1, iz_pv)

# Compute flattened indices
ind_pv = (ix_pv * ny + iy_pv) * nz + iz_pv

# Accumulate values using `.at[]`
ngrid_SDSS_pv = ngrid_SDSS_pv.at[ind_pv].add(1.0)
if pardict["weight"] == "none":
    datagrid_SDSS = datagrid_SDSS.at[ind_pv].add(data_DESI_pv[valid_pv, 3])
    errgrid_SDSS = errgrid_SDSS.at[ind_pv].add(data_DESI_pv[valid_pv, 4] ** 2)
elif pardict["weight"] == "invar":
    datagrid_SDSS_weight = datagrid_SDSS_weight.at[ind_pv].add(
        1.0 / data_DESI_pv[valid_pv, 4] ** 2 * data_DESI_pv[valid_pv, 3]
    )
    errgrid_SDSS_weight = errgrid_SDSS_weight.at[ind_pv].add(
        1.0 / data_DESI_pv[valid_pv, 4] ** 2
    )
elif pardict["weight"] == "full":
    datagrid_SDSS_weight_FKP = datagrid_SDSS_weight_FKP.at[ind_pv].add(
        weight_pv[valid_pv] * data_DESI_pv[valid_pv, 3]
    )
    errgrid_SDSS_weight_FKP = errgrid_SDSS_weight_FKP.at[ind_pv].add(
        (weight_pv[valid_pv] * data_DESI_pv[valid_pv, 4]) ** 2
    )
    weight_sum_SDSS = weight_sum_SDSS.at[ind_pv].add(weight_pv[valid_pv])

print(jnp.sum(ngrid_SDSS_pv))

# Cut out grids where there is no log-distance ratio measurements.
# comp_velocity = jnp.where(
#     (ngrid_SDSS_pv > 0)
#     & (
#         sp.interpolate.splev(datagrid_vec[:, 3], red_spline, der=0)
#         <= jnp.float32(pardict["red_cut"])
#     )
# )[0]
# ncomp_velocity = len(comp_velocity)
# ngrid_SDSS_vel = ngrid_SDSS_pv[comp_velocity]
comp_velocity = jnp.where(ngrid_SDSS_pv > 0)[0]
ncomp_velocity = len(comp_velocity)
ngrid_SDSS_vel = ngrid_SDSS_pv[comp_velocity]
# remove_velocity = jnp.where(
#     (ngrid_SDSS_pv == 0)
#     | (
#         sp.interpolate.splev(datagrid_vec[:, 3], red_spline, der=0)
#         > jnp.float32(pardict["red_cut"])
#     )
# )[0]
remove_velocity = jnp.where(ngrid_SDSS_pv == 0)[0]
if pardict["weight"] == "none":
    data_vel = datagrid_SDSS[comp_velocity]
    errgrid_SDSS = errgrid_SDSS[comp_velocity]

    # Correct the data and covariance matrix for the gridding. Eqs. 19 and 22.
    data_vel /= (
        ngrid_SDSS_vel  # We summed the velocities in each cell, now get the mean
    )
    errgrid_SDSS /= ngrid_SDSS_vel**2  # This is the standard error on the mean.
elif pardict["weight"] == "invar":
    data_vel_weight = datagrid_SDSS_weight[comp_velocity]
    errgrid_SDSS_weight = errgrid_SDSS_weight[comp_velocity]

    data_vel = data_vel_weight / errgrid_SDSS_weight
    errgrid_SDSS = 1.0 / errgrid_SDSS_weight
elif pardict["weight"] == "full":
    datagrid_SDSS_weight_FKP = datagrid_SDSS_weight_FKP[comp_velocity]
    errgrid_SDSS_weight_FKP = errgrid_SDSS_weight_FKP[comp_velocity]
    weight_sum_SDSS = weight_sum_SDSS[comp_velocity]
    data_vel = datagrid_SDSS_weight_FKP / weight_sum_SDSS
    errgrid_SDSS = errgrid_SDSS_weight_FKP / weight_sum_SDSS**2

median_vel = jnp.median(data_vel)

# index_del = jnp.where(jnp.sqrt((data_vel - median_vel)**2/errgrid_SDSS) > jnp.float32(pardict['sigma_cut']))[0]
# index_del = jnp.where(jnp.abs((data_vel - median_vel)/jnp.std(data_vel)) > jnp.float32(pardict['sigma_cut']))[0]
index_del = jnp.array([], dtype=int)
data_vel = jnp.delete(data_vel, index_del, axis=0)
errgrid_SDSS = jnp.delete(errgrid_SDSS, index_del, axis=0)

datagrid_comp = data_vel

datagrid_comp_new = datagrid_comp.reshape((len(datagrid_comp), 1))

length_vel = len(data_vel)

print("The length of the velocity data is " + str(length_vel))

x_2 = jnp.ones(length_vel)

print(len(comp_velocity) + len(remove_velocity))

start = time.time()

conv_vv = []

data_file_conv_vv = str(
    "%s_k0p%03d_0p%03d_gridcorr%02d_vv_sigmau%03d.bin"
    % (
        pardict["covfile_base"],
        (int)(1000.0 * kmin),
        (int)(1000.0 * kmax_velocity),
        int(pardict["gridsize"]),
        int(10.0 * sigma_u),
    )
)

print(data_file_conv_vv)

conv_vv_element = convert_bin_2_float(data_file_conv_vv)
# Delete the grid cells where there is no log-distance ratio measurements.
conv_vv_element = np.delete(
    np.delete(conv_vv_element, remove_velocity, axis=0), remove_velocity, axis=1
)

data_file_conv_vv_ng = str(
    "%s_k0p%03d_0p%03d_gridcorr%02d_vv_ng_sigmau%03d.bin"
    % (
        pardict["covfile_base"],
        (int)(1000.0 * kmin),
        (int)(1000.0 * kmax_velocity),
        int(pardict["gridsize"]),
        int(10.0 * sigma_u),
    )
)

# conv_vv_ng_element = np.array(pd.read_csv(data_file_conv_vv_ng, delim_whitespace=True, header=None, skiprows=1))/1.0e6
conv_vv_ng_element = convert_bin_2_float(data_file_conv_vv_ng)
conv_vv_ng_element = np.delete(
    np.delete(conv_vv_ng_element, remove_velocity, axis=0), remove_velocity, axis=1
)

# Accouting the for the shot-noise of the velocity auto-covariance matrix (equation (31) in the paper).
for k in range(ncomp_velocity):
    conv_vv_element[k][k] += (
        conv_vv_ng_element[k][k] - conv_vv_element[k][k]
    ) / ngrid_SDSS_vel[k]
conv_vv.append([factor_vv * conv_vv_element])
# conv_vv.append([factor_vv*conv_vv_ng_element])

end = time.time()
print(end - start)

# Only use grid cells where there is a log-distance ratio measurement.
datagrid_vec_new = datagrid_vec[comp_velocity, :]

# Convert these to full matrices so that we can write model as a single sum
conv_vv_del = conv_vv[0]
conv_vv_new = np.delete(np.delete(conv_vv_del, index_del, axis=1), index_del, axis=2)
print(np.shape(conv_vv_new), np.shape(datagrid_vec_new), np.shape(datagrid_comp_new))

# Construct each part of the full covariance matrix based on the analytical formula for the full covariance matrix (equation (C2) in the paper).
eye_vv = np.diag((datagrid_vec_new[0:, 4] / np.sqrt(ngrid_SDSS_vel)) ** 2)
eye_vv = np.delete(np.delete(eye_vv, index_del, axis=0), index_del, axis=1)

conv_f = conv_vv_new[0]

conv_sigma_v = eye_vv
conv_noise = np.diag(errgrid_SDSS)

conv_f = jnp.array(conv_f)
conv_sigma_v = jnp.array(conv_sigma_v)
conv_noise = jnp.array(conv_noise)

print(np.shape(conv_f), np.shape(conv_sigma_v), np.shape(conv_noise), np.shape(x_2))

# The fiducial value for the free parameters. They are set quite far away from the truth values.
ffid = 0.4 / fsigma8_old


params_fid = jnp.array([ffid, sigma_v_fid])
# params_fid = jnp.array([ffid, 0.0])


test_matrix = np.random.rand(10, 10)
test_cov = (test_matrix + test_matrix.T) / 2.0
sample = test_cov
test_data = jnp.array(np.random.rand(10, 1))


func_dL = jax.jit(get_max_dL_cholesky, static_argnums=8)
# pre-compile the functions
_ = func_dL(
    params_fid,
    sample,
    sample,
    sample,
    test_data[:, 0],
    sigmab_square,
    test_data[:, 0],
    fsigma8_old,
)

func_d2L = jax.jit(d_dL_dm_dC)
_ = func_d2L(
    test_matrix,
    test_matrix,
    test_data[:, 0],
    test_data[:, 0],
    sigmab_square,
    test_data[:, 0],
    test_data[:, 0],
    test_data[:, 0],
    test_data[:, 0],
    sigmab_square,
    sigmab_square,
    test_matrix,
    sigmab_square,
    sigmab_square,
    sigmab_square,
    sigmab_square,
    sigmab_square,
    test_matrix,
    sigmab_square,
    sigmab_square,
)

func_grad = jax.jit(grad_dL, static_argnums=(8, 9))
_ = func_grad(
    params_fid,
    sample,
    sample,
    sample,
    test_data[:, 0],
    sigmab_square,
    test_data[:, 0],
    fsigma8_old,
    func_dL,
    func_d2L,
)

# hess_func = jax.jacfwd(jax.jit(grad_func))

start = time.time()
score = np.ones_like(params_fid)
diff = np.ones_like(params_fid)
params_result = params_fid
while np.any(np.abs(score) > 1e-3) and np.any(np.abs(diff) > 1e-3):
    # result = minimize(lambda *args: -func_dL(*args), params_fid, method = 'BFGS', args=(conv_b, conv_bf, conv_f, conv_badd, conv_sigma_v, conv_noise, x_2, sigmab_square, datagrid_comp, datagrid_comp_new, False))
    minimizer = ScipyBoundedMinimize(
        fun=lambda *args: -func_dL(*args), method="L-BFGS-B", maxiter=500
    )
    # minimizer = ScipyBoundedMinimize(fun = lambda *args: func_dL(*args), method = 'SLSQP', maxiter = 500)
    # minimizer = LBFGSB(fun = lambda *args: -func_dL(*args), maxiter = 500)
    lower_bounds = jnp.array([0.0, 1.0])
    upper_bounds = jnp.array([1.0, 2000.0])
    bounds = (lower_bounds, upper_bounds)
    # minimizer.init_state(params_fid, bounds, conv_b, conv_bf, conv_f, conv_badd, conv_sigma_v, conv_noise, x_2, sigmab_square, datagrid_comp, datagrid_comp_new)
    result = minimizer.run(
        params_fid,
        bounds,
        conv_f,
        conv_sigma_v,
        conv_noise,
        x_2,
        sigmab_square,
        datagrid_comp,
        fsigma8_old,
    )
    # result = minimizer.run()
    print(result.params, result[1])
    loglike, score, Hessian = func_grad(
        result.params,
        conv_f,
        conv_sigma_v,
        conv_noise,
        x_2,
        sigmab_square,
        datagrid_comp,
        fsigma8_old,
        func_dL,
        func_d2L,
    )
    diff = (result.params - params_result) / result.params
    params_result = result.params
    params_fid = result.params - jnp.matmul(jnp.linalg.inv(Hessian), score)
    print(params_fid, score)
end = time.time()
print(end - start)

# hess_func_new = jax.jacfwd(jax.jit(grad_func_all), has_aux = True)

# hess_jax, (loglike, grad) = jax.block_until_ready(hess_func_new(result.params, conv_b, conv_bf, conv_f, conv_badd, conv_sigma_v, conv_noise, x_2, sigmab_square, datagrid_comp, fsigma8_old, bsigma8_old))

# print(loglike, grad)

# # hess_jax = hess_func(result.params, conv_b, conv_bf, conv_f, conv_badd, conv_sigma_v, conv_noise, x_2, sigmab_square, datagrid_comp, fsigma8_old, bsigma8_old)

# #Correction for Hessian in case the round-off error cause the matrix to be asymmetrical
# hess_jax_corr = 0.5*(hess_jax + hess_jax.T)

# unc = jnp.sqrt(jnp.diagonal(jnp.linalg.inv(-hess_jax_corr), axis1=0, axis2=1))

reduced_chi_2 = func_dL(
    result.params,
    conv_f,
    conv_sigma_v,
    conv_noise,
    x_2,
    sigmab_square,
    datagrid_comp,
    fsigma8_old,
    flag=1,
) / (len(datagrid_comp) - 5.0)

loglike, score, Hessian = func_grad(
    result.params,
    conv_f,
    conv_sigma_v,
    conv_noise,
    x_2,
    sigmab_square,
    datagrid_comp,
    fsigma8_old,
    func_dL,
    func_d2L,
)

unc = jnp.sqrt(jnp.diagonal(jnp.linalg.inv(-Hessian), axis1=0, axis2=1))

print(reduced_chi_2, score, reduced_chi_2 * (len(datagrid_comp) - 2.0), unc)

info = np.append(
    np.array(result.params),
    [
        np.float32(reduced_chi_2),
        np.float32(reduced_chi_2 * (len(datagrid_comp) - 5.0)),
        np.float32(-2.0 * loglike),
        unc[0],
        unc[1],
    ],
)
hessian_mat = Hessian

np.save(
    "./vv_opti/"
    + pardict["PT_method"]
    + "_fsigma8_vv_DESI_sigmau_DR2"
    + str(sigma_u)
    + "_mock_"
    + str(mock_num)
    + "_gridsize_"
    + str(pardict["gridsize"])
    + "_red_cut_"
    + str(pardict["red_cut_low"])
    + "_"
    + str(pardict["red_cut_high"])
    + ".npy",
    info,
)
np.save(
    "./vv_opti/"
    + pardict["PT_method"]
    + "_Hessian_vv_DESI_sigmau_DR2"
    + str(sigma_u)
    + "_mock_"
    + str(mock_num)
    + "_gridsize_"
    + str(pardict["gridsize"])
    + "_red_cut_"
    + str(pardict["red_cut_low"])
    + "_"
    + str(pardict["red_cut_high"])
    + ".npy",
    hessian_mat,
)

# bestfit = result.params

# sigma_bound = 4

# lower_bounds_new = jnp.array([1e-5, 1.0])

# prior_low = bestfit - sigma_bound*unc

# prior_high = bestfit + sigma_bound*unc

# prior_low_nonan = jnp.nan_to_num(prior_low, nan = -1e4)

# prior_high_nonan = jnp.nan_to_num(prior_high, nan = 1e4)

# prior_low_new = jnp.where(prior_low_nonan < lower_bounds_new, lower_bounds_new, prior_low_nonan)

# prior_high_new = jnp.where(prior_high_nonan > upper_bounds, upper_bounds, prior_high_nonan)

# print(prior_low_new)
# print(prior_high_new)

# wrapped_func = partial(
# func_dL, conv_f=conv_f, conv_sigma_v = conv_sigma_v,
# conv_noise=conv_noise,
# x_2=x_2, sigmab_square=sigmab_square, datagrid_comp=datagrid_comp, fsigma8_old = fsigma8_old)

# rng_key = random.PRNGKey(0)
# kernel = SA(model = model)
# mcmc = MCMC(kernel, num_warmup=5000, num_samples=35000, num_chains=1)
# mcmc.run(rng_key, wrapped_func)
# mcmc.print_summary()

# chainfile = str('/global/u2/y/ylai1998/Peculiar_velocity_fitting/fit_pec_vel_' + pardict['name'] + '_' + pardict['PT_method'] + '_k0p%03d_0p%03d_gridcorr_%d_vv_Taylor_data_FP.h5' % (int(1000.0*kmin), int(1000.0*kmax_velocity), gridsize))
# posterior_samples = mcmc.get_samples()
# # Save posterior samples to HDF5
# with h5py.File(chainfile, 'w') as f:
#     for param, chain in posterior_samples.items():
#         f.create_dataset(param, data=chain)
