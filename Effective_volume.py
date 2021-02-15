# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 16:04:12 2021

@author: s4479813
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
import time
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# Speed of light in km/s
LightSpeed = 299792.458

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

def integrand_volume(r, k, PS_vel_spline, sigma_v_spline, nd_spline, red_spline):
    z = sp.interpolate.splev(r, red_spline)
    factor_redshift = ((1.0/(1.0+z))*Ez(z, omega_m, 1.0-omega_m, 0.0, -1.0, 0.0, 0.0))**2
    omega_lambda = 1.0-omega_m
    f = (omega_m*(1+z)**3/(omega_m*(1+z)**3 + omega_lambda))**(6.0/11.0)
    factor_PS_vel = factor_redshift*(100.0*f)**2
    PS_vel = sp.interpolate.splev(k, PS_vel_spline)*factor_PS_vel/(k**2*3.0)
    nd = sp.interpolate.splev(r, nd_spline)
    sigma_v = sp.interpolate.splev(r, sigma_v_spline)
    
    factor = nd/sigma_v
    # result = PS_vel**2/(PS_vel + sigma_v/nd)**2 * r**2
    result = (factor*PS_vel/(1.0+factor*PS_vel))**2 * r**2
    return result

# def integrand_redshift(r, k, PS_vel_spline, sigma_v_spline, nd_spline, red_spline):
#     PS_vel = sp.interpolate.splev(k, PS_vel_spline)*factor_PS_vel/(k**2*3.0)
#     nd = sp.interpolate.splev(r, nd_spline)
#     sigma_v = sp.interpolate.splev(r, sigma_v_spline)
    
#     result = PS_vel**2/(PS_vel + sigma_v/nd)**2 * r**2*sp.interpolate.splev(r, red_spline)
#     return result

def effective_volume(k, r_min, r_max, PS_vel_spline, sigma_v_spline, nd_spline, red_spline):
    V_eff = integrate.quad(integrand_volume, r_min, r_max, args=(k, PS_vel_spline, sigma_v_spline, nd_spline, red_spline), limit=500, epsabs=0.0)[0]
    result = k**2*V_eff
    # result = V_eff
    return result

# def effective_redshift(k, r_min, r_max, PS_vel_spline, sigma_v_spline, nd_spline, red_spline):
#     denominator = integrate.quad(integrand_volume, r_min, r_max, args=(k, PS_vel_spline, sigma_v_spline, nd_spline), limit=500, epsabs=0.0)[0]
#     numerator =  integrate.quad(integrand_redshift, r_min, r_max, args=(k, PS_vel_spline, sigma_v_spline, nd_spline, red_spline), limit=500, epsabs=0.0)[0]
    
#     result = numerator/denominator
#     return result
# def integrand_volume_3D(r, theta, phi, k, PS_vel_spline, nd_interp, sigma_v_interp):
#     PS_vel = sp.interpolate.splev(k, PS_vel_spline)*factor_PS_vel/(k**2*3.0)
#     nd = nd_interp(r, theta, phi)
#     if (nd < 0.0):
#         result = 0.0
#     else: 
#         sigma_v = sigma_v_interp(r, theta, phi)
        
#         result = PS_vel**2/(PS_vel + sigma_v/nd)**2 * r**2*np.sin(theta)
#     return result

# def effective_volume_3D(k, r_min, r_max, theta_min, theta_max, phi_min, phi_max, PS_vel_spline, nd_interp, sigma_v_interp):
#     V_eff = integrate.tplquad(integrand_volume_3D, phi_min, phi_max, theta_min, theta_max, r_min, r_max, args=(k, PS_vel_spline, nd_interp, sigma_v_interp), epsabs=0.0)[0]
#     result = V_eff*k**2
#     return result

# Make sure these match the values used to estimate the covariance matrix
kmin = float(sys.argv[1])    # What kmin to use
kmax = float(sys.argv[2])    # What kmax to use
size = int(sys.argv[3])  # The number of redshift bins
mock_num = int(sys.argv[4]) # Total number of mocks
sky_coverage = float(sys.argv[5])*np.pi # The angular coverage of the sky in terms of pi
dataset = int(sys.argv[6]) #Enter 0 for 6df, 1 for 2MTF and 2 for SDSS. 
omega_m = float(sys.argv[7])             # Used to convert the galaxy redshifts to distances. Should match that used to compute the covariance matrix

if (dataset == 0):
    PSfile = str("matter_6dFGSv.dat")
else:
    PSfile = str("rpt2_waves_input_new.dat")
print(PSfile)

# Compute some useful quantities on the grid. For this we need a redshift-dist lookup table
q_0 = 1.5*omega_m-1.0
nbins = 5000
redmax = 0.5
red = np.empty(nbins)
dist = np.empty(nbins)
for i in range(nbins):
    red[i] = i*redmax/nbins
    dist[i] = DistDc(red[i], omega_m, 1.0-omega_m, 0.0, 100.0, -1.0, 0.0, 0.0)
red_spline = sp.interpolate.splrep(dist, red, s=0) #Interpolate the redshift with comoving distance. 
radial_spline=sp.interpolate.splrep(red, dist, s=0) #Interpolate the comoving distance with redshift.

PS_vel = np.array(pd.read_csv(PSfile, delim_whitespace=True, header=None, skiprows=0))
PS_vel_spline = sp.interpolate.splrep(PS_vel[:, 0], PS_vel[:, 1], s=0)

# factor_redshift = (10.0/11.0*Ez(0.1, omega_m, 1.0-omega_m, 0.0, -1.0, 0.0, 0.0))**2
# factor_PS_vel = factor_redshift*(100.0*omega_m**(6.0/11.0))**2

effective_volume_all = []

for i in range(mock_num):
    # max_red = 0.0
    # min_red = 1000.0
    max_rd = 0.0
    min_rd = 1000.0
    max_theta = -10.0
    min_theta = 10.0
    max_phi = -10.0
    min_phi = 10.0
    data_count = 0
    min_red = 1.0
    max_red = 0.0
    
    # Read in the 6dFGSv data
    data_6dFGSv = []
    if (dataset == 0):
        # datafile = str("./peculiar_velocity_mocks/6dfgsmock"+str(i+1)+"_velocity_data_norsd.dat")
        datafile = str("/data/s4479813/peculiar_velocity_mocks/6dfgsmock"+str(i+1)+"_velocity_data_norsd.dat")
        
        infile = open(datafile, 'r')
        for line in infile:
            ln = line.split()
            if (ln[0][0] == '#'):
                continue
            
            redshift = float(ln[2])
            # PV = float(ln[3])
            # PV_err = 0.0
            
            if (redshift > max_red):
                max_red = redshift
            if (redshift < min_red):
                min_red = redshift
            
            z_m = redshift*(1.0+0.5*(1.0-q_0)*redshift-(1.0/6.0)*(2.0-q_0-3.0*q_0**2)*redshift**2)
            PV_err = float(ln[4])
            
            x = float(ln[6])
            y = float(ln[7])
            z = float(ln[8])
            
            # if (redshift > max_red):
            #     max_red = redshift
            # if (redshift < min_red):
            #     min_red = redshift
            
            rd = np.sqrt(x**2+y**2+z**2)
            
            if (rd > max_rd):
                max_rd = rd
            if (rd < min_rd):
                min_rd = rd
            
            theta = np.arccos(z/rd)
            phi = np.arcsin(y/(rd*np.sin(theta)))
            
            if (theta > max_theta):
                max_theta = theta
            if (theta < min_theta):
                min_theta = theta
            
            if (phi > max_phi):
                max_phi = phi
            if (phi < min_phi):
                min_phi = phi
                
            data_6dFGSv.append((rd,PV_err,redshift, theta, phi))
            
            data_count += 1
    elif (dataset == 1):
        file_num, dot_num = divmod(i, 8)
        datafile = str("/data/s4479813/2MTF_mocks_full/MOCK_HAMHOD_2MTF_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err")
        # datafile = str("./2MTF_mocks/MOCK_HAMHOD_2MTF_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err")
        
        infile = open(datafile, 'r')
        for line in infile:
            ln = line.split()
            if (ln[0][0] == '#'):
                continue
            
            redshift = float(ln[3])/LightSpeed
            # PV = float(ln[3])
            # PV_err = 0.0
            
            if (redshift > max_red):
                max_red = redshift
            if (redshift < min_red):
                min_red = redshift
                
            factor_conversion = (np.log(10.0))*((100.0*Ez(redshift, omega_m, 1.0-omega_m, 0.0, -1.0, 0.0, 0.0)*sp.interpolate.splev(redshift, radial_spline))/(1.0+redshift))
            PV_err = float(ln[6])*factor_conversion
            
            rd = sp.interpolate.splev(redshift, radial_spline)
            
            if (rd > max_rd):
                max_rd = rd
            if (rd < min_rd):
                min_rd = rd
                
            data_6dFGSv.append((rd,PV_err,redshift))
            
            data_count += 1
    
    elif (dataset == 2):
        file_num, dot_num = divmod(i, 8)
        
        datafile = str("./SDSS_mocks/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr")
        
        data = np.array(pd.read_csv(datafile, header=None, skiprows=1))
        
        RA = data[:, 1]/180.0*np.pi
        Dec = data[:, 2]/180.0*np.pi
        redshift = data[:, 4]/LightSpeed
        
        min_red = np.min(redshift)
        max_red = np.max(redshift)
        
        rd = sp.interpolate.splev(redshift, radial_spline)
        data_count = len(rd)
        # log_dist_err = data[:, 37]
        log_dist_err = np.zeros(data_count)
        
        factor_conversion = (np.log(10.0))*((100.0*Ez(redshift, omega_m, 1.0-omega_m, 0.0, -1.0, 0.0, 0.0)*sp.interpolate.splev(redshift, radial_spline))/(1.0+redshift))
        PV_err = log_dist_err*factor_conversion
        
        rd = np.reshape(rd, (data_count, 1))
        PV_err = np.reshape(PV_err, (data_count, 1))
        redshift = np.reshape(redshift, (data_count, 1))
        
        data_6dFGSv = np.concatenate((rd, PV_err, redshift), axis = 1)
        
        min_rd = np.min(rd)
        max_rd = np.max(rd)
        
    print(datafile)
    
    
    print(data_count, min_rd, max_rd)
    
    max_red = max_red + 10e-8
    min_red = min_red - 10e-8
    
    max_rd = max_rd + 10e-8
    min_rd = min_rd - 10e-8
    # max_phi = max_phi + 10e-8
    # min_phi = min_phi-10e-8
    # max_theta = max_theta + 10e-8
    # min_theta = min_theta - 10e-8
    
    # data_6dFGSv = np.array(data_6dFGSv)
    
    # size_rd = 30
    # size_theta = 20
    # size_phi = 35
    
    # rd_bin_size = (max_rd - min_rd)/float(size_rd)
    # theta_bin_size = (max_theta - min_theta)/float(size_theta)
    # phi_bin_size = (max_phi - min_phi)/float(size_phi)
    
    # rd_bin = []
    # theta_bin = []
    # phi_bin = []
    # sigma_v_all = []
    # nd = []
    # count = []
    # volume_all = []
    # a = 0
    # b = 0
    # c = 0
    
    # for i in range(size_rd*size_theta*size_phi):
        
    #     rd_bin_max = (c+1.0)*rd_bin_size + min_rd
    #     rd_bin_min = c*rd_bin_size + min_rd
    #     theta_bin_max = (b+1.0)*theta_bin_size + min_theta
    #     theta_bin_min = b*theta_bin_size + min_theta
    #     phi_bin_max = (a+1.0)*phi_bin_size + min_phi 
    #     phi_bin_min = a*phi_bin_size + min_phi 
        
    #     sigma_v = []
    #     number = 0
    #     volume = (1.0/3.0)*(rd_bin_max**3-rd_bin_min**3)*(-np.cos(theta_bin_max)+np.cos(theta_bin_min))*(phi_bin_max-phi_bin_min)
        
        
    #     for j in range(data_count):
    #         if (data_6dFGSv[j][0] > rd_bin_min) and (data_6dFGSv[j][0] < rd_bin_max):
    #             if (data_6dFGSv[j][3] > theta_bin_min) and (data_6dFGSv[j][3] < theta_bin_max):
    #                 if (data_6dFGSv[j][4] > phi_bin_min) and (data_6dFGSv[j][4] < phi_bin_max):
    #                     sigma_v.append(data_6dFGSv[j][1]**2 + 150.0**2)
    #                     number += 1
        
    #     if (number != 0):
    #         count.append(number)
    #         sigma_v_all.append(np.sum(np.array(sigma_v))/float(number))
    #         nd.append(float(number)/volume)
    #         rd_bin.append((c+0.5)*rd_bin_size + min_rd)
    #         theta_bin.append((b+0.5)*theta_bin_size + min_theta)
    #         phi_bin.append((a+0.5)*phi_bin_size + min_phi)
    #         volume_all.append(volume)
            
    #         # if (phi_bin_min < phi_min):
    #         #     phi_min = phi_bin_min
    #         # if (phi_bin_max > phi_max):
    #         #     phi_max = phi_bin_max
            
    #         # if (theta_bin_min < theta_min):
    #         #     theta_min = theta_bin_min
    #         # if (theta_bin_max > theta_max):
    #         #     theta_max = theta_bin_max
                
            
            
    #     a += 1
    #     if (a == size_phi):
    #         a = 0
    #         b = b + 1
            
    #     if (b == size_theta):
    #         b = 0
    #         c = c + 1
    #         if (c%10 == 0):
    #             print(i)
        
    # print(np.sum(np.array(count)))
    
    # k = 0.05
    # PS_vel = sp.interpolate.splev(k, PS_vel_spline)*factor_PS_vel/(k**2*3.0)
    
    # integrand_v_eff = PS_vel**2/(PS_vel + (np.array(sigma_v_all)**2/np.array(nd)))**2*np.array(volume_all)
    
    
    
    # nd_interp = LinearNDInterpolator(list(zip(rd_bin, theta_bin, phi_bin)), nd, fill_value = -1.0)
    # sigma_v_interp = LinearNDInterpolator(list(zip(rd_bin, theta_bin, phi_bin)), sigma_v_all, fill_value = -1.0)
    
    # rd_min = rd_bin[0]
    # rd_max = rd_bin[-1]
    # theta_min = theta_bin[0]
    # theta_max = theta_bin[-1]
    # phi_min = phi_bin[0]
    # phi_max = phi_bin[-1]
    
    # size_rd_new = 400
    # size_theta_new = 50
    # size_phi_new = 100
    
    # size_all = size_rd_new*size_theta_new*size_phi_new
    
    # rd_gap = (rd_max - rd_min)/(size_rd_new)
    # theta_gap = (theta_max - theta_min)/(size_theta_new)
    # phi_gap = (phi_max - phi_min)/(size_phi_new)
    
    # rd_new = np.zeros(size_all)
    # theta_new = np.zeros(size_all)
    # phi_new = np.zeros(size_all)
    
    # volume = np.zeros(size_all)
    
    # a = 0
    # b = 0
    # c = 0
    
    # for i in range(size_all):
    #     rd_new[i] = (c+0.5)*rd_gap + rd_min
    #     theta_new[i] = (b+0.5)*theta_gap + theta_min
    #     phi_new[i] = (a+0.5)*phi_gap + phi_min
     
    #     rd_min_bin = c*rd_gap + rd_min
    #     rd_max_bin = (c+1)*rd_gap + rd_min
        
    #     theta_min_bin = b*theta_gap + theta_min
    #     theta_max_bin = (b+1)*theta_gap + theta_min 
        
    #     phi_min_bin = a*phi_gap + phi_min 
    #     phi_max_bin = (a+1)*phi_gap + phi_min 
        
    #     volume[i] = ((1.0/3.0)*(rd_max_bin**3-rd_min_bin**3)*(-np.cos(theta_max_bin)+np.cos(theta_min_bin))*(phi_max_bin - phi_min_bin))
        
    #     a += 1
    #     if (a == size_phi_new):
    #         a = 0
    #         b += 1
    #     if (b == size_theta_new):
    #         b = 0
    #         c += 1
            
    #         if (c%20 == 0):
    #             print(i)
                
    # nd_new = nd_interp(rd_new, theta_new, phi_new)
    # sigma_v_bew = sigma_v_interp(rd_new, theta_new, phi_new)
    
    # delete_index
    
    # print(integrate.tplquad(integrand_volume_3D, phi_min, phi_max, theta_min, theta_max, rd_min, rd_max, args=(0.05, PS_vel_spline, nd_interp, sigma_v_interp), epsabs=0.0)[0])
    
    # volume_effective_3D = integrate.quad(effective_volume_3D, kmin, kmax, args=(rd_min, rd_max, theta_min, theta_max, phi_min, phi_max, PS_vel_spline, nd_interp, sigma_v_interp), epsabs=0.0)
    
    
    # print(volume_effective_3D)
    # red_bin_size = (max_red-min_red)/float(size)
    # nd_redshift = []
    # sigma_v_redshift = []
    # radial_redshift = []
    # red_bin = np.zeros(size)
    # count = []
    # for i in range(size):
    #     red_bin[i] = (i+0.5)*red_bin_size + min_red
    #     red_bin_max = (i+1.0)*red_bin_size + min_red
    #     red_bin_min = i*red_bin_size + min_red
    #     redshift_sigma = []
    #     redshift_number = 0
    #     redshift_volume = (4.0*np.pi/3.0)*(sp.interpolate.splev(red_bin_max, radial_spline, der=0)**3 - sp.interpolate.splev(red_bin_min, radial_spline, der=0)**3)*(sky_coverage/(4.0*np.pi))
    #     for j in range(data_count):
    #         if (data_6dFGSv[j][2] > red_bin_min) and (data_6dFGSv[j][2] < red_bin_max):
    #             redshift_sigma.append(data_6dFGSv[j][1]**2 + 150**2)
    #             redshift_number += 1
        
    #     count.append(redshift_number)
    #     # if (redshift_number == 0):
    #     #     nd_redshift.append(0)
    #     #     sigma_v_redshift.append(0.0)
    #     #     radial_redshift.append(sp.interpolate.splev(red_bin[i], radial_spline, der=0))
    #     # else:
    #     #     nd_redshift.append(redshift_number/redshift_volume)
    #     #     sigma_v_redshift.append(np.sum(np.array(redshift_sigma))/redshift_number)
    #     #     radial_redshift.append(sp.interpolate.splev(red_bin[i], radial_spline, der=0))
        
    #     nd_redshift.append(redshift_number/redshift_volume)
    #     sigma_v_redshift.append(np.sum(np.array(redshift_sigma))/redshift_number)
    #     radial_redshift.append(sp.interpolate.splev(red_bin[i], radial_spline, der=0))
    
    rd_bin_size = (max_rd - min_rd)/float(size)
    nd_rd = []
    sigma_v_rd = []
    rd_bin = np.zeros(size)
    count = []
    for i in range(size):
        rd_bin[i] = (i+0.5)*rd_bin_size+min_rd
        rd_bin_max = (i+1.0)*rd_bin_size+min_rd
        rd_bin_min = i*rd_bin_size+min_rd
        rd_sigma = []
        rd_number = 0
        rd_volume = (4.0*np.pi/3.0)*(rd_bin_max**3-rd_bin_min**3)*(sky_coverage/(4.0*np.pi))
        for j in range(data_count):
            if (data_6dFGSv[j][0] > rd_bin_min) and (data_6dFGSv[j][0] < rd_bin_max):
                rd_sigma.append(data_6dFGSv[j][1]**2 + 150**2)
                # rd_sigma.append(data_6dFGSv[j][1]**2)
                rd_number += 1
        count.append(rd_number)
        
        nd_rd.append(rd_number/rd_volume)
        sigma_v_rd.append(np.sum(np.array(rd_sigma))/rd_number)
    print(np.sum(np.array(count)))
    
    # sigma_v_spline = sp.interpolate.splrep(red_bin, sigma_v_redshift, s=0)
    
    # nd_spline = sp.interpolate.splrep(red_bin, nd_redshift, s=0)
    
    sigma_v_spline = sp.interpolate.splrep(rd_bin, sigma_v_rd, s=0)
    
    nd_spline = sp.interpolate.splrep(rd_bin, nd_rd, s=0)
    
    # min_red = red_bin[0]
    # max_red = red_bin[-1] 
     
    min_rd = rd_bin[0]
    max_rd = rd_bin[-1]
    
    # plt.figure(1, dpi=1200)
    # plt.plot(rd_bin, nd_rd, 'r')
    # plt.xlabel('Comoving distance (Mpc/h)')
    # plt.ylabel('Number density ((h/Mpc)^3)')
    # plt.title('Selection function of SDSS mock 1')
    # plt.savefig('SF_SDSS_1_radial.png')    
    
    # plt.plot(sp.interpolate.splev(rd_bin, red_spline), np.sqrt(sigma_v_rd), 'r', label = 'SDSS velocity error')
    # plt.plot(sp.interpolate.splev(rd_bin, red_spline), 0.05*LightSpeed*sp.interpolate.splev(rd_bin, red_spline), 'b', label = '6dFGSv velocity error')
    # plt.xlabel('Redshift')
    # plt.ylabel('Velocity error (km/s)')
    # plt.title('Velocity error of SDSS vs 6dFGSv mocks')
    # plt.legend()
    # plt.savefig('VE_redshift.png')    
    
    # k = np.linspace(kmin, kmax, 100)
    # integrand_er = []
    
    # for i in range(len(k)):
    #     integrand_er.append(effective_volume(k[i], min_rd, max_rd, PS_vel_spline, sigma_v_spline, nd_spline))
        
    # plt.figure(1)
    # plt.plot(k, integrand_er, 'r')
    # plt.xlabel('wavenumber (h/Mpc)')
    # plt.ylabel('Integrand (Mpc/h)')
    # plt.title('The integrand for SDSS mock 1')
    # print(np.trapz(integrand_er, k))
    
    volume_effective = integrate.quad(effective_volume, kmin, kmax, args=(min_rd, max_rd, PS_vel_spline, sigma_v_spline, nd_spline, red_spline), epsabs=0.0)[0]/((1.0/3.0)*(kmax**3-kmin**3))*sky_coverage
    
    print(volume_effective)
    
    effective_volume_all.append(volume_effective)
    
    # print(integrate.quad(effective_volume, kmin, kmax, args=(min_rd, max_rd, PS_vel_spline, sigma_v_spline, nd_spline), epsabs=0.0)[1])
                
    # z = np.linspace(min_red, max_red, 10000)
    # k = kmax
    # PS_vel = sp.interpolate.splev(k, PS_vel_spline)*factor_PS_vel/(k**2*3.0)
    # nd = sp.interpolate.splev(z, nd_spline)
    # sigma_v = sp.interpolate.splev(z, sigma_v_spline)
    # distance = sp.interpolate.splev(z, radial_spline)
    
    # integrand = PS_vel**2/(PS_vel + sigma_v/nd)**2 * distance**2
    
    # plt.plot(z, integrand)

effective_volume_all = np.array(effective_volume_all)
max_volume = 4.0*np.pi/3.0*(max_rd**3)*(sky_coverage/(4.0*np.pi))
delete_index_numerical = np.where(effective_volume_all > max_volume)
delete_index_nan = np.argwhere(np.isnan(effective_volume_all))
if (len(delete_index_nan) != 0):
    delete_index = np.concatenate((delete_index_numerical, delete_index_nan))
else:
    delete_index = delete_index_numerical
EV_after = np.delete(effective_volume_all, delete_index)

print(np.mean(EV_after), np.std(EV_after), len(delete_index_numerical), len(delete_index_nan))