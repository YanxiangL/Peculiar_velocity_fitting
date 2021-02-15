# Author: Abbe Whitford, Cullan Howlett, Yanxiang Lai

# Last updated: 11/02/2021

# This code is essentially a version of PV_fisher by Cullan Howlett, but that has been written in python
# to make it easier for users who do not know C (which the code has been written in originally - the original
# code can be found at https://github.com/CullanHowlett/PV_fisher). 


# import libraries
import matplotlib.pyplot as plt 
import numpy as np 
import scipy as sp 
from scipy.integrate import cumtrapz, quad, simps
from scipy.interpolate import CubicSpline
import pandas as pd
from scipy.special import spherical_jn


# define some variables -----------------------------------------------------------------------------------------

H0 = 100.0                                     # Hubble constant in km/s/MPC
h = H0/100.0                                    # little h
# Obh = 0.022                                     # density of baryonic matter * h*2 (H0/100.0 km/s/MPC)
# Och = 0.120                                     # density of cold dark matter * h*2 (H0/100.0 km/s/MPC)
# m_nu = 0.00                                     # sum of masses of neutrinos 
# Om = (Obh + Och + (m_nu)/(93.14))/((H0/100.0)**2) # density of matter (baryonics + neutrinos + cold dark matter)
Om = 0.3121                              # The matter density at z=0
#print(Om)
c = 299792.458                                  # The speed of light in km/s
gammaval = 0.55                               # The value of gammaval to use in the forecasts (where f(z) = Om(z)^gammaval)
#zin = [0.0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 
#0.175, 0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40, 0.425, 0.45, 0.475, 0.50]            # redshifts to use
zin = np.linspace(0.0, 0.10, 51)
nzin = len(zin)                                 # number of redshifts to use 
num_redshift_bins = 2                        # number of redshift bins                                                
zmax = 0.2                                     # maximum redshift to consider - it gets changed in the code 
zmin = 0.0                                      # minimum redshift to consider - it gets changed in the code 
kmax = 0.20*h                                      # The maximum k to evaluate for dd, dv and vv correlations 
                                                # (Typical values are 0.1 - 0.2, on smaller scales the models are likely to break down).
sigma_g = 4.24                                  # km/s/MPC 
sigma_u = 0.0                                 # km/s/MPC
#sigma80 = 0.826077                                # The value of sigma8 at z=0 
sigma80 = 0.8150
#sigma80 = 1.0584966883819318#0.842908
beta0 = 0.391                                    # The value of beta (at z=0 -
                                                # we'll modify this by the redshift dependent value of bias and f as required)
r_g = 1.0                                       # The cross correlation coefficient between the velocity and density fields
#survey_area = [1.3748, 0.0, 0]                 # We need to know the survey area for each survey and the overlap area between the surveys 
#survey_area = [0.0, 0.0, 1.745]
survey_area = [0.0, 3.65, 0.0] 
                                                # (redshift survey only first, then PV survey only, then overlap. 
                                                # For fully overlapping we would have {0, 0, size_overlap}. For redshift larger than PV, we would have:
                                                # {size_red-size_overlap, 0, size_overlap}). Units are pi steradians, such that full sky is 4.0, half sky is 2.0 etc.
error_rand = 150.0                              # The observational error due to random non-linear velocities 
error_dist = 0.25                              # The percentage error on the distance indicator (Typically 0.05 - 0.10 for SNe IA, 
                                                # 0.2 or more for Tully-Fisher or Fundamental Plane) 
Data = [1]                                   # A vector of flags for the parameters we are interested in (0=beta, 1=fsigma8, 2=r_g, 3=sigma_g, 4=sigma_u). 
nparams = len(Data)                             # The number of free parameters (we can use any of beta, fsigma8, r_g, sigma_g, sigma_u)
verbosity = 1                                  # How much output to give: <= 0 only give minimum information, > 0 give full information

dataset = 1                                    #Which dataset to use, 0 for 6dFGSv, 1 for 2MTF and 2 for SDSS/ 

mock_num = 0                                    #Which mock to use.

# The files containing the number density of the surveys, x10^6. First is the PV survey, then the redshift survey. 
# These files MUST have the same binning and redshift range, 
# so that the sum over redshift bins works (would be fine if we used splines), 
# i.e., if one survey is shallower then that file must contain rows with n(z)=0.
# nbar_file = [r"/home/abbew/honours_codes_28.9.20/DESI_number_densities/14000_BGSurvey_redshifts.csv", 
#               r"/home/abbew/honours_codes_28.9.20/DESI_number_densities/14000_BGSurvey_velocities.csv" ]

# nbar_file = [r"/home/abbew/honours_codes_28.9.20/example_files/example_nbar_red.dat", 
#            r"/home/abbew/honours_codes_28.9.20/example_files/example_nbar_vel.dat" ]

if (dataset == 0):
    powerspectrumfile = str("matter_6dFGSv.dat")
else:
    powerspectrumfile = str("rpt2_waves_input_new.dat")
print(powerspectrumfile)

# powerspectrumfile = r'/home/abbew/honours_codes_28.9.20/DESI_test_files/example_pk_z0'
# powerspectrumfile = r'/home/abbew/honours_codes_28.9.20/zero_neutrinos_power_spectra/example_pk_z0'

skiprows = 1
delimwhitespace = False

# initializing some variables that will change later
# for reading in the power spectra
pmm_array, pmt_array, ptt_array, delta_ks, kvals = [], [], [], [], []

# read in the survey number densities
N_redshifts_arr, N_bar_arr, r_array, delta_r_array, redshift_dist_spline, growth_factor_spline , growth_array = [], [], [], [], [], [], []

# ---------------------------------------------------------------------------------------------------------------

# function used to integrate to compute inverse E(z) = (H/H_0)**2 from Friedmann Equation
def E_z_inverse(z):
    '''
    Compute the inverse of the E(z) function from the first Friedmann Equation.
    '''
    return 1.0/np.sqrt((Om*(1.0+z)**3)+(1.0-Om)) 

def E_z(z):
    '''
    Compute E(z) from the first Friedmann Equation.
    '''
    return np.sqrt((Om*(1.0+z)**3)+(1.0-Om))

# function to integrate to get the normalized growth rate (uses rearranging of f = a/D dD/da)
def norm_growth_rate_func(a):
    '''
    Compute D(z)/D(z=0) by integrating this function over scalefactor from z to z = 0.0.
    '''
    redshift = 1.0/a - 1.0
    Om_z = Om*(E_z_inverse(redshift)**2)/(a**3)
    f = pow(Om_z, gammaval)
    return f/a

# function that computes comoving distance per little h as a function of redshift (dimensionless)
def rz(red):
    '''
    Calculates the proper radial distance to an object at redshift z for our cosmology.
    '''
    #when the input is scalar
    try:
        d_com = (c/H0)*quad(E_z_inverse, 0.0, red, epsabs = 5e-5)[0]
        return d_com
    #When the input is a vector
    except:
        distances = np.zeros(len(red))  
        for i, z in enumerate(red):
            distances[i] = (c/H0)*quad(E_z_inverse, 0.0, z, epsabs = 5e-5)[0]
        return distances 

# function that calculates the normalised growth factor as a function of redshift given a value of gammaval
# ( D(a)/D(a_0) )
def growth_factor_z(red):
    '''

    Integrate norm_growth_rate_func() to get D(z)/D(z=0). 

    '''
    try:       
        a = 1.0/(1.0 + red)                                          
        norm_growth_factor = np.exp(- (quad(norm_growth_rate_func, a, 1.0, epsabs = 5e-5)[0]))
        return norm_growth_factor
    except:
        red = np.array(red)
        a = 1.0/(1.0 + red)
        growth_factors = np.zeros(len(red))  
        for i in range(len(red)):
            growth_factors[i] = np.exp(- (quad(norm_growth_rate_func, a[i], 1.0, epsabs = 5e-5)[0]))
        return growth_factors 


# # function to read in a file with the number density of galaxies as a function of redshift for a
# # velocity survey (values are saved as 1e6 times actual value for density to keep precision, hence
# # multiplication factor of 1e-6 after read in)
# def read_nz():

#     '''
#     Read in the files to get the number density of galaxies for the density field
#     and the velocity field. 

#     This function also computes the distance to each redshift bin we consider, the width of the 
#     distance bins and creates a spline for redshift with distance that can be used globally. A spline
#     for the normalized growth factor with redshift is also computed and an array containing the normalized
#     growth factor as a function of redshift. 

#     '''

#     Nbararr = []
#     Nredshiftsarror = 0
#     for i in np.arange(2): # redshift file then velocity file

#         try: # try to read in the file
#             number_density_data = pd.read_csv(r'%s' % (nbar_file[i]), header=None, engine='python', 
#             delim_whitespace=True, names = ["n_red", "n_bar"], skiprows = skiprows)
#         except: # raise an exception if it cannot be read in
#             raise Exception("File could not be read in: error in (read_nz()).")

#         if i == 0: # save the redshifts array 
#             Nredshiftsarror = np.array(number_density_data["n_red"])

#         Nbararr.append((np.array(number_density_data["n_bar"])*1e-6*(1/(h**3))))  # save the number density for vel. and pos. in a list

#     Nbararr = np.array(Nbararr)
    
#     # need to check the number of redshift bins match for velocity and position data 
#     if (len(Nbararr[0]) != len(Nbararr[1])):
#         raise Exception("The length of the redshift bins for the velocity and position density files are not the same: error in (read_nz()).")


#     # create a redshift-distance spline to use globally 
#     nbins = 500                                 
#     redshifts = np.linspace(0, 2.0, nbins)
#     distances = rz(redshifts)
#     redshiftdistspline = CubicSpline(redshifts, distances)

    
#     # create a redshift-growthrate spline
#     growth_factors = growth_factor_z(redshifts)/growth_factor_z(0.0)
#     growthfactspline = CubicSpline(redshifts, growth_factors)


#     # make the redshift array from the redshift array read in, array of 
#     # distances, array of delta redshifts, array
#     Nredshiftsarror2 = []
#     rarray = []
#     growtharray = []
#     deltararray = []
#     for i in range(len(Nredshiftsarror)-1):
#         Nredshiftsarror2.append((Nredshiftsarror[i+1] + Nredshiftsarror[i])/2.0)
#         rarray.append(rz(Nredshiftsarror2[i]))
#         deltararray.append((rz(Nredshiftsarror[i+1]) - rz(Nredshiftsarror[i])))
#         growtharray.append(growth_factor_z(Nredshiftsarror2[i])/growth_factor_z(0.0))

#     Nredshiftsarr = np.array(Nredshiftsarror2) # make it a numpy array 
#     rarray = np.array(rarray) # make it a numpy array
#     growtharray = np.array(growtharray) # make it a numpy array
#     deltararray = np.array(deltararray)

#     # set the last values for each array 
#     global zmax
#     Nredshiftsarr = np.append(Nredshiftsarr, (zmax + Nredshiftsarror[len(Nredshiftsarror)-1])/2.0 )
#     rarray = np.append(rarray, rz(Nredshiftsarr[len(Nredshiftsarr)-1]) )
#     deltararray = np.append( deltararray, rz(zmax) - rz(Nredshiftsarror[len(Nredshiftsarror)-1]) ) 
#     growtharray = np.append( growtharray, growth_factor_z(Nredshiftsarr[len(Nredshiftsarr)-1])/growth_factor_z(0.0) )


#     return Nredshiftsarr, Nbararr, rarray, deltararray, redshiftdistspline, growthfactspline, growtharray

def read_nz():
    """
    This function calculates read in the datafile of the survey and calculates the redshift, the number density array, the radial distance array, the redshift to 
    radial distance spline, the redshift to growth factor spline and an array of growth factors (return in this order).
    """
    
    nbins = 5000
    redmax = 0.5
    red = np.empty(nbins)
    dist = np.empty(nbins)
    for i in range(nbins):
        red[i] = i*redmax/nbins
        dist[i] = rz(red[i]) #This distance is using the fiducial Hubble constant and its dimension is Mpc/h
    red_spline = sp.interpolate.splrep(dist, red, s=0) #Interpolate the redshift with comoving distance. 
    radial_spline=sp.interpolate.splrep(red, dist, s=0) #Interpolate the comoving distance with redshift.
    
     # create a redshift-growthrate spline
    growth_factors = growth_factor_z(red)/growth_factor_z(0.0)
    growthfactspline = CubicSpline(red, growth_factors)
    
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
        datafile = str("./peculiar_velocity_mocks/6dfgsmock"+str(mock_num+1)+"_velocity_data_norsd.dat")
        # datafile = str("/data/s4479813/peculiar_velocity_mocks/6dfgsmock"+str(i+1)+"_velocity_data_norsd.dat")
        
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
            
            PV_err = float(ln[4])
            
            x = float(ln[6])
            y = float(ln[7])
            z = float(ln[8])
            
            # if (redshift > max_red):
            #     max_red = redshift
            # if (redshift < min_red):
            #     min_red = redshift
            
            rd = np.sqrt(x**2+y**2+z**2)*h
            
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
        file_num, dot_num = divmod(mock_num, 8)
        # datafile = str("/data/s4479813/2MTF_mocks_full/MOCK_HAMHOD_2MTF_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err")
        datafile = str("./2MTF_mocks/MOCK_HAMHOD_2MTF_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err")
        
        infile = open(datafile, 'r')
        for line in infile:
            ln = line.split()
            if (ln[0][0] == '#'):
                continue
            
            redshift = float(ln[3])/c
            # PV = float(ln[3])
            # PV_err = 0.0
            
            if (redshift > max_red):
                max_red = redshift
            if (redshift < min_red):
                min_red = redshift
                
            factor_conversion = (np.log(10.0))*((H0*E_z(redshift)*sp.interpolate.splev(redshift, radial_spline))/(1.0+redshift))
            PV_err = float(ln[6])*factor_conversion
            
            rd = sp.interpolate.splev(redshift, radial_spline)
            
            if (rd > max_rd):
                max_rd = rd
            if (rd < min_rd):
                min_rd = rd
                
            data_6dFGSv.append((rd,PV_err,redshift))
            
            data_count += 1
    
    elif (dataset == 2):
        file_num, dot_num = divmod(mock_num, 8)
        
        datafile = str("./SDSS_mocks/MOCK_HAMHOD_SDSS_v4_R"+str(int(19000+file_num))+"."+str(dot_num)+"_err_corr")
        
        data = np.array(pd.read_csv(datafile, header=None, skiprows=1))

        redshift = data[:, 4]/c
        
        min_red = np.min(redshift)
        max_red = np.max(redshift)
        
        rd = sp.interpolate.splev(redshift, radial_spline)
        data_count = len(rd)
        # log_dist_err = data[:, 36]
        log_dist_err = np.zeros(data_count)
        
        factor_conversion = (np.log(10.0))*((H0*E_z(redshift)*sp.interpolate.splev(redshift, radial_spline))/(1.0+redshift))
        PV_err = log_dist_err*factor_conversion
        
        rd = np.reshape(rd, (data_count, 1))
        PV_err = np.reshape(PV_err, (data_count, 1))
        redshift = np.reshape(redshift, (data_count, 1))
        
        data_6dFGSv = np.concatenate((rd, PV_err, redshift), axis = 1)
        
        min_rd = np.min(rd)
        max_rd = np.max(rd)
        
    print(datafile)
    print(data_count)
    
    max_red = max_red + 10e-12
    min_red = min_red - 10e-12
    
    size = 30
    red_bin_size = (max_red-min_red)/float(size)
    nd_redshift = []
    radial_redshift = []
    growtharray = []
    deltararray = []
    red_bin = np.zeros(size)
    count = []
    for i in range(size):
        red_bin[i] = (i+0.5)*red_bin_size + min_red
        red_bin_max = (i+1.0)*red_bin_size + min_red
        red_bin_min = i*red_bin_size + min_red
        growtharray.append(growth_factor_z(red_bin[i])/growth_factor_z(0.0))
        deltararray.append(rz(red_bin_max)-rz(red_bin_min))
        redshift_number = 0
        redshift_volume = (4.0*np.pi/3.0)*(sp.interpolate.splev(red_bin_max, radial_spline, der=0)**3 - sp.interpolate.splev(red_bin_min, radial_spline, der=0)**3)*(survey_area[1]*np.pi/(4.0*np.pi))
        for j in range(data_count):
            if (data_6dFGSv[j][2] > red_bin_min) and (data_6dFGSv[j][2] <= red_bin_max):
                redshift_number += 1
        
        count.append(redshift_number)
        
        nd_redshift.append(redshift_number/redshift_volume)
        radial_redshift.append(sp.interpolate.splev(red_bin[i], radial_spline, der=0))
    
    print(np.sum(count))
    
    nd_redshift = np.array(nd_redshift) #This is the number density as a function of redshift
    radial_redshift = np.array(radial_redshift) #This is the radial distance as a function of redshift. 
    growtharray = np.array(growtharray) #This contains the growth factor at different redshift. 
    deltararray = np.array(deltararray) #This contains the width of the redshift bin in Mpc. 
    
    return red_bin, nd_redshift, radial_redshift, deltararray, radial_spline, growthfactspline, growtharray, min_red, max_red




# # function to read in files with the power spectra (p_delta,delta , p_delta,theta , p_theta,theta)
# def read_power():
#     '''

#     Function to read in the files for the matter and velocity power spectra. 

#     '''

#     pmmarray, pmtarray, pttarray = [], [], []
#     ks = 0
#     deltaks = []
#     for i in np.arange(nzin): # loop through redshifts we want to read in
        
#         power_spectrum_data = 0
#         try: # try to read in the file 
#             power_spectrum_data = pd.read_csv( powerspectrumfile + (r'p%d.dat' % (int(zin[i]*100)) ), 
#             header=None, engine='python', delim_whitespace=delimwhitespace, names = ["ks", "pmm", "pmt", "ptt"], skiprows = 0)
#         except: # throw an error if it cannot be read in
#             raise Exception("File could not be read in (read_power()).")

#         if i == 0: # saving k values 
#             ks = np.array(power_spectrum_data["ks"]*(h))
#         # saving data for the power spectra 
#         pmmarray.append(np.array(power_spectrum_data["pmm"])*(1/(h**3)))  
#         pmtarray.append(np.array(power_spectrum_data["pmt"])*(1/(h**3)))  
#         pttarray.append(np.array(power_spectrum_data["ptt"])*(1/(h**3)))  


#     pmmarray = np.array(pmmarray)
#     pmtarray = np.array(pmtarray)
#     pttarray = np.array(pttarray)
#     deltaks = ks[1:len(ks)] - ks[0:len(ks)-1]

#     return pmmarray, pmtarray, pttarray, deltaks, ks

#This function reads in the file with power spectrum data.
def read_power():
    """
    This functions reads in the matter-matter, matter-velocity and velocity-velocity power spectrum. For the 6dF mocks, these three are the same (this is just 
    an approximation). This function returns the three power spectra, delta k and k. 

    """
    
    pmmarray, pmtarray, pttarray = [], [], []
    ks = 0
    deltaks = []
    
    if (dataset == 0):
        power_spectrum_data = pd.read_csv(powerspectrumfile, header=None, delim_whitespace=True, names = ['ks', 'pmm'], skiprows = 0)
        pmmarray = (np.array(power_spectrum_data["pmm"]))  
        pmtarray = (np.array(power_spectrum_data["pmm"]))
        pttarray = (np.array(power_spectrum_data["pmm"]))  
    else:
        power_spectrum_data = pd.read_csv(powerspectrumfile, header=None, delim_whitespace=True, names = ['ks', 'pmm', 'pmt', 'ptt'], skiprows = 0)
        pmmarray = (np.array(power_spectrum_data["pmm"]))
        pmtarray = (np.array(power_spectrum_data["pmt"]))
        pttarray = (np.array(power_spectrum_data["ptt"]))
    
    ks = np.array(power_spectrum_data['ks']*h)
    deltaks = ks[1:len(ks)] - ks[0:len(ks)-1]
    
    return pmmarray, pmtarray, pttarray, deltaks, ks

# ---------------------------------------------------------------------------------------------------------------

# def mu_integrand(mu, datalist1): # function to be integrated over mu for Fisher matrix elements, 
#     # at some k and mu values 
#     '''
#     Function to compute the integral of the Fisher matrix elements over mu for a single value of k and z.
#     '''

#     k_index, k, zminv, p1, p2, zmaxv = datalist1  # [numk, k, zmin_iter, Data[i], Data[j], zmax_iter]

#     # getting P_xx(k, z) for full range of redshifts
#     pmm_k = pmm_array[:, k_index]
#     pmt_k = pmt_array[:, k_index]
#     ptt_k = ptt_array[:, k_index]

#     # set up interpolators for each of the power spectra
#     pmm_interp = CubicSpline(zin, pmm_k)
#     pmt_interp = CubicSpline(zin, pmt_k)
#     ptt_interp = CubicSpline(zin, ptt_k)
    
#     D_g = np.sqrt(1.0/(1.0+0.5*(k**2*mu**2*sigma_g**2)))          # This is unitless
#     D_u = np.sin(k*sigma_u)/(k*sigma_u)                           # This is unitless

#     result_sum = 0

#     for zz in range(len(N_redshifts_arr)): # looping through redshift  
#         r_sum = 0
#         zval = N_redshifts_arr[zz] 


#         if (zval < zminv): 
#             continue
#         elif (zval > zmaxv):
#             break

#         rval = r_array[zz]
#         deltarval = delta_r_array[zz]
#         pmm = pmm_interp(zval)
#         pmt = pmt_interp(zval)
#         ptt = ptt_interp(zval)

#         # calculate sigma8, Omz, f, beta
#         sigma8 = sigma80*(growth_array[zz])
#         Omz = Om*(E_z_inverse(zval)**2)*((1.0+zval)**3)
#         f = pow(Omz, gammaval)
#         beta = f*beta0*(growth_array[zz])/pow(Om,0.55)

#         # set up the power spectra 
#         a = (1.0/(1.0+zval))

#         vv_prefac  = a*H0*E_z(zval)*f*mu*(D_u)/k
#         ## set up the power spectra 
        
#         dd_prefac = (1.0/(beta**2) + 2.0*r_g*(mu**2)/beta + (mu**4))*(f**2)*(D_g**2)
#         dv_prefac = (r_g/beta + (mu**2))*f*D_g
#         P_gg = dd_prefac*pmm
#         P_ug = vv_prefac*dv_prefac*pmt
#         P_uu = (vv_prefac**2)*ptt


#         ci1, ci2, ci3, ci4 = 0, 0, 0, 0
#         dcdx1, dcdx2, dcdx3, dcdx4 = 0, 0, 0, 0
#         dcdy1, dcdy2, dcdy3, dcdy4 = 0, 0, 0, 0
    
#         if p1 == 0: # Differential w.r.t betaA
            
#             dcdx1 = -2.0*(1.0/beta + r_g*(mu**2))*(f**2)*(D_g**2)*pmm/(beta**2)
#             dcdx2 = -(vv_prefac*f*r_g*D_g*pmt)/(beta*beta) 
#             dcdx3 = dcdx2
#             dcdx4 = 0

#         elif p1 == 1: # Differential w.r.t. fsigma8
            
#             dcdx1 = 2.0*(f/(beta**2) + 2.0*f*r_g*(mu**2)/beta + f*(mu**4))*(D_g**2)*pmm/sigma8
#             dcdx2 = 2.0*vv_prefac*(r_g/beta + (mu**2))*D_g*pmt/sigma8
#             dcdx3 = dcdx2
#             dcdx4 = (2.0*P_uu)/(f*sigma8)

#         elif p1 == 2: # Differential w.r.t. r_g
            
#             dcdx1 = 2.0*(1.0/beta)*(mu**2)*(f**2)*(D_g**2)*pmm
#             dcdx2 = vv_prefac*(1.0/beta)*f*D_g*pmt
#             dcdx3 = dcdx2
#             dcdx4 = 0

#         elif p1 == 3: # Differential w.r.t. sigma_g
        

#             dcdx1 = -(k**2)*(mu**2)*(D_g**2)*sigma_g*P_gg
#             dcdx2 = -0.5*(k**2)*(mu**2)*(D_g**2)*sigma_g*P_ug
#             dcdx3 = dcdx2
#             dcdx4 = 0

#         elif p1 == 4: # Differential w.r.t. sigma_u
            
#             dcdx1 = 0
#             dcdx2 = P_ug*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
#             dcdx3 = dcdx2
#             dcdx4 = 2.0*P_uu*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)

#         else:
#             raise Exception("p1 is an invalid value.")

#         # ----------------------------------------------------------------------------------------------------

#         if p2 == 0: # Differential w.r.t betaA
            
#             dcdy1 = -2.0*(1.0/beta + r_g*(mu**2))*(f**2)*(D_g**2)*pmm/(beta**2)
#             dcdy2 = -(vv_prefac*f*r_g*D_g*pmt)/(beta*beta) 
#             dcdy3 = dcdy2
#             dcdy4 = 0 

#         elif p2 == 1: # Differential w.r.t. fsigma8
            
#             dcdy1 = 2.0*(f/(beta**2) + 2.0*f*r_g*(mu**2)/beta + f*(mu**4))*(D_g**2)*pmm/sigma8
#             dcdy2 = 2.0*vv_prefac*(r_g/beta + (mu**2))*D_g*pmt/sigma8 
#             dcdy3 = dcdy2
#             dcdy4 = (2.0*P_uu)/(f*sigma8)

#         elif p2 == 2: # Differential w.r.t. r_g
            
#             dcdy1 = 2.0*(1.0/beta)*(mu**2)*(f**2)*(D_g**2)*pmm
#             dcdy2 = vv_prefac*(1.0/beta)*f*D_g*pmt 
#             dcdy3 = dcdy2
#             dcdy4 = 0

#         elif p2 == 3: # Differential w.r.t. sigma_g
            
#             dcdy1 = -(k**2)*(mu**2)*(D_g**2)*sigma_g*P_gg
#             dcdy2 = -0.5*(k**2)*(mu**2)*(D_g**2)*sigma_g*P_ug 
#             dcdy3 = dcdy2
#             dcdy4 = 0

#         elif p2 == 4: # Differential w.r.t. sigma_u
        
#             dcdy1 = 0
#             dcdy2 = P_ug*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
#             dcdy3 = dcdy2
#             dcdy4 = 2.0*P_uu*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
#         else:
#             raise Exception("p2 is an invalid value.")

#         # ----------------------------------------------------------------------------------------------------
#         # need to do overlapping and nonoverlapping surveys seperately 
#         for s in np.arange(3):
#             surv_sum = 0
#             error_obs = 0
#             error_noise = 0
#             n_g = 0
#             n_u = 0

#             if survey_area[s] > 0:

#                 if s == 0: # redshift 
#                     n_g = N_bar_arr[0, zz]

#                 elif s == 1: # velocity 
#                     error_obs = H0*error_dist*rval              # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
#                     error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
#                     n_u = N_bar_arr[1, zz]/error_noise 

#                 else: # s == 2, overlap              
#                     error_obs = H0*error_dist*rval              # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
#                     error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
#                     n_u = N_bar_arr[1, zz]/error_noise                  
#                     n_g = N_bar_arr[0, zz]
                

#                 if (n_g == 0 and n_u == 0):
#                     continue

#                 # get the determinant of the covariance matrix C^-1
#                 det = 1.0 + n_u*n_g*(P_gg*P_uu - P_ug*P_ug) + n_u*P_uu + n_g*P_gg

#                 ci1 = n_u*n_g*P_uu + n_g
#                 ci2 = - n_g*n_u*P_ug
#                 ci3 = ci2 
#                 ci4 = n_g*n_u*P_gg + n_u
                

#                 # multiply the matrices ( dC_dx1 * C^-1 * dC_dx2 * C^-1) and take the trace of the result
               
#                 surv_sum += ci1*dcdx1*(ci1*dcdy1 + ci2*dcdy3) + ci1*dcdx2*(ci3*dcdy1 + ci4*dcdy3)
#                 surv_sum += ci2*dcdx3*(ci1*dcdy1 + ci2*dcdy3) + ci2*dcdx4*(ci3*dcdy1 + ci4*dcdy3)
#                 surv_sum += ci3*dcdx1*(ci1*dcdy2 + ci2*dcdy4) + ci3*dcdx2*(ci3*dcdy2 + ci4*dcdy4)
#                 surv_sum += ci4*dcdx3*(ci1*dcdy2 + ci2*dcdy4) + ci4*dcdx4*(ci3*dcdy2 + ci4*dcdy4)

#                 surv_sum /= det**2
#                 surv_sum *= survey_area[s]
#                 r_sum += surv_sum

#         result_sum += (rval**2)*deltarval*r_sum


#     return result_sum

def mu_integrand(mu, datalist1): # function to be integrated over mu for Fisher matrix elements, 
    # at some k and mu values 
    '''
    Function to compute the integral of the Fisher matrix elements over mu for a single value of k and z.
    '''

    k_index, k, zminv, p1, p2, zmaxv = datalist1  # [numk, k, zmin_iter, Data[i], Data[j], zmax_iter]
    
    D_g = np.sqrt(1.0/(1.0+0.5*(k**2*mu**2*sigma_g**2)))          # This is unitless, Lorenzian parametrization of the finger-of-god effect
    if (sigma_u < 10e-10):
        D_u = 1.0
    else:
        D_u = np.sin(k*sigma_u)/(k*sigma_u)                               # This is unitless

    result_sum = 0



    pmm = pmm_interp(k)
    pmt = pmt_interp(k)
    ptt = ptt_interp(k)
    for zz in range(len(N_redshifts_arr)): # looping through redshift 
        
        r_sum = 0
        zval = N_redshifts_arr[zz] 
        
        if (zval < zminv): 
            continue
        elif (zval > zmaxv):
            break
        
        rval = r_array[zz]
        deltarval = delta_r_array[zz]
        # calculate sigma8, Omz, f, beta
        sigma8 = sigma80*(growth_array[zz])
        Omz = Om*(E_z_inverse(zval)**2)*((1.0+zval)**3)
        f = pow(Omz, gammaval)
        beta = f*beta0*(growth_array[zz])/pow(Om,0.55)
    
        # set up the power spectra 
        a = (1.0/(1.0+zval))
    
        vv_prefac  = a*H0*E_z(zval)*f*mu*(D_u)/k
        ## set up the power spectra 
        
        dd_prefac = (1.0/(beta**2) + 2.0*r_g*(mu**2)/beta + (mu**4))*(f**2)*(D_g**2)
        dv_prefac = (r_g/beta + (mu**2))*f*D_g
        P_gg = dd_prefac*pmm
        P_ug = vv_prefac*dv_prefac*pmt
        P_uu = (vv_prefac**2)*ptt
    
    
        ci1, ci2, ci3, ci4 = 0, 0, 0, 0
        dcdx1, dcdx2, dcdx3, dcdx4 = 0, 0, 0, 0
        dcdy1, dcdy2, dcdy3, dcdy4 = 0, 0, 0, 0
    
        if p1 == 0: # Differential w.r.t betaA
            
            dcdx1 = -2.0*(1.0/beta + r_g*(mu**2))*(f**2)*(D_g**2)*pmm/(beta**2)
            dcdx2 = -(vv_prefac*f*r_g*D_g*pmt)/(beta*beta) 
            dcdx3 = dcdx2
            dcdx4 = 0
    
        elif p1 == 1: # Differential w.r.t. fsigma8
            
            dcdx1 = 2.0*(f/(beta**2) + 2.0*f*r_g*(mu**2)/beta + f*(mu**4))*(D_g**2)*pmm/sigma8
            dcdx2 = 2.0*vv_prefac*(r_g/beta + (mu**2))*D_g*pmt/sigma8
            dcdx3 = dcdx2
            dcdx4 = (2.0*P_uu)/(f*sigma8)
    
        elif p1 == 2: # Differential w.r.t. r_g
            
            dcdx1 = 2.0*(1.0/beta)*(mu**2)*(f**2)*(D_g**2)*pmm
            dcdx2 = vv_prefac*(1.0/beta)*f*D_g*pmt
            dcdx3 = dcdx2
            dcdx4 = 0
    
        elif p1 == 3: # Differential w.r.t. sigma_g
        
    
            dcdx1 = -(k**2)*(mu**2)*(D_g**2)*sigma_g*P_gg
            dcdx2 = -0.5*(k**2)*(mu**2)*(D_g**2)*sigma_g*P_ug
            dcdx3 = dcdx2
            dcdx4 = 0
    
        elif p1 == 4: # Differential w.r.t. sigma_u
            
            dcdx1 = 0
            dcdx2 = P_ug*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
            dcdx3 = dcdx2
            dcdx4 = 2.0*P_uu*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
    
        else:
            raise Exception("p1 is an invalid value.")
    
        # ----------------------------------------------------------------------------------------------------
    
        if p2 == 0: # Differential w.r.t betaA
            
            dcdy1 = -2.0*(1.0/beta + r_g*(mu**2))*(f**2)*(D_g**2)*pmm/(beta**2)
            dcdy2 = -(vv_prefac*f*r_g*D_g*pmt)/(beta*beta) 
            dcdy3 = dcdy2
            dcdy4 = 0 
    
        elif p2 == 1: # Differential w.r.t. fsigma8
            
            dcdy1 = 2.0*(f/(beta**2) + 2.0*f*r_g*(mu**2)/beta + f*(mu**4))*(D_g**2)*pmm/sigma8
            dcdy2 = 2.0*vv_prefac*(r_g/beta + (mu**2))*D_g*pmt/sigma8 
            dcdy3 = dcdy2
            dcdy4 = (2.0*P_uu)/(f*sigma8)
    
        elif p2 == 2: # Differential w.r.t. r_g
            
            dcdy1 = 2.0*(1.0/beta)*(mu**2)*(f**2)*(D_g**2)*pmm
            dcdy2 = vv_prefac*(1.0/beta)*f*D_g*pmt 
            dcdy3 = dcdy2
            dcdy4 = 0
    
        elif p2 == 3: # Differential w.r.t. sigma_g
            
            dcdy1 = -(k**2)*(mu**2)*(D_g**2)*sigma_g*P_gg
            dcdy2 = -0.5*(k**2)*(mu**2)*(D_g**2)*sigma_g*P_ug 
            dcdy3 = dcdy2
            dcdy4 = 0
    
        elif p2 == 4: # Differential w.r.t. sigma_u
        
            dcdy1 = 0
            dcdy2 = P_ug*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
            dcdy3 = dcdy2
            dcdy4 = 2.0*P_uu*(k*np.cos(k*sigma_u)/np.sin(k*sigma_u) - 1.0/sigma_u)
        else:
            raise Exception("p2 is an invalid value.")
    
        # ----------------------------------------------------------------------------------------------------
        # need to do overlapping and nonoverlapping surveys seperately 
        for s in np.arange(3):
            surv_sum = 0
            error_obs = 0
            error_noise = 0
            n_g = 0
            n_u = 0
    
            if survey_area[s] > 0:
    
                if s == 0: # redshift 
                    n_g = N_bar_arr[zz]
    
                elif s == 1: # velocity 
                    error_obs = H0*error_dist*rval              # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
                    error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
                    n_u = N_bar_arr[zz]/error_noise 
                    # n_u = error_noise/N_bar_arr[zz]
    
                else: # s == 2, overlap              
                    error_obs = H0*error_dist*rval              # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
                    error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
                    n_u = N_bar_arr[zz]/error_noise                  
                    n_g = N_bar_arr[zz]
                
    
                if (n_g == 0 and n_u == 0):
                    continue
    
                # get the determinant of the covariance matrix C^-1
                det = 1.0 + n_u*n_g*(P_gg*P_uu - P_ug*P_ug) + n_u*P_uu + n_g*P_gg
    
                ci1 = n_u*n_g*P_uu + n_g
                ci2 = - n_g*n_u*P_ug
                ci3 = ci2 
                ci4 = n_g*n_u*P_gg + n_u
                
    
                # multiply the matrices ( dC_dx1 * C^-1 * dC_dx2 * C^-1) and take the trace of the result
               
                surv_sum += ci1*dcdx1*(ci1*dcdy1 + ci2*dcdy3) + ci1*dcdx2*(ci3*dcdy1 + ci4*dcdy3)
                surv_sum += ci2*dcdx3*(ci1*dcdy1 + ci2*dcdy3) + ci2*dcdx4*(ci3*dcdy1 + ci4*dcdy3)
                surv_sum += ci3*dcdx1*(ci1*dcdy2 + ci2*dcdy4) + ci3*dcdx2*(ci3*dcdy2 + ci4*dcdy4)
                surv_sum += ci4*dcdx3*(ci1*dcdy2 + ci2*dcdy4) + ci4*dcdx4*(ci3*dcdy2 + ci4*dcdy4)
    
                surv_sum /= det**2
                surv_sum *= survey_area[s]
                r_sum += surv_sum
    
        result_sum += (rval**2)*deltarval*r_sum


    return result_sum

# ---------------------------------------------------------------------------------------------------------------

# function to calculate the effective redshift

def z_eff_integrand(mu, datalist1):
    '''

    Function to compute the effective redshift of a redshift bin. 

    '''

    k_index, k, zminv, zmaxv = datalist1  # [numk, k, zmin_iter, zmax_iter]

    # # set up arrays to hold power spectrum information as a function of z for a single k mode 
    # pmm_k = pmm_array[:, k_index]
    # ptt_k = ptt_array[:, k_index]

    # # set up interpolators for each of the power spectra using Cubic splines
    # pmm_interp = CubicSpline(zin, pmm_k)
    # ptt_interp = CubicSpline(zin, ptt_k)

    # calculate some stuff 
    D_g = np.sqrt(1.0/(1.0+0.5*((k*mu*sigma_g)**2)))              # This is unitless
    if (sigma_u < 10e-10):
        D_u = 1.0
    else:
        D_u = np.sin(k*sigma_u)/(k*sigma_u)                               # This is unitless

    
    # initialize variables for integrals 
    dVeff = 0.0         # effective volume element
    zdVeff = 0.0        # z x effective volume element 

    for zz in range(len(N_redshifts_arr)): # loop over all redshifts in redshift bin 

        zval = N_redshifts_arr[zz]

        if (zval < zminv): 
            continue
        elif (zval > zmaxv):
            break

        r_sum = 0.0
        rval = r_array[zz]
        deltarval = delta_r_array[zz]

        # get the power spectra 
        pmm = pmm_interp(k)
        ptt = ptt_interp(k)
        
 
        
        # set up Omz, f, beta 
        Omz = Om*(E_z_inverse(zval)**2)*((1.0+zval)**3)
        f = pow(Omz, gammaval)
        beta = f*beta0*(growth_array[zz])/pow(Om,0.55)

        # set up the power spectra 
        a = (1.0/(1.0+zval))

        vv_prefac  = a*H0*E_z(zval)*f*mu*(D_u)/k
        dd_prefac = (1.0/(beta**2) + 2.0*r_g*(mu**2)/beta + (mu**4))*(f**2)*(D_g**2)
        

        P_gg = dd_prefac*pmm
        P_uu = (vv_prefac**2)*ptt

        # We need to do the overlapping and non-overlapping parts of the redshifts and PV surveys separately
        for s in np.arange(3):
            surv_sum = 0
            error_obs = 0
            error_noise = 0
            n_g = 0
            n_u = 0

            if survey_area[s] > 0:

                if s == 0: # redshift 
                    n_g = N_bar_arr[zz]

                elif s == 1: # velocity 
                    error_obs = H0*error_dist*rval              # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
                    error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
                    n_u = N_bar_arr[zz]/error_noise 

                else: # i == 2, overlap              
                    error_obs = H0*error_dist*rval              # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
                    error_noise = error_rand**2 + error_obs**2     # Error_noise is in km^{2}s^{-2}
                    n_u = N_bar_arr[zz]/error_noise                  
                    n_g = N_bar_arr[zz]

                value1 = n_g/(1.0 + n_g*P_gg)
                value2 = n_u/(1.0 + n_u*P_uu)

                surv_sum += value1**2 + value2**2

                surv_sum *= survey_area[s]*np.pi
                r_sum += surv_sum


        dVeff += rval*rval*deltarval*r_sum
        zdVeff += zval*rval*rval*deltarval*r_sum
    return zdVeff/dVeff, dVeff*P_uu**2

# ---------------------------------------------------------------------------------------------------------------

# run the main code to compute the Fisher matrix 
if __name__ == "__main__": 

    # read in the power spectra
    pmm_array, pmt_array, ptt_array, delta_ks, kvals = read_power() 

    # read in the survey number densities
    N_redshifts_arr, N_bar_arr, r_array, delta_r_array, redshift_dist_spline, growth_factor_spline, growth_array, zmin, zmax = read_nz() 
    
    pmm_interp = CubicSpline(kvals, pmm_array)
    pmt_interp = CubicSpline(kvals, pmt_array)
    ptt_interp = CubicSpline(kvals, ptt_array)

    # do some basic checks on some things:
    if not ((survey_area[0] > 0.0) or (survey_area[2] > 0.0)): # if position and velocity surveys are NOT overlapping AND their is NO density field information,
        # there is no density field information at all

        for i in np.arange(nparams): # (0=beta, 1=fsigma8, 2=r_g, 3=sigma_g, 4=sigma_u)

            if Data[i] == 2: # r_g
                raise Exception("ERROR: r_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular).")

            elif Data[i] == 3: # sigma_g

                raise Exception("ERROR: sigma_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular).")

    if not (((survey_area[1] > 0.0) or (survey_area[2] > 0.0))): # if there is NO overlap between surveys AND there is NO information in the velocity field,
        # there is no velocity field information at all

        for i in np.arange(nparams): # (0=beta, 1=fsigma8, 2=r_g, 3=sigma_g, 4=sigma_u)

            if Data[i] == 4: # sigma_u
                raise Exception("ERROR: sigma_u is a free parameter, but there is no information in the velocity field (Fisher matrix will be singular).")

        
    # Calculate the Fisher matrices for all redshiftbins. ------------------------------------------------------------------------------------------------
    Fisher_total_matrix = np.zeros((nparams, nparams))
    Fisher_total_matrix = np.matrix(Fisher_total_matrix)

    print("Evaluating the Fisher Matrix for %d redshift bins between [z_min = %.3f, z_max = %.3f]" % (num_redshift_bins, zmin, zmax))

    for ziter in range(num_redshift_bins): # iterating through redshift bins --------------------------------------------------------

        # for a single redshift bin we will calculate: 
        # 1) the effective redshift
        # 2) the Fisher matrix for this effective redshift 
        # 3) print out error bars for each parameter of interest to the terminal and effective redshift 

        zbinwidth = (zmax-zmin)/(num_redshift_bins)
        zmin_iter = ziter*zbinwidth + zmin
        zmax_iter = (ziter+1.0)*zbinwidth + zmin

        rzmax = sp.interpolate.splev(zmax_iter, redshift_dist_spline)  # max distance to a galaxy in redshift bin
        # kmin = 2.0*np.pi/rzmax                        # k mode value for max distance 
        if (dataset == 1):
            kmin = 0.007*h
        else:
            kmin = 0.0025*h
        # give information of min and max k modes
        if (verbosity > 0):
            print("Evaluating the Fisher matrix with [k_min = %.6f, k_max = %.6f] and [z_min = %.3f, z_max = %.3f]" % (kmin, kmax, zmin_iter, zmax_iter))

        # Calculate the effective redshift (which has been based on the sum of the S/N for the density and velocity fields)

        # trapezoidal rule over k
        k_sum1, k_sum2 = 0.0, 0.0
        count = 0
        volume = 0.0
        for numk in range(len(delta_ks)):

            k = kvals[numk]+0.5*delta_ks[numk]
            deltak = delta_ks[numk]

            if k < kmin: 
                continue
            elif k > kmax:
                continue 

            datalist1 = [numk, k, zmin_iter, zmax_iter]

            # integration method 1 using quadrature (slow)
            #result = quad(z_eff_integrand, 0.0, 1.0, args = (datalist1), epsabs = 1e-4)[0] # integral over z and mu  

            # integration method 2 using simpson rule (faster)
            mus = np.linspace(0.0, 1.0, 1000)
            zeffs= z_eff_integrand(mus, datalist1)[0]
            result = simps(zeffs, mus)
            
            k_sum1 += k*k*deltak*result
            k_sum2 += k*k*deltak
            

        z_eff = k_sum1/k_sum2
        if (verbosity > 0): 
            print("Effective redshift for this redshift bin, z_eff = %.6f" % z_eff)


        growth_eff = growth_factor_spline(z_eff)

        # Calculate the fisher matrix, integrating over k, then mu, then r (r is last as it means we are effectively integrating over effective volume).
        # As the input spectra are tabulated we'll just use the trapezoid rule to integrate over k
        Fisher_matrix = np.zeros((nparams, nparams))

        number_unique_elements = nparams + nparams*(nparams-1.0)/2.0
        counter = 0
        # here we will just do the integral for mu and k first (integral over redshift within mu_integrand)
        for i in range(0, nparams):
            for j in range(i, nparams): # getting F_ij where i and j are some parameter we want to allow to vary, in a loop and saving 
                # in Fisher matrix 

                counter += 1

                k_sum = 0.0

                for numk in range(len(delta_ks)):

                    k = kvals[numk]+0.5*delta_ks[numk]

                    if k < kmin: 
                        continue
                    elif k > kmax:
                        continue 

                    deltak = delta_ks[numk]

                    datalist1 = [numk, k, zmin_iter, Data[i], Data[j], zmax_iter]

                    # integration method 1 using quadrature 
                    #result = quad(mu_integrand, 0.0, 1.0, args=(datalist1), epsabs = 5e-2)[0]  

                    # integration method 2 using simpson rule
                    mus = np.linspace(0.0, 1.0, 1000)
                    fmus = mu_integrand(mus, datalist1)
                    result = simps(fmus, mus)

                    k_sum += k*k*deltak*result                      # adding up contribution from all ks (with trapezoidal rule)

                Fisher_matrix[i, j] = k_sum/(np.pi*4)
                if i != j:
                    Fisher_matrix[j, i] = k_sum/(np.pi*4)



        Fisher_matrix = np.matrix(Fisher_matrix)


        # add Fisher matrix in this redshift bin to the complete Fisher matrix 
        Fisher_total_matrix = Fisher_total_matrix + Fisher_matrix
                

        # print the Fisher matrix to the terminal for this redshift bin 
        if (verbosity == 2):
            print("Fisher Matrix for this redshift bin:")
            print("==================")
            print(Fisher_matrix) 

        # now invert the Fisher matrix
        Fisher_matrix = np.matrix(Fisher_matrix)
        inverted_Fisher_matrix = np.linalg.inv(Fisher_matrix)

        # calculate sigma8, Omz, f and beta in fiducial cosmology at z_eff
        sigma8 = sigma80 * growth_eff
        Omz = Om*(E_z_inverse(z_eff)**2)*((1.0+z_eff)**3)
        f = pow(Omz, gammaval)
        beta = f*beta0*growth_eff/pow(Om,0.55)

        # print the values of sigma8 to the terminal (at the effective redshift) and the constrain on it
        if 1 in Data and verbosity == 0:
            for i in range(nparams):
                if i == 1:
                    print("#   zmin         zmax         zeff         fsigma8(z_eff)         percentage error(z_eff)")
                    print("%.6f  %.6f  %.6f  %.6f  %.6f" % (zmin_iter, zmax_iter, z_eff, f*sigma8, 100.0*inverted_Fisher_matrix[i, i]/(f*sigma8)) )
            
    
        if verbosity > 0:

            print("============================================")
            for i in range(nparams): # print the error bars for each parameter, as determined from the inverted Fisher matrix inverse for this redshift bin

                if (Data[i] == 0):
                    print("beta = %.6f pm %.6f" % (beta, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    print(" %.4f percent error on beta" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/beta) )
                
                if (Data[i] == 1):
                    print("f*sigma8 = %.6f pm %.6f" % (f*sigma8, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    print(" %.4f percent error on fsigma8" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/(f*sigma8))  )
                
                if (Data[i] == 2):
                    print("r_g = %.6f pm %.6f" % (r_g, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    print(" %.4f percent error on r_g" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/r_g)  )
                
                if (Data[i] == 3):
                    print("sigma_g = %.6f pm %.6f" % (sigma_g, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    print(" %.4f percent error on sigma_g" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/sigma_g)  )
                
                if (Data[i] == 4):
                    print("sigma_u = %.6f pm %.6f" % (sigma_u, np.sqrt(inverted_Fisher_matrix[i,i])) )
                    print(" %.4f percent error on sigma_u" % (100*np.sqrt(inverted_Fisher_matrix[i, i])/sigma_u)    ) 

            print("============================================")

        # print the inverted Fisher matrix to the terminal for this redshift bin 
        if (verbosity == 2):
            print("Covariance Matrix for this redshift bin:")
            print("==================")
            print(inverted_Fisher_matrix)
            
        # -------------------------------------------------------------------------------------------------------------------------------

    # calculate the full Fisher matrix constraints, sum of multiple redshift bins 
    if (num_redshift_bins > 1):

        # rzmax = rz(zmax)
        # kmin = 2.0*np.pi/rzmax
        
        if (dataset == 1):
            kmin = 0.0078*h
        else:
            kmin = 0.0025*h

        if (verbosity > 0):
            print("Finally, evaluating the Fisher Matrix for all redshfit bins: ", \
            " [k_min = %.5f, k_max = %.5f] and [z_min = %.3f, z_max = %.3f]" % (kmin, kmax, zmin, zmax))

          # Calculate the effective redshift (which I base on the sum of the S/N for the density and velocity fields)
        k_sum1, k_sum2 = 0.0, 0.0
        volume_k = 0.0
        
        V_eff_k = []

        for numk in range(len(delta_ks)):

            k = kvals[numk]+0.5*delta_ks[numk]
            
            deltak = delta_ks[numk]

            if k < kmin: 
                continue
            elif k > kmax:
                continue 

            datalist1 = [numk, k, zmin, zmax]
            # result = quad(z_eff_integrand, 0.0, 1.0, args = (datalist1), epsabs = 5e-3)[0] # integral over z and mu  
            mus = np.linspace(0.0, 1.0, 1000)
            zeffs, volume_seg = z_eff_integrand(mus, datalist1)
            result = simps(zeffs, mus)
            volume_part = simps(volume_seg, mus)
            V_eff_k.append(volume_part)

            k_sum1 += k*k*deltak*result
            k_sum2 += k*k*deltak
            volume_k += k*k*deltak*volume_part
        
        z_eff = k_sum1/k_sum2
        volume_eff = volume_k/k_sum2
        if (verbosity > 0): 
            print("Effective redshift z_eff = %.6f" % z_eff)
            print("Effective volume volume_eff = %.6f (Mpc/h^3)" % volume_eff)
            

        growth_eff = growth_factor_spline(z_eff)

        if (verbosity == 2):
            print("Fisher Matrix for all redshift bins:")
            print("======================")
            print(Fisher_total_matrix)


        # Now invert the Fisher matrix
        inverted_Fisher_matrix_total = np.linalg.inv(Fisher_total_matrix)

        # calculate sigma8, Omz, f and beta in fiducial cosmology at z_eff
        sigma8 = sigma80 * growth_eff
        Omz = Om*(E_z_inverse(z_eff)**2)*((1.0+z_eff)**3)
        f = pow(Omz, gammaval)
        beta = f*beta0*growth_eff/pow(Om,0.55)

        if 1 in Data and verbosity == 0:
            for i in range(nparams):
                if i == 1:
                    print("Full redshift range:")
                    print("#     zmin         zmax         zeff      fsigma8(z_eff)   percentage error(z_eff)")
                    print("%.6f  %.6f  %.6f  %.6f  %.6f" % ( zmin, zmax, z_eff, f*sigma8, 100.0*np.sqrt(inverted_Fisher_matrix_total[i, i])/(f*sigma8)) )
            
        if verbosity > 0:

            for i in range(nparams):

                if (Data[i] == 0):
                    print("beta = %f pm %f" % (beta, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    print("%.6f percent error on beta" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/beta) )
                
                if (Data[i] == 1):
                    print(" f*sigma8 = %.6f pm %.6f" % (f*sigma8, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    print(" %.6f percent error on fsigma8" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/(f*sigma8)) )
                
                if (Data[i] == 2):
                    print("r_g = %.6f pm %.6f" % (r_g, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    print(" %.6f percent error on r_g" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/r_g) )
                
                if (Data[i] == 3):
                    print("sigma_g = %.6f pm %.6f" % (sigma_g, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    print(" %.6f percent error on sigma_g" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/sigma_g) )
                
                if (Data[i] == 4):
                    print("sigma_u = %.6f pm %.6f" % (sigma_u, np.sqrt(inverted_Fisher_matrix_total[i,i])) )
                    print(" %.6f percent error on sigma_u" % (100*np.sqrt(inverted_Fisher_matrix_total[i, i])/sigma_u) )
                

        if (verbosity == 2):
            print("Covariance Matrix:")
            print("======================")
            print(inverted_Fisher_matrix_total)

     
     # ----------------------------------------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------------------------------------------
# end main 

    # if (dataset == 1):
    #     kmin = 0.007*h
    # else:
    #     kmin = 0.0025*h


