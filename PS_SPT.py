# -*- coding: utf-8 -*-
"""
Created on Thu May 22 09:59:35 2025

@author: s4479813
"""

import numpy as np
import scipy as sp
from scipy import integrate
import camb
import time
from configobj import ConfigObj
import sys

# Speed of light in km/s
LightSpeed = 299792.458
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
    fz = ((1.0+redshift)**(3*(1.0+w0+wa*ap)))*np.exp(-3*wa*(redshift/(1.0+redshift)))
    omega_k = 1.0-omega_m-omega_lambda-omega_rad
    return np.sqrt(omega_rad*(1.0+redshift)**4+omega_m*(1.0+redshift)**3+omega_k*(1.0+redshift)**2+omega_lambda*fz)

def D_integrand(a, omega_m):
    E = Ez(1.0/a - 1., omega_m, 1.0 - omega_m, 0.0, -1.0, 0.0, 0.0)
    return 1./(a**3*E**3)

def Fmn(m,n,r,x,y):
    if(m==0) and (n==0):
        FMN=(7.0*x + 3.0*r - 10.0*r*x*x)**2/(14.0*14.0*r*r*y*y*y*y);
    if(m==0) and (n==1):
        FMN= (7.0*x + 3.0*r - 10.0*r*x*x)*(7.0*x - r - 6.0*r*x*x)/(14.0*14.0*r*r*y*y*y*y);
    if(m==0) and (n==2):
        FMN= (x*x - 1.0)*(7.0*x + 3.0*r - 10.0*r*x*x)/(14.0*r*y*y*y*y);
    if(m==0) and (n==3):
        FMN= (1.0 - x*x)*(3.0*r*x - 1.0)/(r*r*y*y);       
    if(m==1) and (n==0):
        FMN= x*(7.0*x + 3.0*r - 10.0*r*x*x)/(14.0*r*r*y*y);
    if(m==1) and (n==1):
        FMN= (7.0*x - r - 6.0*r*x*x)**2/(14.0*14.0*r*r*y*y*y*y);
    if(m==1) and (n==2):
        FMN= (x*x - 1.0)*(7.0*x - r - 6.0*r*x*x)/(14.0*r*y*y*y*y);
    if(m==1) and (n==3):
        FMN=( 4.0*r*x + 3.0*x*x - 6.0*r*x*x*x - 1.0)/(2.0*r*r*y*y);             
    if(m==2) and (n==0):
        FMN= (2.0*x + r - 3.0*r*x*x)*(7.0*x + 3.0*r - 10.0*r*x*x)/(14.0*r*r*y*y*y*y);
    if(m==2) and (n==1):
        FMN= (2.0*x + r - 3.0*r*x*x)*(7.0*x - r - 6.0*r*x*x)/(14.0*r*r*y*y*y*y);
    if(m==2) and (n==2):
        FMN= x*(7.0*x - r - 6.0*r*x*x)/(14.0*r*r*y*y);
    if(m==2) and (n==3):
        FMN= 3.0*(1.0-x*x)*(1.0-x*x)/(y*y*y*y); 
    if(m==3) and (n==0):
        FMN= (1.0 - 3.0*x*x - 3.0*r*x + 5.0*r*x*x*x)/(r*r*y*y);
    if(m==3) and (n==1):
        FMN=  (1.0 - 2*r*x)*(1.0 - x*x)/(2.0*r*r*y*y);
    if(m==3) and (n==2):
        FMN=  (1.0 - x*x)*(2.0 - 12.0*r*x - 3.0*r*r + 15.0*r*r*x*x)/(r*r*y*y*y*y);
    if(m==3) and (n==3):
        FMN=  (-4.0 + 12.0*x*x + 24.0*r*x - 40.0*r*x*x*x + 3.0*r*r - 30.0*r*r*x*x + 35.0*r*r*x*x*x*x)/(r*r*y*y*y*y);        
    return FMN

def Imn_inner_integ(x,m,n,k,r,Pl_spline):
    y    = np.sqrt(1.0+r*r-2.0*r*x)#x=cos theta
    Plkq = sp.interpolate.splev(np.log10(k*y), Pl_spline, der=0)
    return Fmn(m,n,r,x,y)*10.0**(Plkq)

def Imn_outer_integ(r,m,n,k,Pl_spline):
    integ,errin= sp.integrate.quad(Imn_inner_integ,-1.0,1.0,epsabs=0.0,epsrel=1.0e-4,args=(m,n,k,r,Pl_spline))
    Pl   = sp.interpolate.splev(np.log10(k*r),Pl_spline, der=0)
    return r*r*integ*10.0**(Pl)

def I_mn(kmod,Pl_spline,ks):
    Imn=np.zeros((len(ks),4,4))
    for i in range(len(ks)):
        I00,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(0,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,0,0]= I00*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        I01,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(0,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,0,1]= I01*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
       
        I11,errin= sp.integrate.quad(Imn_outer_integ,kmod[0],kmod[-1],args=(1,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Imn[i,1,1]= I11*ks[i]*ks[i]*ks[i]/(4.0*np.pi*np.pi)
        
    return Imn

def Gmn(m,n,r):    
    if(m==0) and (n==0):
        GMN= (12.0/(r*r) - 158.0 + 100.0*r*r - 42.0*r*r*r*r + (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(7.0*r*r + 2.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/3024.0;
    if(m==0) and (n==1):
        GMN= (24.0/(r*r) - 202.0 + 56.0*r*r - 30.0*r*r*r*r + (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(5.0*r*r + 4.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/3024.0;
    if(m==0) and (n==2):
        GMN= (2.0*(r*r + 1.0)*(3.0*r*r*r*r - 14.0*r*r + 3.0)/(r*r) - (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/224.0;
    if(m==1) and (n==0):
        GMN= (-38.0 +48.0*r*r - 18.0*r*r*r*r + (9.0/r)*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/1008.0;
    if(m==1) and (n==1):
        GMN= (12.0/(r*r) - 82.0 + 4.0*r*r - 6.0*r*r*r*r + (3.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(r*r + 2.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/1008.0;
    if(m==2) and (n==0):
        GMN= (2.0*(9.0 - 109.0*r*r + 63.0*r*r*r*r - 27.0*r*r*r*r*r*r)/(r*r) + (9.0/(r*r*r))*(r*r - 1.0)*(r*r - 1.0)*(r*r - 1.0)*(3.0*r*r + 1.0)*np.log((r + 1.0)/np.abs(r - 1.0)))/672.0;
    return GMN

def Jmn_integ(q,m,n,k,Pl_spline):
    r    = q/k
    if(r==1.0):
        r=(q+0.00001*q)/k
    Pl   = sp.interpolate.splev(np.log10(q),Pl_spline, der=0)
    return Gmn(m,n,r)*10.0**(Pl)

def J_mn(kmod,Pl_spline,ks):
    Jmn=np.zeros((len(ks),2,3))
    for i in range(len(ks)):
        J00,errin= sp.integrate.quad(Jmn_integ,kmod[0],kmod[-1],args=(0,0,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3)
        Jmn[i,0,0] = J00/(2.0*np.pi*np.pi)
        J01,errin= sp.integrate.quad(Jmn_integ,kmod[0],kmod[-1],args=(0,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3)
        Jmn[i,0,1] = J01/(2.0*np.pi*np.pi)
        
        J11,errin= sp.integrate.quad(Jmn_integ,kmod[0],kmod[-1],args=(1,1,ks[i],Pl_spline),epsabs=0.0,epsrel=1.0e-3) 
        Jmn[i,1,1] = J11/(2.0*np.pi*np.pi)
        
    return Jmn

def get_camb(npoints, ombh2=0.02237, omch2=0.12, tau=0.097, As=2.092988e-09, H0=67.36, ns = 0.9649, kmin = 1e-4, kmax = 1.0, redshifts=0.0):

    # Generate the matter power spectrum
    pars = camb.CAMBparams()
    pars.InitPower.set_params(As=As, ns=ns)
    
    pars.set_matter_power(
        redshifts=[redshifts], kmax=100.0, nonlinear=False
    )
        
    pars.set_cosmology(
        H0=H0,
        omch2=omch2,
        ombh2=ombh2,
        omk=0.0,
        tau=tau,
        mnu=0.06,
        # TCMB=2.895,
        # TCMB = 2.7255, 
        neutrino_hierarchy='degenerate',
    )
    pars.NonLinear = camb.model.NonLinear_none

    # Run CAMB
    results = camb.get_results(pars)

    # Get the power spectrum
    kin, _, Plin = results.get_matter_power_spectrum(
        minkh=kmin,
        # maxkh=0.5,
        # npoints=4000,
        maxkh = kmax,
        npoints=npoints,
    )
    
    sigma8 = results.get_sigma8()
    print(sigma8)
    
    return kin, Plin[0], sigma8

def PSloop_Fun(Nkmod,H0 = 67.36, ombh2 = 0.02237,omch2=0.12,ns=0.9649, As=2.092988e-09, kmin = 1e-4, kmax = 1.0, tau=0.097, redshifts = 0.0): 
    kh,pk,sig8=get_camb(Nkmod, ombh2 = ombh2, omch2 = omch2, tau=tau, As=As, H0=H0, ns=ns, kmin=kmin, kmax=kmax, redshifts = redshifts)
    print( 'sigma8_fid_integ = ',sig8,'\n')
    Pl_spline = sp.interpolate.splrep(np.log10(kh),np.log10(pk), s=0)
    # ks = kh[::4]
    ks = kh[::2]
    # integrations:
    print( '\n Integration-Imn \n')
    Imn=I_mn(kh,Pl_spline,ks)
    print( '\n Integration-Jmn \n')
    Jmn=J_mn(kh,Pl_spline,ks)
    
    return Imn, Jmn, kh, ks, pk, sig8

if __name__ == "__main__":
    # PT, sigma8_fid = np.load('./INTEG_PL.npy', allow_pickle=True)
    # k  =PT[0,0:] ; Pl =PT[1,0:];   
    # I00=PT[2,0:]  ;I01=PT[3,0:]  ;I02=PT[4,0:]  ;I03=PT[5,0:]  ;I10=PT[6,0:]  ;I11=PT[7,0:]  ;I12=PT[8,0:]  ;I13=PT[9,0:] ;
    # I20=PT[10,0:] ;I21=PT[11,0:] ;I22=PT[12,0:] ;I23=PT[13,0:] ;I30=PT[14,0:] ;I31=PT[15,0:] ;I32=PT[16,0:] ;I33=PT[17,0:] ;
    # J00=PT[18,0:] ;J01=PT[19,0:];J02=PT[20,0:]; J10=PT[21,0:]; J11=PT[22,0:];  J20=PT[23,0:]; K00=PT[24,0:]; Ks00=PT[25,0:];
    # K01=PT[26,0:]; Ks01=PT[27,0:];Ks02=PT[28,0:];K10=PT[29,0:];Ks10=PT[30,0:]; K11=PT[31,0:];
    # Ks11=PT[32,0:];K20=PT[33,0:];Ks20=PT[34,0:];K30=PT[35,0:];Ks30=PT[36,0:];  sig3_squre=PT[37,0:];Intg_norm=PT[38,0:];
    
    # configfile = sys.argv[1] #input the location of the configuration file 
    # pardict = ConfigObj(configfile)
    
    # effective_redshift = np.float64(pardict['effective_redshift'])
    # # omega_m = 0.315192
    # omega_m = np.float64(pardict['omega_m'])
    effective_redshift = 0.0
    omega_m = 0.3121
    
    Dz = sp.integrate.quad(D_integrand, 0.0, 1./(1. + effective_redshift), args=(omega_m))[0]/sp.integrate.quad(D_integrand, 0.0, 1., args=(omega_m))[0]*Ez(effective_redshift, omega_m, 1.0 - omega_m, 0.0, -1.0, 0.0, 0.0)
    
    # P_mm = Dz**2*Pl + Dz**4*(2*I00 + 2*3*k**2*Pl*J00)
    # P_mv = Dz**2*Pl + Dz**4*(2*I01 + 2*3*k**2*Pl*J01)
    # p_vv = Dz**2*Pl + Dz**4*(2*I11 + 2*3*k**2*Pl*J11)
    
    start = time.time()
    Imn, Jmn, kh, ks, pk, sig8 = PSloop_Fun(2001)
    end = time.time()
    print(end - start)
    
    Pl_spline = sp.interpolate.splrep(np.log10(kh),np.log10(pk), s=0)

    Plin = 10**sp.interpolate.splev(np.log10(ks), Pl_spline, der=0)

    I00_new = Imn[0, 0:]

    I00_new = Imn[:, 0, 0]

    I01_new = Imn[:, 0, 1]

    I11_new = Imn[:, 1, 1]

    J00_new = Jmn[:, 0, 0]

    J01_new = Jmn[:, 0, 1]

    J11_new = Jmn[:, 1, 1]

    P_mm_new = Plin + (2*I00_new + 2*3*ks**2*Plin*J00_new)

    P_mv_new = Plin + (2*I01_new + 2*3*ks**2*Plin*J01_new)

    P_vv_new = Plin + (2*I11_new + 2*3*ks**2*Plin*J11_new)
    
    
    P_mm_new_z = Dz**2*Plin + Dz**4*(2*I00_new + 2*3*ks**2*Plin*J00_new)
    P_mv_new_z = Dz**2*Plin + Dz**4*(2*I01_new + 2*3*ks**2*Plin*J01_new)
    P_vv_new_z = Dz**2*Plin + Dz**4*(2*I11_new + 2*3*ks**2*Plin*J11_new)
    
    output = np.vstack((ks, P_mm_new_z, P_mv_new_z, P_vv_new_z)).T

    np.savetxt('./grid_correction/PS_model_SPT_z_' + str(round(effective_redshift, 3)) + '_omegam_' + str(round(omega_m, 3)) +'.dat', output)