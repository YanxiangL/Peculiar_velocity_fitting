#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_integration.h>

// Code to calculate the covariance matrix for fitting the peculiar velocities of the 2MTF simulations.
// Generates a grid covering -120<x<120 Mpc/h and similar for y/z so should cover the full data.
// When applied to particular simulations, you need to remove the rows/columns from the matrix
// that contain no galaxies.

// Run parameters
static double LightSpeed = 299792.458;
double omega_m;
double sigma_u;
struct fparams{ 
	gsl_spline * spline_P; 
	gsl_interp_accel * acc_P;
	gsl_spline * spline_grid; 
	gsl_interp_accel * acc_grid; 
	double dist;
	int ell;
	int pq;
	int damping;
	int gridcorr;
};	

gsl_interp_accel * P_mm_acc, * P_vm_acc, * P_vv_acc, * gridcorr_acc;
gsl_spline * P_mm_spline, * P_vm_spline, * P_vv_spline, * gridcorr_spline; 

gsl_interp_accel **** xi_dd_acc, **** xi_dv_acc, *** xi_vv_acc;
gsl_spline **** xi_dd_spline, **** xi_dv_spline, *** xi_vv_spline; 

double ffunc(double x, void *p) {
    double ff = (LightSpeed/100.0)/sqrt(omega_m*(1.0+x)*(1.0+x)*(1.0+x)+(1.0-omega_m)); //This calculates c/H 
    return ff;
}

double rz(double red) {//This calculates the redshift to the galaxy
    double result, error;
    gsl_function F;
    gsl_integration_workspace * w = gsl_integration_workspace_alloc(1000);
    F.function = &ffunc;
    gsl_integration_qags(&F, 0.0, red, 0, 1e-7, 1000, w, &result, &error);
    gsl_integration_workspace_free(w);
    return result;
}

double D_u(double k) {//The damping term due to redshift space distortion
	double result;
	if (k*sigma_u < 1e-10) {
		result = 1.0;
	} else {
		result = sin(k*sigma_u)/(k*sigma_u);
	}
	return result;
}

double conv_integrand(double k, void * p) {
	struct fparams params = *(struct fparams *) p;

	double damping = 1.0;
	if (params.damping == 1) damping = D_u(k);
	if (params.damping == 2) damping = pow(D_u(k), 2.0);

	double gridcorr = 1.0;
	if (params.gridcorr == 1) gridcorr = gsl_spline_eval(gridcorr_spline, k, gridcorr_acc);

	double function = pow(k, params.pq+2)*gsl_spline_eval(params.spline_P, k, params.acc_P)*gridcorr*damping*gsl_sf_bessel_jl(params.ell,k*params.dist);
	return function;
}

double conv_integral(double kmin, double kmax, double dist, int abtype, int ell, int pq, int gridcorr) {
    
    double result, error;
    struct fparams params;
    if (abtype == 0) { params.spline_P = P_mm_spline; params.acc_P = P_mm_acc; }
    if (abtype == 1) { params.spline_P = P_vm_spline; params.acc_P = P_vm_acc; }
    if (abtype == 2) { params.spline_P = P_vv_spline; params.acc_P = P_vv_acc; }
    params.ell = ell;
    params.pq = pq;
    params.dist = dist;
    params.damping = abtype;
    params.gridcorr = gridcorr;

    gsl_function F;
    gsl_integration_cquad_workspace * w = gsl_integration_cquad_workspace_alloc(1000);
    F.function = &conv_integrand;
    F.params = &params;
    gsl_integration_cquad(&F, kmin, kmax, 0, 1e-5, w, &result, &error, NULL);
    gsl_integration_cquad_workspace_free(w);

    return result;
}

double H_vv(int ell, double theta, double phi) {

	if (ell == 0){
		return cos(theta)/3.0;
	} else if (ell == 2) {
		return (3.0*cos(2.0*phi) + cos(theta))/6.0;
	} else {
		return 0;
	}

}

double H_dv(int ell, int p, double theta, double phi) {

    if (ell == 1) {
        if (p == 0) {
            return -cos(phi + theta/2.);
        } else if (p == 1) {
            return (-cos(phi - (3*theta)/2.) - 2*cos(phi + theta/2.))/5.;
        } else if (p == 2) {
            return (-3*(2*cos(phi - (3*theta)/2.) + 3*cos(phi + theta/2.)))/35.;
        } else if (p == 3) {
            return (-3*cos(phi - (3*theta)/2.) - 4*cos(phi + theta/2.))/21.;
        } else if (p == 4) {
            return (-4*cos(phi - (3*theta)/2.) - 5*cos(phi + theta/2.))/33.;
        }
    } else if (ell == 3) {
        if (p == 1) {
            return (-cos(phi - (3*theta)/2.) - 5*cos(3*phi - theta/2.) - 2*cos(phi + theta/2.))/20.;
        } else if (p == 2) {
            return (-5*cos(3*phi - (5*theta)/2.) - 6*cos(phi - (3*theta)/2.) - 20*cos(3*phi - theta/2.) - 9*cos(phi + theta/2.))/90.;
        } else if (p == 3) {
            return (-10*cos(3*phi - (5*theta)/2.) - 9*cos(phi - (3*theta)/2.) - 25*cos(3*phi - theta/2.) - 12*cos(phi + theta/2.))/132.;
        } else if (p == 4) {
            return (-7*(5*cos(3*phi - (5*theta)/2.) + 4*cos(phi - (3*theta)/2.) + 10*cos(3*phi - theta/2.) + 5*cos(phi + theta/2.)))/429.;
        }
    } else if (ell == 5) {
        if (p == 2) {
            return (-7*cos(3*phi - (5*theta)/2.) - 12*cos(phi - (3*theta)/2.) - 63*cos(5*phi - (3*theta)/2.) - 28*cos(3*phi - theta/2.) - 18*cos(phi + theta/2.))/1008.;
        } else if (p == 3) {
            return (-63*cos(5*phi - (7*theta)/2.) - 70*cos(3*phi - (5*theta)/2.) - 90*cos(phi - (3*theta)/2.) - 378*cos(5*phi - (3*theta)/2.) - 175*cos(3*phi - theta/2.) - 120*cos(phi + theta/2.))/4368.;
        } else if (p == 4) {
            return (-42*cos(5*phi - (7*theta)/2.) - 35*cos(3*phi - (5*theta)/2.) - 40*cos(phi - (3*theta)/2.) - 147*cos(5*phi - (3*theta)/2.) - 70*cos(3*phi - theta/2.) - 50*cos(phi + theta/2.))/1560.;
        }
    } else if (ell == 7) {
        if (p == 3) {
            return (-33*cos(5*phi - (7*theta)/2.) - 54*cos(3*phi - (5*theta)/2.) - 429*cos(7*phi - (5*theta)/2.) - 75*cos(phi - (3*theta)/2.) - 198*cos(5*phi - (3*theta)/2.) - 135*cos(3*phi - theta/2.) - 100*cos(phi + theta/2.))/27456.;
        } else if (p == 4) {
            return (-429*cos(7*phi - (9*theta)/2.) - 462*cos(5*phi - (7*theta)/2.) - 567*cos(3*phi - (5*theta)/2.) - 3432*cos(7*phi - (5*theta)/2.) - 700*cos(phi - (3*theta)/2.) - 1617*cos(5*phi - (3*theta)/2.) - 1134*cos(3*phi - theta/2.) - 875*cos(phi + theta/2.))/116688.;
        }
    } else if (ell == 9) {
        if (p == 4) {
            return (-715*cos(7*phi - (9*theta)/2.) - 1144*cos(5*phi - (7*theta)/2.) - 12155*cos(9*phi - (7*theta)/2.) - 1540*cos(3*phi - (5*theta)/2.) - 5720*cos(7*phi - (5*theta)/2.) - 1960*cos(phi - (3*theta)/2.) - 4004*cos(5*phi - (3*theta)/2.) - 3080*cos(3*phi - theta/2.) - 2450*cos(phi + theta/2.))/3.11168e6;
        }
    }

    return 0;
}

double H_dd_0(int p, int q, double theta, double phi) {

	switch(p) {
		case 0:
			switch(q) {
				case 0:
					return 1.0;
				case 1:
				    return 1.0/3.0;
				case 2:
				    return 1.0/5.0;
				case 3:
				    return 1.0/7.0;
				case 4:
				    return 1.0/9.0;
				default:
					return 0.0;
			}
		case 1:
			switch(q) {
				case 0:
					return 1.0/3.0;
				case 1:
        			return (2 + cos(2*theta))/15.;
				case 2:
        			return (3 + 2*cos(2*theta))/35.;
				case 3:
        			return (4 + 3*cos(2*theta))/63.;
				case 4:
        			return (5 + 4*cos(2*theta))/99.;
				default:
					return 0.0;
			}
		case 2:
			switch(q) {
				case 0:
					return 1.0/5.0;
				case 1:
        			return (3 + 2*cos(2*theta))/35.;
				case 2:
        			return (18 + 16*cos(2*theta) + cos(4*theta))/315.;
				case 3:
        			return (10 + 10*cos(2*theta) + cos(4*theta))/231.;
				case 4:
        			return (15 + 16*cos(2*theta) + 2*cos(4*theta))/429.;
				default:
					return 0.0;
			}
		case 3:
			switch(q) {
				case 0:
					return 1.0/7.0;
				case 1:
        			return (4 + 3*cos(2*theta))/63.;
				case 2:
        			return (10 + 10*cos(2*theta) + cos(4*theta))/231.;
				case 3:
        			return (200 + 225*cos(2*theta) + 36*cos(4*theta) + cos(6*theta))/6006.;
				case 4:
        			return (175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta))/6435.;
				default:
					return 0.0;
			}
		case 4:
			switch(q) {
				case 0:
        			return 1.0/9.0;
				case 1:
                	return (5 + 4*cos(2*theta))/99.;
				case 2:
                	return (15 + 16*cos(2*theta) + 2*cos(4*theta))/429.;
				case 3:
                	return (175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta))/6435.;
				case 4:
                	return (2450 + 3136*cos(2*theta) + 784*cos(4*theta) + 64*cos(6*theta) + cos(8*theta))/109395.;
				default:
					return 0.0;
			}
		default:
			return 0.0;
	}
	return 0.0;
}

double H_dd_2(int p, int q, double theta, double phi) {

	switch(p) {
		case 0:
			switch(q) {
				case 1:
				    return (1 + 3*cos(2*phi + theta))/6.;
				case 2:
				    return (1 + 3*cos(2*phi + theta))/7.;
				case 3:
				    return (5*(1 + 3*cos(2*phi + theta)))/42.;
				case 4:
				    return (10*(1 + 3*cos(2*phi + theta)))/99.;
				default:
					return 0.0;
			}
		case 1:
			switch(q) {
				case 0:
					return (1 + 3*cos(2*phi - theta))/6.;
				case 1:
        			return (2 + 9*cos(2*phi)*cos(theta) + cos(2*theta))/21.;
				case 2:
        			return (3 + 6*cos(2*phi - theta) + 2*cos(2*theta) + 8*cos(2*phi + theta) + cos(2*phi + 3*theta))/42.;
				case 3:
        			return (5*(16 + 30*cos(2*phi - theta) + 12*cos(2*theta) + 45*cos(2*phi + theta) + 9*cos(2*phi + 3*theta)))/1386.;
				case 4:
        			return (5*(25 + 45*cos(2*phi - theta) + 20*cos(2*theta) + 72*cos(2*phi + theta) + 18*cos(2*phi + 3*theta)))/2574.;
				default:
					return 0.0;
			}
		case 2:
			switch(q) {
				case 0:
					return (1 + 3*cos(2*phi - theta))/7.;
				case 1:
        			return (3 + cos(2*phi - 3*theta) + 14*cos(2*phi)*cos(theta) + 2*cos(2*theta) + 2*sin(2*phi)*sin(theta))/42.;
				case 2:
        			return (2*(18 + 16*cos(2*theta) + 15*cos(2*phi)*(6*cos(theta) + cos(3*theta)) + cos(4*theta)))/693.;
				case 3:
        			return (5*(100 + 45*cos(2*phi - 3*theta) + 240*cos(2*phi - theta) + 100*cos(2*theta) + 10*cos(4*theta) + 270*cos(2*phi + theta) + 72*cos(2*phi + 3*theta) + 3*cos(2*phi + 5*theta)))/12012.;
				case 4:
        			return (15 + 7*cos(2*phi - 3*theta) + 35*cos(2*phi - theta) + 16*cos(2*theta) + 2*cos(4*theta) + 42*cos(2*phi + theta) + 14*cos(2*phi + 3*theta) + cos(2*phi + 5*theta))/429.;
				default:
					return 0.0;
			}
		case 3:
			switch(q) {
				case 0:
					return (5*(1 + 3*cos(2*phi - theta)))/42.;
				case 1:
        			return (5*(16 + 9*cos(2*phi - 3*theta) + 75*cos(2*phi)*cos(theta) + 12*cos(2*theta) + 15*sin(2*phi)*sin(theta)))/1386.;
				case 2:
        			return (5*(3*cos(2*phi - 5*theta) + 3*cos(2*phi)*(170*cos(theta) + 39*cos(3*theta)) + 10*(10 + 10*cos(2*theta) + cos(4*theta)) + 3*sin(2*phi)*(10*sin(theta) + 9*sin(3*theta))))/12012.;
				case 3:
        			return (200 + 225*cos(2*theta) + 36*cos(4*theta) + 42*cos(2*phi)*cos(theta)*(18 + 14*cos(2*theta) + cos(4*theta)) + cos(6*theta))/6006.;
				case 4:
        			return (1225 + 84*cos(2*phi - 5*theta) + 1008*cos(2*phi - 3*theta) + 3150*cos(2*phi - theta) + 1470*cos(2*theta) + 294*cos(4*theta) + 14*cos(6*theta) + 3360*cos(2*phi + theta) + 1260*cos(2*phi + 3*theta) + 144*cos(2*phi + 5*theta) + 3*cos(2*phi + 7*theta))/43758.;
				default:
					return 0.0;
			}
		case 4:
			switch(q) {
				case 0:
        			return (10*(1 + 3*cos(2*phi - theta)))/99.;
				case 1:
                	return (5*(25 + 18*cos(2*phi - 3*theta) + 117*cos(2*phi)*cos(theta) + 20*cos(2*theta) + 27*sin(2*phi)*sin(theta)))/2574.;
				case 2:
                	return (15 + cos(2*phi - 5*theta) + 16*cos(2*theta) + 2*cos(4*theta) + 14*cos(theta)*(cos(2*phi)*(4 + 3*cos(2*theta)) + sin(2*phi)*sin(2*theta)))/429.;
				case 3:
                	return (3*cos(2*phi - 7*theta) + 6*cos(2*phi)*(1085*cos(theta) + 378*cos(3*theta) + 38*cos(5*theta)) + 7*(175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta)) + 6*sin(2*phi)*(35*sin(theta) + 42*sin(3*theta) + 10*sin(5*theta)))/43758.;
				case 4:
                	return (4*(2450 + 3136*cos(2*theta) + 784*cos(4*theta) + 64*cos(6*theta) + 27*cos(2*phi)*(490*cos(theta) + 28*(7*cos(3*theta) + cos(5*theta)) + cos(7*theta)) + cos(8*theta)))/415701.;
				default:
					return 0.0;
			}
		default:
			return 0.0;
	}
	return 0.0;
}
	
double H_dd_4(int p, int q, double theta, double phi) {

	switch(p) {
		case 0:
			switch(q) {
				case 2:
				    return (9 + 20*cos(2*phi + theta) + 35*cos(2*(2*phi + theta)))/280.;
				case 3:
				    return (3*(9 + 20*cos(2*phi + theta) + 35*cos(2*(2*phi + theta))))/616.;
				case 4:
				    return (3*(9 + 20*cos(2*phi + theta) + 35*cos(2*(2*phi + theta))))/572.;
				default:
					return 0.0;
			}
		case 1:
			switch(q) {
				case 1:
        			return (6 + 35*cos(4*phi) + 20*cos(2*phi)*cos(theta) + 3*cos(2*theta))/280.;
				case 2:
        			return (81 + 350*cos(4*phi) + 120*cos(2*phi - theta) + 54*cos(2*theta) + 160*cos(2*phi + theta) + 175*cos(2*(2*phi + theta)) + 20*cos(2*phi + 3*theta))/3080.;
				case 3:
        			return (3*(144 + 525*cos(4*phi) + 200*cos(2*phi - theta) + 108*cos(2*theta) + 35*cos(4*(phi + theta)) + 300*cos(2*phi + theta) + 420*cos(2*(2*phi + theta)) + 60*cos(2*phi + 3*theta)))/16016.;
				case 4:
        			return (15 + 49*cos(4*phi) + 20*cos(2*phi - theta) + 12*cos(2*theta) + 7*cos(4*(phi + theta)) + 32*cos(2*phi + theta) + 49*cos(2*(2*phi + theta)) + 8*cos(2*phi + 3*theta))/572.;
				default:
					return 0.0;
			}
		case 2:
			switch(q) {
				case 0:
					return (9 + 35*cos(4*phi - 2*theta) + 20*cos(2*phi - theta))/280.;
				case 1:
        			return (81 + 350*cos(4*phi) + 20*cos(2*phi - 3*theta) + 175*cos(4*phi - 2*theta) + 280*cos(2*phi)*cos(theta) + 54*cos(2*theta) + 40*sin(2*phi)*sin(theta))/3080.;
				case 2:
        			return (3*(175*cos(4*phi)*(4 + 3*cos(2*theta)) + 100*cos(2*phi)*(6*cos(theta) + cos(3*theta)) + 9*(18 + 16*cos(2*theta) + cos(4*theta))))/20020.;
				case 3:
        			return (180 + 735*cos(4*phi) + 60*cos(2*phi - 3*theta) + 245*cos(4*phi - 2*theta) + 320*cos(2*phi - theta) + 180*cos(2*theta) + 18*cos(4*theta) + 49*cos(4*(phi + theta)) + 360*cos(2*phi + theta) + 441*cos(2*(2*phi + theta)) + 96*cos(2*phi + 3*theta) + 4*cos(2*phi + 5*theta))/8008.;
				case 4:
        			return (405 + 1568*cos(4*phi) + 140*cos(2*phi - 3*theta) + 490*cos(4*phi - 2*theta) + 700*cos(2*phi - theta) + 432*cos(2*theta) + 54*cos(4*theta) + 224*cos(4*(phi + theta)) + 840*cos(2*phi + theta) + 1176*cos(2*(2*phi + theta)) + 280*cos(2*phi + 3*theta) + 20*cos(2*phi + 5*theta) + 7*cos(4*phi + 6*theta))/19448.;
				default:
					return 0.0;
			}
		case 3:
			switch(q) {
				case 0:
					return (3*(9 + 35*cos(4*phi - 2*theta) + 20*cos(2*phi - theta)))/616.;
				case 1:
        			return (3*(144 + 525*cos(4*phi) + 60*cos(2*phi - 3*theta) + 420*cos(4*phi - 2*theta) + 35*cos(4*(phi - theta)) + 500*cos(2*phi)*cos(theta) + 108*cos(2*theta) + 100*sin(2*phi)*sin(theta)))/16016.;
				case 2:
        			return (4*cos(2*phi - 5*theta) + 4*cos(2*phi)*(170*cos(theta) + 39*cos(3*theta)) + 18*(10 + 10*cos(2*theta) + cos(4*theta)) + 49*cos(4*phi)*(15 + 14*cos(2*theta) + cos(4*theta)) + 4*sin(2*phi)*(10*sin(theta) + 9*sin(3*theta)) + 49*sin(4*phi)*(4*sin(2*theta) + sin(4*theta)))/8008.;
				case 3:
        			return (3*(490*cos(4*phi)*(15 + 16*cos(2*theta) + 2*cos(4*theta)) + 140*cos(2*phi)*(50*cos(theta) + 15*cos(3*theta) + cos(5*theta)) + 9*(200 + 225*cos(2*theta) + 36*cos(4*theta) + cos(6*theta))))/272272.;
				case 4:
        			return (3*(44100*cos(4*phi) + 560*cos(2*phi - 5*theta) + 6720*cos(2*phi - 3*theta) + 22050*cos(4*phi - 2*theta) + 2940*cos(4*(phi - theta)) + 21000*cos(2*phi - theta) + 13230*cos(2*theta) + 2646*cos(4*theta) + 126*cos(6*theta) + 5*(1260*cos(4*(phi + theta)) + 4480*cos(2*phi + theta) + 5880*cos(2*(2*phi + theta)) + 1680*cos(2*phi + 3*theta) + 192*cos(2*phi + 5*theta) + 63*(35 + cos(4*phi + 6*theta)) + 4*cos(2*phi + 7*theta))))/1.84756e6;
				default:
					return 0.0;
			}
		case 4:
			switch(q) {
				case 0:
        			return (3*(9 + 35*cos(4*phi - 2*theta) + 20*cos(2*phi - theta)))/572.;
				case 1:
                	return (15 + 49*cos(4*phi) + 8*cos(2*phi - 3*theta) + 49*cos(4*phi - 2*theta) + 7*cos(4*(phi - theta)) + 52*cos(2*phi)*cos(theta) + 12*cos(2*theta) + 12*sin(2*phi)*sin(theta))/572.;
				case 2:
                	return (405 + 7*cos(4*phi - 6*theta) + 20*cos(2*phi - 5*theta) + 224*cos(4*(phi - theta)) + 432*cos(2*theta) + 98*cos(4*phi)*(16 + 17*cos(2*theta)) + 54*cos(4*theta) + 280*cos(theta)*(4*cos(2*phi) + 2*cos(2*(phi - theta)) + cos(2*(phi + theta))) + 686*sin(4*phi)*sin(2*theta))/19448.;
				case 3:
                	return (3*(20*cos(2*phi - 7*theta) + 40*cos(2*phi)*(1085*cos(theta) + 378*cos(3*theta) + 38*cos(5*theta)) + 63*(175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta)) + 105*cos(4*phi)*(420 + 490*cos(2*theta) + 88*cos(4*theta) + 3*cos(6*theta)) + 40*sin(2*phi)*(35*sin(theta) + 42*sin(3*theta) + 10*sin(5*theta)) + 105*sin(4*phi)*(70*sin(2*theta) + 32*sin(4*theta) + 3*sin(6*theta))))/1.84756e6;
				case 4:
                	return (3*(2450 + 3136*cos(2*theta) + 784*cos(4*theta) + 64*cos(6*theta) + 175*cos(4*phi)*(56 + 70*cos(2*theta) + 16*cos(4*theta) + cos(6*theta)) + 20*cos(2*phi)*(490*cos(theta) + 28*(7*cos(3*theta) + cos(5*theta)) + cos(7*theta)) + cos(8*theta)))/461890.;
				default:
					return 0.0;
			}
		default:
			return 0.0;
	}
	return 0.0;
}

double H_dd_6(int p, int q, double theta, double phi) {

	switch(p) {
		case 0:
			switch(q) {
				case 3:
				    return (50 + 105*cos(2*phi + theta) + 126*cos(2*(2*phi + theta)) + 231*cos(3*(2*phi + theta)))/7392.;
				case 4:
				    return (50 + 105*cos(2*phi + theta) + 126*cos(2*(2*phi + theta)) + 231*cos(3*(2*phi + theta)))/3960.;
				default:
					return 0.0;
			}
		case 1:
			switch(q) {
				case 2:
        			return (30 + 84*cos(4*phi) + 42*cos(2*phi - theta) + 20*cos(2*theta) + 56*cos(2*phi + theta) + 42*cos(2*(2*phi + theta)) + 231*cos(6*phi + theta) + 7*cos(2*phi + 3*theta))/7392.;
				case 3:
        			return (800 + 1890*cos(4*phi) + 1050*cos(2*phi - theta) + 600*cos(2*theta) + 126*cos(4*(phi + theta)) + 1575*cos(2*phi + theta) + 1512*cos(2*(2*phi + theta)) + 1617*cos(3*(2*phi + theta)) + 4851*cos(6*phi + theta) + 315*cos(2*phi + 3*theta))/110880.;
				case 4:
        			return (1250 + 2646*cos(4*phi) + 1575*cos(2*phi - theta) + 1000*cos(2*theta) + 378*cos(4*(phi + theta)) + 2520*cos(2*phi + theta) + 2646*cos(2*(2*phi + theta)) + 3696*cos(3*(2*phi + theta)) + 6468*cos(6*phi + theta) + 630*cos(2*phi + 3*theta) + 231*cos(6*phi + 5*theta))/134640.;
				default:
					return 0.0;
			}
		case 2:
			switch(q) {
				case 1:
        			return (30 + 84*cos(4*phi) + 7*cos(2*phi - 3*theta) + 42*cos(4*phi - 2*theta) + 56*cos(2*phi - theta) + 231*cos(6*phi - theta) + 20*cos(2*theta) + 42*cos(2*phi + theta))/7392.;
				case 2:
        			return (1617*cos(6*phi)*cos(theta) + 126*cos(4*phi)*(4 + 3*cos(2*theta)) + 105*cos(2*phi)*(6*cos(theta) + cos(3*theta)) + 10*(18 + 16*cos(2*theta) + cos(4*theta)))/27720.;
				case 3:
        			return (5000 + 13230*cos(4*phi) + 1575*cos(2*phi - 3*theta) + 4410*cos(4*phi - 2*theta) + 8400*cos(2*phi - theta) + 16170*cos(6*phi - theta) + 5000*cos(2*theta) + 500*cos(4*theta) + 882*cos(4*(phi + theta)) + 9450*cos(2*phi + theta) + 7938*cos(2*(2*phi + theta)) + 6468*cos(3*(2*phi + theta)) + 25872*cos(6*phi + theta) + 2520*cos(2*phi + 3*theta) + 105*cos(2*phi + 5*theta))/628320.;
				case 4:
        			return (3750 + 9408*cos(4*phi) + 1225*cos(2*phi - 3*theta) + 2940*cos(4*phi - 2*theta) + 6125*cos(2*phi - theta) + 9702*cos(6*phi - theta) + 4000*cos(2*theta) + 500*cos(4*theta) + 1344*cos(4*(phi + theta)) + 7350*cos(2*phi + theta) + 7056*cos(2*(2*phi + theta)) + 8316*cos(3*(2*phi + theta)) + 19404*cos(6*phi + theta) + 2450*cos(2*phi + 3*theta) + 175*cos(2*phi + 5*theta) + 693*cos(6*phi + 5*theta) + 42*cos(4*phi + 6*theta))/426360.;
				default:
					return 0.0;
			}
		case 3:
			switch(q) {
				case 0:
					return (50 + 231*cos(6*phi - 3*theta) + 126*cos(4*phi - 2*theta) + 105*cos(2*phi - theta))/7392.;
				case 1:
        			return (800 + 1890*cos(4*phi) + 315*cos(2*phi - 3*theta) + 1617*cos(6*phi - 3*theta) + 1512*cos(4*phi - 2*theta) + 126*cos(4*(phi - theta)) + 1575*cos(2*phi - theta) + 4851*cos(6*phi - theta) + 600*cos(2*theta) + 1050*cos(2*phi + theta))/110880.;
				case 2:
        			return (105*cos(2*phi - 5*theta) + 882*cos(4*(phi - theta)) + 882*cos(4*phi)*(15 + 14*cos(2*theta)) + 3234*cos(6*phi)*(13*cos(theta) + 2*cos(3*theta)) + 105*cos(2*phi)*(170*cos(theta) + 39*cos(3*theta)) + 500*(10 + 10*cos(2*theta) + cos(4*theta)) + 3528*sin(4*phi)*sin(2*theta) + 3234*sin(6*phi)*(3*sin(theta) + 2*sin(3*theta)) + 105*sin(2*phi)*(10*sin(theta) + 9*sin(3*theta)))/628320.;
				case 3:
        			return (1764*cos(4*phi)*(15 + 16*cos(2*theta) + 11*cos(2*phi)*cos(theta)*(7 + 4*cos(2*theta)) + 2*cos(4*theta)) + 147*cos(2*phi)*(-344*cos(theta) - 57*cos(3*theta) + 5*cos(5*theta)) + 50*(200 + 225*cos(2*theta) + 36*cos(4*theta) + cos(6*theta)))/1.193808e6;
				case 4:
        			return (8750 + 22680*cos(4*phi) + 420*cos(2*phi - 5*theta) + 5040*cos(2*phi - 3*theta) + 6930*cos(6*phi - 3*theta) + 11340*cos(4*phi - 2*theta) + 1512*cos(4*(phi - theta)) + 15750*cos(2*phi - theta) + 33264*cos(6*phi - theta) + 10500*cos(2*theta) + 2100*cos(4*theta) + 100*cos(6*theta) + 3240*cos(4*(phi + theta)) + 16800*cos(2*phi + theta) + 15120*cos(2*(2*phi + theta)) + 15840*cos(3*(2*phi + theta)) + 41580*cos(6*phi + theta) + 6300*cos(2*phi + 3*theta) + 720*cos(2*phi + 5*theta) + 1485*cos(6*phi + 5*theta) + 162*cos(4*phi + 6*theta) + 15*cos(2*phi + 7*theta))/1.023264e6;
				default:
					return 0.0;
			}
		case 4:
			switch(q) {
				case 0:
        			return (50 + 231*cos(6*phi - 3*theta) + 126*cos(4*phi - 2*theta) + 105*cos(2*phi - theta))/3960.;
				case 1:
                	return (2646*cos(4*phi) + 231*cos(6*phi - 5*theta) + 630*cos(2*phi - 3*theta) + 3696*cos(6*phi - 3*theta) + 2646*cos(4*phi - 2*theta) + 378*cos(4*(phi - theta)) + 2520*cos(2*phi - theta) + 6468*cos(6*phi - theta) + 25*(50 + 40*cos(2*theta) + 63*cos(2*phi + theta)))/134640.;
				case 2:
                	return (42*cos(4*phi - 6*theta) + 175*cos(2*phi - 5*theta) + 693*cos(6*phi - 5*theta) + 1344*cos(4*(phi - theta)) + 588*cos(4*phi)*(16 + 17*cos(2*theta)) + 250*(15 + 16*cos(2*theta) + 2*cos(4*theta)) + 4116*sin(4*phi)*sin(2*theta) + 7*(594*cos(6*phi)*(7*cos(theta) + 2*cos(3*theta)) + 175*cos(2*phi)*(11*cos(theta) + 3*cos(3*theta)) + 700*pow(cos(theta),2)*sin(2*phi)*sin(theta) + 198*sin(6*phi)*(7*sin(theta) + 6*sin(3*theta))))/426360.;
				case 3:
                	return (99*cos(6*phi)*(756*cos(theta) + 230*cos(3*theta) + 15*cos(5*theta)) + 50*(175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta)) + 54*cos(4*phi)*(420 + 490*cos(2*theta) + 88*cos(4*theta) + 3*cos(6*theta)) + 15*cos(2*phi)*(2170*cos(theta) + 756*cos(3*theta) + 76*cos(5*theta) + cos(7*theta)) + 297*sin(6*phi)*(28*sin(theta) + 30*sin(3*theta) + 5*sin(5*theta)) + 54*sin(4*phi)*(70*sin(2*theta) + 32*sin(4*theta) + 3*sin(6*theta)) + 15*sin(2*phi)*(70*sin(theta) + 84*sin(3*theta) + 20*sin(5*theta) + sin(7*theta)))/1.023264e6;
				case 4:
                	return (7623*cos(6*phi)*(28*cos(theta) + 10*cos(3*theta) + cos(5*theta)) + 1134*cos(4*phi)*(56 + 70*cos(2*theta) + 16*cos(4*theta) + cos(6*theta)) + 189*cos(2*phi)*(490*cos(theta) + 28*(7*cos(3*theta) + cos(5*theta)) + cos(7*theta)) + 10*(2450 + 3136*cos(2*theta) + 784*cos(4*theta) + 64*cos(6*theta) + cos(8*theta)))/2.941884e6;
				default:
					return 0.0;
			}
		default:
			return 0.0;
	}
	return 0.0;
}

double H_dd_8(int p, int q, double theta, double phi) {

	switch(p) {
		case 0:
			switch(q) {
				case 4:
				    return (1225 + 2520*cos(2*phi + theta) + 2772*cos(2*(2*phi + theta)) + 3432*cos(3*(2*phi + theta)) + 6435*cos(4*(2*phi + theta)))/823680.;
				default:
					return 0.0;
			}
		case 1:
			switch(q) {
				case 3:
        			return (700 + 1485*cos(4*phi) + 900*cos(2*phi - theta) + 525*cos(2*theta) + 99*cos(4*(phi + theta)) + 1350*cos(2*phi + theta) + 1188*cos(2*(2*phi + theta)) + 858*cos(3*(2*phi + theta)) + 6435*cos(2*(4*phi + theta)) + 2574*cos(6*phi + theta) + 270*cos(2*phi + 3*theta))/823680.;
				case 4:
        			return (30625 + 58212*cos(4*phi) + 37800*cos(2*phi - theta) + 24500*cos(2*theta) + 8316*cos(4*(phi + theta)) + 60480*cos(2*phi + theta) + 58212*cos(2*(2*phi + theta)) + 54912*cos(3*(2*phi + theta)) + 57915*cos(4*(2*phi + theta)) + 231660*cos(2*(4*phi + theta)) + 96096*cos(6*phi + theta) + 15120*cos(2*phi + 3*theta) + 3432*cos(6*phi + 5*theta))/1.564992e7;
				default:
					return 0.0;
			}
		case 2:
			switch(q) {
				case 2:
        			return (6435*cos(8*phi) + 3432*cos(6*phi)*cos(theta) + 396*cos(4*phi)*(4 + 3*cos(2*theta)) + 360*cos(2*phi)*(6*cos(theta) + cos(3*theta)) + 35*(18 + 16*cos(2*theta) + cos(4*theta)))/823680.;
				case 3:
        			return (20790*cos(4*phi) + 57915*cos(8*phi) + 2700*cos(2*phi - 3*theta) + 6930*cos(4*phi - 2*theta) + 14400*cos(2*phi - theta) + 17160*cos(6*phi - theta) + 8750*cos(2*theta) + 875*cos(4*theta) + 2*(4375 + 693*cos(4*(phi + theta)) + 8100*cos(2*phi + theta) + 6237*cos(2*(2*phi + theta)) + 3432*cos(3*(2*phi + theta)) + 19305*cos(2*(4*phi + theta)) + 13728*cos(6*phi + theta) + 2160*cos(2*phi + 3*theta) + 90*cos(2*phi + 5*theta)))/5.21664e6;
				case 4:
        			return (206976*cos(4*phi) + 450450*cos(8*phi) + 29400*cos(2*phi - 3*theta) + 64680*cos(4*phi - 2*theta) + 147000*cos(2*phi - theta) + 144144*cos(6*phi - theta) + 98000*cos(2*theta) + 12250*cos(4*theta) + 3*(30625 + 9856*cos(4*(phi + theta)) + 58800*cos(2*phi + theta) + 51744*cos(2*(2*phi + theta)) + 41184*cos(3*(2*phi + theta)) + 32175*cos(4*(2*phi + theta)) + 171600*cos(2*(4*phi + theta)) + 96096*cos(6*phi + theta) + 19600*cos(2*phi + 3*theta) + 1400*cos(2*phi + 5*theta) + 3432*cos(6*phi + 5*theta) + 308*cos(4*phi + 6*theta)))/3.651648e7;
				default:
					return 0.0;
			}
		case 3:
			switch(q) {
				case 1:
        			return (700 + 1485*cos(4*phi) + 270*cos(2*phi - 3*theta) + 858*cos(6*phi - 3*theta) + 1188*cos(4*phi - 2*theta) + 6435*cos(8*phi - 2*theta) + 99*cos(4*(phi - theta)) + 1350*cos(2*phi - theta) + 2574*cos(6*phi - theta) + 525*cos(2*theta) + 900*cos(2*phi + theta))/823680.;
				case 2:
        			return (19305*cos(8*phi)*(3 + 2*cos(2*theta)) + 875*(10 + 10*cos(2*theta) + cos(4*theta)) + 66*cos(4*phi)*(104*cos(2*phi)*(13*cos(theta) + 2*cos(3*theta)) + 21*(15 + 14*cos(2*theta) + cos(4*theta))) + 12*cos(2*phi)*(-1168*cos(theta) + 13*cos(3*theta) + 15*cos(5*theta)) + 38610*sin(8*phi)*sin(2*theta) + 3432*sin(6*phi)*(3*sin(theta) + 2*sin(3*theta)) + 1386*sin(4*phi)*(4*sin(2*theta) + sin(4*theta)) + 180*sin(2*phi)*(10*sin(theta) + 9*sin(3*theta) + sin(5*theta)))/5.21664e6;
				case 3:
        			return (38610*cos(8*phi)*(6 + 5*cos(2*theta)) + 5544*cos(4*phi)*(15 + 16*cos(2*theta) + 2*cos(4*theta)) + 72*(286*cos(6*phi)*(9*cos(theta) + 2*cos(3*theta)) + 35*cos(2*phi)*(50*cos(theta) + 15*cos(3*theta) + cos(5*theta))) + 175*(200 + 225*cos(2*theta) + 36*cos(4*theta) + cos(6*theta)))/1.4606592e7;
				case 4:
        			return (7783776*cos(6*phi)*cos(theta) + 212355*cos(8*phi)*(42 + 44*cos(2*theta) + 5*cos(4*theta)) + 8575*(175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta)) + 360*cos(2*phi)*(15190*cos(theta) - 1286*cos(3*theta) + 103*cos(5*theta) + 7*cos(7*theta)) + 396*cos(4*phi)*(8820 + 5980*cos(2*phi - 3*theta) + 10290*cos(2*theta) + 1848*cos(4*theta) + 63*cos(6*theta) + 5980*cos(2*phi + 3*theta) + 780*cos(2*phi + 5*theta)) - 61776*sin(6*phi)*(14*sin(theta) + 15*sin(3*theta)) - 212355*sin(8*phi)*(16*sin(2*theta) + 5*sin(4*theta)) - 8316*sin(4*phi)*(70*sin(2*theta) + 32*sin(4*theta) + 3*sin(6*theta)) - 360*sin(2*phi)*(490*sin(theta) + 588*sin(3*theta) + 569*sin(5*theta) + 7*sin(7*theta)))/5.03927424e8;
				default:
					return 0.0;
			}
		case 4:
			switch(q) {
				case 0:
        			return (6435*cos(8*phi - 4*theta) + 3432*cos(6*phi - 3*theta) + 7*(175 + 396*cos(4*phi - 2*theta) + 360*cos(2*phi - theta)))/823680.;
				case 1:
                	return (58212*cos(4*phi) + 3432*cos(6*phi - 5*theta) + 57915*cos(8*phi - 4*theta) + 15120*cos(2*phi - 3*theta) + 54912*cos(6*phi - 3*theta) + 58212*cos(4*phi - 2*theta) + 231660*cos(8*phi - 2*theta) + 7*(1188*cos(4*(phi - theta)) + 8640*cos(2*phi - theta) + 13728*cos(6*phi - theta) + 875*(5 + 4*cos(2*theta)) + 5400*cos(2*phi + theta)))/1.564992e7;
				case 2:
                	return (924*cos(4*phi - 6*theta) + 4200*cos(2*phi - 5*theta) + 10296*cos(6*phi - 5*theta) + 96525*cos(8*phi - 4*theta) + 29568*cos(4*(phi - theta)) + 64350*cos(8*phi)*(7 + 8*cos(2*theta)) + 264*cos(4*phi)*(784 + 833*cos(2*theta) + 468*cos(2*phi)*(7*cos(theta) + 2*cos(3*theta))) + 6125*(15 + 16*cos(2*theta) + 2*cos(4*theta)) + 264*(343*sin(4*phi) + 1950*sin(8*phi))*sin(2*theta) - 24*(cos(2*phi)*(4543*cos(theta) + 1473*cos(3*theta)) - 4900*pow(cos(theta),2)*sin(2*phi)*sin(theta) - 858*sin(6*phi)*(7*sin(theta) + 6*sin(3*theta))))/3.651648e7;
				case 3:
                	return (7783776*cos(6*phi)*cos(theta) + 212355*cos(8*phi)*(42 + 44*cos(2*theta) + 5*cos(4*theta)) + 8575*(175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta)) + 360*cos(2*phi)*(15190*cos(theta) - 1286*cos(3*theta) + 103*cos(5*theta) + 7*cos(7*theta)) + 61776*sin(6*phi)*(14*sin(theta) + 15*sin(3*theta)) + 212355*sin(8*phi)*(16*sin(2*theta) + 5*sin(4*theta)) + 396*cos(4*phi)*(260*cos(2*phi)*(46*cos(3*theta) + 3*cos(5*theta)) + 21*(420 + 490*cos(2*theta) + 88*cos(4*theta) + 3*cos(6*theta)) + 780*sin(2*phi)*sin(5*theta)) + 8316*sin(4*phi)*(70*sin(2*theta) + 32*sin(4*theta) + 3*sin(6*theta)) + 360*sin(2*phi)*(490*sin(theta) + 588*sin(3*theta) + 569*sin(5*theta) + 7*sin(7*theta)))/5.03927424e8;
				case 4:
                	return (127413*cos(8*phi)*(28 + 32*cos(2*theta) + 5*cos(4*theta)) + 396*cos(4*phi)*(572*cos(2*phi)*(28*cos(theta) + cos(5*theta)) + 63*(56 + 70*cos(2*theta) + 16*cos(4*theta) + cos(6*theta))) + 72*(15730*cos(6*phi)*cos(3*theta) + cos(2*phi)*(-13174*cos(theta) + 12348*cos(3*theta) + 191*cos(5*theta) + 63*cos(7*theta))) + 245*(2450 + 3136*cos(2*theta) + 784*cos(4*theta) + 64*cos(6*theta) + cos(8*theta)))/1.7997408e8;
				default:
					return 0.0;
			}
		default:
			return 0.0;
	}
	return 0.0;
}

double H_dd_10(int p, int q, double theta, double phi) {

	switch(p) {
		case 1:
			switch(q) {
				case 4:
        			return (4410 + 8008*cos(4*phi) + 5390*cos(2*phi - theta) + 3528*cos(2*theta) + 1144*cos(4*(phi + theta)) + 8624*cos(2*phi + theta) + 8008*cos(2*(2*phi + theta)) + 6864*cos(3*(2*phi + theta)) + 4862*cos(4*(2*phi + theta)) + 19448*cos(2*(4*phi + theta)) + 12012*cos(6*phi + theta) + 2156*cos(2*phi + 3*theta) + 46189*cos(10*phi + 3*theta) + 429*cos(6*phi + 5*theta))/2.3648768e7;
				default:
					return 0.0;
			}
		case 2:
			switch(q) {
				case 3:
        			return (3780 + 8580*cos(4*phi) + 14586*cos(8*phi) + 1155*cos(2*phi - 3*theta) + 2860*cos(4*phi - 2*theta) + 6160*cos(2*phi - theta) + 6435*cos(6*phi - theta) + 3780*cos(2*theta) + 378*cos(4*theta) + 572*cos(4*(phi + theta)) + 6930*cos(2*phi + theta) + 5148*cos(2*(2*phi + theta)) + 2574*cos(3*(2*phi + theta)) + 9724*cos(2*(4*phi + theta)) + 10296*cos(6*phi + theta) + 46189*cos(10*phi + theta) + 1848*cos(2*phi + 3*theta) + 77*cos(2*phi + 5*theta))/2.3648768e7;
				case 4:
        			return (119070 + 256256*cos(4*phi) + 340340*cos(8*phi) + 37730*cos(2*phi - 3*theta) + 80080*cos(4*phi - 2*theta) + 188650*cos(2*phi - theta) + 162162*cos(6*phi - theta) + 127008*cos(2*theta) + 15876*cos(4*theta) + 36608*cos(4*(phi + theta)) + 226380*cos(2*phi + theta) + 192192*cos(2*(2*phi + theta)) + 138996*cos(3*(2*phi + theta)) + 72930*cos(4*(2*phi + theta)) + 388960*cos(2*(4*phi + theta)) + 324324*cos(6*phi + theta) + 1016158*cos(10*phi + theta) + 75460*cos(2*phi + 3*theta) + 508079*cos(10*phi + 3*theta) + 5390*cos(2*phi + 5*theta) + 11583*cos(6*phi + 5*theta) + 1144*cos(4*phi + 6*theta))/2.71960832e8;
				default:
					return 0.0;
			}
		case 3:
			switch(q) {
				case 2:
        			return (3780 + 8580*cos(4*phi) + 14586*cos(8*phi) + 77*cos(2*phi - 5*theta) + 572*cos(4*(phi - theta)) + 22*cos(2*phi)*(1934 - 2678*cos(4*phi) + 4199*cos(8*phi))*cos(theta) + 4*(945 + 2002*cos(4*phi) + 2431*cos(8*phi))*cos(2*theta) + 429*(7*cos(2*phi) + 6*cos(6*phi))*cos(3*theta) + 378*cos(4*theta) + 11*(70*sin(2*phi) + 351*sin(6*phi) + 4199*sin(10*phi))*sin(theta) + 572*(4*sin(4*phi) + 17*sin(8*phi))*sin(2*theta) + 99*(7*sin(2*phi) + 26*sin(6*phi))*sin(3*theta))/2.3648768e7;
				case 3:
        			return (3*(4862*cos(8*phi)*(30 + 209*cos(2*phi)*cos(theta) + 25*cos(2*theta)) - 572*cos(4*phi)*(cos(2*phi)*(1169*cos(theta) - 135*cos(3*theta)) - 10*(15 + 16*cos(2*theta) + 2*cos(4*theta))) + 11*cos(2*phi)*(42644*cos(theta) + 165*cos(3*theta) + 245*cos(5*theta)) + 189*(200 + 225*cos(2*theta) + 36*cos(4*theta) + cos(6*theta))))/2.71960832e8;
				case 4:
        			return (3*(267410*cos(8*phi)*(42 + 44*cos(2*theta) + 57*cos(2*phi)*cos(3*theta) + 5*cos(4*theta)) + 18522*(175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta)) + 4290*cos(4*phi)*(-1483*cos(2*phi)*cos(3*theta) + 4*(420 + 490*cos(2*theta) + 88*cos(4*theta) + 3*cos(6*theta))) + 11*(3510364*cos(10*phi)*cos(theta) + 5265*cos(6*phi)*(252*cos(theta) + 5*cos(5*theta)) + 5*cos(2*phi)*(212660*cos(theta) + 131925*cos(3*theta) + 98*(76*cos(5*theta) + cos(7*theta)))) - 2540395*sin(10*phi)*(4*sin(theta) + 3*sin(3*theta)) - 267410*sin(8*phi)*(16*sin(2*theta) + 5*sin(4*theta)) - 57915*sin(6*phi)*(28*sin(theta) + 5*(6*sin(3*theta) + sin(5*theta))) - 17160*sin(4*phi)*(70*sin(2*theta) + 32*sin(4*theta) + 3*sin(6*theta)) - 5390*sin(2*phi)*(70*sin(theta) + 84*sin(3*theta) + 20*sin(5*theta) + sin(7*theta))))/1.35980416e10;
				default:
					return 0.0;
			}
		case 4:
			switch(q) {
				case 1:
                	return (8008*cos(4*phi) + 429*cos(6*phi - 5*theta) + 4862*cos(8*phi - 4*theta) + 2156*cos(2*phi - 3*theta) + 6864*cos(6*phi - 3*theta) + 46189*cos(10*phi - 3*theta) + 2*(4004*cos(4*phi - 2*theta) + 9724*cos(8*phi - 2*theta) + 572*cos(4*(phi - theta)) + 7*(315 + 616*cos(2*phi - theta) + 858*cos(6*phi - theta) + 252*cos(2*theta) + 385*cos(2*phi + theta))))/2.3648768e7;
				case 2:
                	return (256256*cos(4*phi) + 340340*cos(8*phi) + 1144*cos(4*phi - 6*theta) + 5390*cos(2*phi - 5*theta) + 11583*cos(6*phi - 5*theta) + 72930*cos(8*phi - 4*theta) + 75460*cos(2*phi - 3*theta) + 138996*cos(6*phi - 3*theta) + 508079*cos(10*phi - 3*theta) + 2*(96096*cos(4*phi - 2*theta) + 194480*cos(8*phi - 2*theta) + 18304*cos(4*(phi - theta)) + 113190*cos(2*phi - theta) + 162162*cos(6*phi - theta) + 508079*cos(10*phi - theta) + 7*(8505 + 9072*cos(2*theta) + 1134*cos(4*theta) + 13475*cos(2*phi + theta) + 5720*cos(2*(2*phi + theta)) + 11583*cos(6*phi + theta) + 2695*cos(2*phi + 3*theta))))/2.71960832e8;
				case 3:
                	return (3*(267410*cos(8*phi)*(42 + 44*cos(2*theta) + 57*cos(2*phi)*cos(3*theta) + 5*cos(4*theta)) + 18522*(175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta)) + 4290*cos(4*phi)*(-1483*cos(2*phi)*cos(3*theta) + 4*(420 + 490*cos(2*theta) + 88*cos(4*theta) + 3*cos(6*theta))) + 11*(3510364*cos(10*phi)*cos(theta) + 5265*cos(6*phi)*(252*cos(theta) + 5*cos(5*theta)) + 5*cos(2*phi)*(212660*cos(theta) + 131925*cos(3*theta) + 98*(76*cos(5*theta) + cos(7*theta)))) + 2540395*sin(10*phi)*(4*sin(theta) + 3*sin(3*theta)) + 267410*sin(8*phi)*(16*sin(2*theta) + 5*sin(4*theta)) + 57915*sin(6*phi)*(28*sin(theta) + 5*(6*sin(3*theta) + sin(5*theta))) + 17160*sin(4*phi)*(70*sin(2*theta) + 32*sin(4*theta) + 3*sin(6*theta)) + 5390*sin(2*phi)*(70*sin(theta) + 84*sin(3*theta) + 20*sin(5*theta) + sin(7*theta))))/1.35980416e10;
				case 4:
                	return (7*(53482*cos(8*phi)*(247*cos(2*phi)*(4*cos(theta) + cos(3*theta)) + 5*(28 + 32*cos(2*theta) + 5*cos(4*theta))) - 286*cos(4*phi)*(11*cos(2*phi)*(10496*cos(theta) + 1949*cos(3*theta) - 225*cos(5*theta)) - 300*(56 + 70*cos(2*theta) + 16*cos(4*theta) + cos(6*theta))) + 11*cos(2*phi)*(2221228*cos(theta) + 566827*cos(3*theta) + 8985*cos(5*theta) + 1470*cos(7*theta)) + 882*(2450 + 3136*cos(2*theta) + 784*cos(4*theta) + 64*cos(6*theta) + cos(8*theta))))/1.52977968e10;
				default:
					return 0.0;
			}
		default:
			return 0.0;
	}
	return 0.0;
}

double H_dd_12(int p, int q, double theta, double phi) {

	switch(p) {
		case 2:
			switch(q) {
				case 4:
        			return (48510 + 101920*cos(4*phi) + 117572*cos(8*phi) + 15288*cos(2*phi - 3*theta) + 31850*cos(4*phi - 2*theta) + 76440*cos(2*phi - theta) + 61880*cos(6*phi - theta) + 51744*cos(2*theta) + 6468*cos(4*theta) + 14560*cos(4*(phi + theta)) + 91728*cos(2*phi + theta) + 76440*cos(2*(2*phi + theta)) + 53040*cos(3*(2*phi + theta)) + 25194*cos(4*(2*phi + theta)) + 134368*cos(2*(4*phi + theta)) + 123760*cos(6*phi + theta) + 676039*cos(2*(6*phi + theta)) + 235144*cos(10*phi + theta) + 30576*cos(2*phi + 3*theta) + 117572*cos(10*phi + 3*theta) + 2184*cos(2*phi + 5*theta) + 4420*cos(6*phi + 5*theta) + 455*cos(4*phi + 6*theta))/1.384527872e9;
				default:
					return 0.0;
			}
		case 3:
			switch(q) {
				case 3:
					return (676039*cos(12*phi) + 25194*cos(8*phi)*(6 + 28*cos(2*phi)*cos(theta) + 5*cos(2*theta)) + 13*cos(4*phi)*(272*cos(2*phi)*(-87*cos(theta) + 25*cos(3*theta)) + 525*(15 + 16*cos(2*theta) + 2*cos(4*theta))) + 52*cos(2*phi)*(6108*cos(theta) + 95*cos(3*theta) + 63*cos(5*theta)) + 231*(200 + 225*cos(2*theta) + 36*cos(4*theta) + cos(6*theta)))/1.384527872e9;
				case 4:
        			return (8599500*cos(4*phi) + 11639628*cos(8*phi) + 35154028*cos(12*phi) + 183456*cos(2*phi - 5*theta) + 2201472*cos(2*phi - 3*theta) + 1547000*cos(6*phi - 3*theta) + 4299750*cos(4*phi - 2*theta) + 3879876*cos(8*phi - 2*theta) + 573300*cos(4*(phi - theta)) + 6879600*cos(2*phi - theta) + 7425600*cos(6*phi - theta) + 9876048*cos(10*phi - theta) + 4753980*cos(2*theta) + 950796*cos(4*theta) + 45276*cos(6*theta) + 1228500*cos(4*(phi + theta)) + 7338240*cos(2*phi + theta) + 5733000*cos(2*(2*phi + theta)) + 3536000*cos(3*(2*phi + theta)) + 3*(461890*cos(4*(2*phi + theta)) + 2771340*cos(2*(4*phi + theta)) + 3094000*cos(6*phi + theta) + 8788507*cos(2*(6*phi + theta)) + 5643456*cos(10*phi + theta) + 917280*cos(2*phi + 3*theta) + 1763580*cos(10*phi + 3*theta) + 104832*cos(2*phi + 5*theta) + 25*(52822 + 4420*cos(6*phi + 5*theta) + 819*cos(4*phi + 6*theta)) + 2184*cos(2*phi + 7*theta)))/3.7382252544e10;
				default:
					return 0.0;
			}
		case 4:
			switch(q) {
				case 2:
                	return (101920*cos(4*phi) + 117572*cos(8*phi) + 455*cos(4*phi - 6*theta) + 2184*cos(2*phi - 5*theta) + 4420*cos(6*phi - 5*theta) + 25194*cos(8*phi - 4*theta) + 30576*cos(2*phi - 3*theta) + 53040*cos(6*phi - 3*theta) + 117572*cos(10*phi - 3*theta) + 76440*cos(4*phi - 2*theta) + 134368*cos(8*phi - 2*theta) + 676039*cos(12*phi - 2*theta) + 14*(3465 + 1040*cos(4*(phi - theta)) + 6552*cos(2*phi - theta) + 8840*cos(6*phi - theta) + 16796*cos(10*phi - theta) + 3696*cos(2*theta) + 462*cos(4*theta) + 5460*cos(2*phi + theta) + 2275*cos(2*(2*phi + theta)) + 4420*cos(6*phi + theta) + 1092*cos(2*phi + 3*theta)))/1.384527872e9;
				case 3:
                	return (8788507*cos(12*phi)*(4 + 3*cos(2*theta)) + 25194*cos(8*phi)*(462 + 484*cos(2*theta) + 420*cos(2*phi)*cos(3*theta) + 55*cos(4*theta)) + 65*cos(4*phi)*(136*cos(2*phi)*(-47*cos(3*theta) + 75*cos(5*theta)) + 315*(420 + 490*cos(2*theta) + 88*cos(4*theta) + 3*cos(6*theta))) + 52*(1428*(225*cos(6*phi) + 361*cos(10*phi))*cos(theta) + cos(2*phi)*(273420*cos(theta) + 99251*cos(3*theta) + 3201*cos(5*theta) + 126*cos(7*theta))) + 3*(7546*(175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta)) + 8788507*sin(12*phi)*sin(2*theta) + 587860*sin(10*phi)*(4*sin(theta) + 3*sin(3*theta)) + 92378*sin(8*phi)*(16*sin(2*theta) + 5*sin(4*theta)) + 22100*sin(6*phi)*(28*sin(theta) + 5*(6*sin(3*theta) + sin(5*theta))) + 6825*sin(4*phi)*(70*sin(2*theta) + 32*sin(4*theta) + 3*sin(6*theta)) + 2184*sin(2*phi)*(70*sin(theta) + 84*sin(3*theta) + 20*sin(5*theta) + sin(7*theta))))/3.7382252544e10;
				case 4:
                	return (8788507*cos(12*phi)*(8 + 7*cos(2*theta)) + 75582*cos(8*phi)*(4*(77 + 88*cos(2*theta) + 91*cos(2*phi)*(4*cos(theta) + cos(3*theta))) + 55*cos(4*theta)) + 104*cos(2*phi)*cos(theta)*(356750 + 245808*cos(2*theta) + 7243*cos(4*theta) + 1134*cos(6*theta)) + 13*cos(4*phi)*(136*cos(2*phi)*(-23744*cos(theta) - 1811*cos(3*theta) + 1375*cos(5*theta)) + 23625*(56 + 70*cos(2*theta) + 16*cos(4*theta) + cos(6*theta))) + 3234*(2450 + 3136*cos(2*theta) + 784*cos(4*theta) + 64*cos(6*theta) + cos(8*theta)))/3.8717332992e10;
				default:
					return 0.0;
			}
		default:
			return 0.0;
	}
	return 0.0;
}

double H_dd_14(int p, int q, double theta, double phi) {

	switch(p) {
		case 3:
			switch(q) {
				case 4:
        			return (642600*cos(4*phi) + 813960*cos(8*phi) + 1485800*cos(12*phi) + 13860*cos(2*phi - 5*theta) + 166320*cos(2*phi - 3*theta) + 113050*cos(6*phi - 3*theta) + 321300*cos(4*phi - 2*theta) + 271320*cos(8*phi - 2*theta) + 42840*cos(4*(phi - theta)) + 519750*cos(2*phi - theta) + 542640*cos(6*phi - theta) + 624036*cos(10*phi - theta) + 360360*cos(2*theta) + 72072*cos(4*theta) + 3432*cos(6*theta) + 91800*cos(4*(phi + theta)) + 554400*cos(2*phi + theta) + 428400*cos(2*(2*phi + theta)) + 258400*cos(3*(2*phi + theta)) + 3*(32300*cos(4*(2*phi + theta)) + 193800*cos(2*(4*phi + theta)) + 226100*cos(6*phi + theta) + 371450*cos(2*(6*phi + theta)) + 356592*cos(10*phi + theta) + 5*(20020 + 334305*cos(14*phi + theta) + 13860*cos(2*phi + 3*theta) + 22287*cos(10*phi + 3*theta) + 1584*cos(2*phi + 5*theta) + 1615*cos(6*phi + 5*theta) + 306*cos(4*phi + 6*theta) + 33*cos(2*phi + 7*theta))))/4.10793984e10;
				default:
					return 0.0;
			}
		case 4:
			switch(q) {
				case 3:
                	return (3*(358050*cos(2*phi) + 323*(1260*cos(6*phi) + 1748*cos(10*phi) + 5175*cos(14*phi)))*cos(theta) + 371450*cos(12*phi)*(4 + 3*cos(2*theta)) + 337075*cos(2*phi)*cos(3*theta) + 9690*cos(8*phi)*(84 + 88*cos(2*theta) + 69*cos(2*phi)*cos(3*theta) + 10*cos(4*theta)) + 13395*cos(2*phi)*cos(5*theta) + 170*cos(4*phi)*(19*cos(2*phi)*(23*cos(3*theta) + 15*cos(5*theta)) + 9*(420 + 490*cos(2*theta) + 88*cos(4*theta) + 3*cos(6*theta))) + 495*cos(2*phi)*cos(7*theta) + 3*(572*(175 + 210*cos(2*theta) + 42*cos(4*theta) + 2*cos(6*theta)) + 1671525*sin(14*phi)*sin(theta) + 371450*sin(12*phi)*sin(2*theta) + 37145*sin(10*phi)*(4*sin(theta) + 3*sin(3*theta)) + 6460*sin(8*phi)*(16*sin(2*theta) + 5*sin(4*theta)) + 1615*sin(6*phi)*(28*sin(theta) + 30*sin(3*theta) + 5*sin(5*theta)) + 510*sin(4*phi)*(70*sin(2*theta) + 32*sin(4*theta) + 3*sin(6*theta)) + 165*sin(2*phi)*(70*sin(theta) + 84*sin(3*theta) + 20*sin(5*theta) + sin(7*theta))))/4.10793984e10;
				case 4:
                	return ((15280650*cos(2*phi) + 323*(53900*cos(6*phi) + 75348*cos(10*phi) + 232875*cos(14*phi)))*cos(theta) + 2600150*cos(12*phi)*(8 + 7*cos(2*theta)) + 5978861*cos(2*phi)*cos(3*theta) + 40698*cos(8*phi)*(280 + 320*cos(2*theta) + 299*cos(2*phi)*cos(3*theta) + 50*cos(4*theta)) + 251405*cos(2*phi)*cos(5*theta) + 238*cos(4*phi)*(19*cos(2*phi)*(59*cos(3*theta) + 275*cos(5*theta)) + 675*(56 + 70*cos(2*theta) + 16*cos(4*theta) + cos(6*theta))) + 31185*cos(2*phi)*cos(7*theta) + 1716*(2450 + 3136*cos(2*theta) + 784*cos(4*theta) + 64*cos(6*theta) + cos(8*theta)))/1.591826688e11;
				default:
					return 0.0;
			}
		default:
			return 0.0;
	}
	return 0.0;
}   

double H_dd_16(int p, int q, double theta, double phi) {

	switch(p) {
		case 4:
			switch(q) {
				case 4:
                	return (300540195*cos(16*phi) + 8023320*cos(12*phi)*(8 + 7*cos(2*theta)) + 208012*cos(8*phi)*(196 + 224*cos(2*theta) + 200*cos(2*phi)*cos(3*theta) + 35*cos(4*theta)) + 18088*cos(4*phi)*(4*cos(2*phi)*(55*cos(3*theta) + 63*cos(5*theta)) + 33*(56 + 70*cos(2*theta) + 16*cos(4*theta) + cos(6*theta))) + 272*(19*(12348*cos(6*phi) + 16100*cos(10*phi) + 30015*cos(14*phi))*cos(theta) + cos(2*phi)*(210210*cos(theta) + 76769*cos(3*theta) + 3633*cos(5*theta) + 429*cos(7*theta))) + 6435*(2450 + 3136*cos(2*theta) + 784*cos(4*theta) + 64*cos(6*theta) + cos(8*theta)))/9.84810110976e12;
				default:
					return 0.0;
			}
		default:
			return 0.0;
	}
	return 0.0;
}   

double H_dd(int ell, int p, int q, double theta, double phi) {

	switch(ell) {
		case 0:
			return H_dd_0(p, q, theta, phi);
		case 2: 
			return H_dd_2(p, q, theta, phi);
		case 4: 
			return H_dd_4(p, q, theta, phi);
		case 6: 
			return H_dd_6(p, q, theta, phi);
		case 8: 
			return H_dd_8(p, q, theta, phi);
		case 10: 
			return H_dd_10(p, q, theta, phi);
		case 12: 
			return H_dd_12(p, q, theta, phi);
		case 14: 
			return H_dd_14(p, q, theta, phi);
		case 16: 
			return H_dd_16(p, q, theta, phi);
		default:
			return 0;
	}

	return 0;
}

void write_cov(char * covfile, int ncov, double ** cov) {

	FILE * fp;
	int i, j;

    if(!(fp = fopen(covfile, "w"))) {
      printf("\nERROR: Can't write in file '%s'.\n\n", covfile);
      exit(0);
    }
    fprintf(fp, "%d\n", ncov);
    for (i=0; i<ncov; i++) {
		for (j = 0; j < ncov; j++) fprintf(fp, "%12.6lf  ", 1.0e6*cov[i][j]);
        fprintf(fp, "\n");
    }
    fclose(fp);

}

int main(int argc, char **argv) {

    FILE * fp;
    char buf[500];
    int i, j, k, ell, veltype;
    char gridcorrfile[500], covfile[500];
    char * pkvelfile, * covfile_base;

    if (argc < 6) {
        printf("Error: 6 command line arguments required\n");
        exit(0);
    }

    double omega_m = 0.3121;    // The value of omega_m used to generate the simulations
    double kmin = atof(argv[1]);    // The minimum k-value to include information for
    double kmax = atof(argv[2]);    // The maximum k-value to include information for
    int gridsize = atoi(argv[3]);   // The size of each grid cell
    pkvelfile = argv[4];            // The file containing the input velocity power spectrum
    covfile_base = argv[5];         // The base for the output file name (other stuff will get added to the name)
	double job_num = atof(argv[6]);	// The job number of getafix

	sigma_u = job_num;

    // Read in the tabulated correction for the gridding
    sprintf(gridcorrfile, "./gridcorr_%d.dat", gridsize);

    //*****************************************************************************************//
    // I've decided to centre the grid on 0,0,0 so that we don't have any cell centres that are very close to the origin
	//These are the grid cells for the SDSS survey.
    double xmin = -175.0, xmax = 215.0;
    double ymin = -260.0, ymax = 280.0;
    double zmin = -300.0, zmax = 0.0;
    double smin = 0.0, smax = sqrt((xmax-xmin)*(xmax-xmin) + (ymax-ymin)*(ymax-ymin) + (zmax-zmin)*(zmax-zmin))+gridsize;
    int nx = (int)ceil((xmax-xmin)/gridsize);
    int ny = (int)ceil((ymax-ymin)/gridsize);
    int nz = (int)ceil((zmax-zmin)/gridsize);
    int nelements = nx*ny*nz;
    printf("%lf, %lf\n", smin, smax);
	printf("%d, %d, %d\n", nx, ny, nz);

    // Compute some useful quantities on the grid. For this we need a redshift-distance lookup table
    int nbins = 5000;
    double redmax = 0.5;//maximum redshift
    double * redarray = (double*)malloc(nbins*sizeof(double));//redshift array
    double * distarray = (double*)malloc(nbins*sizeof(double));//distance array
    for (i=0; i<nbins; i++) {
        redarray[i] = i*redmax/nbins;
        distarray[i] = rz(redarray[i]);
    }

    gsl_interp_accel * red_acc = gsl_interp_accel_alloc();
    gsl_spline * red_spline = gsl_spline_alloc(gsl_interp_cspline, nbins);
    gsl_spline_init(red_spline, distarray, redarray, nbins); //spline interpolation of redshift
    free(redarray);
    free(distarray);

    // Some arrays to hold values for each cell
    double * datagrid_x = (double *)malloc(nelements*sizeof(double));
    double * datagrid_y = (double *)malloc(nelements*sizeof(double));
    double * datagrid_z = (double *)malloc(nelements*sizeof(double));
    double * datagrid_r = (double *)malloc(nelements*sizeof(double));
	double * datagrid_dm = (double *)malloc(nelements*sizeof(double));
    for (i=0; i<nx; i++) {
        for (j=0; j<ny; j++) {
            for (k=0; k<nz; k++) {
                int ind = (i*ny+j)*nz+k;
                double x = (i+0.5)*gridsize+xmin;
                double y = (j+0.5)*gridsize+ymin;
                double z = (k+0.5)*gridsize+zmin;
                double R_dis = sqrt(x*x + y*y + z*z);
                datagrid_x[ind] = x;
                datagrid_y[ind] = y;
                datagrid_z[ind] = z;
                datagrid_r[ind] = R_dis; //distance to the galaxy
				double red = gsl_spline_eval(red_spline, R_dis, red_acc);//calculate redshift
                double ez = sqrt(omega_m*(1.0+red)*(1.0+red)*(1.0+red) + (1.0 - omega_m));//Friedman first equation
                datagrid_dm[ind] = (1.0/log(10))*(1.0+red)/(100.0*ez*R_dis);//The prefactor for the log-distance ratio
            }
        }
    }

    //*****************************************************************************************//
    // Read in the velocity divergence power spectrum and multiply by (aHf)^2. We set f=1, a=1 and H=100h.
    if(!(fp = fopen(pkvelfile, "r"))) {
        printf("\nERROR: Can't open power file '%s'.\n\n", pkvelfile);
        exit(0);
    }

    int npk = 0;
    while(fgets(buf,500,fp)) {
        if(strncmp(buf,"#",1)!=0) {
            double tk, pkgal, pkgal_vel, pkvel;
            if(sscanf(buf, "%lg  %lg %lg %lg\n", &tk, &pkgal, &pkgal_vel, &pkvel) != 4) {printf("Pvel read error\n"); exit(0);};
            npk++;
        }
    }
    fclose(fp);

    double * karray = (double *)malloc(npk*sizeof(double));
    double * pmmarray = (double *)malloc(npk*sizeof(double));
	double * pmvarray = (double *)malloc(npk*sizeof(double));
	double * pvvarray = (double *)malloc(npk*sizeof(double));

    npk = 0;
    fp = fopen(pkvelfile, "r");
    while(fgets(buf,500,fp)) {
        if(strncmp(buf,"#",1)!=0) {
            double tk, pmm, pmv, pvv;
            if(sscanf(buf, "%lg  %lg %lg %lg\n", &tk, &pmm, &pmv, &pvv) != 4) {printf("Pvel read error\n"); exit(0);};
            karray[npk] = tk;
			pmmarray[npk] = pmm;
			pmvarray[npk] = pmv;
			pvvarray[npk] = pvv;
            npk++;
        }
    }
    fclose(fp);

    // Read in and spline the grid corrections
    if(!(fp = fopen(gridcorrfile, "r"))) {
        printf("\nERROR: Can't open grid_corr file '%s'.\n\n", gridcorrfile);
        exit(0);
    }

	P_mm_acc = gsl_interp_accel_alloc();
    P_mm_spline = gsl_spline_alloc(gsl_interp_cspline, npk);
    gsl_spline_init(P_mm_spline, karray, pmmarray, npk);

	P_vm_acc = gsl_interp_accel_alloc();
    P_vm_spline = gsl_spline_alloc(gsl_interp_cspline, npk);
    gsl_spline_init(P_vm_spline, karray, pmvarray, npk);

	P_vv_acc = gsl_interp_accel_alloc();
    P_vv_spline = gsl_spline_alloc(gsl_interp_cspline, npk);
    gsl_spline_init(P_vv_spline, karray, pvvarray, npk);
	free(karray);
	free(pmmarray);
	free(pmvarray);
	free(pvvarray);

    int ngridcorr = 0;
    while(fgets(buf,500,fp)) {
        if(strncmp(buf,"#",1)!=0) {
            double tk, gridcorr;
            if(sscanf(buf, "%lf %lf\n", &tk, &gridcorr) != 2) {printf("Pvel read error\n"); exit(0);};
            ngridcorr++;
        }
    }
    fclose(fp);

    double * gridkarray = (double *)malloc(ngridcorr*sizeof(double));
    double * gridcorrarray = (double *)malloc(ngridcorr*sizeof(double));

    ngridcorr = 0;
    fp = fopen(gridcorrfile, "r");
    while(fgets(buf,500,fp)) {
        if(strncmp(buf,"#",1)!=0) {
            double tk, gridcorr;
            if(sscanf(buf, "%lf %lf\n", &tk, &gridcorr) != 2) {printf("Pvel read error\n"); exit(0);};
            gridkarray[ngridcorr] = tk;
            gridcorrarray[ngridcorr] = gridcorr;
            ngridcorr++;
        }
    }
    fclose(fp);

    gridcorr_acc = gsl_interp_accel_alloc();
    gridcorr_spline = gsl_spline_alloc(gsl_interp_cspline, ngridcorr);
    gsl_spline_init(gridcorr_spline, gridkarray, gridcorrarray, ngridcorr);
    free(gridkarray);
    free(gridcorrarray);

    //*****************************************************************************************//
    // Now compute the covariance matrix. This is symmetric so we only need to actually calculate the forward half. However to make it more easily parallelisable
    // The double loop is converted to a single loop and the indices of the matrix calculated from the single iterator. For this though we need to know which indices correspond to which index
    int niter = nelements*(nelements+1)/2;

    int * indexi = (int *)malloc(niter*sizeof(int));
    int * indexj = (int *)malloc(niter*sizeof(int));
    int counteri = 0;

	int sum = 0;
    for (i=0; i<nelements; i++) {
        for (j=0; j<nelements-i; j++) {
            indexi[j+counteri] = i;
            indexj[j+counteri] = i+j;
			sum += 1;
        }
        counteri += nelements-i;
    }

	printf("%d\n", niter);

	printf("Computing diagonals\n");

	// Compute the diagonal correlation functions. This only needs doing once as it is just an integral over the power spectrum
	// and is only non-zero for l=0
	double theta_diag = 0.0; 
	double phi_diag = M_PI/2.0; 
	//At the diagonal of the matrix, line of sight to two galaxies overlap, so theta = 0, and phi = pi/2. This is because cos(phi) = s dot d = 0;
	
	double diag_vv = conv_integral(kmin, kmax, 0.0, 2, 0, -2, 1)*H_vv(0, theta_diag, phi_diag);
	double diag_vv_ng = conv_integral(kmin, kmax, 0.0, 2, 0, -2, 0)*H_vv(0, theta_diag, phi_diag);

    //*****************************************************************************************//

    printf("Precomputing vv correlation functions\n");

	xi_vv_acc = (gsl_interp_accel ***)malloc(2*sizeof(gsl_interp_accel **));
	for (i=0; i<2; i++) xi_vv_acc[i] = (gsl_interp_accel **)malloc(2*sizeof(gsl_interp_accel *));

	xi_vv_spline = (gsl_spline ***)malloc(2*sizeof(gsl_spline **));
	for (i=0; i<2; i++) xi_vv_spline[i] = (gsl_spline **)malloc(2*sizeof(gsl_spline *));

	int rbins = 10000;
	double * svals = (double *)malloc(rbins*sizeof(double));
	double * xivals = (double *)malloc(rbins*sizeof(double));
	for (i=0; i<rbins; i++) svals[i] = i*(smax-smin)/(rbins-1.0) + smin;
	for (veltype=0; veltype<2; veltype++) { //veltype = 0 for non-gridded velocity auto-covariance matrix, veltype = 1 for the gridded version.
		for (ell=0; ell<2; ell++) { //ell is the order of the spherical bessel function.
			for (j=0; j<rbins; j++) xivals[j] = conv_integral(kmin, kmax, svals[j], 2, 2*ell, -2, veltype);
			printf("%d, %d\n", veltype, 2*ell);
			xi_vv_acc[veltype][ell] = gsl_interp_accel_alloc();
   	 		xi_vv_spline[veltype][ell] = gsl_spline_alloc(gsl_interp_cspline, rbins);
    		gsl_spline_init(xi_vv_spline[veltype][ell], svals, xivals, rbins); //This spline interpolate the result of the integrals.
		}
	}
	free(svals);
	free(xivals);

	printf("Starting vv covariance\n");

    double ** conv_pk_vel = (double **)malloc(nelements*sizeof(double*));
    double ** conv_pk_vel_ng = (double **)malloc(nelements*sizeof(double*));
    for (i=0; i<nelements; i++) {
        conv_pk_vel[i] = (double *)malloc(nelements*sizeof(double));
        conv_pk_vel_ng[i] = (double *)malloc(nelements*sizeof(double));
    }

    for (i=0; i<niter; i++) {
        int indi = indexi[i];
        int indj = indexj[i];
        if (indi == indj) {
            conv_pk_vel[indi][indi] = 10000.0*diag_vv/(2.0*M_PI*M_PI)*datagrid_dm[indi]*datagrid_dm[indi];
            conv_pk_vel_ng[indi][indi] = 10000.0*diag_vv_ng/(2.0*M_PI*M_PI)*datagrid_dm[indi]*datagrid_dm[indi];
            printf("%d, %lf, %lf, %lf, %lf, %lf\n", indi, datagrid_x[indi], datagrid_y[indi], datagrid_z[indi], conv_pk_vel[indi][indi], conv_pk_vel_ng[indi][indi]);
        } else {
            double xi = datagrid_x[indi];
            double yi = datagrid_y[indi];
            double zi = datagrid_z[indi];
            double ri = datagrid_r[indi];
            double xj = datagrid_x[indj];
            double yj = datagrid_y[indj];
            double zj = datagrid_z[indj];
            double rj = datagrid_r[indj];
			double phi, dx, dy, dz, dl, phi_arg;
            double s = sqrt((xi - xj)*(xi - xj) + (yi - yj)*(yi - yj) + (zi - zj)*(zi - zj));
			double costheta = (xi*xj + yi*yj + zi*zj)/(ri*rj);
			double theta;
			if ((costheta >= -1.0) && (costheta <= 1.0)) {
				theta = acos(costheta);
				dx = ri*rj/(ri+rj)*(xi/ri + xj/rj);
	            dy = ri*rj/(ri+rj)*(yi/ri + yj/rj);
	            dz = ri*rj/(ri+rj)*(zi/ri + zj/rj);
	            dl = sqrt(dx*dx+dy*dy+dz*dz);
				phi_arg = (dx*(xi-xj) + dy*(yi-yj) + dz*(zi-zj))/(dl*s);
				if ((phi_arg >=-1.0) && (phi_arg <= 1.0)) {
					phi = acos(phi_arg);
				}
				else if (phi_arg > 1.0) {
					phi = 0.0;
				}
				else {
					phi = M_PI;
				}
			}
			else if (costheta > 1.0) {
				theta = 0.0;
				dx = ri*rj/(ri+rj)*(xi/ri + xj/rj);
	            dy = ri*rj/(ri+rj)*(yi/ri + yj/rj);
	            dz = ri*rj/(ri+rj)*(zi/ri + zj/rj);
	            dl = sqrt(dx*dx+dy*dy+dz*dz);
				phi_arg = (dx*(xi-xj) + dy*(yi-yj) + dz*(zi-zj))/(dl*s);
				if ((phi_arg >=-1.0) && (phi_arg <= 1.0)) {
					phi = acos(phi_arg);
				}
				else if (phi_arg > 1.0) {
					phi = 0.0;
				}
				else {
					phi = M_PI;
				}
			}
			else {
				theta = M_PI;
				dx = ri*rj/(ri+rj)*(xi/ri + xj/rj);
	            dy = ri*rj/(ri+rj)*(yi/ri + yj/rj);
	            dz = ri*rj/(ri+rj)*(zi/ri + zj/rj);
	            dl = sqrt(dx*dx+dy*dy+dz*dz);
				phi_arg = (dx*(xi-xj) + dy*(yi-yj) + dz*(zi-zj))/(dl*s);
				if ((phi_arg >=-1.0) && (phi_arg <= 1.0)) {
					phi = acos(phi_arg);
				}
				else if (phi_arg > 1.0) {
					phi = 0.0;
				}
				else {
					phi = M_PI;
				}
			}

            /*double theta = acos((xi*xj + yi*yj + zi*zj)/(ri*rj));
            double phi;
            if (indi + indj == nelements-1) {
            	phi = M_PI/2.0;
            } else {
	            double dx = ri*rj/(ri+rj)*(xi/ri + xj/rj);
	            double dy = ri*rj/(ri+rj)*(yi/ri + yj/rj);
	            double dz = ri*rj/(ri+rj)*(zi/ri + zj/rj);
	            double dl = sqrt(dx*dx+dy*dy+dz*dz);
			   	double phi = acos((dx*(xi-xj) + dy*(yi-yj) + dz*(zi-zj))/(dl*s));
			}*/

			double cov = 0, cov_ng = 0;
			for (ell=0; ell<4; ell+=2) {
				cov += creal(cpow(I,ell))*gsl_spline_eval(xi_vv_spline[1][ell/2], s, xi_vv_acc[1][ell/2])*H_vv(ell, theta, phi);
				cov_ng += creal(cpow(I,ell))*gsl_spline_eval(xi_vv_spline[0][ell/2], s, xi_vv_acc[0][ell/2])*H_vv(ell, theta, phi);
				/*cov += -creal(cpow(I,ell+2))*conv_integral(kmin, kmax, s, 2, ell, -2, 1)*H_vv(ell, theta, phi);
				cov_ng += -creal(cpow(I,ell+2))*conv_integral(kmin, kmax, s, 2, ell, -2, 0)*H_vv(ell, theta, phi);*/
			}
            conv_pk_vel[indi][indj] = 10000.0*cov/(2.0*M_PI*M_PI)*datagrid_dm[indi]*datagrid_dm[indj];
            conv_pk_vel[indj][indi] = conv_pk_vel[indi][indj];
            conv_pk_vel_ng[indi][indj] = 10000.0*cov_ng/(2.0*M_PI*M_PI)*datagrid_dm[indi]*datagrid_dm[indj];
            conv_pk_vel_ng[indj][indi] = conv_pk_vel_ng[indi][indj];
            //printf("%d, %lf, %lf, %lf, %d, %lf, %lf, %lf, %lf, %lf\n", indi, datagrid_x[indi], datagrid_y[indi], datagrid_z[indi], indj, datagrid_x[indj], datagrid_y[indj], datagrid_z[indj], conv_pk_vel[indi][indj], conv_pk_vel_ng[indi][indj]);
        }
    }

	sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_vv_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel);
    sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_vv_ng_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel_ng);

	printf("Done vv covariance\n");
    for (i=0; i<nelements; i++) {
    	free(conv_pk_vel[i]);
    	free(conv_pk_vel_ng[i]);
    }
    free(conv_pk_vel);
    free(conv_pk_vel_ng);
    for (veltype=0; veltype<2; veltype++) {
		for (ell=0; ell<2; ell++) {
			gsl_spline_free(xi_vv_spline[veltype][ell]);
    		gsl_interp_accel_free(xi_vv_acc[veltype][ell]);
		}
	}
	for (i=0; i<2; i++) {
		free(xi_vv_acc[i]);
		free(xi_vv_spline[i]);
	}
	free(xi_vv_acc);
	free(xi_vv_spline);

    //*****************************************************************************************//

    printf("Precomputing dv correlation functions\n");

	xi_dv_acc = (gsl_interp_accel ****)malloc(2*sizeof(gsl_interp_accel ***));
	for (i=0; i<2; i++) {
		xi_dv_acc[i] = (gsl_interp_accel ***)malloc(4*sizeof(gsl_interp_accel **)); // i = 0 gives the velocity-matter cross-power spectrum and i = 1 gives the velocity auto-power spectrum.
		for (j=0; j<4; j++) xi_dv_acc[i][j] = (gsl_interp_accel **)malloc((j+2)*sizeof(gsl_interp_accel *));
	}
	xi_dv_spline = (gsl_spline ****)malloc(2*sizeof(gsl_spline ***));
	for (i=0; i<2; i++) {
		xi_dv_spline[i] = (gsl_spline ***)malloc(4*sizeof(gsl_spline **));
		for (j=0; j<4; j++) xi_dv_spline[i][j] = (gsl_spline **)malloc((j+2)*sizeof(gsl_spline *));
	}

	svals = (double *)malloc(rbins*sizeof(double));
	xivals = (double *)malloc(rbins*sizeof(double));
	for (i=0; i<rbins; i++) svals[i] = i*(smax-smin)/(rbins-1.0) + smin;
	for (veltype=0; veltype<2; veltype++) {
		for (i=0; i<4; i++) {//2*i-1 is the power of k in the integral.
			for (ell=0; ell<i+2; ell++) {
				for (j=0; j<rbins; j++) xivals[j] = conv_integral(kmin, kmax, svals[j], veltype+1, 2*ell+1, 2*i-1, 1); //veltype+1 gives determine which power spectrum to use.
				printf("%d, %d, %d\n", veltype+1, 2*i-1, 2*ell+1);
				xi_dv_acc[veltype][i][ell] = gsl_interp_accel_alloc();
   	 			xi_dv_spline[veltype][i][ell] = gsl_spline_alloc(gsl_interp_cspline, rbins);
    			gsl_spline_init(xi_dv_spline[veltype][i][ell], svals, xivals, rbins);
			}
		}
	}
	free(svals);
	free(xivals);

   	printf("Starting dv covariance\n");
	double ** conv_pk_vel_gal_1_0 = (double **)malloc(nelements*sizeof(double*));
	double ** conv_pk_vel_gal_1_2 = (double **)malloc(nelements*sizeof(double*));
	double ** conv_pk_vel_gal_1_4 = (double **)malloc(nelements*sizeof(double*));
	double ** conv_pk_vel_gal_1_6 = (double **)malloc(nelements*sizeof(double*));
    double ** conv_pk_vel_gal_2_0 = (double **)malloc(nelements*sizeof(double*));
    double ** conv_pk_vel_gal_2_2 = (double **)malloc(nelements*sizeof(double*));
    double ** conv_pk_vel_gal_2_4 = (double **)malloc(nelements*sizeof(double*));
    double ** conv_pk_vel_gal_2_6 = (double **)malloc(nelements*sizeof(double*));
    for (i=0; i<nelements; i++) {
		conv_pk_vel_gal_1_0[i] = (double *)malloc(nelements*sizeof(double));
		conv_pk_vel_gal_1_2[i] = (double *)malloc(nelements*sizeof(double));
		conv_pk_vel_gal_1_4[i] = (double *)malloc(nelements*sizeof(double));
		conv_pk_vel_gal_1_6[i] = (double *)malloc(nelements*sizeof(double));
        conv_pk_vel_gal_2_0[i] = (double *)malloc(nelements*sizeof(double));
        conv_pk_vel_gal_2_2[i] = (double *)malloc(nelements*sizeof(double));
        conv_pk_vel_gal_2_4[i] = (double *)malloc(nelements*sizeof(double));
        conv_pk_vel_gal_2_6[i] = (double *)malloc(nelements*sizeof(double));
    }

    for (i=0; i<niter; i++) {
        int indi = indexi[i];
        int indj = indexj[i];
        if (indi == indj) {
            conv_pk_vel_gal_1_0[indi][indi] = 0.0;
            conv_pk_vel_gal_1_2[indi][indi] = 0.0;
            conv_pk_vel_gal_1_4[indi][indi] = 0.0;
            conv_pk_vel_gal_1_6[indi][indi] = 0.0;
            conv_pk_vel_gal_2_0[indi][indi] = 0.0;
            conv_pk_vel_gal_2_2[indi][indi] = 0.0;
            conv_pk_vel_gal_2_4[indi][indi] = 0.0;
            conv_pk_vel_gal_2_6[indi][indi] = 0.0;
            printf("%d, %lf, %lf, %lf, %lf, %lf\n", indi, datagrid_x[indi], datagrid_y[indi], datagrid_z[indi], conv_pk_vel_gal_1_0[indi][indi], conv_pk_vel_gal_1_6[indi][indi]);
        } else {
            double xi = datagrid_x[indi];
            double yi = datagrid_y[indi];
            double zi = datagrid_z[indi];
            double ri = datagrid_r[indi];
            double xj = datagrid_x[indj];
            double yj = datagrid_y[indj];
            double zj = datagrid_z[indj];
            double rj = datagrid_r[indj];
			double phi, dx, dy, dz, dl, phi_arg, phi_opp;
            double s = sqrt((xi - xj)*(xi - xj) + (yi - yj)*(yi - yj) + (zi - zj)*(zi - zj));
			double costheta = (xi*xj + yi*yj + zi*zj)/(ri*rj);
			double theta;
			if ((costheta >= -1.0) && (costheta <= 1.0)) {
				theta = acos(costheta);
				dx = ri*rj/(ri+rj)*(xi/ri + xj/rj);
	            dy = ri*rj/(ri+rj)*(yi/ri + yj/rj);
	            dz = ri*rj/(ri+rj)*(zi/ri + zj/rj);
	            dl = sqrt(dx*dx+dy*dy+dz*dz);
				phi_arg = (dx*(xi-xj) + dy*(yi-yj) + dz*(zi-zj))/(dl*s);
				if ((phi_arg >=-1.0) && (phi_arg <= 1.0)) {
					phi = acos(phi_arg);
					phi_opp = acos(-phi_arg);
				}
				else if (phi_arg > 1.0) {
					phi = 0.0;
					phi_opp = M_PI;
				}
				else {
					phi = M_PI;
					phi_opp = 0.0;
				}
			}
			else if (costheta > 1.0) {
				theta = 0.0;
				dx = ri*rj/(ri+rj)*(xi/ri + xj/rj);
	            dy = ri*rj/(ri+rj)*(yi/ri + yj/rj);
	            dz = ri*rj/(ri+rj)*(zi/ri + zj/rj);
	            dl = sqrt(dx*dx+dy*dy+dz*dz);
				phi_arg = (dx*(xi-xj) + dy*(yi-yj) + dz*(zi-zj))/(dl*s);
				if ((phi_arg >=-1.0) && (phi_arg <= 1.0)) {
					phi = acos(phi_arg);
					phi_opp = acos(-phi_arg);
				}
				else if (phi_arg > 1.0) {
					phi = 0.0;
					phi_opp = M_PI;
				}
				else {
					phi = M_PI;
					phi_opp = 0.0;
				}
			}
			else {
				theta = M_PI;
				dx = ri*rj/(ri+rj)*(xi/ri + xj/rj);
	            dy = ri*rj/(ri+rj)*(yi/ri + yj/rj);
	            dz = ri*rj/(ri+rj)*(zi/ri + zj/rj);
	            dl = sqrt(dx*dx+dy*dy+dz*dz);
				phi_arg = (dx*(xi-xj) + dy*(yi-yj) + dz*(zi-zj))/(dl*s);
				if ((phi_arg >=-1.0) && (phi_arg <= 1.0)) {
					phi = acos(phi_arg);
					phi_opp = acos(-phi_arg);
				}
				else if (phi_arg > 1.0) {
					phi = 0.0;
					phi_opp = M_PI;
				}
				else {
					phi = M_PI;
					phi_opp = M_PI; 
				}
			}

            /*double theta = acos((xi*xj + yi*yj + zi*zj)/(ri*rj));
            double phi;
            if (indi + indj == nelements-1) {
            	phi = M_PI/2.0;
            } else {
	            double dx = ri*rj/(ri+rj)*(xi/ri + xj/rj);
	            double dy = ri*rj/(ri+rj)*(yi/ri + yj/rj);
	            double dz = ri*rj/(ri+rj)*(zi/ri + zj/rj);
	            double dl = sqrt(dx*dx+dy*dy+dz*dz);
			   	double phi = acos((dx*(xi-xj) + dy*(yi-yj) + dz*(zi-zj))/(dl*s));
			}*/

			double cov_1_0 = 0, cov_1_2 = 0, cov_1_4 = 0, cov_1_6 = 0;
			double cov_2_0 = 0, cov_2_2 = 0, cov_2_4 = 0, cov_2_6 = 0;
			double cov_1_0_opp = 0, cov_1_2_opp = 0, cov_1_4_opp = 0, cov_1_6_opp = 0;
			double cov_2_0_opp = 0, cov_2_2_opp = 0, cov_2_4_opp = 0, cov_2_6_opp = 0;
			double integral_1, integral_2;

			for (ell=9; ell>0; ell-=2) {
				integral_1 = creal(cpow(I, ell+1))*gsl_spline_eval(xi_dv_spline[0][3][(ell-1)/2], s, xi_dv_acc[0][3][(ell-1)/2]);
				integral_2 = creal(cpow(I, ell+1))*gsl_spline_eval(xi_dv_spline[1][3][(ell-1)/2], s, xi_dv_acc[1][3][(ell-1)/2]);
				cov_1_6 += integral_1*H_dv(ell, 3, theta, phi);
				cov_2_6 += integral_2*H_dv(ell, 4, theta, phi);
				cov_1_6_opp += integral_1*H_dv(ell, 3, theta, phi_opp);
				cov_2_6_opp += integral_2*H_dv(ell, 4, theta, phi_opp);
				if (ell < 8) {
					integral_1 = creal(cpow(I, ell+1))*gsl_spline_eval(xi_dv_spline[0][2][(ell-1)/2], s, xi_dv_acc[0][2][(ell-1)/2]);
					integral_2 = creal(cpow(I, ell+1))*gsl_spline_eval(xi_dv_spline[1][2][(ell-1)/2], s, xi_dv_acc[1][2][(ell-1)/2]);
					cov_1_4 += integral_1*H_dv(ell, 2, theta, phi);
					cov_2_4 += integral_2*H_dv(ell, 3, theta, phi);
					cov_1_4_opp += integral_1*H_dv(ell, 2, theta, phi_opp);
					cov_2_4_opp += integral_2*H_dv(ell, 3, theta, phi_opp);
				}
				if (ell < 6) {
					integral_1 = creal(cpow(I, ell+1))*gsl_spline_eval(xi_dv_spline[0][1][(ell-1)/2], s, xi_dv_acc[0][1][(ell-1)/2]);
					integral_2 = creal(cpow(I, ell+1))*gsl_spline_eval(xi_dv_spline[1][1][(ell-1)/2], s, xi_dv_acc[1][1][(ell-1)/2]);
					cov_1_2 += integral_1*H_dv(ell, 1, theta, phi);
				 	cov_2_2 += integral_2*H_dv(ell, 2, theta, phi);
					cov_1_2_opp += integral_1*H_dv(ell, 1, theta, phi_opp);
				 	cov_2_2_opp += integral_2*H_dv(ell, 2, theta, phi_opp);
				}
				if (ell < 4) {
					integral_1 = creal(cpow(I, ell+1))*gsl_spline_eval(xi_dv_spline[0][0][(ell-1)/2], s, xi_dv_acc[0][0][(ell-1)/2]);
					integral_2 = creal(cpow(I, ell+1))*gsl_spline_eval(xi_dv_spline[1][0][(ell-1)/2], s, xi_dv_acc[1][0][(ell-1)/2]);
					cov_1_0 += integral_1*H_dv(ell, 0, theta, phi);
					cov_2_0 += integral_2*H_dv(ell, 1, theta, phi);
					cov_1_0_opp += integral_1*H_dv(ell, 0, theta, phi_opp);
					cov_2_0_opp += integral_2*H_dv(ell, 1, theta, phi_opp);
				}
			}
            conv_pk_vel_gal_1_0[indi][indj] =  100.0*cov_1_0/(2.0*M_PI*M_PI)*datagrid_dm[indi];
            conv_pk_vel_gal_1_2[indi][indj] = -100.0*cov_1_2/(2.0*M_PI*M_PI)/2.0*datagrid_dm[indi];
            conv_pk_vel_gal_1_4[indi][indj] =  100.0*cov_1_4/(2.0*M_PI*M_PI)/8.0*datagrid_dm[indi];
            conv_pk_vel_gal_1_6[indi][indj] = -100.0*cov_1_6/(2.0*M_PI*M_PI)/48.0*datagrid_dm[indi];
            conv_pk_vel_gal_2_0[indi][indj] =  100.0*cov_2_0/(2.0*M_PI*M_PI)*datagrid_dm[indi];
            conv_pk_vel_gal_2_2[indi][indj] = -100.0*cov_2_2/(2.0*M_PI*M_PI)/2.0*datagrid_dm[indi];
            conv_pk_vel_gal_2_4[indi][indj] =  100.0*cov_2_4/(2.0*M_PI*M_PI)/8.0*datagrid_dm[indi];
            conv_pk_vel_gal_2_6[indi][indj] = -100.0*cov_2_6/(2.0*M_PI*M_PI)/48.0*datagrid_dm[indi];
            conv_pk_vel_gal_1_0[indj][indi] =  100.0*cov_1_0_opp/(2.0*M_PI*M_PI)*datagrid_dm[indj];
            conv_pk_vel_gal_1_2[indj][indi] = -100.0*cov_1_2_opp/(2.0*M_PI*M_PI)/2.0*datagrid_dm[indj];
            conv_pk_vel_gal_1_4[indj][indi] =  100.0*cov_1_4_opp/(2.0*M_PI*M_PI)/8.0*datagrid_dm[indj];
            conv_pk_vel_gal_1_6[indj][indi] = -100.0*cov_1_6_opp/(2.0*M_PI*M_PI)/48.0*datagrid_dm[indj];
            conv_pk_vel_gal_2_0[indj][indi] =  100.0*cov_2_0_opp/(2.0*M_PI*M_PI)*datagrid_dm[indj];
            conv_pk_vel_gal_2_2[indj][indi] = -100.0*cov_2_2_opp/(2.0*M_PI*M_PI)/2.0*datagrid_dm[indj];
            conv_pk_vel_gal_2_4[indj][indi] =  100.0*cov_2_4_opp/(2.0*M_PI*M_PI)/8.0*datagrid_dm[indj];
            conv_pk_vel_gal_2_6[indj][indi] = -100.0*cov_2_6_opp/(2.0*M_PI*M_PI)/48.0*datagrid_dm[indj];
            //printf("%d, %lf, %lf, %lf, %d, %lf, %lf, %lf, %lf, %lf\n", indi, datagrid_x[indi], datagrid_y[indi], datagrid_z[indi], indj, datagrid_x[indj], datagrid_y[indj], datagrid_z[indj], 1.0e6*conv_pk_vel_gal_1_2[indi][indj], 1.0e6*conv_pk_vel_gal_2_6[indi][indj]);
        }
    }

	sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_dv_1_0_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel_gal_1_0);
	sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_dv_1_2_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel_gal_1_2);
   	sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_dv_1_4_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel_gal_1_4);
    sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_dv_1_6_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel_gal_1_6);
	sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_dv_2_0_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel_gal_2_0);
	sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_dv_2_2_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel_gal_2_2);
   	sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_dv_2_4_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel_gal_2_4);
    sprintf(covfile, "%s_k0p%03d_0p%03d_gridcorr%02d_dv_2_6_sigmau%03d.dat", covfile_base, (int)(1000.0*kmin), (int)(1000.0*kmax), gridsize, (int) (10.0*sigma_u));
    write_cov(covfile, nelements, conv_pk_vel_gal_2_6);

	printf("Done dv covariance\n");
    for (i=0; i<nelements; i++) {
    	free(conv_pk_vel_gal_1_0[i]);
    	free(conv_pk_vel_gal_1_2[i]);
    	free(conv_pk_vel_gal_1_4[i]);
    	free(conv_pk_vel_gal_1_6[i]);
    	free(conv_pk_vel_gal_2_0[i]);
    	free(conv_pk_vel_gal_2_2[i]);
    	free(conv_pk_vel_gal_2_4[i]);
    	free(conv_pk_vel_gal_2_6[i]);
    }
    free(conv_pk_vel_gal_1_0);
    free(conv_pk_vel_gal_1_2);
    free(conv_pk_vel_gal_1_4);
    free(conv_pk_vel_gal_1_6);
    free(conv_pk_vel_gal_2_0);
    free(conv_pk_vel_gal_2_2);
    free(conv_pk_vel_gal_2_4);
    free(conv_pk_vel_gal_2_6);
    for (veltype=0; veltype<2; veltype++) {
		for (i=0; i<4; i++) {
			for (ell=0; ell<i+2; ell++) {
				gsl_spline_free(xi_dv_spline[veltype][i][ell]);
    			gsl_interp_accel_free(xi_dv_acc[veltype][i][ell]);
			}
		}
	}
	for (i=0; i<2; i++) {
		for (j=0; j<4; j++) {
			free(xi_dv_acc[i][j]);
			free(xi_dv_spline[i][j]);
		}
		free(xi_dv_acc[i]);
		free(xi_dv_spline[i]);
	}
	free(xi_dv_acc);
	free(xi_dv_spline);

	free(datagrid_x);
    free(datagrid_y);
    free(datagrid_z);
    free(datagrid_r);
	gsl_spline_free (gridcorr_spline);
    gsl_interp_accel_free (gridcorr_acc);
	gsl_spline_free (P_mm_spline);
    gsl_interp_accel_free (P_mm_acc);
	gsl_spline_free (P_vm_spline);
    gsl_interp_accel_free (P_vm_acc);
	gsl_spline_free (P_vv_spline);
    gsl_interp_accel_free (P_vv_acc);

    return 0;
}