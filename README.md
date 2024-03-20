# Wide-angle peculiar velocity fitting code
1. This code uses INIH (INI Not Invented Here) to read in the configuration file (config.ini) in c. Please see https://github.com/benhoyt/inih for more detail on INIH. <br />
2. To compile the c codes, use the following command: "gcc name_of_the_code.c ./inih/ini.c -lgsl -lgslcblas -lm -o name_of_the_executable. <br />
3. This repository includes the c code to generate the covariance matrix and the python code to fit fsigma8 with the covariance matrices. The output covariance matrix assumes f=b=1, they will be rescaled in the python fitting code. <br />
4. To run the python code, you need to install numpy, scipy, emcee and pandas. It reads in a config file (config.ini) that contains the input cosmological parameters and the location of the data and random files. Remember to change the config file if you are using a different dataset. <br />
5. For the random file, the code assumes the first three columns are RA, Dec, and redshift. If not, change line 538 to 540. <br />
6. For the data file, we assume the heading for RA is "RA", heading for Dec is Dec, heading for redshift is "zcmb", heading for log-distance ratio is "logdist_corr", and the heading for the uncertainty for the log-distance ratio is "logdist_corr_err". If not, change line 575 to 577 and line 598 and 599. <br />
7. The grid correction files and the power spectrum we used to analyse the SDSS PV catalogue is in the grid_correction folder. The data file is in the data folder.  <br />
8. Our python and c code only calculate the covariance matrix with the Taylor expansion of D_g up to the third order. You can use the Mathematica note book provided to calculate the covariance matrix with higher order of Taylor expansion. <br />
9. The mocks and random can be accessed through this website https://zenodo.org/record/6640513.
10. To copy this repository run "git clone <url> --branch desi".

