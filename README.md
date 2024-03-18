# Wide-angle peculiar velocity fitting code
1. This repository includes the c code to generate the covariance matrix and the python code to fit fsigma8 with the covariance matrices. The output covariance matrix assumes f=b=1, they will be rescaled in the python fitting code. <br />
2. To run the python code, you need to install numpy, scipy, emcee and pandas. It reads in a config file that contains the input cosmological parameters and the location of the data and random files. Remember to change the config file if you are using a different dataset. <br />
3. For the random file, the code assumes the first three columns are RA, Dec, and redshift. If not, change line 538 to 540. <br />
4. For the data file, we assume the heading for RA is "RA", heading for Dec is Dec, heading for redshift is "zcmb", heading for log-distance ratio is "logdist_corr", and the heading for the uncertainty for the log-distance ratio is "logdist_corr_err". If not, change line 575 to 577 and line 598 and 599. <br />
5. The grid correction files and the power spectrum we used to analyse the SDSS PV catalogue is in the grid_correction folder. The data file is in the data folder.  <br />
6. Our python and c code only calculate the covariance matrix with the Taylor expansion of D_g up to the third order. You can use the Mathematica note book provided to calculate the covariance matrix with higher order of Taylor expansion. <br />
7. The mocks and random can be accessed through this website https://zenodo.org/record/6640513. 

