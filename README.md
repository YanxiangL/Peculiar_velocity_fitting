# Wide-angle peculiar velocity fitting code
This repository includes the c code to generate the covariance matrix and the python code to fit fsigma8 with the covariance matrices. 
To run the python code, you need to install numpy, scipy, emcee and pandas. 
The grid correction files and the power spectrum we used to analyse the SDSS PV catalogue is in the grid_correction folder. The data file is in the data folder. 
You can modify the code to fit other surveys as well. I have indicated which lines you need to change if you are using a different survey in the python and c code. 
Our python and c code only calculate the covariance matrix with the Taylor expansion of D_g up to the third order. You can use the Mathematica note book provided to calculate the covariance matrix with higher order of Taylor expansion. 
The mocks and random can be accessed through this website https://zenodo.org/record/6640513. 
