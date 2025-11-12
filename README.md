# Wide-angle peculiar velocity fitting code
1. This code uses INIH (INI Not Invented Here) to read in the configuration file (config.ini) in c. Please see https://github.com/benhoyt/inih for more detail on INIH. <br />
2. To compile the c codes, use the following command: "gcc name_of_the_code.c ./inih/ini.c -lgsl -lgslcblas -lm -o name_of_the_executable. <br />
3. This repository includes the c code to generate the covariance matrix and the python code to fit fsigma8 with the covariance matrices. The output covariance matrix assumes f=b=1; they will be rescaled in the Python fitting code. <br />
4. To run the python code, the easiest way is to create a new python virtual environment and run pip install -r requirements.txt to install all the required paskages. The python code reads in a config file (config.ini) that contains the input cosmological parameters and the location of the data and random files. Remember to change the config file if you are using a different dataset. <br />
5. The grid correction files and the power spectrum we used to analyse the SDSS PV catalogue is in the grid_correction folder. The data file is in the data folder.  <br />
6. Our python and c code only calculate the covariance matrix with the Taylor expansion of D_g up to the third order. You can use the Mathematica notebook provided to calculate the covariance matrix with a higher order of Taylor expansion. <br />
7. The mocks and random of DESI can be accessed through this website https://zenodo.org/record/6640513. <br />
8. To copy this repository, run "git clone https://github.com/YanxiangL/Peculiar_velocity_fitting.git". <br />
9. The derivation of the analytical covariance matrix is given in this paper: https://arxiv.org/abs/2209.04166. Please cite this paper if you are using the code. 
