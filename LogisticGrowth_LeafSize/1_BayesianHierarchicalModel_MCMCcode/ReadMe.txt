------------------------------------------------------------------------------------------------------------
Requirements:
All scripts were written in the latest Python version 2.7.x.
Bayesian Statistical Modeling Python package (PyMC) version 2.3 was used to write and run the Bayesian models. 
------------------------------------------------------------------------------------------------------------

Short description of the python files:

Main_HBayesian.py : This is the main file where users specify filenames, and provide initial values for the required parameters for the Bayesian model. Run this code to fit the Bayesian model with MCMC algorithms. 

LogisticModel.py  : It contains function for the solution of the logistic differential equation.

InitialModel_Logistic_*_m3c1.py  : They are codes for the three-level Hierarchical Bayesian Initial Model. Their variations are due to slight modification in the Initial Model. 