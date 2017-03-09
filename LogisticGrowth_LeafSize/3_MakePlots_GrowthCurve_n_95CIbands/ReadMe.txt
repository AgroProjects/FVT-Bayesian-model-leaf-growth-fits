------------------------------------------------------------------------------------------------------------
Requirements:
All scripts were written in the latest Python version 2.7.x.
Bayesian Statistical Modeling Python package (PyMC) version 2.3 was used to write and run the Bayesian models. 
------------------------------------------------------------------------------------------------------------

Short description of the python files:

Plot_Growthcurves_histograms_csv.py : User can use this code to generate growth curves, posterior distributions (histogram) plots from the MCMC outputs (i.e, binary file in pickle format). A csv file with all the estimated parameters for the individual plants will also generated. 

Plot_95_CI_n_PI.py: This code allows user to generate growth curves plots with 95% credible and predictive interval bands. 


LogisticModel.py  : It contains function for the solution of the logistic differential equation.

InitialModel_Logistic_*_m3c1.py  : They are codes for the three-level Hierarchical Bayesian Initial Model. Their variations are due to slight modification in the Initial Model. 