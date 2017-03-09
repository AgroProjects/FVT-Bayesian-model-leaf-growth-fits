------------------------------------------------------------------------------------------------------------
Requirements:
All scripts were written in the latest Python version 2.7.x.
Bayesian Statistical Modeling Python package (PyMC) version 2.3 was used to write and run the Bayesian models. 
------------------------------------------------------------------------------------------------------------

The number on the folders’ name indicates the steps in the Bayesian analysis. 
In each folder, there is a ReadMe.txt file that briefly describes the purpose of the Python codes.
 Comments were included in each Python code file to describe the purpose of the specific code block or fragment.      

“1_BayesianHierarchicalModel_MCMCcode” folder :  It contains codes to fit the Bayesian models with MCMC algorithms. User should look into this folder first. 

“2_DignosticCheck_n_Evaluation” folder: Once the MCMC run is completed, user can use some simple diagnostic methods written in a Python code file to evaluate the quality of the MCMC outputs.

“3_MakePlots_GrowthCurve_n_95CIbands” folder : If the quality of MCMC output is satisfactory, user can generate relevant plots for further analysis. The plots are leaf growth curves, posterior distributions of the model parameters, and the 95% credible and predictive bands.

“4_Bayesian_Model_Selection” folder : It contains python code to calculate the pseudo Bayes Factor (PBF), which is a method to compare two Bayesian models.