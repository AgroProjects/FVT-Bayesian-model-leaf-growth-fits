#-------------------------------------------------------------------------------
# Name:      MCMC code to run the hierarchical Bayesian model for Logistic leaf
#               Growth model
# Purpose:  Estimate the Logistic curve parameters via Bayesian appraoch
#
#-------------------------------------------------------------------------------
import scipy.stats as ss
import scipy.interpolate as si
import numpy as np
import pymc      # pymc2.3 package


"""
This code runs the Bayesian Model for Logistic Growth function via MCMC with pymc2.3 package
"""
#-------------------------------------------------------------------------------------
# Provide input files
#   Datafile   - Contains measurement data And MAY/MAY NOT contain Line-specific Cofactors
#   KDEfile    - kernel density distribution saved in Binary file (.npy format)
#
# Output file: Provide a name for pickle
#   output_name - This binary file (pickle format) stores all the parameter values estimated via MCMC
#------------------------------------------------------------------------------------
Path        = "..\\"
# The measurement data file is in the following format:
#          Line	| ID	|  Time (in Degree Days) | Size | Cofactor
Datafile    = "LeafSizeData.csv"
output_name = "Logistic_Model3v1_500000_thin20.pickle"

# The customized likelihood functionn: kernel density distribution
#   IF your likelihood function are standard probability distribution found in
#   the pymc package, you can delete the following 4 lines.
KDEfile    = "Residual_Y.npy"
# Set min and max residuals (x-axis) in the customized likelihood function
minv       = -14
maxv       = 14



#-------------------------------------------------------------------------------------
# Set Options:
# Select whicn Bayesian Model, BModel:
#   Initial Modle for 2012 data = 0;
#   Initial Model with Cofactor for 2012 data = 1;
#   Initial model for 2011 data = 2
#
# Select which Co-factor (Amax or other indices), index:
#   NO coactor = 0
#   Line-Specific Amax = 1
#   Line-Specific Index = 2
#-------------------------------------------------------------------------------------
BModel = 2
index  = 0

if BModel == 0:     # Initial Modle for 2012 data
    from InitialModel_Logistic_Y2012_m3v1 import *
    #-------------------------------------------------------------
    # Set relevant parameters and assign values for Global priors
    #   Depend on the trait and year
    #-------------------------------------------------------------
    seed     = 0
    t0       = 80.98        # Set time at initial Leaf size, L0, i.e. the minimum measurement time
    # The values assigned to the GLobal prior for r. THe prior is modeled with Gaussian Distrbution
    r_0      = 0.02         # Set the mean in the Gaussian Distribution for r
    r0_var   = 3.           # Set the variance in the Gaussian Distribution for r

    # The values assigned to the GLobal prior for Lmax. THe prior is modeled with Gaussian Distrbution
    Lmax_0   = 50.          # Set the mean in the Gaussian Distribution for Lmax
    Lmax_var = 25.          # Set the variance in the Gaussian Distribution for Lmax

    # The values assigned to the Global prior for the nuisance parameter, initial leaf size L0,
    #       which its prior distribution is modeled with Uniform Distribution
    L0_0_l   = 0.0001       # Set lower bound of Uniform Distribution
    L0_0_u   = 10.          # Set upper bound of Uniform Distribution

    pprior = [r_0,r0_var,Lmax_0,Lmax_var,L0_0_l,L0_0_u]

    #----------------------------------------------------------------------------------
    # Read the measurement data
    #----------------------------------------------------------------------------------
    Data  = np.genfromtxt(Path+Datafile, dtype="S10, S10,f8, f8 ", delimiter=',', names=True)
    # Extract Line and Individual names
    Lines = list(set(Data['Line']))
    IDs   = list(set(Data['ID']))

elif BModel == 1:   #Initial Model with Cofactor for 2012 data
    from InitialModel_Logistic_cofactor_Y2012_m3v1 import *
    #-------------------------------------------------------------
    # Set relevant parameters and assign values for Global priors
    #   Depend on the trait and year
    #-------------------------------------------------------------
    seed     = 0
    t0       = 80.98        # Set time at initial Leaf size, L0, i.e. the minimum measurement time
    # The values assigned to the GLobal prior for r. THe prior is modeled with Gaussian Distrbution
    r_0      = 0.02         # Set the mean in the Gaussian Distribution for r
    r0_var   = 3.           # Set the variance in the Gaussian Distribution for r

    # The values assigned to the GLobal prior for Lmax. THe prior is modeled with Gaussian Distrbution
    Lmax_0   = 50.          # Set the mean in the Gaussian Distribution for Lmax
    Lmax_var = 25.          # Set the variance in the Gaussian Distribution for Lmax

    # The values assigned to the Global prior for the nuisance parameter, initial leaf size L0,
    #       which its prior distribution is modeled with Uniform Distribution
    L0_0_l   = 0.0001       # Set lower bound of Uniform Distribution
    L0_0_u   = 10.          # Set upper bound of Uniform Distribution

    # The values assigned to the br and bLmax parameter
    br_0        = 0.0       # Set the mean in the Gaussian Distribution for br
    br0_var     = 1.        # Set the variance in the Gaussian Distribution for br
    blmax_0     = 0.8       # Set the mean in the Gaussian Distribution for blmax
    blmax0_var  = 1.        # Set the variance in the Gaussian Distribution for blmax

    pprior = [r_0,r0_var,Lmax_0,Lmax_var,L0_0_l,L0_0_u,br_0,br0_var,blmax_0,blmax0_var]

    #----------------------------------------------------------------------------------
    # Read the measurement data
    #----------------------------------------------------------------------------------
    Data  = np.genfromtxt(Path+Datafile, dtype="S10, S10,f8, f8, f8", delimiter=',', names=True)
    # Extract Line and Individual names
    Lines = list(set(Data['Line']))
    IDs   = list(set(Data['ID']))

    #---------------------------------------------------------------------------
    # Pre-process the Line-Specific index values by apply log-transformation
    #   and then centered at zero
    #---------------------------------------------------------------------------
    if index == 1 :      # Amax
        takelog = {l: np.log(Data['Cofactor'][Data['Line']==l][0]) for l in Lines}
        logindex_l = {line:takelog[line] - np.mean([takelog[l] for l in takelog.keys()]) for line in takelog.keys()}

    elif index == 2:    # Other indices such as PRI and MTCI
        takelog = {l: np.log(0.01*np.exp(12.7*Data['Cofactor'][Data['Line']==l][0])) for l in Lines}
        logindex_l = {line:takelog[line] - np.mean([takelog[l] for l in takelog.keys()]) for line in takelog.keys()}

elif BModel == 2:   #Initial model for 2011 data
    from InitialModel_Logistic_Y2011_m3v1 import *
    #-------------------------------------------------------------
    # Set relevant parameters and assign values for Global priors
    #   Depend on the trait and year
    #-------------------------------------------------------------
    seed     = 0
    t0       = 80.98        # Set time at initial Leaf size, L0, i.e. the minimum measurement time
    # The values assigned to the GLobal prior for r. THe prior is modeled with Gaussian Distrbution
    r_0      = 0.02         # Set the mean in the Gaussian Distribution for r
    r0_var   = 3.           # Set the variance in the Gaussian Distribution for r

    # The values assigned to the GLobal prior for Lmax. THe prior is modeled with Gaussian Distrbution
    Lmax_0_l = 25.          # Set lower bound of Uniform Distribution for Lmax
    Lmax_0_u = 50.          # Set upper bound of Uniform Distribution for Lmax

    # The values assigned to the Global prior for the nuisance parameter, initial leaf size L0,
    #       which its prior distribution is modeled with Uniform Distribution
    L0_0_l   = 0.0001       # Set lower bound of Uniform Distribution
    L0_0_u   = 10.          # Set upper bound of Uniform Distribution

    pprior = [r_0,r0_var,Lmax_0_l,Lmax_0_u,L0_0_l,L0_0_u]

    #----------------------------------------------------------------------------------
    # Read the measurement data
    #----------------------------------------------------------------------------------
    Data  = np.genfromtxt(Path+Datafile, dtype="S10, S10,f8, f8 ", delimiter=',', names=True)
    # Extract Line and Individual names
    Lines = list(set(Data['Line']))
    IDs   = list(set(Data['ID']))

#----------------------------------------------------------------------------------
# Customized Likelihood function: kernel density distribution
#   Load the Residuals and set the min-max of the Residual distribution
#   Then, calculate the KDE for the distribution and use scipy.interpolate.splrep
#   function to find the B-spline representation of the KDE curve
    """
    IF your likelihood function are standard probability distribution found in
    the pymc package, you can the codes in this section BUT make sure you make
    necessary changes on the "InitialModel_Logistic_*_m3v1.py".
    """
#----------------------------------------------------------------------------------
Residual_Y = list(np.load(KDEfile))
rangm      = [minv,maxv]

# For Y-axis
KDE_Object_Y = ss.gaussian_kde(Residual_Y, bw_method='silverman')
ind          = np.linspace(minv,maxv,1024)
kdepdf       = KDE_Object_Y.evaluate(ind)
splrep_tck_Y = si.splrep(ind, kdepdf)

KDEminmax = [KDE_Object_Y.evaluate(minv),KDE_Object_Y.evaluate(maxv)]



#---------------------------------------------------------------------------------------------------------
# Create the "FVT_Bayesian_Model" model object (The Bayesian hierarchical Initial odel) for MCMC routine.
#   The code for the "FVT_Bayesian_Model" function is named "InitialModel_Logistic_m3v1.py"
#---------------------------------------------------------------------------------------------------------
if BModel == 1: # Initial Model with Cofactor
    model_objects = FVT_Bayesian_Model_cofactor(Data,Lines,IDs,logindex_l,t0,splrep_tck_Y,rangm,KDEminmax,KDE_Object_Y,pprior)
else:           # BModel = 0 or 2
    model_objects = FVT_Bayesian_Model(Data,Lines,IDs,t0,splrep_tck_Y,rangm,KDEminmax,KDE_Object_Y,pprior)

# The following syntaxs are to delete redundant data
del model_objects['Data']
del model_objects['Lines']
del model_objects['IDs']


#------------------------------------------------------------------------------
# The following lines set up the MCMC run and the all estimated parameters
#   are saved in a pickle. The MCMC sampler is Metropolis-Hastings algorithm.
# User need to set up the following variables:
#   iter = Total number of iterations
#   burn = number of iterations is dedicated for burn-in period
#   thin = Set the # of thinning to reduce autocorrelation and reduce
#          number of saved iterations
# pymc.assign_step_methods[AdaptiveMetropolis]: Set to use Adaptive Metropolis, which is
#                                                   better at handling highly-correlated variables
#------------------------------------------------------------------------------
pymc.numpy.random.seed(seed)
M = pymc.MCMC(model_objects,db = 'pickle',dbname = output_name)
# In case you want to use the Step_methods - Adjust the proposal density for e_r_ij
#   e_Lmax_ij and e_L0_ij at the same time.
for ind in IDs:
    M.assign_step_methods(pymc.AdaptiveMetropolis,[
        model_objects['e_r_ij']['%s'%ind],
        model_objects['e_Lmax_ij']['%s'%ind],
        model_objects['e_L0_ij']['%s'%ind]
        ])
#M.sample(iter=500000,burn=440000,thin=20)
M.sample(iter=100,burn=20,thin=2)
M.db.commit()
M.db.close()
