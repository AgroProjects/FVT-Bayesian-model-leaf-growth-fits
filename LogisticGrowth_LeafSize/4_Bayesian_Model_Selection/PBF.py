
#-------------------------------------------------------------------------------
# Name:        Example: Criterion for Bayesian Model Selection for Logistoc model
# Purpose:
#
# Author:      welchlab1
#
# Created:     31/07/2013
# Copyright:   (c) welchlab1 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np
import pymc
import scipy.stats as ss
import scipy.interpolate as si

"""
This code calculate the pseudo-Bayes factor (PBF) using the pseudo-marginal
likelihood of the models (LPML) for the three-level hierarchical Bayesian Model.

Currently, we assumes there are 2 Bayesian Models for the Logistic growth model.
Both use the same measurement data and Likelihood Function.

You may have to customize the code for your Bayesian Models
"""

#-------------------------------------------------------------------------------------
# Set input files
#   Datafile   - Contains measurement data And MAY/MAY NOT contain Line-specific Cofactors
#   KDEfile    - kernel density distribution saved in Binary file (.npy format)
#
#   mcmcfile1 - This pickle stores all the parameter values estimated via MCMC for Bayesian Model 1
#   mcmcfile2 - This pickle stores all the parameter values estimated via MCMC for Bayesian Model 2
#
# Then, read the input files
#------------------------------------------------------------------------------------
Path        = "..\\"
# The measurement data file is in the following format:
#          Line	| ID	|  Time (in Degree Days) | Size | Cofactor
Datafile    = "LeafSizeData.csv"
mcmcfile1   = "Logistic_500000_thin20_BayesianModel1.pickle"
mcmcfile2   = "Logistic_500000_thin20_BayesianModel2.pickle"


# The customized likelihood functionn: kernel density distribution
#   IF your likelihood function are standard probability distribution found in
#   the pymc package, you can delete the following 4 lines.
KDEfile    = "Residual_Y.npy"
# Set min and max residuals (x-axis) in the customized likelihood function
minv       = -14
maxv       = 14



"""
BAYSIAN MODEL 1: Initial Modle for 2012 data
"""
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

pprior1 = [r_0,r0_var,Lmax_0,Lmax_var,L0_0_l,L0_0_u]


"""
BAYSIAN MODEL 2: Initial Model with "Cofactor" for 2012 data
"""
from InitialModel_Logistic_cofactor_Y2012_m3v1 import *
# Select which Co-factor (Amax or other indices), index:
#   Line-Specific Amax = 1
#   Line-Specific Index = 2
index  = 1
#-------------------------------------------------------------
# Set relevant parameters and assign values for Global priors
#   Depend on the trait and year
#-------------------------------------------------------------
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

pprior2 = [r_0,r0_var,Lmax_0,Lmax_var,L0_0_l,L0_0_u,br_0,br0_var,blmax_0,blmax0_var]

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



#----------------------------------------------------------------------------------
# Customized Likelihood function: kernel density distribution
#   Load the Residuals and set the min-max of the Residual distribution
#   Then, calculate the KDE for the distribution and use scipy.interpolate.splrep
#   function to find the B-spline representation of the KDE curve
# Assume Bayesian model1 and Bayesian model2 have the same Likehood function

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




"""
First, calculate the Log Pseudo-Marginal Likelihood (LPML) for the two Bayesian
Models, i.e., LPML1 and LPML2.

LPML = SUM_j=n ( log( (1/G) * SUM_i=G ( 1 / prob(y_j | theta_j, Model)) )^-1

Then, calculate the the log10 of pseudo-Bayes factor (PBF) for Models 1 and 2.
    log10(PBF12) = log10(exp(LPML1-LPML2)
"""

# Sort the Lines - Assuming the Lines are numbers (no Alphabets)
sortedlines = np.sort([int(l) for l in Lines])

LPML = [0,0]       # An empty List to store the LPML
for model in [0,1]:
    if model == 0:
        """
        BAYSIAN MODEL 1: Initial Modle for 2012 data
        """
        #---------------------------------------------------------------------------------------------------------
        # Create the "FVT_Bayesian_Model" model object (The Bayesian hierarchical Initial odel) for MCMC routine.
        #   The code for the "FVT_Bayesian_Model" function is named "InitialModel_Logistic_m3v1.py"
        #---------------------------------------------------------------------------------------------------------
        model_objects = FVT_Bayesian_Model(Data,Lines,IDs,t0,splrep_tck_Y,rangm,KDEminmax,KDE_Object_Y,pprior1)
        del model_objects['Data']
        del model_objects['Lines']
        del model_objects['IDs']

        #----------------------------------------------------------------------
        #  pymc load the model, the pickle file and retrive the saved data
        #----------------------------------------------------------------------
        Mdb = pymc.database.pickle.load(Path + mcmcfile1)
        M   = pymc.MCMC(model_objects,db=Mdb)
    else:
        """
        BAYSIAN MODEL 2: Initial Model with "Cofactor" for 2012 data
        """
        #------------------------------------------------------------------------------------------------------------------
        # Create the "FVT_Bayesian_Model_cofactor" model object (The Bayesian hierarchical Initial odel) for MCMC routine.
        #   The code for the "FVT_Bayesian_Model_cofactor" function is named "InitialModel_Logistic_cofactor_Y2012_m3v1.py"
        #------------------------------------------------------------------------------------------------------------------
        model_objects = FVT_Bayesian_Model_cofactor(Data,Lines,IDs,logindex_l,t0,splrep_tck_Y,rangm,KDEminmax,KDE_Object_Y,pprior2)
        del model_objects['Data']
        del model_objects['Lines']
        del model_objects['IDs']

        #----------------------------------------------------------------------
        #  pymc load the model, the pickle file and retrive the saved data
        #----------------------------------------------------------------------
        Mdb = pymc.database.pickle.load(Path + mcmcfile2)
        M   = pymc.MCMC(model_objects,db=Mdb)

    #----------------------------------------------------------------------
    # Routine to calculate the PBF
    #----------------------------------------------------------------------
    n_meanLk = []
    for line in sortedlines:
        str_ids = list(set(Data['ID'][Data['Line']==str(line)]))
        ids     = np.sort([int(l) for l in str_ids])
        G      = len(M.mu_r_0.trace()[:])

        for id in ids:
            index_i = Data['ID']==str(id)
            X       = Data['Time'][index_i]
            Y       = Data['Size'][index_i]

            for i in xrange(G):
                # Extract the parameters at sample-i
                L0_ij   = np.exp(M.e_L0_ij[str(id)].trace()[i])
                r_ij    = np.exp(M.e_r_ij[str(id)].trace()[i])
                Lmax_ij = np.exp(M.e_Lmax_ij[str(id)].trace()[i])

                est_curve = Logistic_Curve(r_ij,Lmax_ij,L0_ij,t0,X)

                # Calculate The Likelihood
                res         = est_curve-Y
                p           = np.array([float(KDEminmax[0]*0.99**((-r)-rangm[0])) if r <= rangm[0] else float(KDEminmax[1]*0.99**(r-rangm[1])) if r>= rangm[1] else float(si.splev(r,splrep_tck_Y)) for r in res])

                if i == 0:
                    L = 1./p
                else:
                    L += 1./p

            n_meanLk += list((1./G)*L)

    del model_objects, Mdb, M

    #---------------------------
    #   Calculate LPML
    #---------------------------
    CPO = 1./np.array(n_meanLk)
    LPML[model] = np.sum(np.log(CPO))

#---------------------------
#   Calculate LOG10(PB_12F)
#---------------------------
print " log10(PNF_12) = ", np.log10(np.exp(LPML[0]-LPML[1]))



