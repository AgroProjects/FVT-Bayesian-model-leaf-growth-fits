#-------------------------------------------------------------------------------
# Name:   Convergence Diagnostics
# Purpose:  Check if the MCMC chain has reached convergence after the run
#-------------------------------------------------------------------------------
import numpy as np
import scipy.stats as ss
from pymc.Matplot import plot as mcplot
import pymc     #pymc2.3 package

"""
Convergence Diagnostics after a MCMC run

After the MCMC run completed, you have to check if the chain has reached
convergence. Luckily, several diagnositc methods are avaible in the pymc package.
They are listed in this article
     https://pymc-devs.github.io/pymc/modelchecking.html

In this program. we apply popular and easy methods to help you quickly check your
MCMC outputs. The methods used here are:

1) Trace plot - A visual way to assess convergence, which is if the chain has
                reached stationarity. A chain that is converged indicate the
                well-mixed and has the characteristic of small fluctuation around
                a mean value.

2) Autocorrelation plot - Another quick way to check the degree of correlations
                          between long lags. High correlation indicates
                          poor mixing, which needs other ways to improve the run,
                          such as increase number of iterations or increase
                          thinning size, or reparametrization, or  revise your model

3) Raftery and Lewis Diagnostic - Returns the number of iterations, burn-in, and
                                  thinning needed to estimate the specific
                                  parameter's posterior cdf of the q-quantile
                                  to within +/- 0.01 accuracy with
                                  probability 0.95 of attaining this level
                                  of accuracy.
                                - Estimate the Dependence Factor. Dependence
                                  Factor > 5 indicates strong autocorrelation
                                  and convergence failure, which indicate the
                                  model needs revision or need reparametrization.
"""

#-------------------------------------------------------------------------------------
# Set input files
#   Datafile   - Contains measurement data And MAY/MAY NOT contain Line-specific Cofactors
#   KDEfile    - kernel density distribution saved in Binary file (.npy format)
#   mcmcfile -   pickle file stores all the parameter values estimated via MCMC
#------------------------------------------------------------------------------------
Path     = "..\\"
#The measurement data file is in the following format:
#          Line	| ID	|  Time (in Degree Days) | Size | Cofactor
Datafile = "BrapaData_QC_AWOLfix_2012_CR_LL_QC.csv" #"LeafSizeData.csv"
mcmcfile = "Logistic_Modelv501_500000_thin20.pickle"

# The customized likelihood functionn: kernel density distribution
#   IF your likelihood function are standard probability distribution found in
#   the pymc package, you can delete the following 4 lines.
KDEfile    = "Residual_Y.npy"
# Set min and max residuals (x-axis) in the customized likelihood function
minv       = -14
maxv       = 14


#--------------------------------------------------------
# Set fixed parameters for Raftery and Lewis Diagnostic
#--------------------------------------------------------
quantile = 0.975
accuracy = 0.01
v        = 0


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
BModel = 0
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


#----------------------------------------------------------------------
#  pymc load the model, the pickle file and retrive the saved data
#----------------------------------------------------------------------
Mdb = pymc.database.pickle.load(Path + mcmcfile)
M   = pymc.MCMC(model_objects,db=Mdb)




"""
A. Generate composite figure with trace plot, autocorrelation plot, and
   posterior distribution for the Global-specific parameters. The figure is
   displayed and also automatically saved in your current folder.

   For Example:
   The following few lines of code generate the composite figures for the
   globalprameters. You can add more lines for other parameters.
"""
print " Generating Trace plots ,autocorrelation plots, posterior distribution ..... "


mcplot(M.trace("mu_r_G"),common_scale=False)
mcplot(M.trace("mu_Lmax_G"),common_scale=False)
mcplot(M.trace("mu_L0_G"),common_scale=False)
mcplot(M.trace('tau_r_line'),common_scale=False)
mcplot(M.trace('tau_r_indv'),common_scale=False)
mcplot(M.trace('tau_Lmax_line'),common_scale=False)
mcplot(M.trace('tau_Lmax_indv'),common_scale=False)
mcplot(M.trace('tau_L0_line'),common_scale=False)
mcplot(M.trace('tau_L0_indv'),common_scale=False)
plt.show()

# NOTE: You can print and save the summary of the parameters' posterior stats.
#       To print the summary stats for a global parameter, "mu_r_0", here is the
#       example syntax:
#
#       M.mu_r_0.summary()
#
#       To save the summary stats for global parameter in a .csv file, here is
#       the example syntax:
#
#       M.write_csv("globalparam_posStats.csv", variables=["mu_L0_0", "mu_r_0", "mu_Lmax_0"])
#



"""
B. Raftery and Lewis Diagnostic
Calculates the # of iterations needed to estimate the posterior
cdf of the q-quantile to within +/- r accuracy with probability s

# From StatLib -- gibbsit.f
def raftery_lewis(x, q, r, s=.95, epsilon=.001, verbose=1):

    Return the number of iterations needed to achieve a given
    precision.

    :Parameters:
        x : sequence
            Sampled series.
        q : float
            Quantile.
        r : float
            Accuracy requested for quantile.
        s (optional): float
            Probability of attaining the requested accuracy (defaults to 0.95).
        epsilon (optional) : float
             Half width of the tolerance interval required for the q-quantile (defaults to 0.001).
        verbose (optional) : int
            Verbosity level for output (defaults to 1).

    :Return:
        nmin : int
            Minimum number of independent iterates required to achieve
            the specified accuracy for the q-quantile.
        kthin : int
            Skip parameter sufficient to produce a first-order Markov
            chain.
        nburn : int
            Number of iterations to be discarded at the beginning of the
            simulation, i.e. the number of burn-in iterations.
        nprec : int
            Number of iterations not including the burn-in iterations which
            need to be obtained in order to attain the precision specified
            by the values of the q, r and s input parameters.
        kmind : int
            Minimum skip parameter sufficient to produce an independence
            chain.

    :Example:
        >>> raftery_lewis(x, q=.025, r=.005)

    :Reference:
        Raftery, A.E. and Lewis, S.M. (1995).  The number of iterations,
        convergence diagnostics and generic Metropolis algorithms.  In
        Practical Markov Chain Monte Carlo (W.R. Gilks, D.J. Spiegelhalter
        and S. Richardson, eds.). London, U.K.: Chapman and Hall.

        See the fortran source file `gibbsit.f` for more details and references.

    output = nmin, kthin, nburn, nprec, kmind = pymc.flib.gibbmain(x, q, r, s, epsilon)

    if verbose:

        print "========================"
        print "Raftery-Lewis Diagnostic"
        print "========================"
        print
        print "%s iterations required (assuming independence) to achieve %s accuracy with %i percent probability." % (nmin, r, 100*s)
        print
        print "Thinning factor of %i required to produce a first-order Markov chain." % kthin
        print
        print "%i iterations to be discarded at the beginning of the simulation (burn-in)." % nburn
        print
        print "%s subsequent iterations required." % nprec
        print
        print "Thinning factor of %i required to produce an independence chain." % kmind

Read : Raftery and Lewis, StatSci 1992.pdf for I = measures the increase in the number of iterations due to dependence in the sequence
"""
for s in list(M.stochastics):
    if 'predictive' in str(s) : continue
    elif 'mu_r_i' in str(s) or 'mu_L0_i' in str(s) or 'mu_Lmax_i' in str(s) :
        print "Line    : ", str(s)
        pymc.raftery_lewis(s,q=quantile, r=accuracy)
        Nmin, kthin, Nburn, Nprec, Kmind = pymc.raftery_lewis(s,q=quantile, r=accuracy, verbose=0)
        print "  Calculated Dependence Factor = ", Nprec/float(Nmin)
        print "-----------------------------------"
        # NOTE: if there are too many parameters, you may need to save them in a .csv file
    elif 'e_' in str(s):
        print "Plant ID : ", str(s)
        pymc.raftery_lewis(s,q=quantile, r=accuracy)
        Nmin, kthin, Nburn, Nprec, Kmind = pymc.raftery_lewis(s,q=quantile, r=accuracy, verbose=0)
        print "  Calculated Dependence Factor = ", Nprec/float(Nmin)
        print "-----------------------------------"
        # NOTE: if there are too many parameters, you may need to save them in a .csv file
    else:
        print "Global parameters :",str(s)
        pymc.raftery_lewis(s,q=quantile, r=accuracy)
        Nmin, kthin, Nburn, Nprec, Kmind = pymc.raftery_lewis(s,q=quantile, r=accuracy, verbose=0)
        print "  Calculated Dependence Factor = ", Nprec/float(Nmin)
        print "-----------------------------------"
        # NOTE: if there are too many parameters, you may need to save them in a .csv file

