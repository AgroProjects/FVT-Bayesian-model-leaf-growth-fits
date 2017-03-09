#-------------------------------------------------------------------------------
# Name:        Three-Level hierarchical Bayesian Model for Logistic Leaf growth
# Purpose:     THe Initial Model for 2012 data
#-------------------------------------------------------------------------------
import pymc
import numpy as np
import math
import scipy.interpolate as si
from LogisticModel import *
import random

"""
Coding the Bayesian model for Logistic Leaf Gwoth model with pymc ersion 2.3
"""

def FVT_Bayesian_Model(Data,Lines,IDs,t0,splrep_tck_Y,rangm,KDEminmax,KDE_Object_Y,pprior):
    #------------------------------------------------------------------------------------------------------
    # Submodel for parameter r
    #   Set the prior distributions for the global-specific parameters of r
    # ------------------------------------------------------------------------------------------------------
    prior_mu_r_0 = math.log(pprior[0])
    prior_variance_mu_r_0 = math.log(pprior[1])
    tau_r_alpha = 2.
    tau_r_beta = 0.3

    mu_r_G = pymc.Normal(name='mu_r_G',mu = prior_mu_r_0,tau = 1./prior_variance_mu_r_0)
    tau_r_line = pymc.Gamma(name='tau_r_line',alpha = tau_r_alpha, beta = tau_r_beta)
    tau_r_indv = pymc.Gamma(name='tau_r_indv',alpha = tau_r_alpha, beta = tau_r_beta)

    #------------------------------------------------------------------------------------------------------
    # Set the prior distributions for line-specific parameters of r
    # ------------------------------------------------------------------------------------------------------
    mu_r_i = {}
    for l in Lines:
        mu_r_i[l] = pymc.Normal(name='mu_r_i_%s' % l, mu = mu_r_G, tau = tau_r_line)

    #------------------------------------------------------------------------------------------------------
    # Set the prior distributions for Individual-specific parameters of r
    # ------------------------------------------------------------------------------------------------------
    e_r_ij ={}
    for i in IDs:
        l = Data['Line'][Data['ID']==i][0]
        e_r_ij[i] = pymc.Normal(name='e_r_ij_%s' % i, mu = mu_r_i[l], tau = tau_r_indv)

    #--------------------------------------------------------------------------
    #  This function calculates the parameter r_ij from e_r_ij
    #--------------------------------------------------------------------------
    def rij_eval(e_ij = e_r_ij[IDs[0]],l=Lines[0]):
        return math.exp(e_ij)

    r_ij = {}
    for i in IDs:
        l = Data['Line'][Data['ID']==i][0]
        r_ij[i] = pymc.Deterministic(eval = rij_eval,
                            name = 'r_ij_%s' % i,
                            parents = {'e_ij': e_r_ij[i],
                                       'l': l
                                        },
                            doc='individual r_ij value',
                            trace = False,
                            verbose=0,dtype=float,plot=False,cache_depth = 2)


    #------------------------------------------------------------------------------------------------------
    # Submodel for parameter Lmax
    #   Set the prior distributions for the global-specific parameters of Lmax
    # ------------------------------------------------------------------------------------------------------
    prior_variance_mu_Lmax_0 = math.log(pprior[2])
    prior_mu_Lmax_0 = math.log(pprior[3])
    tau_Lmax_alpha = 2.
    tau_Lmax_beta = 0.3

    mu_Lmax_G = pymc.Normal(name='mu_Lmax_G',mu=prior_mu_Lmax_0,tau = 1./prior_variance_mu_Lmax_0)
    tau_Lmax_line = pymc.Gamma(name='tau_Lmax_line',alpha = tau_Lmax_alpha, beta = tau_Lmax_beta)
    tau_Lmax_indv = pymc.Gamma(name='tau_Lmax_indv',alpha = tau_Lmax_alpha, beta = tau_Lmax_beta)

    #------------------------------------------------------------------------------------------------------
    # Set the prior distributions for line-specific parameters of Lmax
    # ------------------------------------------------------------------------------------------------------
    mu_Lmax_i = {}
    for l in Lines:
        mu_Lmax_i[l] = pymc.Normal(name='mu_Lmax_i_%s' % l, mu = mu_Lmax_G, tau = tau_Lmax_line)

    #------------------------------------------------------------------------------------------------------
    # Set the prior distributions for Individual-specific parameters of Lmax
    # ------------------------------------------------------------------------------------------------------
    e_Lmax_ij ={}
    for i in IDs:
        l = Data['Line'][Data['ID']==i][0]
        e_Lmax_ij[i] = pymc.Normal(name='e_Lmax_ij_%s' % i, mu = mu_Lmax_i[l], tau = tau_Lmax_indv)

    #--------------------------------------------------------------------------
    #  This function calculates the parameter Lmax_ij from e_Lmax_ij
    #--------------------------------------------------------------------------
    def Lmax_ij_eval(e_ij = e_Lmax_ij[IDs[0]],l=Lines[0]):
        return math.exp(e_ij)

    Lmax_ij = {}
    for i in IDs:
        l = Data['Line'][Data['ID']==i][0]
        Lmax_ij[i] = pymc.Deterministic(eval = Lmax_ij_eval,
                            name = 'Lmax_ij_%s' % i,
                            parents = {'e_ij': e_Lmax_ij[i],
                                       'l': l
                                        },
                            doc='individual Lmax value',
                            trace = False,
                            verbose=0,dtype=float,plot=False,cache_depth = 2)


    #------------------------------------------------------------------------------------------------------
    #   Submodel for nuisance parameter - initial leaf size, L0
    # Set the prior distributions for the global-specific parameters of L0
    # ------------------------------------------------------------------------------------------------------
    prior_mu_L0_0_lower = math.log(pprior[4])
    prior_mu_L0_0_upper = math.log(pprior[5])
    tau_L0_alpha = 2.
    tau_L0_beta = 0.3

    mu_L0_G = pymc.Uniform(name='mu_L0_G',lower=prior_mu_L0_0_lower, upper=prior_mu_L0_0_upper)
    tau_L0_line = pymc.Gamma(name='tau_L0_line',alpha = tau_L0_alpha, beta = tau_L0_beta)
    tau_L0_indv = pymc.Gamma(name='tau_L0_indv',alpha = tau_L0_alpha, beta = tau_L0_beta)

    #------------------------------------------------------------------------------------------------------
    # Set the prior distributions for line-specific parameters of L0
    # ------------------------------------------------------------------------------------------------------
    mu_L0_i = {}
    for l in Lines:
        mu_L0_i[l] = pymc.Normal(name='mu_L0_i_%s' % l, mu = mu_L0_G, tau = tau_L0_line)

    #------------------------------------------------------------------------------------------------------
    # Set the prior distributions for Individual-specific parameters of L0
    # ------------------------------------------------------------------------------------------------------
    e_L0_ij ={}
    for i in IDs:
        l = Data['Line'][Data['ID']==i][0]
        e_L0_ij[i] = pymc.Normal(name='e_L0_ij_%s' % i, mu = mu_L0_i[l], tau = tau_L0_indv)

    #-----------------------------------------------------------------------------------------------
    #  This function calculates the parameter L0_ij from e_L0_ij (exponentiated the log L0 values)
    #-----------------------------------------------------------------------------------------------
    def L0_eval(e_L0_ij = e_L0_ij[IDs[0]],l=Lines[0]):
        return math.exp(e_L0_ij)

    L0_ij = {}
    for i in IDs:
        l = Data['Line'][Data['ID']==i][0]
        L0_ij[i] = pymc.Deterministic(eval = L0_eval,
                         name = 'L0_ij_%s' % i,
                         parents = {'e_L0_ij': e_L0_ij[i],
                                    'l': l
                                     },
                         doc='individual L0_ij value',
                         trace = False,
                         verbose=0,dtype=float,plot=False,cache_depth = 2)



    #-------------------------------------------------------------------------------------------------------------
    # ResidualModel() is a function block that contains code to calculate the
    #           log-likelihoods using a Customized likelihood function, i.e., kernel density distribution
    """
      NOTE: DELETE this function block IF your likelihood function are standard probability distribution
            found in the pymc package.
    """
    #-------------------------------------------------------------------------------------------------------------
    def ResidualModel(name,value,predicted,splrep_tck_Y,rangm,KDEminmax,KDE_Object_Y,observed=False,trace=False):
        def logp(value,predicted,rangm,splrep_tck_Y,KDEminmax,KDE_Object_Y):
            res = predicted-value
            p = np.array([float(KDEminmax[0]*0.99**((-r)-rangm[0])) if r <= rangm[0] else float(KDEminmax[1]*0.99**(r-rangm[1])) if r>= rangm[1] else float(si.splev(r,splrep_tck_Y)) for r in res])

            try:        # Check to make sure there in illogical values
                if math.isnan(sum(np.log(p))) == True or math.isinf(sum(np.log(p))) == True:
                    logp = float(-1.7976931348623157e+300)
                else:
                    logp = sum(np.log(p))
            except:
                print p, sum(p)

            return logp

        def random(predicted,KDE_Object_Y):
            result = KDE_Object_Y.resample(len(predicted))[0][0]
            print KDE_Object_Y
            return result

        result = pymc.Stochastic(logp = logp,
                      name =  name,
                      parents = {'predicted' : predicted,
                                 'splrep_tck_Y' : splrep_tck_Y,
                                 'rangm' : rangm,
                                 'KDEminmax' : KDEminmax,
                                 'KDE_Object_Y':KDE_Object_Y
                                },
                      value = value,
                      random = random,
                      observed = observed,
                      doc='Likelihood Probability leaf lengths P(L|L_hat)= P(Residuals)',
                      trace = trace,
                      verbose=0,dtype=float,plot=False,cache_depth = 2)
        return result

    #--------------------------------------------------------------------------------------------------
    # This section calculate the predicted Y values from the proposed parameters, r, Lmax & L0
    #--------------------------------------------------------------------------------------------------
    loglike = {}
    predicted_Y = {}
    for i in IDs:
        index_i = Data['ID']==i
        l = Data['Line'][Data['ID']==i][0]

        predicted_Y[i] = pymc.Deterministic(eval =  Logistic_Curve,
							name = 'predicted_size_%s' % i,
							parents = {'r': r_ij[i],
									   'Lmax': Lmax_ij[i],
									   'L0': L0_ij[i],
                                       't0': t0,
									   'X':Data['Time'][index_i]
										},
							doc='predicted size',
							trace = True,
							verbose=0,dtype=float,plot=False,cache_depth = 2)


        #------------------------------------------------------------------------------
        # ResidualModel() is a Customized Likelihood Function, i.e., kernel density distribution:
        #        To Evaluate the proposed parameters and the log likelihood values for the
        #        predicted Y values are calculated
        #  REPLACE this function with your own likelihood function
        """
           NOTE: Likelihood function depends on your problem that you're solving.
                 If your likelihood function is standard distributions,
                 e.g. Gaussian, Poisson. etc, then you can use these standard
                 probability distributions that are in the pymc package. You
                 ONLY need to customized your likelihood if you cannot find any
                 standard distributions in the package.
        """
        #-------------------------------------------------------------------------------
        loglike[i] = ResidualModel('loglike_%s'%i,Data['Size'][index_i],predicted_Y[i],splrep_tck_Y,rangm,KDEminmax,KDE_Object_Y,observed=True,trace=True)


    #--------------------------------------------------------------------------------------
    # This few lines of code is added to sample for Predictive Prosterior Distributions
    #--------------------------------------------------------------------------------------
    predictive_e_r_ij = {}
    predictive_e_Lmax_ij = {}
    predictive_e_L0_ij = {}
    for i in IDs:
        predictive_e_r_ij[i] = pymc.Normal( "predictive_e_r_ij_%s"% i, mu = e_r_ij[i], tau = tau_r_indv)                # For r
        predictive_e_Lmax_ij[i] = pymc.Normal( "predictive_e_Lmax_ij_%s"% i, mu = e_Lmax_ij[i], tau = tau_Lmax_indv)    # For Lmax
        predictive_e_L0_ij[i] = pymc.Normal( "predictive_e_L0_ij_%s"% i, mu = e_L0_ij[i], tau = tau_L0_indv)            # For nuisance parameter L0

    return locals()
