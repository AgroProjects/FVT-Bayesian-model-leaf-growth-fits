#-------------------------------------------------------------------------------
# Name:         Plot credible and predictive bands from the
#                     Posterior samples generated for the Logistic growth model
# Purpose:      Plotting 95% credible and predictive bands for the Logistic
#               growth model
#-------------------------------------------------------------------------------
import numpy as np
import math
import random
import matplotlib.pylab as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
from scipy.stats.mstats import mquantiles
import scipy.stats as ss
from matplotlib.backends.backend_pdf import PdfPages
import pymc     #pymc2.3 package
"""
This code aims to plot creadible band and Predictive band for the Logistic model
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
Path2 = "C:\\Users\\Leong\\Desktop\\Python_scripts\\Leaf_length_model\\Model2\\freeze4_OtherTest\\"
mcmcfile = "Model3_v1_BrapaData_QC_AWOLfix_2012_CR_LL_500000_thin20_QC_pred_ind_noAmax_CV.pickle"
#"Logistic_Modelv501_500000_thin20.pickle"

# The customized likelihood functionn: kernel density distribution
#   IF your likelihood function are standard probability distribution found in
#   the pymc package, you can delete the following 4 lines.
KDEfile    = "Residual_Y.npy"
# Set min and max residuals (x-axis) in the customized likelihood function
minv       = -14
maxv       = 14



#-----------------------------------
# Set fixed parameters for plots
#-----------------------------------
ymax      = 115.                     # Set maximum y-axis limit in matplotlib
xmin      = 0.                       # Set minimum value for X_data (Plotting)
xmax      = 1200.                    # Set maximum value for X_data (Plotting)
n_points  = 300                      # Set number of points-to-be randomly selected
xaxisname = "Time"                   # Set X axes label
yaxisname = "Y values"               # Set Y axes label

# The plots are converted to pdf format
pp        = PdfPages('Bayesian_Logistic_95Cre_Pred_Bands.pdf')


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
    remove_lines = ['206','89','340','21','360']
    for l in remove_lines:
        Data = Data[Data['Line']!=l]
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


# Create a list of color code for plotting the curves
colorVal = ['orchid','chocolate','darkolivegreen','orangered',\
    'darkturquoise','steelblue','tomato','goldenrod','purple','saddlebrown',\
    'slategray','magenta','darkgreen','darkcyan','sienna', \
    'navy','olive','maroon','darkorange','limegreen','darkkhaki',\
    'teal','aquamarine','red']

# Sort the Lines - Assuming the Lines are numbers (no Alphabets)
sortedlines = np.sort([int(l) for l in Lines])

# Generate x data value
X_data = np.arange(xmin,xmax)

for line in sortedlines:
    line    = str(line)
    str_ids = list(set(Data['ID'][Data['Line']==line]))
    ids     = np.sort([int(l) for l in str_ids])

    # Figure settings
    f = plt.figure(1)
    plt.title('95 Credible and Predictive Bands, Line %s, #plant=%s'%(line,str(len(ids))),fontsize=10)

    # Set up matrix to hold the curves generated using the posterior samples
    Ys = np.zeros((len(X_data),n_points*len(ids)))      # For COnfidence Band
    Yp = np.zeros((len(X_data),n_points*len(ids)))      # For Predictive Band

    # Randomly generate indeces for the posterior samples
    k_samples = np.array(random.sample(xrange(n_points), n_points))

    iter_cR = 0
    count = 0
    for i in ids:
        index_i = Data['ID']==str(i)

        for ks in k_samples:
            # Select posterior sample for r, L0, and Lmax (For Confidence Band)
            L0_ij = np.exp(M.e_L0_ij[str(i)].trace()[ks])
            r_ij = np.exp(M.e_r_ij[str(i)].trace()[ks])
            Lmax_ij = np.exp(M.e_Lmax_ij[str(i)].trace()[ks])

            # Caculate curve with the selected parameters (For Confidence Band)
            estY = Logistic_Curve(r_ij,Lmax_ij,L0_ij,t0,X_data)
            Ys[:,iter_cR] = estY

            # Select posterior predictive sample for r, L0, and Lmax (For Predictive Band)
            L0 = np.exp(M.predictive_e_L0_ij[str(i)].trace()[ks])
            r = np.exp(M.predictive_e_r_ij[str(i)].trace()[ks])
            Lmax = np.exp(M.predictive_e_Lmax_ij[str(i)].trace()[ks])

            # Caculate curve with the selected parameters (For Predictive Band)
            estY = Logistic_Curve(r,Lmax,L0,t0,X_data)
            Yp[:,iter_cR] = estY

            iter_cR += 1

        # Plot the measurement data
        plt.plot(Data['Time'][index_i],Data['Size'][index_i],'--o',ms= 4.5, color=colorVal[count],label=i)

        count +=1

    #----------------------------------------------------
    # Calculated the 95% predictive intervals
    #----------------------------------------------------
    lower_quant = mquantiles(Yp,prob=.025,axis=1)
    upper_quant = mquantiles(Yp,prob=.975,axis=1)

    # Plot the predictive band
    plt.fill_between(X_data,np.array(lower_quant[0:,0]),np.array(upper_quant[0:,0]),facecolor='lightgreen',alpha = 0.5)
    plt.plot(X_data, np.array(upper_quant[0:,0]),linestyle='-',color='lightgreen',linewidth=1)
    plt.plot(X_data, np.array(lower_quant[0:,0]),linestyle='-',color='lightgreen',linewidth=1)


    #----------------------------------------------------
    # Calculated the Means and 95% confidence intervals
    #----------------------------------------------------
    mean_Y = mquantiles(Ys,prob=.5,axis=1)
    lower_quant = mquantiles(Ys,prob=.025,axis=1)
    upper_quant = mquantiles(Ys,prob=.975,axis=1)

    # Plot the confidence band, including the mean
    plt.plot(X_data, list(mean_Y),linestyle='-',color='k',linewidth=1)
    plt.fill_between(X_data,np.array(lower_quant[0:,0]),np.array(upper_quant[0:,0]),facecolor='yellow',alpha = 0.5)
    plt.plot(X_data, np.array(upper_quant[0:,0]),linestyle='-',color='yellow',linewidth=1)
    plt.plot(X_data, np.array(lower_quant[0:,0]),linestyle='-',color='yellow',linewidth=1)

    # Add Common Title, X-label and Y-Label
    plt.ylim([0,ymax])
    plt.xlim([0,xmax+50])
    plt.legend(loc="upper center",ncol=5, mode="expand",fontsize=10)
    plt.xlabel('%s'%(xaxisname), fontsize=12)
    plt.ylabel('%s'%(yaxisname),fontsize=12)
    plt.grid()
    pp.savefig(f)
    plt.close()
    plt.clf()


pp.close()

