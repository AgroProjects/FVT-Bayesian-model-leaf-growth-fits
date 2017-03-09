#-------------------------------------------------------------------------------
# Name:        Plot leaf growth curves and model parameters' histograms from
#                   their posterior distributions
# Purpose:     Making relevant plots and calculate estimated parameters
#              after completing a MCMC run.
#-------------------------------------------------------------------------------
import numpy as np
import scipy.stats as ss
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import pymc     #pymc2.3 package

"""
This code aims to plot Logistic growth curves and the model parameters'
    prosterior distributions (i.e., the histograms), and to output the estimates
    of individual plants (i.e., r, Lmax, d, iD).

    # Duration, d, is the time at which 95% of the Lmax
    # Parameter iD is the time (degree days) when growth curves reach the
        inflection point. Inflection point = Lmax/2
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
Datafile = "LeafSizeData.csv"
mcmcfile = "Logistic_Modelv501_500000_thin20.pickle"

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
xaxisname = "Degree Days"       # Set X axes label (Time)
yaxisname = "Leaf Size"         # Set Y axes label (Either Leaf Length or Leaf Width)
ymax      = 115.                # Set maximum y-axis limit in matplotlib
xmin      = 0.                  # Set minimum value for X_data (For lotting in Degree Days)
xmax      = 1200.               # Set maximum value for X_data (For lotting in Degree Days)
bracketr  = [0.,0.05]           # Set histogram "range" variable for parameter r
bracketm  = [10.,100.]          # Set histogram "range" variable for parameter Lmax
bracketl  = [0.,25.]            # Set histogram "range" variable for parameter L0
binsize   = 50                  # Set bin size for Histogram
ncol      = 3                   # Set axes array for Hisrogram plots: # of columns
nrow      = 2                   # Set axes array for Hisrogram plots: # of rows
cutOff    = 0.95                # Set the percentage of the final leaf size (Lmax) was reached


#-------------------------------------------
# Provide filenames for pdfs and.csv file
#-------------------------------------------
# The plots for growth curves and parameters' posterior distributions
#       are converted to pdf files
ppB            = PdfPages('Bayesian_LogisticCurve.pdf')
ppH            = PdfPages('Bayesian_LogisticHistograms.pdf')

# The estimated parameters are recorded in a csv file
outputfilename = "Bayesian_Logistic_Est_Parameter.csv"
ofile          = open(outputfilename, "wb")
file_writer    = csv.writer(ofile)
file_writer.writerow(["Line","ID","r","Lmax","d","iD"])



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



# Create a list of color code for plotting the curves
colorVal = ['orchid','chocolate','darkolivegreen','orangered',\
    'darkturquoise','steelblue','tomato','goldenrod','purple','saddlebrown',\
    'slategray','magenta','darkgreen','darkcyan','sienna', \
    'navy','olive','maroon','darkorange','limegreen','darkkhaki',\
    'teal','aquamarine','red']

# Sort the Lines - Assuming the Lines are numbers (no Alphabets)
sortedlines = np.sort([int(l) for l in Lines])


"""
Plotting curves and record the parameter means (i.e., the average of the estimated parameters)
"""
# Generate x data value (in Degree Days)
X_data = np.arange(xmin,xmax)
X_data_hi = np.arange(xmin,xmax,0.5)#0.002)

for line in sortedlines:
    count   = 0
    str_ids = list(set(Data['ID'][Data['Line']==str(line)]))
    ids     = np.sort([int(l) for l in str_ids])

    # Figure settings
    f = plt.figure(1)
    plt.title('Line %s'%(line),fontsize=10)

    for i in ids:
        index_i = Data['ID']==str(i)
        X = Data['Time'][index_i]
        Y = Data['Size'][index_i]

        # Calculate the mean for parameters and plot the estimated curve
        rij_mean = np.mean(np.exp(M.e_r_ij[str(i)].trace()[:]))
        Lmaxij_mean = np.mean(np.exp(M.e_Lmax_ij[str(i)].trace()[:]))
        L0ij_mean = np.mean(np.exp(M.e_L0_ij[str(i)].trace()[:]))

        est_curve = Logistic_Curve(rij_mean,Lmaxij_mean,L0ij_mean,t0,X_data)

        # Plot the measurement data
        plt.plot(X,Y,'o',color=colorVal[count],label=int(i))
        # Plot the estimated curve
        plt.plot(X_data,est_curve,linestyle='-',color=colorVal[count],linewidth=1.5)

        # Calculate Time at Inflection point (iD)
        iD = t0 + (np.log((Lmaxij_mean-L0ij_mean)/L0ij_mean)/rij_mean)

        # Calulcate the duration (d)
        mcurve = Logistic_Curve(rij_mean,Lmaxij_mean,L0ij_mean,t0,X_data_hi)
        Lmax_Threshold = cutOff*Lmaxij_mean
        xx=[x for x in mcurve if x <= Lmax_Threshold]
        d = round(X_data_hi[len(xx)-1],2)

        # Record the calculated parameter values
        file_writer.writerow([line,i,rij_mean,Lmaxij_mean,d,iD])

        count+= 1

    # Set Y-axis limit; Add Common Title, X-label and Y-Label
    # Save plots
    plt.ylim([0,ymax])
    plt.xlim([0,xmax])
    plt.legend(loc="upper center",ncol=5, mode="expand",fontsize=10)
    plt.xlabel('%s'%(xaxisname),fontsize=12)
    plt.ylabel('%s'%(yaxisname),fontsize=12)
    plt.grid()

    ppB.savefig(f)
    plt.close()
    plt.clf()


# Close pdf and csv files
ppB.close()
ofile.close()


"""
Plotting parameters' prosterior distributions (i.e., the histograms)
"""

for j in np.arange(int(np.round(float(len(sortedlines)/nrow)))):
    lines    = sortedlines[j*nrow:nrow*(j+1)]
    f, axarr = plt.subplots(nrow, ncol, sharex='col', sharey='row',figsize=(20,15))
    count    = 0
    for line in lines:
        str_ids = list(set(Data['ID'][Data['Line']==str(line)]))
        ids     = np.sort([int(l) for l in str_ids])


        for h in np.arange(ncol):
            # Keep track of the color codes
            color_count = 0
            # Determine which axes to plot
            axis = axarr[count // ncol,count % ncol ]

            # Plot the parameters' prosterior distributions
            if h%ncol == 0:     # For Parameter r
                for i in ids:
                    axis.hist(np.exp(M.e_r_ij[str(i)].trace()[:]), bins=binsize, range=bracketr,facecolor=colorVal[color_count])
                    color_count += 1
                axis.set_title("line %s: Parameter r"%(str(line)),fontsize=14)

            elif h%ncol == 1:   # For Parameter Lmax
                for i in ids:
                    axis.hist(np.exp(M.e_Lmax_ij[str(i)].trace()[:]), bins=binsize, range=bracketm, facecolor=colorVal[color_count],label=str(i))
                    color_count += 1
                axis.set_title("line %s: Parameter Lmax"%(str(line)),fontsize=14)
                axis.legend(loc="upper center",ncol=5, mode="expand",fontsize=10)

            elif h%ncol == 2:
                for i in ids:
                    axis.hist(np.exp(M.e_L0_ij[str(i)].trace()[:]), bins=binsize, range=bracketl, facecolor=colorVal[color_count])
                    color_count += 1
                axis.set_title("line %s: Parameter L0"%(str(line)),fontsize=14)

            count+= 1

    # Add Common Title, X-label and Y-Label; Save plots
    big_ax = f.add_subplot(111)
    big_ax.set_axis_bgcolor('none')
    big_ax.spines['top'].set_color('none')
    big_ax.spines['bottom'].set_color('none')
    big_ax.spines['left'].set_color('none')
    big_ax.spines['right'].set_color('none')
    big_ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    big_ax.set_xlabel('Parameters', fontsize=16)
    big_ax.set_ylabel('Counts',fontsize=16)
    f.suptitle('Prosterior Distributions',fontsize=16)
    ppH.savefig(f)
    plt.close()
    plt.clf()


# Close pdf file
ppH.close()
