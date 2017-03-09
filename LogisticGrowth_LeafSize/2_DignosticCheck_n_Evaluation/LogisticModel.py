#-------------------------------------------------------------------------------
# Name:        Logistic Growth Model
# Purpose:     Calculate the desired output values of a Logistic Growth Model
#
# Author:
#
# Created:     20/01/2014
# Copyright:   (c) 2014
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import numpy as np

"""
f(degree_days) = Logistic_Curve(r,Lmax,L0,t0,X):
    This Logistic function calculates the values on the Y-axis (Curve) from the
    input parameters and measured Time data (X-axis).
"""
def Logistic_Curve(r,Lmax,L0,t0,X):
    '''
    Input:
        X - Data on the X-axis in array or List format. In this case, it's measured Time data.
        r - growth rate parameter
        Lmax -The curve's maximum value
        L0 - Initial curve value
        t0 - The minimum measured time value (in degree days)

    Output:
        Curve - Calculated values on the Y-axis
    '''

    Curve = [((L0*Lmax)/(( (Lmax-L0)*np.exp(-(t-t0)*r) ) + L0)) for t in X]

    return np.array(Curve)
