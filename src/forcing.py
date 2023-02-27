import numpy as np
import xarray as xr

from constants import ModelConstants

def create_forcing(object):

    object.z    = np.arange(-5000.,0,1)
    object.T0   = object.l1*object.S0+object.l2       # [degC] surface freezing temperature

    if object.forcop == "tanh":
        tanh(object)
    elif object.forcop == "linear":
        linear(object)
    elif object.forcop == "isomip":
        isomip(object)
    elif object.forcop == "linear2":
        linear2(object)    

    print(object.Tz)

    return

def tanh(object):
    drho = .01*np.abs(object.z)**.5

    object.Tz = object.T1 + (object.T0-object.T1) * (1+np.tanh((object.z-object.z0)/object.z1))/2
    object.Sz = object.S0 + object.alpha*(object.Tz-object.T0)/object.beta + drho/(object.beta*object.rho0)

    return 

def linear(object):
    object.Tz = object.T0 + object.z*(object.T1-object.T0)/object.z0 
    object.Sz = object.S0 + object.z*(object.S1-object.S0)/object.z0

    return

def linear2(object):
    if object.T1>object.T0:
        object.Tz = np.minimum(object.T0 + object.z*(object.T1-object.T0)/object.z0,object.T1)
    else:
        object.Tz = np.maximum(object.T0 + object.z*(object.T1-object.T0)/object.z0,object.T1)
    object.Sz = np.minimum(object.S0 + object.z*(object.S1-object.S0)/object.z0,object.S1)

    return


def isomip(object):
    #Hardcoded parameters
    z1 = -720
    T0 = -1.9
    S0 = 33.8
    if object.isomipcond == 'warm':
        T1 = 1.0
        S1 = 34.7
    elif object.isomipcond == 'cold':
        T1 = -1.9
        S1 = 34.55

    object.Tz = T0 + object.z*(T1-T0)/z1 
    object.Sz = S0 + object.z*(S1-S0)/z1

    return