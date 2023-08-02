import sys
import numpy as np
import xarray as xr

from constants import ModelConstants

def read_forcing(object):
    #Called if forcing is provided from input file
    try:
        ds = xr.open_dataset(object.forcfile)

        object.z = ds.z
        object.Tz = ds.T
        object.Sz = ds.S

        ds.close()

    except:
        object.print2log(f"Error: could not open forcing input file {object.forcfile}. Check whether filename is correct and file contains variables z, T and S")
        sys.exit()

    #Do checks on validity

    object.forcname = object.forcfile

    object.print2log(f"Succesfully read forcing input file {object.forcfile}")

    return

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

    object.print2log(f"Finished creating forcing {object.forcname}")

    return

def tanh(object):
    drho = .01*np.abs(object.z)**.5

    object.Tz = object.T1 + (object.T0-object.T1) * (1+np.tanh((object.z-object.z0)/object.z1))/2
    object.Sz = object.S0 + object.alpha*(object.Tz-object.T0)/object.beta + drho/(object.beta*object.rho0)

    object.forcname = f"{object.forcop}_{object.T1:.1f}_{object.z0}"

    return 

def linear(object):
    object.Tz = object.T0 + object.z*(object.T1-object.T0)/object.z0 
    object.Sz = object.S0 + object.z*(object.S1-object.S0)/object.z0

    object.forcname = f"{object.forcop}_{object.T1:.1f}_{object.z0}"

    return

def linear2(object):
    if object.T1>object.T0:
        object.Tz = np.minimum(object.T0 + object.z*(object.T1-object.T0)/object.z0,object.T1)
    else:
        object.Tz = np.maximum(object.T0 + object.z*(object.T1-object.T0)/object.z0,object.T1)
    object.Sz = np.minimum(object.S0 + object.z*(object.S1-object.S0)/object.z0,object.S1)

    object.forcname = f"{object.forcop}_{object.T1:.1f}_{object.z0}"

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

    object.forcname = f"{object.forcop}_{object.isomipcond}"

    return