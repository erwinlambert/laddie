import sys
import numpy as np
import xarray as xr

def read_forcing(object):
    """Read external forcing file"""

    try:
        ds = xr.open_dataset(object.forcfile)

        #Read temperature and salinity fields or profiles
        object.z = ds.z
        object.Tz = ds.T.values
        object.Sz = ds.S.values

        #Extract required info on x,y-dimensions of 3D external 
        if len(object.Tz.shape)==3:
            object.Tax1 = np.arange(object.Tz.shape[1])[:,None]
            object.Tax2 = np.arange(object.Tz.shape[2])[:,None]

        ds.close()

    except:
        object.print2log(f"Error: could not open forcing input file {object.forcfile}. Check whether filename is correct and file contains variables z, T and S")
        sys.exit()

    #Do checks on validity
    check_inputforcing(object)

    #Save forcing name to store in output
    object.forcname = object.forcfile

    object.print2log(f"Succesfully read forcing input file {object.forcfile}")

    return

def check_inputforcing(object):
    """Check validity of input """

    #Check whether step in z is 1 meter throughout, required for interpolation to find Ta and Sa. More flexibility or pre-interpolation to be added
    dz = object.z[1:]-object.z[:-1]
    if (dz != 1).any():
        print('FORCING ERROR: input variable z must be incremental with steps 1')
        sys.exit()

    object.zmin = min(object.z)
    """To be expanded """

    object.print2log(f"Finished checking forcing. All OK")

    return

def create_forcing(object):
    """Create internal 1D forcing based on provided parameters"""

    #Create depth dimension
    object.z    = np.arange(-5000.,0,1)

    #Call forcing-specific function
    if object.forcop == "tanh":
        tanh(object)
    elif object.forcop == "linear":
        linear(object)
    elif object.forcop == "isomip":
        isomip(object)
    elif object.forcop == "linear2":
        linear2(object)    

    #Do checks on validity
    check_inputforcing(object)

    object.print2log(f"Finished creating forcing {object.forcname}")

    return

def tanh(object):
    """Tangent hyperbolic function representing a two-layer ambient profile separated by a smooth thermocline"""

    #Surface freezing temperature
    object.T0   = object.l1*object.S0+object.l2

    #Quadratic density profile
    drho = object.drho0*np.abs(object.z)**.5

    #Temperature profile: smooth thermocline separating two layers
    object.Tz = object.T1 + (object.T0-object.T1) * (1+np.tanh((object.z-object.z0)/object.z1))/2

    #Salinity profile, compensating to maintain prescribed density profile
    object.Sz = object.S0 + object.alpha*(object.Tz-object.T0)/object.beta + drho/(object.beta*object.rho0)

    #Save forcing name to store in output
    object.forcname = f"{object.forcop}_{object.T1:.1f}_{object.z0}"

    return 

def linear(object):
    """Linear profile in both temperature and salinity"""

    #Surface freezing temperature
    object.T0   = object.l1*object.S0+object.l2

    #Temperature profile
    object.Tz = object.T0 + object.z*(object.T1-object.T0)/object.z0

    #Salinity profile
    object.Sz = object.S0 + object.z*(object.S1-object.S0)/object.z0

    #Save forcing name to store in output
    object.forcname = f"{object.forcop}_{object.T1:.1f}_{object.z0}"

    return

def linear2(object):
    """Linear profile down to prescribed z0, uniform below that depth """

    #Surface freezing temperature
    object.T0   = object.l1*object.S0+object.l2
    
    #Temperature profile
    if object.T1>object.T0:
        object.Tz = np.minimum(object.T0 + object.z*(object.T1-object.T0)/object.z0,object.T1)
    else:
        object.Tz = np.maximum(object.T0 + object.z*(object.T1-object.T0)/object.z0,object.T1)
    
    #Salinity profile
    object.Sz = np.minimum(object.S0 + object.z*(object.S1-object.S0)/object.z0,object.S1)

    #Save forcing name to store in output
    object.forcname = f"{object.forcop}_{object.T1:.1f}_{object.z0}"

    return


def isomip(object):
    """ISOMIP+ forcing from Asay-Davis et al. (2016). doi: 10.5194/gmd-9-2471-2016"""

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

    #Temperature and salinity profiles
    object.Tz = T0 + object.z*(T1-T0)/z1 
    object.Sz = S0 + object.z*(S1-S0)/z1

    #Save forcing name to store in output
    object.forcname = f"{object.forcop}_{object.isomipcond}"

    return