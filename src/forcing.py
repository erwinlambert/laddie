import sys
import numpy as np
import xarray as xr

def get_forcing(object):
    if object.newdir: object.print2log("================= Starting preparing forcing ======================")

    #Read or create forcing
    if object.forcop == "file":
        #Read forcing from external file
        read_forcing(object)
    else:
        #Create forcing using an internal routine
        create_forcing(object)

    #Do checks on validity
    check_inputforcing(object)

    object.print2log(f"================= Successfully got forcing {object.forcname} ==========================")
    if object.newdir: object.print2log(f"===========================================================================")

    return

def read_forcing(object):
    """Read external forcing file"""

    try:
        ds = xr.open_dataset(object.forcfile)

        #Read temperature and salinity fields or profiles
        object.z = ds.z.values
        object.Tz = ds.T.values
        object.Sz = ds.S.values

        #Extract required info on x,y-dimensions of 3D external 
        if len(object.Tz.shape)==3:
            object.Tax1 = np.arange(object.Tz.shape[1])[:,None]
            object.Tax2 = np.arange(object.Tz.shape[2])[:,None]

        ds.close()

    except:
        print(f"INPUT ERROR: could not open forcing input file {object.forcfile}. Check whether filename is correct and file contains variables z, T and S")
        sys.exit()

    #Save forcing name to store in output
    object.forcname = object.forcfile

    if object.newdir: object.print2log(f"Finished reading forcing from input file")

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

    if object.newdir: object.print2log(f"Finished creating forcing from internal routine {object.forcop}")

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

def check_inputforcing(object):
    """Check validity of input. Only functions for 1D input forcing for now """

    if object.newdir: object.print2log(f"Started checking forcing")

    #Skip check if 3D input
    if len(object.Tz.shape)==3:
        object.print2log(f"WARNING: 3D input forcing is not checked. Big chance that something goes wrong. Trying to proceed anyway..")
        return

    #Check whether z, Tz and Sz are of equal length
    if len(object.Tz) != len(object.z):
        print('FORCING ERROR: input variable Tz must be of equal length as input variable z. Check and correct input forcing file')
        sys.exit()
    if len(object.Sz) != len(object.z):
        print('FORCING ERROR: input variable Sz must be of equal length as input variable z. Check and correct input forcing file')
        sys.exit()

    #Check whether step in z is 1 meter throughout, required for interpolation to find Ta and Sa. More flexibility or pre-interpolation to be added
    dz = object.z[1:]-object.z[:-1]
    if (dz != 1).any():
        #Not equal steps of 1m, try interpolation
        try:
            zin = object.z.copy()
            Tzin = object.Tz.copy()
            Szin = object.Sz.copy()
            object.z = np.arange(-5000.,0,1)
            object.Tz = np.interp(object.z,zin,Tzin)
            object.Sz = np.interp(object.z,zin,Szin)
            if object.newdir: object.print2log(f"NOTE: modified input forcing through inteprolation to 1m grid. Provide 1m grid as input to prevent this.")
        except:
            print('FORCING ERROR: input variable z must be incremental with steps 1 meter, could not interpolate. Check whether z is monotonic')
            sys.exit()

    #Extract minimum value in z for interpolation
    object.zmin = min(object.z)

    """To be expanded """

    if object.newdir: object.print2log(f"Finished checking forcing. All OK")

    return