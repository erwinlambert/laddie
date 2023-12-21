import os, sys
import numpy as np

def div0(a,b):
    """Divide to variables allowing for divide by zero"""
    return np.divide(a,b,out=np.zeros_like(a), where=b!=0)

def div0_NN(a,b):
    """Divide to variables allowing for divide by zero, set output to np.nan if divided by zero"""
    return np.divide(a,b,out=np.zeros_like(a)*np.nan, where=b!=0)

def im(var):
    """Value at i-1/2 """
    return .5*(var+np.roll(var,1,axis=1))

def ip(var):
    """Value at i+1/2"""
    return .5*(var+np.roll(var,-1,axis=1))

def jm(var):
    """Value at j-1/2"""
    return .5*(var+np.roll(var,1,axis=0))

def jp(var):
    """Value at j+1/2"""
    return .5*(var+np.roll(var,-1,axis=0))

def im_t(object,var):
    """Value at i-1/2, no gradient across boundary"""
    return div0((var*object.tmask + np.roll(var*object.tmask, 1,axis=1)),object.tmask+object.tmaskxp1)

def ip_t(object,var):
    """Value at i+1/2, no gradient across boundary"""   
    return div0((var*object.tmask + np.roll(var*object.tmask,-1,axis=1)),object.tmask+object.tmaskxm1)

def jm_t(object,var):
    """Value at j-1/2, no gradient across boundary"""
    return div0((var*object.tmask + np.roll(var*object.tmask, 1,axis=0)),object.tmask+object.tmaskyp1)

def jp_t(object,var):
    """Value at j+1/2, no gradient across boundary"""
    return div0((var*object.tmask + np.roll(var*object.tmask,-1,axis=0)),object.tmask+object.tmaskym1)

def im_(var,mask):
    """Value at i-1/2, no gradient across boundary"""
    return div0((var*mask + np.roll(var*mask, 1,axis=1)),(mask+np.roll(mask, 1,axis=1)))

def ip_(var,mask):
    """Value at i+1/2, no gradient across boundary"""
    return div0((var*mask + np.roll(var*mask,-1,axis=1)),(mask+np.roll(mask,-1,axis=1)))

def jm_(var,mask):
    """Value at j-1/2, no gradient across boundary"""
    return div0((var*mask + np.roll(var*mask, 1,axis=0)),(mask+np.roll(mask, 1,axis=0)))

def jp_(var,mask):
    """Value at j+1/2, no gradient across boundary"""
    return div0((var*mask + np.roll(var*mask,-1,axis=0)),(mask+np.roll(mask,-1,axis=0)))    

def tryread(object,category,parameter,reqtype,valid=None,allowconversion=True,checkfile=False,checkdir=False,default=None):
    """Function to read values from config-file, check type and values, and aborting or defaulting if missing"""

    #Make sure integers 0 or 1 are not interpreted as boolean
    if reqtype==bool:
        allowconversion=False

    #Check whether input parameter exists
    try:
        out = object.config[category][parameter]
    except:
        if default == None:
            print(f"INPUT ERROR: missing input parameter '{parameter}' in [{category}]. Please add to config-file")
            sys.exit()
        else:
            if object.newdir:
                object.print2log(f"Note: missing input parameter '{parameter}' in [{category}], using default value {default}")
            out = default

    #Check whether input parameter is of the correct type
    if isinstance(out,reqtype) == False:
        if allowconversion:
            try:
                #Convert to required type, for example float to int or vice versa
                out2 = reqtype(out)
                if object.newdir:
                    object.print2log(f"Note: changing input parameter '{parameter}' from {type(out)} to {reqtype}")
                out = out2
            except:
                if default == None:
                    print(f"INPUT ERROR: input parameter '{parameter}' in [{category}] is of wrong type. Is {type(out)}, should be {reqtype}")
                    sys.exit()
                else:
                    print(f"WARNING: wrong type '{parameter}' in [{category}], using default value {default}")
                    out = default
        else:
            if default == None:
                print(f"INPUT ERROR: input parameter '{parameter}' in [{category}] is of wrong type. Is {type(out)}, should be {reqtype}")
                sys.exit()            
            else:
                print(f"WARNING: wrong type '{parameter}' in [{category}], using default value {default}")
                out = default          

    #Check whether value of input is valid
    if valid != None:
        if isinstance(valid,list):
            if out not in valid:
                if default == None:
                    print(f"INPUT ERROR: invalid value for '{parameter}' in [{category}]; choose from {valid}")
                    sys.exit()
                else:
                    print(f"WARNING: invalid value '{parameter}' in [{category}], using default value {default}")
                    out = default
        if isinstance(valid,tuple):
            if out < valid[0]:
                if default == None:
                    print(f"INPUT ERROR: invalid value for '{parameter}' in [{category}]; should be >= {valid[0]} ")
                    sys.exit()
                else:
                    print(f"WARNING: invalid value '{parameter}' in [{category}]; should be >= {valid[0]}, using default value {default}")
                    out = default
            if out > valid[1]:
                if default == None:
                    print(f"INPUT ERROR: invalid value for '{parameter}' in [{category}]; should be <= {valid[1]} ")
                    sys.exit()
                else:
                    print(f"WARNING: invalid value '{parameter}' in [{category}]; should be <= {valid[1]}, using default value {default}")
                    out = default

    #Check whether file exists
    if checkfile:
        if os.path.isfile(out) == False:
            print(f"INPUT ERROR: non-existing file for '{parameter}' in [{category}]; check filename")
            sys.exit()
        if out[-3:] != ".nc":
            print(f"INPUT ERROR: file '{parameter}' in [{category}] must be '.nc'; check filename")
            sys.exit()

    #Check whether directory exists
    if checkdir:
        if os.path.isdir(out) == False:
            try:
                os.mkdir(out)
                print('WARNING: making a new results directory')
            except:
                print(f"INPUT ERROR: could not create directory '{parameter}' in [{category}]; check directory name")
                sys.exit()

    return out

def extrapolate_initvals(object,object_variable,object_mask,init_variable,init_mask):
    """Copy variable from restart file and extrapolate into new grid cells"""

    #Get a temporary mask from restart file
    imask = init_mask.copy()

    #Copy restart field where valid
    object_variable[:] = np.where(np.logical_and(object_mask==1, imask==1), init_variable, object_variable[:])

    #Set to nan where values are missing (newly opened grid cells)
    object_variable[:] = np.where(np.logical_and(object_mask[:]==1, imask[:]==0), np.nan, object_variable[:])

    #Get number of empty cells
    N_empty_cells = np.sum(np.isnan(object_variable[1]))

    object.print2log(f'empty mask: {N_empty_cells:.0f}')

    #Loop until no more empty cells
    while N_empty_cells > 0:
        #Field denoting where missing grid cells are
        condition = np.logical_and(object_mask[:]==1, imask[:]==0)

        #Extrapolate 1 neighbour using nearest neighbour averaging
        object_variable[:] = np.where(condition, compute_average_NN(object_variable, imask), object_variable[:])

        #Redefine temporary mask, setting 0 to 1 for cells that were just filled
        imask[:] = np.where(np.logical_and(np.isnan(object_variable[1])==False, imask[:]==0), 1, imask[:])

        #Get number of remaining empty cells
        N_empty_cells = np.sum(np.isnan(object_variable[1]))
        object.print2log(f'empty mask: {N_empty_cells:.0f}')

    return object_variable


def compute_average_NN(object_variable, mask):
    """
    Compute the average of nearest neighbouring cells
    
    object_variable: variable for which the NN average is to be computed, for example: object.T
    mask: mask that corresponds to object_variable, either object.tmask, object.umask, or object.vmask

    """

    # Create nn_average array to store average nearest neighbour values
    nn_average = object_variable * 0

    # Loop over time dimension
    for i in range(3):
        var = object_variable[i,:,:]

        # Only take values from cells within shelf mask
        vari = np.where(mask==1, var, 0)

        # Take the sum of the values in neighbouring cells for nt = 1 
        nn_total = np.roll(vari,-1,axis=0)+ np.roll(vari,1,axis=0) + np.roll(vari,-1,axis=1) + np.roll(vari,1,axis=1)

        # Compute the weight using the mask (the weight is the number of neighbouring cells which contain values within the shelf mask)
        weight = np.roll(mask,-1, axis=0)+ np.roll(mask,1, axis=0) + np.roll(mask,-1, axis=1) + np.roll(mask,1, axis=1)

        # Divide sum of neighbours by the weight and fill nn_average array
        nn_average[i,:,:] = div0_NN(nn_total,weight)

    return nn_average