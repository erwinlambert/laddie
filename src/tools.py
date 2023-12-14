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
            object.print2log(f"Note: missing input parameter '{parameter}' in [{category}], using default value {default}")
            out = default

    #Check whether input parameter is of the correct type
    if isinstance(out,reqtype) == False:
        if allowconversion:
            try:
                #Convert to required type, for example float to int or vice versa
                out2 = reqtype(out)
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