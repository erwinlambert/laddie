import numpy as np

def div0(a,b):
    """Divide to variables allowing for divide by zero"""
    return np.divide(a,b,out=np.zeros_like(a), where=b!=0)

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
