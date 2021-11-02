import numpy as np

from physics import *
from tools import *

def integrate(object):
    """Integration of 2 time steps, now-centered Leapfrog scheme"""
    intD(object,2*object.dt)
    intu(object,2*object.dt)
    intv(object,2*object.dt) 
    intT(object,2*object.dt)
    intS(object,2*object.dt)        

def timefilter(object):
    """Time filter, Robert Asselin scheme"""
    object.D[1,:,:] += object.nu/2 * (object.D[0,:,:]+object.D[2,:,:]-2*object.D[1,:,:]) * object.tmask
    object.u[1,:,:] += object.nu/2 * (object.u[0,:,:]+object.u[2,:,:]-2*object.u[1,:,:]) * object.umask
    object.v[1,:,:] += object.nu/2 * (object.v[0,:,:]+object.v[2,:,:]-2*object.v[1,:,:]) * object.vmask
    object.T[1,:,:] += object.nu/2 * (object.T[0,:,:]+object.T[2,:,:]-2*object.T[1,:,:]) * object.tmask
    object.S[1,:,:] += object.nu/2 * (object.S[0,:,:]+object.S[2,:,:]-2*object.S[1,:,:]) * object.tmask
    
def updatevars(object):
    """Update temporary variables"""
    object.D = np.roll(object.D,-1,axis=0)
    object.u = np.roll(object.u,-1,axis=0)
    object.v = np.roll(object.v,-1,axis=0)
    object.T = np.roll(object.T,-1,axis=0)
    object.S = np.roll(object.S,-1,axis=0)
    updatesecondary(object)
    
def cutforstability(object):
    """Cut D, U, and V when exceeding specified thresholds"""
    object.D = np.where(object.D>object.maxD,object.maxD,object.D)
    object.u = np.where(object.u>object.vcut,object.vcut,object.u)
    object.u = np.where(object.u<-object.vcut,-object.vcut,object.u)
    object.v = np.where(object.v>object.vcut,object.vcut,object.v)
    object.v = np.where(object.v<-object.vcut,-object.vcut,object.v)   
    
def intD(object,delt):
    """Integrate D"""
    object.D[2,:,:] = object.D[0,:,:] \
                    + (convT(object,object.D[1,:,:]) \
                    +  object.melt \
                    +  object.entr \
                    ) * object.tmask * delt    
    
def intu(object,delt):
    """Integrate u"""
    object.u[2,:,:] = object.u[0,:,:] \
                    +div0((-object.u[1,:,:] * ip_t(object,(object.D[2,:,:]-object.D[0,:,:]))/(2*object.dt) \
                    +  convu(object) \
                    +  -object.g*ip_t(object,object.drho*object.D[1,:,:])*(object.Dxm1-object.D[1,:,:])/object.dx * object.tmask*object.tmaskxm1 \
                    +  object.g*ip_t(object,object.drho*object.D[1,:,:]*object.dzdx) \
                    +  -.5*object.g*ip_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx * object.tmask * object.tmaskxm1 \
                    +  object.f*ip_t(object,object.D[1,:,:]*jm_(object.v[1,:,:],object.vmask)) \
                    +  -object.Cd* object.u[1,:,:] *(object.u[1,:,:]**2 + ip(jm(object.v[1,:,:]))**2)**.5 \
                    +  object.Ah*lapu(object)
                    ),ip_t(object,object.D[1,:,:])) * object.umask * delt

def intv(object,delt):
    """Integrate v"""
    object.v[2,:,:] = object.v[0,:,:] \
                    +div0((-object.v[1,:,:] * jp_t(object,(object.D[2,:,:]-object.D[0,:,:]))/(2*object.dt) \
                    + convv(object) \
                    + -object.g*jp_t(object,object.drho*object.D[1,:,:])*(object.Dym1-object.D[1,:,:])/object.dy * object.tmask*object.tmaskym1 \
                    + object.g*jp_t(object,object.drho*object.D[1,:,:]*object.dzdy) \
                    + -.5*object.g*jp_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy * object.tmask * object.tmaskym1 \
                    + -object.f*jp_t(object,object.D[1,:,:]*im_(object.u[1,:,:],object.umask)) \
                    + -object.Cd* object.v[1,:,:] *(object.v[1,:,:]**2 + jp(im(object.u[1,:,:]))**2)**.5 \
                    + object.Ah*lapv(object)
                    ),jp_t(object,object.D[1,:,:])) * object.vmask * delt
    
def intT(object,delt):
    """Integrate T"""
    object.T[2,:,:] = object.T[0,:,:] \
                    +div0((-object.T[1,:,:] * (object.D[2,:,:]-object.D[0,:,:])/(2*object.dt) \
                    +  convT(object,object.D[1,:,:]*object.T[1,:,:]) \
                    +  object.entr*object.Ta \
                    #+  object.melt*(object.Tf - object.L/object.cp) \
                    +  object.melt*object.Tb - object.gamT*(object.T[1,:,:]-object.Tb) \
                    +  object.Kh*lapT(object,object.T[0,:,:]) \
                    ),object.D[1,:,:]) * object.tmask * delt

def intS(object,delt):
    """Integrate S"""
    object.S[2,:,:] = object.S[0,:,:] \
                    +div0((-object.S[1,:,:] * (object.D[2,:,:]-object.D[0,:,:])/(2*object.dt) \
                    +  convT(object,object.D[1,:,:]*object.S[1,:,:]) \
                    +  object.entr*object.Sa \
                    +  object.Kh*lapT(object,object.S[0,:,:]) \
                    ),object.D[1,:,:]) * object.tmask * delt
