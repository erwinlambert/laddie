import numpy as np
from physics import *
from tools import *

def integrate(object):
    """Integration of 2 time steps, now-centered Leapfrog scheme"""
    intD(object,2*object.dt)
    intU(object,2*object.dt)
    intV(object,2*object.dt) 
    intT(object,2*object.dt)
    intS(object,2*object.dt)        

def timefilter(object):
    """Time filter, Robert Asselin scheme"""
    object.D[1,:,:] += object.nu/2 * (object.D[0,:,:]+object.D[2,:,:]-2*object.D[1,:,:]) * object.tmask
    object.U[1,:,:] += object.nu/2 * (object.U[0,:,:]+object.U[2,:,:]-2*object.U[1,:,:]) * object.umask
    object.V[1,:,:] += object.nu/2 * (object.V[0,:,:]+object.V[2,:,:]-2*object.V[1,:,:]) * object.vmask
    object.T[1,:,:] += object.nu/2 * (object.T[0,:,:]+object.T[2,:,:]-2*object.T[1,:,:]) * object.tmask
    object.S[1,:,:] += object.nu/2 * (object.S[0,:,:]+object.S[2,:,:]-2*object.S[1,:,:]) * object.tmask

    object.drho = (object.beta*(object.Sa-object.S[1,:,:]) - object.alpha*(object.Ta-object.T[1,:,:])) * object.tmask
    if object.convop == 0:
    #    #Prescribe minimum stratification
        object.drho = np.maximum(object.drho,object.mindrho/object.rho0)
    elif object.convop == 1:        
    #    #Apply instantaneous convection
        object.T[1,:,:] = np.where(object.drho<0,object.Ta,object.T[1,:,:]) #Convective heating unlimited by available heat underneath layer. May overstimate convective melt
        object.S[1,:,:] = np.where(object.drho<0,object.Sa,object.S[1,:,:])
        object.drho = (object.beta*(object.Sa-object.S[1,:,:]) - object.alpha*(object.Ta-object.T[1,:,:])) * object.tmask

def updatevars(object):
    """Update temporary variables"""
    object.D = np.roll(object.D,-1,axis=0)
    object.U = np.roll(object.U,-1,axis=0)
    object.V = np.roll(object.V,-1,axis=0)
    object.T = np.roll(object.T,-1,axis=0)
    object.S = np.roll(object.S,-1,axis=0)
    updatesecondary(object)
    
def cutforstability(object):
    """Cut U, and V when exceeding specified thresholds"""
    object.U = np.where(object.U>object.vcut,object.vcut,object.U)
    object.U = np.where(object.U<-object.vcut,-object.vcut,object.U)
    object.V = np.where(object.V>object.vcut,object.vcut,object.V)
    object.V = np.where(object.V<-object.vcut,-object.vcut,object.V)   
    
def intD(object,delt):
    """Integrate D"""
    object.D[2,:,:] = object.D[0,:,:] \
                    + (convT(object,object.D[1,:,:]) \
                    +  object.melt \
                    +  object.entr \
                    +  object.ent2 \
                    -  object.detr \
                    ) * object.tmask * delt    
    
def intU(object,delt):
    """Integrate U"""
    object.U[2,:,:] = object.U[0,:,:] \
                    +div0((-object.U[1,:,:] * ip_t(object,(object.D[2,:,:]-object.D[0,:,:]))/(2*object.dt) \
                    +  convU(object) \
                    +  -object.g*ip_t(object,object.drho*object.D[1,:,:])*(object.Dxm1-object.D[1,:,:])/object.dx \
                    +  object.g*ip_t(object,object.drho*object.D[1,:,:]*object.dzdx) \
                    +  -.5*object.g*ip_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=1)-object.drho)/object.dx \
                    +  object.f*ip_t(object,object.D[1,:,:]*jm_(object.V[1,:,:],object.vmask)) \
                    +  -object.Cd* object.U[1,:,:] *(object.U[1,:,:]**2 + ip(jm(object.V[1,:,:]))**2)**.5 \
                    +  object.Ah*lapU(object) \
                    +  -object.detr* object.U[1,:,:] \
                    ),ip_t(object,object.D[1,:,:])) * object.umask * delt

def intV(object,delt):
    """Integrate V"""
    object.V[2,:,:] = object.V[0,:,:] \
                    +div0((-object.V[1,:,:] * jp_t(object,(object.D[2,:,:]-object.D[0,:,:]))/(2*object.dt) \
                    + convV(object) \
                    + -object.g*jp_t(object,object.drho*object.D[1,:,:])*(object.Dym1-object.D[1,:,:])/object.dy \
                    + object.g*jp_t(object,object.drho*object.D[1,:,:]*object.dzdy) \
                    + -.5*object.g*jp_t(object,object.D[1,:,:])**2*(np.roll(object.drho,-1,axis=0)-object.drho)/object.dy \
                    + -object.f*jp_t(object,object.D[1,:,:]*im_(object.U[1,:,:],object.umask)) \
                    + -object.Cd* object.V[1,:,:] *(object.V[1,:,:]**2 + jp(im(object.U[1,:,:]))**2)**.5 \
                    + object.Ah*lapV(object) \
                    +  -object.detr* object.V[1,:,:] \
                    ),jp_t(object,object.D[1,:,:])) * object.vmask * delt
    
def intT(object,delt):
    """Integrate T"""
    object.T[2,:,:] = object.T[0,:,:] \
                    +div0((-object.T[1,:,:] * (object.D[2,:,:]-object.D[0,:,:])/(2*object.dt) \
                    +  convT(object,object.D[1,:,:]*object.T[1,:,:]) \
                    +  object.entr*object.Ta \
                    +  object.ent2*object.Ta \
                    -  object.detr*object.Ta \
                    +  object.melt*object.Tb - object.gamT*(object.T[1,:,:]-object.Tb) \
                    +  object.Kh*lapT(object,object.T[0,:,:]) \
                    -  (object.T[0,:,:]-object.Ta)*np.where(object.drho<0,1,0)*object.D[1,:,:]/object.convtime *np.where(object.convop==2,1,0) \
                    ),object.D[1,:,:]) * object.tmask * delt

def intS(object,delt):
    """Integrate S"""
    object.S[2,:,:] = object.S[0,:,:] \
                    +div0((-object.S[1,:,:] * (object.D[2,:,:]-object.D[0,:,:])/(2*object.dt) \
                    +  convT(object,object.D[1,:,:]*object.S[1,:,:]) \
                    +  object.entr*object.Sa \
                    +  object.ent2*object.Sa \
                    -  object.detr*object.Sa \
                    +  object.Kh*lapT(object,object.S[0,:,:]) \
                    -  (object.S[0,:,:]-object.Sa)*np.where(object.drho<0,1,0)*object.D[1,:,:]/object.convtime *np.where(object.convop==2,1,0)\
                    ),object.D[1,:,:]) * object.tmask * delt
