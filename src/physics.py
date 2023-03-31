import numpy as np

from tools import *

def lapT(object,var):
    """Laplacian operator for DT and DS"""
    
    tN = jp_t(object,object.D[0,:,:])*(np.roll(var,-1,axis=0)-var)*object.tmaskym1/object.dy**2
    tS = jm_t(object,object.D[0,:,:])*(np.roll(var, 1,axis=0)-var)*object.tmaskyp1/object.dy**2
    tE = ip_t(object,object.D[0,:,:])*(np.roll(var,-1,axis=1)-var)*object.tmaskxm1/object.dx**2
    tW = im_t(object,object.D[0,:,:])*(np.roll(var, 1,axis=1)-var)*object.tmaskxp1/object.dx**2    
    
    return tN+tS+tE+tW

def lapU(object):
    """Laplacian operator for DU"""
    Dcent = ip_t(object,object.D[0,:,:])
    var = object.U[0,:,:]

    tN = jp_t(object,Dcent)                            * (np.roll(var,-1,axis=0)-var)/object.dy**2 * (1-object.ocnym1) - object.slip*Dcent*var*object.grdNu/object.dy**2
    tS = jm_t(object,Dcent)                            * (np.roll(var, 1,axis=0)-var)/object.dy**2 * (1-object.ocnyp1) - object.slip*Dcent*var*object.grdSu/object.dy**2  
    tE = np.roll(object.D[0,:,:]*object.tmask,-1,axis=1) * (np.roll(var,-1,axis=1)-var)/object.dx**2 * (1-object.ocnxm1)
    tW = object.D[0,:,:]                               * (np.roll(var, 1,axis=1)-var)/object.dx**2 * (1-object.ocn   )
    
    return (tN+tS+tE+tW) * object.umask

def lapV(object):
    """Laplacian operator for DV"""
    Dcent = jp_t(object,object.D[0,:,:])
    var = object.V[0,:,:]
    
    tN = np.roll(object.D[0,:,:]*object.tmask,-1,axis=0) * (np.roll(var,-1,axis=0)-var)/object.dy**2 * (1-object.ocnym1) 
    tS = object.D[0,:,:]                               * (np.roll(var, 1,axis=0)-var)/object.dy**2 * (1-object.ocn   )
    tE = ip_t(object,Dcent)                            * (np.roll(var,-1,axis=1)-var)/object.dx**2 * (1-object.ocnxm1) - object.slip*Dcent*var*object.grdEv/object.dx**2
    tW = im_t(object,Dcent)                            * (np.roll(var, 1,axis=1)-var)/object.dx**2 * (1-object.ocnxp1) - object.slip*Dcent*var*object.grdWv/object.dx**2  
    
    return (tN+tS+tE+tW) * object.vmask

def convT(object,var):
    """Upstream convergence scheme for D, DT, DS"""
    
    if object.boundop ==1:
        #Option 1: zero gradient for inflow
        tN = - (np.maximum(object.V[1,:,:],0)*var                   + np.minimum(object.V[1,:,:],0)*(np.roll(var,-1,axis=0)*object.tmaskym1+var*object.ocnym1)) / object.dy * object.vmask
        tS =   (np.maximum(object.Vyp1    ,0)*(np.roll(var,1,axis=0)*object.tmaskyp1+var*object.ocnyp1) + np.minimum(object.Vyp1    ,0)*var                   ) / object.dy * object.vmaskyp1
        tE = - (np.maximum(object.U[1,:,:],0)*var                   + np.minimum(object.U[1,:,:],0)*(np.roll(var,-1,axis=1)*object.tmaskxm1+var*object.ocnxm1)) / object.dx * object.umask
        tW =   (np.maximum(object.Uxp1    ,0)*(np.roll(var,1,axis=1)*object.tmaskxp1+var*object.ocnxp1) + np.minimum(object.Uxp1    ,0)*var                   ) / object.dx * object.umaskxp1
    elif object.boundop ==2:
        #Opion 2: zero inflow
        tN = - (np.maximum(object.V[1,:,:],0)*var                   + np.minimum(object.V[1,:,:],0)*np.roll(var,-1,axis=0)) / object.dy * object.vmask
        tS =   (np.maximum(object.Vyp1    ,0)*np.roll(var,1,axis=0) + np.minimum(object.Vyp1    ,0)*var                   ) / object.dy * object.vmaskyp1
        tE = - (np.maximum(object.U[1,:,:],0)*var                   + np.minimum(object.U[1,:,:],0)*np.roll(var,-1,axis=1)) / object.dx * object.umask
        tW =   (np.maximum(object.Uxp1    ,0)*np.roll(var,1,axis=1) + np.minimum(object.Uxp1    ,0)*var                   ) / object.dx * object.umaskxp1               
    return (tN+tS+tE+tW) * object.tmask

def convT2(object,var,vara):
    """Upstream convergence scheme for D, DT, DS; allowing for inflow of ambient water"""
    tN = - (np.maximum(object.V[1,:,:],0)*var                   + np.minimum(object.V[1,:,:],0)*(np.roll(var,-1,axis=0)*object.tmaskym1+vara*object.ocnym1)) / object.dy * object.vmask
    tS =   (np.maximum(object.Vyp1    ,0)*(np.roll(var,1,axis=0)*object.tmaskyp1+vara*object.ocnyp1) + np.minimum(object.Vyp1    ,0)*var                   ) / object.dy * object.vmaskyp1
    tE = - (np.maximum(object.U[1,:,:],0)*var                   + np.minimum(object.U[1,:,:],0)*(np.roll(var,-1,axis=1)*object.tmaskxm1+vara*object.ocnxm1)) / object.dx * object.umask
    tW =   (np.maximum(object.Uxp1    ,0)*(np.roll(var,1,axis=1)*object.tmaskxp1+vara*object.ocnxp1) + np.minimum(object.Uxp1    ,0)*var                   ) / object.dx * object.umaskxp1
    return (tN+tS+tE+tW) * object.tmask


def convU(object):
    """Convergence for DU"""
   
    #Get D at north, south, east, west points
    DN = div0((object.D[1,:,:]*object.tmask + object.Dxm1 + object.Dym1 + object.Dxm1ym1),(object.tmask + object.tmaskxm1 + object.tmaskym1 + object.tmaskxm1ym1))
    DS = div0((object.D[1,:,:]*object.tmask + object.Dxm1 + object.Dyp1 + object.Dxm1yp1),(object.tmask + object.tmaskxm1 + object.tmaskyp1 + object.tmaskxm1yp1))
    DE = object.Dxm1                 + object.ocnxm1 * object.D[1,:,:]*object.tmask
    DW = object.D[1,:,:]*object.tmask  + object.ocn    * object.Dxm1
    
    tN = -DN *        ip_(object.V[1,:,:],object.vmask)           *(jp_(object.U[1,:,:],object.umask)-object.slip*object.U[1,:,:]*object.grdNu) /object.dy
    tS =  DS *np.roll(ip_(object.V[1,:,:],object.vmask),1,axis=0) *(jm_(object.U[1,:,:],object.umask)-object.slip*object.U[1,:,:]*object.grdSu) /object.dy
    tE = -DE *        ip_(object.U[1,:,:],object.umask)           *(ip_(object.U[1,:,:],object.umask)-object.ocnxm1*(1-np.sign(object.U[1,:,:]))*object.U[1,:,:])                                     /object.dx
    tW =  DW *        im_(object.U[1,:,:],object.umask)           *(im_(object.U[1,:,:],object.umask)-object.ocn*np.sign(object.U[1,:,:])*object.U[1,:,:])                                     /object.dx
    
    return (tN+tS+tE+tW) * object.umask

def convV(object):
    """Convergence for DV"""
    
    #Get D at north, south, east, west points
    DE = div0((object.D[1,:,:]*object.tmask + object.Dym1 + object.Dxm1 + object.Dxm1ym1),(object.tmask + object.tmaskym1 + object.tmaskxm1 + object.tmaskxm1ym1))
    DW = div0((object.D[1,:,:]*object.tmask + object.Dym1 + object.Dxp1 + object.Dxp1ym1),(object.tmask + object.tmaskym1 + object.tmaskxp1 + object.tmaskxp1ym1))
    DN = object.Dym1                 + object.ocnym1 * object.D[1,:,:]*object.tmask
    DS = object.D[1,:,:]*object.tmask  + object.ocn    * object.Dym1 
    
    tN = -DN *        jp_(object.V[1,:,:],object.vmask)           *(jp_(object.V[1,:,:],object.vmask)-object.ocnym1*(1-np.sign(object.V[1,:,:]))*object.V[1,:,:])                                      /object.dy
    tS =  DS *        jm_(object.V[1,:,:],object.vmask)           *(jm_(object.V[1,:,:],object.vmask)-object.ocn*np.sign(object.V[1,:,:])*object.V[1,:,:])                                     /object.dy
    tE = -DE *        jp_(object.U[1,:,:],object.umask)           *(ip_(object.V[1,:,:],object.vmask)-object.slip*object.V[1,:,:]*object.grdEv) /object.dx
    tW =  DW *np.roll(jp_(object.U[1,:,:],object.umask),1,axis=1) *(im_(object.V[1,:,:],object.vmask)-object.slip*object.V[1,:,:]*object.grdWv) /object.dx
    
    return (tN+tS+tE+tW) * object.vmask     

def updatesecondary(object):
    """Update a bunch of secondary variables"""    

    #Ambient fields
    if len(object.Tz.shape)==1:
        object.Ta   = np.interp(object.zb-object.D[1,:,:],object.z,object.Tz)
        object.Sa   = np.interp(object.zb-object.D[1,:,:],object.z,object.Sz)
    elif len(object.Tz.shape)==3:
        object.Ta = object.Tz[np.int_(.01*np.minimum(4999,-object.zb+object.D[1,:,:])),object.ind[0],object.ind[1]]
        object.Sa = object.Sz[np.int_(.01*np.minimum(4999,-object.zb+object.D[1,:,:])),object.ind[0],object.ind[1]]
        
    object.Tf   = (object.l1*object.S[1,:,:]+object.l2+object.l3*object.zb).values
    
    object.drho = (object.beta*(object.Sa-object.S[1,:,:]) - object.alpha*(object.Ta-object.T[1,:,:])) * object.tmask

    #Ensure stable stratification
    object.convection = np.where(object.drho<0*object.tmask,1,0)
    if object.convop == 0:
    #    #Prescribe minimum stratification
        object.drho = np.maximum(object.drho,object.mindrho/object.rho0)
    elif object.convop == 1:        
    #    #Apply instantaneous convection
        object.T[1,:,:] = np.where(object.drho<0,object.Ta,object.T[1,:,:]) #Convective heating unlimited by available heat underneath layer. May overstimate convective melt
        object.S[1,:,:] = np.where(object.drho<0,object.Sa,object.S[1,:,:])
        object.drho = (object.beta*(object.Sa-object.S[1,:,:]) - object.alpha*(object.Ta-object.T[1,:,:])) * object.tmask

    #Melt
    object.ustar = (object.Cdtop*(im(object.U[1,:,:])**2+jm(object.V[1,:,:])**2+object.utide**2))**.5 * object.tmask
    
    if object.usegamtfix:
        object.gamT = object.gamTfix
        object.gamS = object.gamT/35.
    else: 
        object.gamT = object.ustar/(2.12*np.log(object.ustar*np.maximum(object.D[1,:,:],object.minD)/object.nu0+1e-12)+12.5*object.Pr**(2./3)-8.68) * object.tmask
        object.gamS = object.ustar/(2.12*np.log(object.ustar*np.maximum(object.D[1,:,:],object.minD)/object.nu0+1e-12)+12.5*object.Sc**(2./3)-8.68) * object.tmask

    That = (object.l2+object.l3*object.zb).values
    Chat = object.cp/(object.L-object.ci*object.Ti)
    Ctil = object.ci/object.cp

    b = Chat*object.gamT*(That-object.T[1,:,:])+object.gamS*(1+Chat*Ctil*(That+object.l1*object.S[1,:,:]))
    c = Chat*object.gamT*object.gamS*(That-object.T[1,:,:]+object.l1*object.S[1,:,:])

    object.melt = .5*(-b+np.sqrt(b**2-4*c)) * object.tmask
    object.Tb = div0(Chat*object.gamT*object.T[1,:,:]-object.melt,Chat*object.gamT+Chat*Ctil*object.melt) * object.tmask

    #Entrainment
    #object.entr = object.E0*np.maximum(0,(im(object.U[1,:,:])*object.dzdx + jm(object.V[1,:,:])*object.dzdy)) * object.tmask
    if object.entpar == 'Holland':
        object.entr = object.cl*object.Kh/object.Ah**2*(np.maximum(0,im(object.U[1,:,:])**2+jm(object.V[1,:,:])**2-object.g*object.drho*object.Kh/object.Ah*object.D[1,:,:]))**.5 * object.tmask
        object.detr = 0.*object.entr
    elif object.entpar == 'Gaspar':
        object.Sb = (object.Tb-object.l2-object.l3*object.zb).values/object.l1
        object.drhob = (object.beta*(object.S[1,:,:]-object.Sb) - object.alpha*(object.T[1,:,:]-object.Tb)) * object.tmask
        object.ent = 2*object.mu/object.g * div0(object.ustar**3,object.D[1,:,:]*np.maximum(.0001,object.drho)) - div0(object.drhob,np.maximum(.0001,object.drho))*object.melt * object.tmask
        object.entr = np.maximum(object.ent,0)
        object.detr = np.minimum(object.maxdetr,np.maximum(-object.ent,0))
    
    object.Dym1    = np.roll(        object.D[1,:,:]*object.tmask,-1,axis=0)
    object.Dyp1    = np.roll(        object.D[1,:,:]*object.tmask, 1,axis=0)
    object.Dxm1    = np.roll(        object.D[1,:,:]*object.tmask,-1,axis=1)
    object.Dxp1    = np.roll(        object.D[1,:,:]*object.tmask, 1,axis=1)
    object.Dxm1ym1 = np.roll(np.roll(object.D[1,:,:]*object.tmask,-1,axis=1),-1,axis=0)
    object.Dxp1ym1 = np.roll(np.roll(object.D[1,:,:]*object.tmask, 1,axis=1),-1,axis=0)
    object.Dxm1yp1 = np.roll(np.roll(object.D[1,:,:]*object.tmask,-1,axis=1), 1,axis=0)
    
    object.Vyp1    = np.roll(object.V[1,:,:],1,axis=0)  
    object.Uxp1    = np.roll(object.U[1,:,:],1,axis=1)      

    #Additional entrainment to prevent D<minD
    object.ent2 = np.maximum(0,(object.minD-object.D[0,:,:])/(2*object.dt)-(convT(object,object.D[1,:,:])+object.melt+object.entr-object.detr)) *object.tmask