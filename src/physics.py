import numpy as np
from tools import *

def lapT(object,var):
    """Laplacian operator for DT and DS"""
    
    #Operator at all four sides
    tN = object.D0jp * (np.roll(var,-1,axis=0)-var)*object.tmaskym1/object.dy**2
    tS = object.D0jm * (np.roll(var, 1,axis=0)-var)*object.tmaskyp1/object.dy**2
    tE = object.D0ip * (np.roll(var,-1,axis=1)-var)*object.tmaskxm1/object.dx**2
    tW = object.D0im * (np.roll(var, 1,axis=1)-var)*object.tmaskxp1/object.dx**2    
    
    return tN+tS+tE+tW

def lapU(object):
    """Laplacian operator for DU"""

    #Get D on U-grid
    Dcent = object.D0ip*object.tmask
    var = object.U[0,:,:]
    AA = object.slip*Dcent*var/object.dy**2

    #Operator at all four sides
    tN = jp_t(object,Dcent)                 * (np.roll(var,-1,axis=0)-var)/object.dy**2 * (1-object.ocnym1) - AA*object.grdNu
    tS = jm_t(object,Dcent)                 * (np.roll(var, 1,axis=0)-var)/object.dy**2 * (1-object.ocnyp1) - AA*object.grdSu  
    tE = np.roll(object.D[0,:,:],-1,axis=1) * (np.roll(var,-1,axis=1)-var)/object.dx**2 * (1-object.ocnxm1)
    tW = object.D[0,:,:]                    * (np.roll(var, 1,axis=1)-var)/object.dx**2 * (1-object.ocn   )
    
    return tN+tS+tE+tW

def lapV(object):
    """Laplacian operator for DV"""

    #Get D on V-grid
    Dcent = object.D0jp*object.tmask
    var = object.V[0,:,:]
    AA = object.slip*Dcent*var/object.dx**2
    
    #Operator on all four sides
    tN = np.roll(object.D[0,:,:],-1,axis=0) * (np.roll(var,-1,axis=0)-var)/object.dy**2 * (1-object.ocnym1) 
    tS = object.D[0,:,:]                    * (np.roll(var, 1,axis=0)-var)/object.dy**2 * (1-object.ocn   )
    tE = ip_t(object,Dcent)                 * (np.roll(var,-1,axis=1)-var)/object.dx**2 * (1-object.ocnxm1) - AA*object.grdEv
    tW = im_t(object,Dcent)                 * (np.roll(var, 1,axis=1)-var)/object.dx**2 * (1-object.ocnxp1) - AA*object.grdWv  
    
    return tN+tS+tE+tW

def convT(object,var):
    """Upstream convergence scheme for D, DT, DS"""
    
    if object.boundop ==1:
        #Option 1: assume inflow from open ocean to cavity is determined by zero gradient in D, T, and S
        tN = - (object.Vpos*var                          + object.Vneg*(np.roll(var,-1,axis=0)*object.tmaskym1+var*object.ocnym1)) / object.dy * object.vmask
        tS =   (object.Vyp1pos*(np.roll(var,1,axis=0)*object.tmaskyp1+var*object.ocnyp1) + object.Vyp1neg *var                   ) / object.dy * object.vmaskyp1
        tE = - (object.Upos*var                          + object.Uneg*(np.roll(var,-1,axis=1)*object.tmaskxm1+var*object.ocnxm1)) / object.dx * object.umask
        tW =   (object.Uxp1pos*(np.roll(var,1,axis=1)*object.tmaskxp1+var*object.ocnxp1) + object.Uxp1neg *var                   ) / object.dx * object.umaskxp1
    elif object.boundop ==2:
        #Opion 2: assume zero inflow from open ocean to cavity
        tN = - (object.Vpos*var                      + object.Vneg   *np.roll(var,-1,axis=0)) / object.dy * object.vmask
        tS =   (object.Vyp1pos*np.roll(var,1,axis=0) + object.Vyp1neg*var                   ) / object.dy * object.vmaskyp1
        tE = - (object.Upos*var                      + object.Uneg   *np.roll(var,-1,axis=1)) / object.dx * object.umask
        tW =   (object.Uxp1pos*np.roll(var,1,axis=1) + object.Uxp1neg*var                   ) / object.dx * object.umaskxp1               
    return tN+tS+tE+tW

def convU(object):
    """Convergence for DU"""
    
    #Get D at north, south, east, west points of U-grid
    DN = div0((object.D[1,:,:] + object.Dxm1 + object.Dym1 + object.Dxm1ym1),(object.tmask + object.tmaskxm1 + object.tmaskym1 + object.tmaskxm1ym1))
    DS = div0((object.D[1,:,:] + object.Dxm1 + object.Dyp1 + object.Dxm1yp1),(object.tmask + object.tmaskxm1 + object.tmaskyp1 + object.tmaskxm1yp1))
    DE = object.Dxm1     + object.ocnxm1 * object.D[1,:,:]
    DW = object.D[1,:,:] + object.ocn    * object.Dxm1
    
    tN = -DN *        object.Vip           *(object.Ujp -    object.slip  * object.U[1,:,:] * object.grdNu ) /object.dy
    tS =  DS *np.roll(object.Vip,1,axis=0) *(object.Ujm -    object.slip  * object.U[1,:,:] * object.grdSu ) /object.dy
    tE = -DE *        object.Uip           *(object.Uip - (1-object.signU)* object.U[1,:,:] * object.ocnxm1) /object.dx
    tW =  DW *        object.Uim           *(object.Uim -    object.signU * object.U[1,:,:] * object.ocn   ) /object.dx
    
    return tN+tS+tE+tW

def convV(object):
    """Convergence for DV"""
    
    #Get D at north, south, east, west points of V-grid
    DE = div0((object.D[1,:,:] + object.Dym1 + object.Dxm1 + object.Dxm1ym1),(object.tmask + object.tmaskym1 + object.tmaskxm1 + object.tmaskxm1ym1))
    DW = div0((object.D[1,:,:] + object.Dym1 + object.Dxp1 + object.Dxp1ym1),(object.tmask + object.tmaskym1 + object.tmaskxp1 + object.tmaskxp1ym1))
    DN = object.Dym1     + object.ocnym1 * object.D[1,:,:]
    DS = object.D[1,:,:] + object.ocn    * object.Dym1 
    
    tN = -DN *        object.Vjp           *(object.Vjp - (1-object.signV)* object.V[1,:,:] * object.ocnym1) /object.dy
    tS =  DS *        object.Vjm           *(object.Vjm -    object.signV * object.V[1,:,:] * object.ocn   ) /object.dy
    tE = -DE *        object.Ujp           *(object.Vip -    object.slip  * object.V[1,:,:] * object.grdEv ) /object.dx
    tW =  DW *np.roll(object.Ujp,1,axis=1) *(object.Vim -    object.slip  * object.V[1,:,:] * object.grdWv ) /object.dx
    
    return tN+tS+tE+tW    

def updatesecondary(object):
    """Update a bunch of secondary variables"""    

    #Ambient temperature and salinity
    update_ambientfields(object)
    
    #Freezing temperature
    update_freezing_temperature(object)
    
    #Density difference with ambient fields
    update_density(object)

    #Apply convection
    update_convection(object)

    #Melt
    update_melt(object)

    #Compute some reused fields for convT, convU, convV
    prepare_convergence(object)

    #Compute some reused fields for lapT, lapU, lapV
    prepare_laplacian(object)

    #Entrainment and detrainment
    update_entrainment(object)

    return

def update_ambientfields(object):
    """Compute Ta and Sa from Tz and Sz at depth of mixed layer (zb - D)"""

    #Find index of z-array at base of layer for each grid point
    idx = -object.z[0] + (object.zb-object.D[1,:,:])/object.dz #Exact index on zgrid
    idx0 = np.maximum(0,np.minimum(len(object.z)-1,np.int_(idx))) #Integer index before exact index
    idx1 = np.maximum(0,np.minimum(len(object.z)-1,np.int_(idx+1))) #Integer index after exact index
    w1 = idx-idx0 #Weight of integer index after exact index

    #Get ambient Ta and Sa based on forcing profiles
    if len(object.Tz.shape)==1:
        #1D forcing profiles, linearly interpolated on z grid
        object.Ta = w1*object.Tz[idx1] + (1-w1)*object.Tz[idx0]
        object.Sa = w1*object.Sz[idx1] + (1-w1)*object.Sz[idx0]
    elif len(object.Tz.shape)==3:
        #3D forcing fields, linearly interpolated on z grid
        object.Ta = w1*object.Tz[idx1,object.Tax1,object.Tax2]+(1-w1)*object.Tz[idx0,object.Tax1,object.Tax2]
        object.Sa = w1*object.Sz[idx1,object.Tax1,object.Tax2]+(1-w1)*object.Sz[idx0,object.Tax1,object.Tax2]
    
    return

def update_freezing_temperature(object):
    """Freezing temperature"""

    object.Tf   = (object.l1*object.S[1,:,:]+object.l2+object.l3*object.zb)

    return

def update_density(object):
    """Density difference with ambient fields. drho = (rho_ambient - rho_mixedlayer)/rho0"""

    #Linear EOS. Other options to be added later
    object.drho = (object.beta*(object.Sa-object.S[1,:,:]) - object.alpha*(object.Ta-object.T[1,:,:])) * object.tmask

    return

def update_convection(object):
    """Convection module to prevent unstable stratification (drho<0) """

    #Get number of grid points with unstable stratification
    object.convection = np.where(object.drho<0*object.tmask,1,0)

    if object.convop == 0:
        #Prescribe minimum stratification
        object.drho = np.maximum(object.drho,object.mindrho/object.rho0)
    elif object.convop == 1:        
        #Apply instantaneous convection
        object.T[1,:,:] = np.where(object.drho<object.mindrho/object.rho0,object.Ta,object.T[1,:,:]) #Convective heating unlimited by available heat underneath layer. May overstimate convective melt
        object.S[1,:,:] = np.where(object.drho<object.mindrho/object.rho0,object.Sa-object.mindrho/(object.rho0*object.beta),object.S[1,:,:])
        #Recompute density difference
        update_density(object)

    return

def update_melt(object):
    """Routine to compute melt rate based on three equations (Jenkins)"""

    #Friction velocity
    object.ustar = (object.Cdtop*(im(object.U[1,:,:])**2+jm(object.V[1,:,:])**2+object.utide**2))**.5 * object.tmask
    
    #Gamma_T and Gamma_S
    if object.usegamtfix:
        object.gamT = object.gamTfix
        object.gamS = object.gamT/35. #Ratio gamT/gamS from Asay-Davis (ISOMIP)
    else: 
        AA = 2.12*np.log(object.ustar*object.D[1,:,:]/object.nu0+1e-12)
        object.gamT = object.ustar/(AA + 12.5*object.Pr**(2./3) - 8.68)
        object.gamS = object.ustar/(AA + 12.5*object.Sc**(2./3) - 8.68)

    #Solve three equations
    That = (object.l2+object.l3*object.zb)
    Chat = object.cp/(object.L-object.ci*object.Ti)
    Ctil = object.ci/object.cp

    b = Chat*object.gamT*(That-object.T[1,:,:])+object.gamS*(1+Chat*Ctil*(That+object.l1*object.S[1,:,:]))
    c = Chat*object.gamT*object.gamS*(That-object.T[1,:,:]+object.l1*object.S[1,:,:])

    #Melt rate
    object.melt = .5*(-b+np.sqrt(b**2-4*c)) * object.tmask

    #Temperature at ice shelf base
    object.Tb = div0(Chat*object.gamT*object.T[1,:,:]-object.melt,Chat*object.gamT+Chat*Ctil*object.melt) * object.tmask

    return

def update_entrainment(object):
    """Entrainment rate of ambient water into mixed layer"""

    if object.entpar == 'Holland':
        #Holland et al (2006) doi:10.1175/JPO2970.1
        object.entr = object.cl*object.Kh/object.Ah**2*(np.maximum(0,im(object.U[1,:,:])**2+jm(object.V[1,:,:])**2-object.g*object.drho*object.Kh/object.Ah*object.D[1,:,:]))**.5 * object.tmask
        object.detr = 0.*object.entr
    elif object.entpar == 'Gaspar':
        #Gaspar et al (1988) doi:10.1175/1520-0485(1988)018<0161:MTSCOT>2.0.CO;2
        #Gladish et al (2012) doi:10.3189/2012JoG12J003
        object.Sb = (object.Tb-object.l2-object.l3*object.zb)/object.l1
        object.drhob = (object.beta*(object.S[1,:,:]-object.Sb) - object.alpha*(object.T[1,:,:]-object.Tb)) * object.tmask
        drhopos = np.maximum(.0001,object.drho)
        object.ent = 2*object.mu/object.g * div0(object.ustar**3,object.D[1,:,:]*drhopos) - div0(object.drhob,drhopos)*object.melt * object.tmask
        object.entr = np.maximum(object.ent,0)
        object.detr = np.minimum(object.maxdetr,np.maximum(-object.ent,0))

    #Additional entrainment to prevent D<minD
    object.convD = convT(object,object.D[1,:,:])
    object.ent2 = np.maximum(0,(object.minD-object.D[0,:,:])/(2*object.dt)-(object.convD+object.melt+object.entr-object.detr)) *object.tmask

    #Net entrainment
    object.nentr = object.entr+object.ent2-object.detr
    return

def prepare_convergence(object):
    """Compute a bunch of variables that are used several times each time step"""

    object.Dym1    = np.roll(        object.D[1,:,:]*object.tmask,-1,axis=0)
    object.Dyp1    = np.roll(        object.D[1,:,:]*object.tmask, 1,axis=0)
    object.Dxm1    = np.roll(        object.D[1,:,:]*object.tmask,-1,axis=1)
    object.Dxp1    = np.roll(        object.D[1,:,:]*object.tmask, 1,axis=1)
    object.Dxm1ym1 = np.roll(np.roll(object.D[1,:,:]*object.tmask,-1,axis=1),-1,axis=0)
    object.Dxp1ym1 = np.roll(np.roll(object.D[1,:,:]*object.tmask, 1,axis=1),-1,axis=0)
    object.Dxm1yp1 = np.roll(np.roll(object.D[1,:,:]*object.tmask,-1,axis=1), 1,axis=0)

    object.Vyp1    = np.roll(object.V[1,:,:],1,axis=0)  
    object.Uxp1    = np.roll(object.U[1,:,:],1,axis=1)    

    object.Upos = np.maximum(object.U[1,:,:],0)
    object.Uneg = np.minimum(object.U[1,:,:],0)
    object.Vpos = np.maximum(object.V[1,:,:],0)
    object.Vneg = np.minimum(object.V[1,:,:],0)

    object.Vyp1pos = np.maximum(object.Vyp1,0)
    object.Vyp1neg = np.minimum(object.Vyp1,0)
    object.Uxp1pos = np.maximum(object.Uxp1,0)
    object.Uxp1neg = np.minimum(object.Uxp1,0)

    object.Vip = ip_v(object,object.V[1,:,:])
    object.Vim = im_v(object,object.V[1,:,:])
    object.Uip = ip_u(object,object.U[1,:,:])
    object.Uim = im_u(object,object.U[1,:,:])

    object.Vjp = jp_v(object,object.V[1,:,:])
    object.Vjm = jm_v(object,object.V[1,:,:])
    object.Ujp = jp_u(object,object.U[1,:,:])
    object.Ujm = jm_u(object,object.U[1,:,:])

    object.signU = np.sign(object.U[1,:,:])
    object.signV = np.sign(object.V[1,:,:])

    return

def prepare_laplacian(object):

    object.D0ip = ip_t(object,object.D[0,:,:])
    object.D0im = im_t(object,object.D[0,:,:])    
    object.D0jp = jp_t(object,object.D[0,:,:])
    object.D0jm = jm_t(object,object.D[0,:,:]) 

    return
