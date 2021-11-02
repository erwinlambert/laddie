import numpy as np
import xarray as xr

from integrate import updatesecondary,intD,intu,intv,intT,intS

def create_mask(object):
    """Create mask
    
    tmask: mask of T grid below ice shelf
    grd:   mask of grounded ice
    ocn:   mask of open ocean
    
    """

    object.tmask = np.where(object.mask==3,1,0)
    object.grd   = np.where(object.mask==2,1,0)
    object.grd   = np.where(object.mask==1,1,object.grd)
    object.ocn   = np.where(object.mask==0,1,0)

    #Rolled masks
    object.tmaskym1    = np.roll(object.tmask,-1,axis=0)
    object.tmaskyp1    = np.roll(object.tmask, 1,axis=0)
    object.tmaskxm1    = np.roll(object.tmask,-1,axis=1)
    object.tmaskxp1    = np.roll(object.tmask, 1,axis=1)    
    object.tmaskxm1ym1 = np.roll(np.roll(object.tmask,-1,axis=0),-1,axis=1)
    object.tmaskxm1yp1 = np.roll(np.roll(object.tmask, 1,axis=0),-1,axis=1)
    object.tmaskxp1ym1 = np.roll(np.roll(object.tmask,-1,axis=0), 1,axis=1)

    object.ocnym1      = np.roll(object.ocn,-1,axis=0)
    object.ocnyp1      = np.roll(object.ocn, 1,axis=0)
    object.ocnxm1      = np.roll(object.ocn,-1,axis=1)
    object.ocnxp1      = np.roll(object.ocn, 1,axis=1)

    object.grdNu = 1-np.roll((1-object.grd)*(1-np.roll(object.grd,-1,axis=1)),-1,axis=0)
    object.grdSu = 1-np.roll((1-object.grd)*(1-np.roll(object.grd,-1,axis=1)), 1,axis=0)
    object.grdEv = 1-np.roll((1-object.grd)*(1-np.roll(object.grd,-1,axis=0)),-1,axis=1)
    object.grdWv = 1-np.roll((1-object.grd)*(1-np.roll(object.grd,-1,axis=0)), 1,axis=1)

    #Extract ice shelf front
    object.isfE = object.ocn*object.tmaskxp1
    object.isfN = object.ocn*object.tmaskyp1
    object.isfW = object.ocn*object.tmaskxm1
    object.isfS = object.ocn*object.tmaskym1
    object.isf  = object.isfE+object.isfN+object.isfW+object.isfS
    
    #Extract grounding line 
    object.grlE = object.grd*object.tmaskxp1
    object.grlN = object.grd*object.tmaskyp1
    object.grlW = object.grd*object.tmaskxm1
    object.grlS = object.grd*object.tmaskym1
    object.grl  = object.grlE+object.grlN+object.grlW+object.grlS

    #Create masks for U- and V- velocities at N/E faces of grid points
    object.umask = (object.tmask+object.isfW)*(1-np.roll(object.grlE,-1,axis=1))
    object.vmask = (object.tmask+object.isfS)*(1-np.roll(object.grlN,-1,axis=0))

    object.umaskxp1 = np.roll(object.umask,1,axis=1)
    object.vmaskyp1 = np.roll(object.vmask,1,axis=0)
    

def create_grid(object):   
    #Spatial parameters
    object.nx = len(object.x)
    object.ny = len(object.y)
    object.dx = (object.x[1]-object.x[0]).values
    object.dy = (object.y[1]-object.y[0]).values
    object.xu = object.x+object.dx
    object.yv = object.y+object.dy

    # Assure free-slip is used in 1D simulation
    if (len(object.y)==3 or len(object.x)==3):
        print('1D run, using free slip')
        object.slip = 0  

def initialize_vars(object,days,restartfile):
    #Major variables. Three arrays for storage of previous timestep, current timestep, and next timestep
    object.u = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.v = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.D = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.T = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.S = np.zeros((3,object.ny,object.nx)).astype('float64')
    
    #Draft dz/dx and dz/dy on t-grid
    object.dzdx = np.gradient(object.zb,object.dx,axis=1)
    object.dzdy = np.gradient(object.zb,object.dy,axis=0)

    #Initial values
    try:
        dsinit = xr.open_dataset(f"../../results/restart/{object.ds['name_geo'].values}_{restartfile}.nc")
        object.tstart = dsinit.tend.values
        object.u = dsinit.u.values
        object.v = dsinit.v.values
        object.D = dsinit.D.values
        object.T = dsinit.T.values
        object.S = dsinit.S.values
        print(f'Starting from restart file at day {object.tstart:.3f}')
    except:    
        object.tstart = 0
        if len(object.Tz.shape)==1:
            object.Ta   = np.interp(object.zb,object.z,object.Tz)
            object.Sa   = np.interp(object.zb,object.z,object.Sz)
        elif len(object.Tz.shape)==3:
            object.Ta = object.Tz[np.int_(-object.zb),object.ind[0],object.ind[1]]
            object.Sa = object.Sz[np.int_(-object.zb),object.ind[0],object.ind[1]]
            
        #object.Tf   = (object.l1*object.Sa+object.l2+object.l3*object.zb).values
        
        object.D += object.Dinit
        for n in range(3):
            object.T[n,:,:] = object.Ta-.1
            object.S[n,:,:] = object.Sa-.1
        print('Starting from noflow')
        
        #Perform first integration step with 1 dt
        updatesecondary(object)
        intD(object,object.dt)
        intu(object,object.dt)
        intv(object,object.dt)
        intT(object,object.dt)
        intS(object,object.dt)

    #For storing time averages
    object.count = 0
    object.saveint = int(object.saveday*3600*24/object.dt)
    object.diagint = int(object.diagday*3600*24/object.dt)
    object.restint = int(object.restday*3600*24/object.dt)
    
    object.dsav = object.ds
    object.dsav['U'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['V'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['D'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['T'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['S'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['melt'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['entr'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['tmask'] = (['y','x'], object.tmask)
    object.dsav['mask']  = (['y','x'], object.mask.data)
    object.dsav['name_model'] = 'Layer'
    object.dsav['tstart'] = object.tstart
    
    #For storing restart file
    object.dsre = object.ds
    object.dsre = object.dsre.assign_coords({'n':np.array([0,1,2])})

    object.nt = int(days*24*3600/object.dt)+1    # Number of time steps
    object.tend = object.tstart+days
    object.time = np.linspace(object.tstart,object.tend,object.nt)  # Time in days
        