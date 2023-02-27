import os,sys
import numpy as np
import xarray as xr

from integrate import updatesecondary,intD,intU,intV,intT,intS

def create_rundir(object):
    #Create run directory and logfile
    object.name = object.config["Run"]["name"]
    object.rundir = os.path.join(object.config["Directories"]["results"],object.name)
    object.logfile = os.path.join(object.rundir,object.config["Filenames"]["logfile"])

    try:
        os.mkdir(object.rundir)
    except:
        sys.exit('Error: cannot create run dir, aborting. Try choosing a new runname')

    object.print2log("Rundir created")

    return

def read_config(object):

    #Inherit all config keys and check whether they are of the correct form / type
    object.print2log("Starting to read config file")

    #Run
    object.days = object.config["Run"]["days"]

    #Time
    object.dt      = object.config["Time"]["dt"]
    object.restday = object.config["Time"]["restday"]
    object.saveday = object.config["Time"]["saveday"]

    #Geometry
    object.geomfile   = object.config["Geometry"]["filename"]
    object.lonlat     = object.config["Geometry"]["lonlat"]
    object.projection = object.config["Geometry"]["projection"]
    object.coarsen    = object.config["Geometry"]["coarsen"]

    #Forcing
    object.forcop = object.config["Forcing"]["option"]
    assert object.forcop in ["tanh","linear","linear2","isomip"], "Invalid input for Forcing.option"
    object.z0     = object.config["Forcing"]["z0"]
    if object.z0>0: object.z0 = -object.z0
    object.S0     = object.config["Forcing"]["S0"]
    object.S1     = object.config["Forcing"]["S1"]
    object.T1     = object.config["Forcing"]["T1"]
    object.z1     = object.config["Forcing"]["z1"]
    object.isomipcond = object.config["Forcing"]["isomipcond"]
    assert object.isomipcond in ["warm","cold"]

    #Options
    object.correctisf = object.config["Options"]["correctisf"]
    object.slip       = object.config["Options"]["slip"]
    assert object.slip >= 0, "Invalid input for Options.slip"
    assert object.slip <= 2, "Invalid input for Options.slip"
    object.convop     = object.config["Options"]["convop"]
    assert object.convop in [0,1,2], "Invalid input for Options.convop"

    #Directories
    object.resultdir = object.config["Directories"]["results"]

    #Filenames
    object.restartfile = object.config["Filenames"]["restartfile"]

    #Parameters
    object.Dinit    = object.config["Initialisation"]["Dinit"]
    
    object.utide    = object.config["Parameters"]["utide"]
    object.Ti       = object.config["Parameters"]["Ti"]
    object.f        = object.config["Parameters"]["f"]
    object.rhofw    = object.config["Parameters"]["rhofw"]
    object.rho0     = object.config["Parameters"]["rho0"]
    object.rhoi     = object.config["Parameters"]["rhoi"]
    object.gamTfix  = object.config["Parameters"]["gamTfix"]
    object.Cd       = object.config["Parameters"]["Cd"]
    object.Cdtop    = object.config["Parameters"]["Cdtop"]
    object.Ah       = object.config["Parameters"]["Ah"]
    object.Kh       = object.config["Parameters"]["Kh"]
    object.entpar   = object.config["Parameters"]["entpar"]
    object.mu       = object.config["Parameters"]["mu"]
    object.maxdetr  = object.config["Parameters"]["maxdetr"]
    object.minD     = object.config["Parameters"]["minD"]
    object.vcut     = object.config["Parameters"]["vcut"]
    
    object.alpha    = object.config["EOS"]["alpha"]
    object.beta     = object.config["EOS"]["beta"]
    object.l1       = object.config["EOS"]["l1"]
    object.l2       = object.config["EOS"]["l2"]
    object.l3       = object.config["EOS"]["l3"]

    object.g        = object.config["Constants"]["g"]
    object.L        = object.config["Constants"]["L"]
    object.cp       = object.config["Constants"]["cp"]
    object.ci       = object.config["Constants"]["ci"]
    object.CG       = object.config["Constants"]["CG"]
    object.Pr       = object.config["Constants"]["Pr"]
    object.Sc       = object.config["Constants"]["Sc"]
    object.nu0      = object.config["Constants"]["nu0"]

    object.nu       = object.config["Numerics"]["nu"]
    object.spy      = object.config["Numerics"]["spy"]
    object.dpm      = object.config["Numerics"]["dpm"]

    object.mindrho  = object.config["Convection"]["mindrho"]
    object.convtime = object.config["Convection"]["convtime"]

    object.print2log("Finished reading config. All input correct")

    return






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

    #Define ocean rolled masks
    object.ocnym1      = np.roll(object.ocn,-1,axis=0)
    object.ocnyp1      = np.roll(object.ocn, 1,axis=0)
    object.ocnxm1      = np.roll(object.ocn,-1,axis=1)
    object.ocnxp1      = np.roll(object.ocn, 1,axis=1)
    
    if object.correctisf:
        #Mask ice shelf points sticking out as ocean
        #print('removing points sticking out')
        #print(np.sum(object.tmask*(object.ocnym1*object.ocnyp1+object.ocnxm1*object.ocnxp1)>0))
        while np.sum(object.tmask*(object.ocnym1*object.ocnyp1+object.ocnxm1*object.ocnxp1)>0) > 0:
            object.ocn = np.where(object.tmask*(object.ocnym1*object.ocnyp1+object.ocnxm1*object.ocnxp1)>0,1,object.ocn)
            object.tmask = np.where(object.ocn==1,0,object.tmask)
            object.mask  = np.where(object.ocn==1,0,object.mask)
            #Redefine rolled masks
            object.ocnym1      = np.roll(object.ocn,-1,axis=0)
            object.ocnyp1      = np.roll(object.ocn, 1,axis=0)
            object.ocnxm1      = np.roll(object.ocn,-1,axis=1)
            object.ocnxp1      = np.roll(object.ocn, 1,axis=1)
            #print(np.sum(object.tmask*(object.ocnym1*object.ocnyp1+object.ocnxm1*object.ocnxp1)>0))
    
    #Rolled tmasks
    object.tmaskym1    = np.roll(object.tmask,-1,axis=0)
    object.tmaskyp1    = np.roll(object.tmask, 1,axis=0)
    object.tmaskxm1    = np.roll(object.tmask,-1,axis=1)
    object.tmaskxp1    = np.roll(object.tmask, 1,axis=1)    
    
    if object.correctisf:
        #Mask isolated ice shelf points as grounded
        #print('removing isolated points')
        #print(np.sum(np.logical_and(object.tmask==1,object.tmaskym1+object.tmaskyp1+object.tmaskxm1+object.tmaskxp1==0)))
        while np.sum(np.logical_and(object.tmask==1,object.tmaskym1+object.tmaskyp1+object.tmaskxm1+object.tmaskxp1==0)):
            object.mask = np.where(np.logical_and(object.tmask==1,object.tmaskym1+object.tmaskyp1+object.tmaskxm1+object.tmaskxp1==0),2,object.mask)
            object.tmask = np.where(object.mask==3,1,0)
            object.grd   = np.where(object.mask==2,1,0)
            #Redefine rolled masks
            object.tmaskym1    = np.roll(object.tmask,-1,axis=0)
            object.tmaskyp1    = np.roll(object.tmask, 1,axis=0)
            object.tmaskxm1    = np.roll(object.tmask,-1,axis=1)
            object.tmaskxp1    = np.roll(object.tmask, 1,axis=1)   
            #print(np.sum(np.logical_and(object.tmask==1,object.tmaskym1+object.tmaskyp1+object.tmaskxm1+object.tmaskxp1==0)))

        #Update rolled masks
        object.ocnym1      = np.roll(object.ocn,-1,axis=0)
        object.ocnyp1      = np.roll(object.ocn, 1,axis=0)
        object.ocnxm1      = np.roll(object.ocn,-1,axis=1)
        object.ocnxp1      = np.roll(object.ocn, 1,axis=1)    
        object.tmaskym1    = np.roll(object.tmask,-1,axis=0)
        object.tmaskyp1    = np.roll(object.tmask, 1,axis=0)
        object.tmaskxm1    = np.roll(object.tmask,-1,axis=1)
        object.tmaskxp1    = np.roll(object.tmask, 1,axis=1)      

    object.tmaskxm1ym1 = np.roll(np.roll(object.tmask,-1,axis=0),-1,axis=1)
    object.tmaskxm1yp1 = np.roll(np.roll(object.tmask, 1,axis=0),-1,axis=1)
    object.tmaskxp1ym1 = np.roll(np.roll(object.tmask,-1,axis=0), 1,axis=1)    

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

    #Temporal parameters
    object.nt = int(object.days*24*3600/object.dt)+1    # Number of time steps
    object.tend = object.tstart+object.days
    object.time = np.linspace(object.tstart,object.tend,object.nt)  # Time in days

def initialise_vars(object):
    
    #Check whether entrainment parameterisation is valid
    assert object.entpar in ['Holland','Gaspar']
    
    #Major variables. Three arrays for storage of previous timestep, current timestep, and next timestep
    object.U = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.V = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.D = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.T = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.S = np.zeros((3,object.ny,object.nx)).astype('float64')
    
    #Include ice shelf front gradient
    object.zb = xr.where(object.isf,0,object.zb)
    
    #Remove positive values. Set shallowest ice shelf draft to 10 meters
    object.zb = xr.where(np.logical_and(object.tmask==1,object.zb>-10),-10,object.zb)
    
    #Draft dz/dx and dz/dy on t-grid
    object.dzdx = np.gradient(object.zb,object.dx,axis=1)
    object.dzdy = np.gradient(object.zb,object.dy,axis=0)

    #For dynamic ice module
    object.Ussa = np.zeros((2,len(object.y),len(object.x)))
    object.Vssa = np.zeros((2,len(object.y),len(object.x)))

    #Initial values
    try:
        dsinit = xr.open_dataset(f"../results/restart/{object.ds['name_geo'].values}_{object.restartfile}.nc")
        object.tstart = dsinit.tend.values
        object.U = dsinit.U.values
        object.V = dsinit.V.values
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
            object.Ta = object.Tz[np.int_(.01*-object.zb),object.ind[0],object.ind[1]]
            object.Sa = object.Sz[np.int_(.01*-object.zb),object.ind[0],object.ind[1]]
                    
        object.D += object.Dinit
        for n in range(3):
            object.T[n,:,:] = object.Ta
            object.S[n,:,:] = object.Sa-.1
        print('Starting from noflow')
        
        #Perform first integration step with 1 dt
        updatesecondary(object)
        intD(object,object.dt)
        intU(object,object.dt)
        intV(object,object.dt)
        intT(object,object.dt)
        intS(object,object.dt)
    return

def prepare_output(object):

    #For storing time averages
    object.count = 0
    object.saveint = int(object.saveday*3600*24/object.dt)
    object.diagint = int(object.diagday*3600*24/object.dt)
    object.restint = int(object.restday*3600*24/object.dt)
    
    object.dsav = object.ds
    object.dsav = object.dsav.drop_vars(['Tz','Sz'])
    object.dsav = object.dsav.drop_dims(['z'])
    object.dsav['U'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['V'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['D'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['T'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['S'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['melt'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['entr'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['ent2'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))
    object.dsav['detr'] = (['y','x'], np.zeros((object.ny,object.nx)).astype('float64'))    
    object.dsav['tmask'] = (['y','x'], object.tmask)
    object.dsav['mask']  = (['y','x'], object.mask.data)
    object.dsav['zb'] = (['y','x'], object.zb.data)
    object.dsav['H'] = (['y','x'], object.H.data)
    object.dsav['Ui'] = (['y','x'], object.Ussa[0,:,:])
    object.dsav['Vi'] = (['y','x'], object.Vssa[0,:,:])
    object.dsav['name_model'] = 'LADDIE'
    object.dsav['tstart'] = object.tstart

    #For storing restart file
    object.dsre = object.ds
    object.dsre = object.dsre.drop_vars(['Tz','Sz'])
    object.dsre = object.dsre.drop_dims(['z'])    
    object.dsre = object.dsre.assign_coords({'n':np.array([0,1,2])})

    return