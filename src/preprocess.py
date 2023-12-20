import os,sys
import numpy as np
import xarray as xr
import datetime as dt

from integrate import updatesecondary,intD,intU,intV,intT,intS
from tools import compute_average_NN, tryread

def create_rundir(object,configfile):
    """Create run directory and logfile"""

    #Read run name
    object.name = tryread(object,"Run","name",str)
    #Read directory to store output
    object.resultdir = tryread(object,"Directories","results",str,checkdir=True)
    #Read flag to force new folder
    object.forcenewdir = tryread(object,"Directories","forcenewdir",bool,default=True)
    #Read logfile
    object.logfilename = tryread(object,"Filenames","logfile",str,default="log.txt")

    #Derive desired run directory:
    object.rundir = os.path.join(object.resultdir,object.name)
    if object.forcenewdir or os.path.isdir(object.rundir) == False:
        try:
            #Create rundirectory
            os.mkdir(object.rundir)
        except:
            try:
                #Create rundirectory with current date
                object.rundir = os.path.join(object.resultdir,dt.datetime.today().strftime(f"{object.name}_%Y-%m-%d"))
                os.mkdir(object.rundir)
            except:
                for n in range(100):
                    try:
                        #Create rundirectory with current date and incremental number. Give up after 100 tries
                        object.rundir = os.path.join(object.resultdir,dt.datetime.today().strftime(f"{object.name}_%Y-%m-%d_{n}"))
                        os.mkdir(object.rundir)
                        break
                    except:
                        continue
        result = f"Created new run directory {object.rundir}"
    else:
        result = f"Continuing run in existing directory {object.rundir}"

    #Create log file
    object.logfile = os.path.join(object.rundir,object.logfilename)

    #Copy config file to run directory
    os.system(f"cp {configfile} {object.rundir}")

    object.print2log(result)

    return

def read_config(object):
    """Inherit all config keys and check whether they are of the correct form / type"""

    object.print2log("========= Starting to read config file =============")

    #Run
    object.days = tryread(object,"Run","days",float)

    #Time
    object.dt      = tryread(object,"Time","dt",int,allowconversion=False)
    object.restday = tryread(object,"Time","restday",float)
    object.saveday = tryread(object,"Time","saveday",float)
    object.diagday = tryread(object,"Time","diagday",float)

    #Geometry
    object.geomfile       = tryread(object,"Geometry","filename",str,checkfile=True)
    object.maskoption     = tryread(object,"Geometry","maskoption",str,["BM","UFEMISM","ISOMIP"],default="BM")
    object.geomyear       = tryread(object,"Geometry","geomyear",int,(0,1e20),allowconversion=False,default=0)
    object.lonlat         = tryread(object,"Geometry","lonlat",bool,default=False)
    if object.lonlat:
        object.projection = tryread(object,"Geometry","projection",str,default="epsg:3031")
    object.coarsen        = tryread(object,"Geometry","coarsen",int,default=1)
    object.calvthresh     = tryread(object,"Geometry","calvthresh",float,(0,1e20),default=0)
    object.removebergs    = tryread(object,"Geometry","removebergs",bool,default=False)
    object.correctisf     = tryread(object,"Geometry","correctisf",bool,default=False)
    object.fillisolated   = tryread(object,"Geometry","fillisolated",bool,default=False)

    #Forcing
    object.forcop         = tryread(object,"Forcing","option",str,["tanh","linear","linear2","isomip","file"])
    #Read forcing-specific parameters
    if object.forcop == "file": 
        object.forcfile   = tryread(object,"Forcing","filename",str,checkfile=True)
    if object.forcop == "tanh":
        object.z1         = tryread(object,"Forcing","z1",float,(-5000,0))
        object.drho0      = tryread(object,"Forcing","drho0",float,(0,100))
    if object.forcop in ["tanh","linear","linear2"]:
        object.z0         = tryread(object,"Forcing","z0",float,(-5000,0))
        object.S0         = tryread(object,"Forcing","S0",float,(0,100))
        object.S1         = tryread(object,"Forcing","S1",float,(0,100))
        object.T1         = tryread(object,"Forcing","T1",float,(-100,100))
    if object.forcop == "isomip":
        object.isomipcond = tryread(object,"Forcing","isomipcond",str,['warm','cold'])

    #Options
    object.slip           = tryread(object,"Options","slip",float,(0,2))
    object.convop         = tryread(object,"Options","convop",int,[0,1,2],allowconversion=False)
    if object.convop == 2: object.print2log("WARNING: convop = 2 (convective restoring) is experimental and may lead to instabilities. Consider using convop = 0 or 1")
    object.boundop        = tryread(object,"Options","boundop",int,[1,2],allowconversion=False)
    object.usegamtfix     = tryread(object,"Options","usegamtfix",bool)

    #Filenames
    object.fromrestart      = tryread(object,"Initialisation","fromrestart",bool)
    if object.fromrestart:
        object.restartfile  = tryread(object,"Filenames","restartfile",str)
    else:
        object.Dinit        = tryread(object,"Initialisation","Dinit",float,default=10.0)
        object.dTinit       = tryread(object,"Initialisation","dTinit",float,default=0.0)
        object.dSinit       = tryread(object,"Initialisation","dSinit",float,(-100,0),default=-0.1)

    #Parameters
    object.utide    = tryread(object,"Parameters","utide",float,(0,1e20))
    object.Ti       = tryread(object,"Parameters","Ti",float)
    object.f        = tryread(object,"Parameters","f",float)
    object.rhofw    = tryread(object,"Parameters","rhofw",float,(0,1e20))
    object.rho0     = tryread(object,"Parameters","rho0",float,(0,1e20))
    object.rhoi     = tryread(object,"Parameters","rhoi",float,(0,1e20))
    if object.usegamtfix:
        object.gamTfix  = tryread(object,"Parameters","gamTfix",float,(0,1e20))
    object.Cd       = tryread(object,"Parameters","Cd",float,(0,1e20))
    object.Cdtop    = tryread(object,"Parameters","Cdtop",float,(0,1e20))
    object.Ah       = tryread(object,"Parameters","Ah",float,(0,1e20))
    object.Kh       = tryread(object,"Parameters","Kh",float,(0,1e20))
    object.entpar   = tryread(object,"Parameters","entpar",str,['Holland','Gaspar'])
    if object.entpar == 'Gaspar':
        object.mu       = tryread(object,"Parameters","mu",float,(0,1e20))
    if object.entpar == 'Holland':
        object.cl   = tryread(object,"Parameters","cl",float,(0,1e20),default=.01775)
    object.maxdetr  = tryread(object,"Parameters","maxdetr",float,(0,1e20))
    object.minD     = tryread(object,"Parameters","minD",float,(0,1e20))
    object.vcut     = tryread(object,"Parameters","vcut",float,(0,1e20),default=1.414) 
    
    object.alpha    = tryread(object,"EOS","alpha",float,(0,1e20))
    object.beta     = tryread(object,"EOS","beta",float,(0,1e20))
    object.l1       = tryread(object,"EOS","l1",float,(-1e20,0),default=-5.73e-2)
    object.l2       = tryread(object,"EOS","l2",float,(0,1e20),default=8.32e-2)
    object.l3       = tryread(object,"EOS","l3",float,(0,1e20),default=7.61e-4)

    object.g        = tryread(object,"Constants","g",float,(0,1e20),default=9.81)
    object.L        = tryread(object,"Constants","L",float,(0,1e20),default=3.34e5)
    object.cp       = tryread(object,"Constants","cp",float,(0,1e20),default=3.974e3)
    object.ci       = tryread(object,"Constants","ci",float,(0,1e20),default=2009)
    object.CG       = tryread(object,"Constants","CG",float,(0,1e20),default=5.9e-4)
    object.Pr       = tryread(object,"Constants","Pr",float,(0,1e20),default=13.8)
    object.Sc       = tryread(object,"Constants","Sc",float,(0,1e20),default=2432.0)
    object.nu0      = tryread(object,"Constants","nu0",float,(0,1e20),default=1.95e-6)

    object.nu       = tryread(object,"Numerics","nu",float,(0,1))
    object.spy      = tryread(object,"Numerics","spy",int,(0,1e20),default=31536000)
    object.dpm      = tryread(object,"Numerics","dpm",list,default=[31,28,31,30,31,30,31,31,30,31,30,31])

    if object.convop == 0:
        object.mindrho  = tryread(object,"Convection","mindrho",float,(0,1e20))
    object.convtime     = tryread(object,"Convection","convtime",float,(0,1e20),default=1000)

    object.print2log("============= Finished reading config. All input correct =======")
    object.print2log("================================================================")
    return

def create_mask(object):
    """Create masks and extract ice shelf front, grounding line, etc"""

    #Main masks
    object.tmask = np.where(object.mask==3,1,0)             #Grid cells with floating ice, on which computations are applied
    object.grd   = np.where(object.mask==2,1,0)             #Grid cells with grounded ice or bare rock, treated the same
    object.grd   = np.where(object.mask==1,1,object.grd)    #Grid cells with grounded ice or bare rock, treated the same
    object.ocn   = np.where(object.mask==0,1,0)             #Grid cells with ocean

    #Define ocean neighbour masks, used to compute gradients and boundaries (ice shelf front)
    #ym1 indicates mask shifted by -1 grid cell in the y-direction; in other words: the ocean mask in the North
    object.ocnym1      = np.roll(object.ocn,-1,axis=0)
    object.ocnyp1      = np.roll(object.ocn, 1,axis=0)
    object.ocnxm1      = np.roll(object.ocn,-1,axis=1)
    object.ocnxp1      = np.roll(object.ocn, 1,axis=1)
    
    #If required, smoothen ice shelf front by converting some grid cells from ice shelf to ocean
    if object.correctisf:
        apply_correct_isf(object)

    #Define ice shelf neighbour masks, used to compute gradients and boundaries
    object.tmaskym1    = np.roll(object.tmask,-1,axis=0)
    object.tmaskyp1    = np.roll(object.tmask, 1,axis=0)
    object.tmaskxm1    = np.roll(object.tmask,-1,axis=1)
    object.tmaskxp1    = np.roll(object.tmask, 1,axis=1)    
    
    #If required, fill isolated ice shelf grid cells surrounded by grounded ice (subglacial lakes of size 1 grid cell)
    if object.fillisolated:
        apply_fill_isolated(object)

    #Define ice shelf neighbour masks at NW etc
    object.tmaskxm1ym1 = np.roll(np.roll(object.tmask,-1,axis=0),-1,axis=1)
    object.tmaskxm1yp1 = np.roll(np.roll(object.tmask, 1,axis=0),-1,axis=1)
    object.tmaskxp1ym1 = np.roll(np.roll(object.tmask,-1,axis=0), 1,axis=1)    

    #Define grounded ice cells along the grounding line at U- and V-grids, used to compute slip.
    #grdNu indicates grounded grid cells bordering the ice shelf on the Northern side (positive y-direction),
    #where the U-grid falls on the grounding line.
    object.grdNu = 1-np.roll((1-object.grd)*(1-np.roll(object.grd,-1,axis=1)),-1,axis=0)
    object.grdSu = 1-np.roll((1-object.grd)*(1-np.roll(object.grd,-1,axis=1)), 1,axis=0)
    object.grdEv = 1-np.roll((1-object.grd)*(1-np.roll(object.grd,-1,axis=0)),-1,axis=1)
    object.grdWv = 1-np.roll((1-object.grd)*(1-np.roll(object.grd,-1,axis=0)), 1,axis=1)

    #Define ocean grid cells along ice shelf front at each side
    #isfE indicates ocean grid cells bordering the ice shelf on the Eastern side (positive x-direction)
    object.isfE = object.ocn*object.tmaskxp1
    object.isfN = object.ocn*object.tmaskyp1
    object.isfW = object.ocn*object.tmaskxm1
    object.isfS = object.ocn*object.tmaskym1
    object.isf  = object.isfE+object.isfN+object.isfW+object.isfS
    
    #Define grounded grid cells along grounding line at each side
    #grlE indicates grounded grid cells bordering the ice shelf on the Eastern side (positive x-direction)
    object.grlE = object.grd*object.tmaskxp1
    object.grlN = object.grd*object.tmaskyp1
    object.grlW = object.grd*object.tmaskxm1
    object.grlS = object.grd*object.tmaskym1
    object.grl  = object.grlE+object.grlN+object.grlW+object.grlS

    #Create masks for U- and V- velocities at N/E faces of grid points
    object.umask = (object.tmask+object.isfW)*(1-np.roll(object.grlE,-1,axis=1))
    object.vmask = (object.tmask+object.isfS)*(1-np.roll(object.grlN,-1,axis=0))

    #Rolled u- and v- masks
    object.umaskxp1 = np.roll(object.umask,1,axis=1)
    object.vmaskyp1 = np.roll(object.vmask,1,axis=0)
    
    object.print2log("Finished creating mask")

    return

def apply_correct_isf(object):
    """Convert ice shelf points sticking out into ocean to ocean, for stability"""

    object.print2log("======== Removing grid points sticking out from ice shelf front ========")

    #Compute number of ice shelf points sticking out
    object.print2log(f"{np.sum(object.tmask*(object.ocnym1*object.ocnyp1+object.ocnxm1*object.ocnxp1)>0)} remaining...")

    #Loop while number > 0
    while np.sum(object.tmask*(object.ocnym1*object.ocnyp1+object.ocnxm1*object.ocnxp1)>0) > 0:
        #Convert current points sticking out to ocean
        object.ocn = np.where(object.tmask*(object.ocnym1*object.ocnyp1+object.ocnxm1*object.ocnxp1)>0,1,object.ocn)

        #Recompute main masks
        object.tmask = np.where(object.ocn==1,0,object.tmask)
        object.mask  = np.where(object.ocn==1,0,object.mask)

        #Recompute rolled masks
        object.ocnym1      = np.roll(object.ocn,-1,axis=0)
        object.ocnyp1      = np.roll(object.ocn, 1,axis=0)
        object.ocnxm1      = np.roll(object.ocn,-1,axis=1)
        object.ocnxp1      = np.roll(object.ocn, 1,axis=1)

        #Print number of remaining ice shelf points sticking out
        object.print2log(f"{np.sum(object.tmask*(object.ocnym1*object.ocnyp1+object.ocnxm1*object.ocnxp1)>0)} remaining...")
    object.print2log("======== Finished removing grid points sticking out from ice shelf front======== ")

    return

def apply_fill_isolated(object):
    """Mask isolated ice shelf points as grounded"""

    object.print2log("======== Removing isolated grid points surrounded by grounded ice ========")
    object.print2log(f"{np.sum(np.logical_and(object.tmask==1,object.tmaskym1+object.tmaskyp1+object.tmaskxm1+object.tmaskxp1==0))} remaining...")
    while np.sum(np.logical_and(object.tmask==1,object.tmaskym1+object.tmaskyp1+object.tmaskxm1+object.tmaskxp1==0)):
        object.mask = np.where(np.logical_and(object.tmask==1,object.tmaskym1+object.tmaskyp1+object.tmaskxm1+object.tmaskxp1==0),2,object.mask)
        object.tmask = np.where(object.mask==3,1,0)
        object.grd   = np.where(object.mask==2,1,0)
        #Redefine rolled masks
        object.tmaskym1    = np.roll(object.tmask,-1,axis=0)
        object.tmaskyp1    = np.roll(object.tmask, 1,axis=0)
        object.tmaskxm1    = np.roll(object.tmask,-1,axis=1)
        object.tmaskxp1    = np.roll(object.tmask, 1,axis=1)   
        object.print2log(f"{np.sum(np.logical_and(object.tmask==1,object.tmaskym1+object.tmaskyp1+object.tmaskxm1+object.tmaskxp1==0))} remaining...")
    
    #Update rolled masks
    object.ocnym1      = np.roll(object.ocn,-1,axis=0)
    object.ocnyp1      = np.roll(object.ocn, 1,axis=0)
    object.ocnxm1      = np.roll(object.ocn,-1,axis=1)
    object.ocnxp1      = np.roll(object.ocn, 1,axis=1)    
    object.tmaskym1    = np.roll(object.tmask,-1,axis=0)
    object.tmaskyp1    = np.roll(object.tmask, 1,axis=0)
    object.tmaskxm1    = np.roll(object.tmask,-1,axis=1)
    object.tmaskxp1    = np.roll(object.tmask, 1,axis=1)

    object.print2log("======== Finished removing isolated grid points surrounded by grounded ice ========")

    return

def create_grid(object):
    """Create grid"""

    #Spatial parameters
    object.nx = len(object.x)
    object.ny = len(object.y)
    object.dx = (object.x[1]-object.x[0])
    object.dy = (object.y[1]-object.y[0])
    object.xu = object.x + 0.5*object.dx
    object.yv = object.y + 0.5*object.dy

    #Check if domain is large enough
    if  (object.ny<3 or object.nx<3):
        print(f"INPUT ERROR: domain is too small, make sure domain is at least 3x3 grid cells")
        sys.exit()

    #Make sure free-slip is used in 1D simulation
    if (object.ny==3 or object.nx==3):
        object.print2log("WARNING: one dimension contains 3 grid cells, interpreting as 1D run and using free slip")
        object.slip = 0  

    object.print2log("Finished creating grid")

    return

def dsinit_to_new_geometry(object, dsinit):

    #Inherit start time from restart file
    object.tstart = dsinit.time

    # Check whether object geometry matches restart file geometry
    difftmask = np.sum(np.abs(object.tmask - dsinit.tmask))
    diffumask = np.sum(np.abs(object.umask - dsinit.umask))
    diffvmask = np.sum(np.abs(object.vmask - dsinit.vmask))
    
    totaldiff = difftmask.values+diffumask.values+diffvmask.values
    
    if totaldiff==0:
        # If input geometry and restart geometry match
        object.print2log('Input file geometry matches restart file geometry.')

        object.U = dsinit.U.values
        object.V = dsinit.V.values
        object.D = dsinit.D.values
        object.T = dsinit.T.values
        object.S = dsinit.S.values

    else:
        # If input geometry and restart geometry do not match
        object.print2log(f'Input file geometry does not match restart file geometry: (dtmask + dumask + dvmask) = {totaldiff:.0f} cells. Extrapolate restart file variables to mask from input file.')

        # Inherit values for variables in grid cells that were already marked as iceshelf
        object.T[:] = np.where(np.logical_and(object.tmask==1, dsinit.tmask==1), dsinit.T[:], object.T[:])
        object.S[:] = np.where(np.logical_and(object.tmask==1, dsinit.tmask==1), dsinit.S[:], object.S[:])
        object.D[:] = np.where(np.logical_and(object.tmask==1, dsinit.tmask==1), dsinit.D[:], object.D[:])
        object.U[:] = np.where(np.logical_and(object.umask==1, dsinit.umask==1), dsinit.U[:], object.U[:])
        object.V[:] = np.where(np.logical_and(object.vmask==1, dsinit.vmask==1), dsinit.V[:], object.V[:]) 
  
        # Find new ice shelf cells and fill them with np.nan
        object.T[:] = np.where(np.logical_and(object.tmask[:]==1, dsinit.tmask[:]==0), np.nan, object.T[:])
        object.S[:] = np.where(np.logical_and(object.tmask[:]==1, dsinit.tmask[:]==0), np.nan, object.S[:])
        object.D[:] = np.where(np.logical_and(object.tmask[:]==1, dsinit.tmask[:]==0), np.nan, object.D[:])
        object.U[:] = np.where(np.logical_and(object.umask[:]==1, dsinit.umask[:]==0), np.nan, object.U[:])
        object.V[:] = np.where(np.logical_and(object.vmask[:]==1, dsinit.vmask[:]==0), np.nan, object.V[:])    
    
        # Count empty cells in different masks 
        N_empty_cells_tmask = np.sum(np.isnan(object.T[1]))
        N_empty_cells_umask = np.sum(np.isnan(object.U[1]))
        N_empty_cells_vmask = np.sum(np.isnan(object.V[1]))

        # Fill new cells with NN average, use a while loop to make sure every cell is filled
        object.print2log(f'empty tmask: {N_empty_cells_tmask:.0f}')
        while N_empty_cells_tmask > 0:
            conditiont = np.logical_and(object.tmask[:]==1, dsinit.tmask[:]==0)
            object.T[:] = np.where(conditiont, compute_average_NN(object.T, dsinit.tmask), object.T[:])
            object.S[:] = np.where(conditiont, compute_average_NN(object.S, dsinit.tmask), object.S[:])
            object.D[:] = np.where(conditiont, compute_average_NN(object.D, dsinit.tmask), object.D[:])
            # Update tmask
            dsinit.tmask[:] = np.where(np.logical_and(np.isnan(object.T[1])==False, dsinit.tmask[:]==0), 1, dsinit.tmask[:])
            N_empty_cells_tmask = np.sum(np.isnan(object.T[1]))
            object.print2log(f'empty tmask: {N_empty_cells_tmask:.0f}')

        object.print2log(f'empty umask: {N_empty_cells_umask:.0f}')
        while N_empty_cells_umask > 0:
            conditionu = np.logical_and(object.umask[:]==1, dsinit.umask[:]==0)
            object.U[:] = np.where(conditionu, compute_average_NN(object.U, dsinit.umask),object.U[:])
            # Update umask
            dsinit.umask[:] = np.where(np.logical_and(np.isnan(object.U[1])==False, dsinit.umask[:]==0), 1, dsinit.umask[:])
            N_empty_cells_umask = np.sum(np.isnan(object.U[1]))
            object.print2log(f'empty umask: {N_empty_cells_umask:.0f}')

        object.print2log(f'empty vmask: {N_empty_cells_vmask:.0f}')
        while N_empty_cells_vmask > 0:
            conditionv = np.logical_and(object.vmask[:]==1, dsinit.vmask[:]==0)
            object.V[:] = np.where(conditionv, compute_average_NN(object.V, dsinit.vmask),object.V[:])
            # Update vmask
            dsinit.vmask[:] = np.where(np.logical_and(np.isnan(object.V[1])==False, dsinit.vmask[:]==0), 1, dsinit.vmask[:])
            N_empty_cells_vmask = np.sum(np.isnan(object.V[1]))
            object.print2log(f'empty vmask: {N_empty_cells_vmask:.0f}')

    return object

def initialise_vars(object):
    
    #Major variables. Three arrays for storage of previous timestep, current timestep, and next timestep
    object.U = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.V = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.D = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.T = np.zeros((3,object.ny,object.nx)).astype('float64')
    object.S = np.zeros((3,object.ny,object.nx)).astype('float64')
    
    #Include ice shelf front gradient
    object.zb = np.where(object.isf,0,object.zb)
    
    #Remove positive values. Set shallowest ice shelf draft to 10 meters
    object.zb = np.where(np.logical_and(object.tmask==1,object.zb>-10),-10,object.zb)
    
    #Draft dz/dx and dz/dy on t-grid
    object.dzdx = np.gradient(object.zb,object.dx,axis=1)
    object.dzdy = np.gradient(object.zb,object.dy,axis=0)

    if object.fromrestart:
        object.print2log(f'Restart file: {object.restartfile}')

    try:
        dsinit = xr.open_dataset(object.restartfile)
        dsinit_to_new_geometry(object, dsinit)

        object.print2log(f'Starting from restart file at day {object.tstart:.0f}')
    except:    
        object.tstart = 0.
        if len(object.Tz.shape)==1:
            object.Ta = np.interp(object.zb,object.z,object.Tz)
            object.Sa = np.interp(object.zb,object.z,object.Sz)
        elif len(object.Tz.shape)==3:
            object.Ta = object.Tz[np.maximum(0,np.minimum(4999,np.int_(5000+(object.zb-object.D[1,:,:])))),object.Tax1,object.Tax2]
            object.Sa = object.Sz[np.maximum(0,np.minimum(4999,np.int_(5000+(object.zb-object.D[1,:,:])))),object.Tax1,object.Tax2]
           
        object.D += object.Dinit
        for n in range(3):
            object.T[n,:,:] = object.Ta + object.dTinit
            object.S[n,:,:] = object.Sa + object.dSinit
        object.print2log(f'Starting from scratch with zero velocity and uniform thickness {object.Dinit:.0f} m')
        
        #Perform first integration step with 1 dt
        updatesecondary(object)
        intD(object,object.dt)
        intU(object,object.dt)
        intV(object,object.dt)
        intT(object,object.dt)
        intS(object,object.dt)

    return

def prepare_output(object):

    #Temporal parameters
    object.nt = int(object.days*24*3600/object.dt)+1    # Number of time steps
    object.tend = object.tstart+object.days
    object.time = np.linspace(object.tstart,object.tend,object.nt)  # Time in days

    #For storing time averages
    object.count = 0
    object.saveint = int(object.saveday*3600*24/object.dt)
    object.diagint = int(object.diagday*3600*24/object.dt)
    object.restint = int(object.restday*3600*24/object.dt)

    object.dsav = xr.Dataset()
    object.dsav = object.dsav.assign_coords({'x':object.x,'y':object.y})
    if object.lonlat:
        object.dsav = object.dsav.assign_coords({'lon':object.lon,'lat':object.lat})

    object.Uav = np.zeros((object.ny,object.nx))
    object.dsav['U'] = (['y','x'], object.Uav.astype('float64'))
    object.dsav['U'].attrs['name'] = 'Ocean velocity in x-direction'
    object.dsav['U'].attrs['units'] = 'm/s'

    object.Vav = np.zeros((object.ny,object.nx))
    object.dsav['V'] = (['y','x'], object.Vav.astype('float64'))
    object.dsav['V'].attrs['name'] = 'Ocean velocity in y-direction'
    object.dsav['V'].attrs['units'] = 'm/s'

    object.Dav = np.zeros((object.ny,object.nx))
    object.dsav['D'] = (['y','x'], object.Dav.astype('float64'))
    object.dsav['D'].attrs['name'] = 'Mixed layer thickness'
    object.dsav['D'].attrs['units'] = 'm'

    object.Tav = np.zeros((object.ny,object.nx))
    object.dsav['T'] = (['y','x'], object.Tav.astype('float64'))
    object.dsav['T'].attrs['name'] = 'Layer-averaged potential temperature'
    object.dsav['T'].attrs['units'] = 'degrees C'

    object.Sav = np.zeros((object.ny,object.nx))
    object.dsav['S'] = (['y','x'], object.Sav.astype('float64'))
    object.dsav['S'].attrs['name'] = 'Layer-averaged salinity'
    object.dsav['S'].attrs['units'] = 'psu'

    object.meltav = np.zeros((object.ny,object.nx))
    object.dsav['melt'] = (['y','x'], object.meltav.astype('float64'))
    object.dsav['melt'].attrs['name'] = 'Basal melt rate'
    object.dsav['melt'].attrs['units'] = 'm/yr'

    object.entrav = np.zeros((object.ny,object.nx))
    object.dsav['entr'] = (['y','x'], object.entrav.astype('float64'))
    object.dsav['entr'].attrs['name'] = 'Entrainment rate of ambient water'
    object.dsav['entr'].attrs['units'] = 'm/yr'

    object.ent2av = np.zeros((object.ny,object.nx))
    object.dsav['ent2'] = (['y','x'], object.ent2av.astype('float64'))
    object.dsav['ent2'].attrs['name'] = 'Additional entrainment to ensure minimum layer thickness'
    object.dsav['ent2'].attrs['units'] = 'm/yr'

    object.detrav = np.zeros((object.ny,object.nx))
    object.dsav['detr'] = (['y','x'], object.detrav.astype('float64'))
    object.dsav['detr'].attrs['name'] = 'Detrainment rate'
    object.dsav['detr'].attrs['units'] = 'm/yr'

    object.dsav['tmask'] = (['y','x'], object.tmask)
    object.dsav['tmask'].attrs['name'] = 'Mask at grid center (t-grid)'
    
    object.dsav['umask'] = (['y','x'], object.umask)
    object.dsav['umask'].attrs['name'] = 'Mask at grid side + 1/2 dx (u-grid)'

    object.dsav['vmask'] = (['y','x'], object.vmask)
    object.dsav['vmask'].attrs['name'] = 'Mask at grid side + 1/2 dy (v-grid)'

    object.dsav['mask']  = (['y','x'], object.mask)
    object.dsav['mask'].attrs['name'] = 'Mask at grid center (t-grid)'
    object.dsav['mask'].attrs['values'] = '0: open ocean. 1 and 2: grounded ice or bare rock. 3: ice shelf and cavity'
    
    object.dsav['zb'] = (['y','x'], object.zb)
    object.dsav['zb'].attrs['name'] = 'Ice shelf draft depth'
    object.dsav['zb'].attrs['units'] = 'm'    

    if object.readsavebed:
        object.dsav['B'] = (['y','x'], object.B)
        object.dsav['B'].attrs['name'] = 'Bedrock depth'
        object.dsav['B'].attrs['units'] = 'm'  

    #Add attributes
    object.dsav.attrs['model_name'] = 'LADDIE'
    object.dsav.attrs['model_version'] = object.modelversion
    object.dsav.attrs['time_start'] = object.tstart

    #For storing restart file
    object.dsre = xr.Dataset()
    object.dsre = object.dsav.assign_coords({'x':object.x,'y':object.y,'n':np.array([0,1,2])})
    object.dsre['tmask'] = (['y','x'], object.tmask)
    object.dsre['umask'] = (['y','x'], object.umask)
    object.dsre['vmask'] = (['y','x'], object.vmask)
    object.dsre['mask']  = (['y','x'], object.mask)
    object.dsre['zb'] = (['y','x'], object.zb)
    if object.readsavebed:
        object.dsre['B'] = (['y','x'], object.B)
    object.dsre.attrs['name_model'] = 'LADDIE'
    object.dsre.attrs['model_version'] = object.modelversion

    object.print2log("Prepared datasets for output and restart files")
    return