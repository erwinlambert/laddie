import numpy as np
import xarray as xr
import sys

def read_geom(object):
    #Read input file

    try:
        ds = xr.open_dataset(object.geomfile)

        #Check for time dimension
        if len(ds.dims) >2:
            if object.maskoption == "ISOMIP":
                ds = ds.isel(t=object.geomyear)
            elif object.maskoption in ["UFEMISM","IMAUICE"]:
                ds = ds.isel(time=object.geomyear)
            object.print2log(f'selecting geometry time index {object.geomyear}')

        #Get grid cell size
        object.dx = ds.x[1]-ds.x[0]
        object.dy = ds.y[1]-ds.y[0]

        #Check order of x and y
        if object.dx<0:
            object.print2log('inverting x-coordinates')
            ds = ds.reindex(x=list(reversed(ds.x)))
            object.dx = -object.dx
        if object.dy<0:
            object.print2log('inverting y-coordinates')
            ds = ds.reindex(y=list(reversed(ds.y)))
            object.dy = -object.dy

        #If required, coarsen grid by given factor
        if object.coarsen>1:
            ds = apply_coarsen(object,ds)

        #If required, include longitude and latitude
        if object.lonlat:
            ds = add_lonlat(object,ds,object.projection)

        #Read variables
        object.dx = object.dx.values
        object.dy = object.dy.values
        object.x_full    = ds.x.values
        object.y_full    = ds.y.values
        object.xu_full   = object.x_full + 0.5*object.dx
        object.yv_full   = object.y_full + 0.5*object.dy

        object.nx_full = len(object.x_full)
        object.ny_full = len(object.y_full)
        
        #Try to read draft
        gotdraft = False
        for v in ['draft','Hib','zb','lowerSurface']:
            if v in ds.variables:
                object.zb_full = ds[v].values
                gotdraft = True
                object.print2log(f"Got ice shelf draft from '{v}'")

        #Get ice thickness to compute mask and, if needed, draft
        if gotdraft == False or object.maskoption in ["UFEMISM","IMAUICE"]:
            gotthick = False
            #Get thickness
            for v in ['thickness','Hi']:
                if v in ds.variables:
                    object.H = ds[v].values
                    gotthick = True

        #If reading draft failed, try to get draft from thickness and surface
        if gotdraft == False:
            gotsurf  = False
            #Get surface
            for v in ['surface','Hs']:
                if v in ds.variables:
                    object.zs = ds[v].values
                    gotsurf = True
            if gotthick and gotsurf:
                #Extract draft
                object.zb_full = object.zs-object.H
                object.H_full = object.H
                gotdraft = True
                object.print2log(f"Got ice shelf draft from thickness and surface")
            else:
                print(f"INPUT ERROR: Could not find or extract ice shelf draft from input geometry. Check variable names.")
                print(f"Need either draft ('draft', 'Hb','zb',or 'lowerSurface') or thickness ('thickness' or 'Hi') and surface ('surface' or 'Hs')")
                sys.exit()

        #Try to read bed if requested for saving or needed to compute mask
        if object.save_B or object.maskoption in ["UFEMISM","IMAUICE"]:
            gotbed = False
            for v in ['bed','Hb','bedrockTopography']:
                if v in ds.variables:
                    object.B = ds[v].values
                    gotbed = True
            if gotbed == False:
                if object.maskoption in ["UFEMISM","IMAUICE"]:
                    print(f"INPUT ERROR: Could not find Bed in input file, needed to compute mask. Check variable names")
                    sys.exit()
                object.save_B = False
                object.print2log("Warning: no Bed included in input file, so omitted from output")

        #Read mask and convert to BedMachine standard (0: ocean, 1 and/or 2: grounded, 3: ice shelf)
        if object.maskoption == "BM":
            object.mask_full = ds.mask.values
        elif object.maskoption in ["UFEMISM","IMAUICE"]:
            object.mask_full = np.where(object.H>0,1,0)
            buff = .1 #.1
            object.mask_full = np.where(np.logical_and(object.mask_full==1,object.zb_full>object.B+buff),3,object.mask_full)
        elif object.maskoption == "ISOMIP":
            object.mask_full = ds.groundedMask.values
            object.mask_full = np.where(ds.floatingMask,3,object.mask_full)

        ds.close()

        #Cut out minimal region
        if object.cutdomain:
            cut_domain(object)
        else:
            object.mask = object.mask_full.copy()
            object.zb   = object.zb_full.copy()
            object.x    = object.x_full.copy()
            object.y    = object.y_full.copy()
            object.imin = 0
            object.jmin = 0
            object.imax = object.nx_full-1
            object.jmax = object.ny_full-1
            object.nx   = len(object.x)
            object.ny   = len(object.y)

        #Add boundaries here to prevent effects of periodic boundary conditions
        add_border(object)            

        #Apply calving threshold
        if object.calvthresh>0:
            try:
                ncalv = sum(sum(np.logical_and(object.mask==3,object.H<object.calvthresh)))
                object.mask = np.where(np.logical_and(object.mask==3,object.H<object.calvthresh),0,object.mask)
                object.print2log(f"Removed {ncalv} grid points with ice thickness below than {object.calvthresh} m")
            except:
                draftlim = -object.rhoi/object.rho0*object.calvthresh
                ncalv = sum(sum(np.logical_and(object.mask==3,object.zb>draftlim)))
                object.mask = np.where(np.logical_and(object.mask==3,object.zb>draftlim),0,object.mask)
                object.print2log(f"Removed {ncalv} grid points with draft shallower than {draftlim} m")

        #Remove icebergs
        if object.removebergs:
            remove_icebergs(object)

        #Set draft to 0 at new ocean grid points
        object.zb = np.where(object.mask==0,0,object.zb)

    except:
        print(f"INPUT ERROR, cannot read geometry file {object.geomfile}. Check whether it exists and contains the correct variables")
        sys.exit()

    object.res = (object.x[1]-object.x[0])/1000
    object.print2log(f"Finished reading geometry {object.geomfile} at resolution {object.res} km. All good")

    return

def cut_domain(object):
    """Determine boundaries around ice shelf and shrink domain for computation"""

    #Get imin, imax, jmin, jmax
    tmask = np.where(object.mask_full==3,1,0)
    tmaskx = np.sum(tmask,axis=0)
    sargsx = np.argwhere(tmaskx>0)
    object.imin = np.maximum(0,sargsx[0][0]-1)
    object.imax = np.minimum(object.nx_full-1,sargsx[-1][0]+1)
    tmasky = np.sum(tmask,axis=1)
    sargsy = np.argwhere(tmasky>0)
    object.jmin = np.maximum(0,sargsy[0][0]-1)
    object.jmax = np.minimum(object.ny_full-1,sargsy[-1][0]+1)

    #Cut out region
    object.mask = object.mask_full[object.jmin:object.jmax+1,object.imin:object.imax+1]
    object.zb   = object.zb_full[object.jmin:object.jmax+1,object.imin:object.imax+1]
    object.H    = object.H_full[object.jmin:object.jmax+1,object.imin:object.imax+1]

    object.x    = object.x_full[object.imin:object.imax+1]
    object.y    = object.y_full[object.jmin:object.jmax+1]

    object.nx = len(object.x)
    object.ny = len(object.y)

    #Print reduced size
    reducedsize = 100*(1-(object.nx*object.ny)/(object.nx_full*object.ny_full))
    object.print2log(f"Finished cutting domain. Reduced size by {reducedsize:.0f} percent")

    return

def add_border(object):
    """Add border along all sides of 1 grid cell of specified mask value"""

    #Add north
    object.mask = np.append(object.mask,object.borderN+np.zeros((1,object.nx)),axis=0)
    object.zb   = np.append(object.zb,np.zeros((1,object.nx)),axis=0)
    object.H    = np.append(object.H,np.zeros((1,object.nx)),axis=0)
    object.y    = np.append(object.y,object.y[-1]+object.dy)

    #Add south
    object.mask = np.append(object.borderS+np.zeros((1,object.nx)),object.mask,axis=0)
    object.zb   = np.append(np.zeros((1,object.nx)),object.zb,axis=0)
    object.H   = np.append(np.zeros((1,object.nx)),object.H,axis=0)
    object.y    = np.append(object.y[0]-object.dy,object.y)

    #Add east
    object.mask = np.append(object.mask,object.borderE+np.zeros((object.ny+2,1)),axis=1)
    object.zb   = np.append(object.zb,np.zeros((object.ny+2,1)),axis=1)
    object.H   = np.append(object.H,np.zeros((object.ny+2,1)),axis=1)
    object.x    = np.append(object.x,object.x[-1]+object.dx)

    #Add west
    object.mask = np.append(object.borderW+np.zeros((object.ny+2,1)),object.mask,axis=1)
    object.zb   = np.append(np.zeros((object.ny+2,1)),object.zb,axis=1)
    object.H   = np.append(np.zeros((object.ny+2,1)),object.H,axis=1)
    object.x    = np.append(object.x[0]-object.dx,object.x)

    return

def apply_coarsen(object,ds):
    """Coarsen grid resolution by a factor given by object.coarsen"""

    #Check whether maskoption is BM (bedmachine), otherwise: ignore coarsening request
    if object.maskoption == "BM":
        ds['mask'] = xr.where(ds.mask==0,np.nan,ds.mask)
        ds['thickness'] = xr.where(np.isnan(ds.mask),np.nan,ds.thickness)
        ds['surface'] = xr.where(np.isnan(ds.mask),np.nan,ds.surface)
        ds['bed'] = xr.where(np.isnan(ds.mask),np.nan,ds.bed)
        ds = ds.coarsen(x=object.coarsen,y=object.coarsen,boundary='trim').median()
        ds['mask'] = np.round(ds.mask)
        ds['mask'] = xr.where(np.isnan(ds.mask),0,ds.mask)
        ds['thickness'] = xr.where(ds.mask==0,0,ds.thickness)
        ds['surface'] = xr.where(ds.mask==0,0,ds.surface)
        ds['thickness'] = xr.where(np.isnan(ds.thickness),0,ds.thickness)
        ds['surface'] = xr.where(np.isnan(ds.surface),0,ds.surface)

        object.print2log(f"Coarsened geometry by a factor of {object.coarsen}")
    else:
        print("WARNING: coarsening required, which only works for BedMachine-like input. So proceeding without coarsening")

    """To be expanded for other mask options"""

    return ds

def add_lonlat(object,ds,proj):
    """Compute longitude and latitude based on x,y, and include in output. Only useful in realistic settings"""

    import pyproj
    project = pyproj.Proj(proj)
    xx, yy = np.meshgrid(ds.x, ds.y)
    lons, lats = project(xx, yy, inverse=True)
    dims = ['y','x']
    ds = ds.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})

    object.lon = ds.lon
    object.lat = ds.lat
    object.print2log(f"Added longitude and latitude dimensions")
    return ds

def remove_icebergs(object):
    """Remove ice shelf clusters that are not directly or indirectly connected to grounded ice"""

    mmask = object.mask.copy() #Copy of mask on which shelf points are overwritten as grounded points
    kmask = object.mask.copy() #Unclassified shelf points
    gmask = object.mask.copy() #Mask of grounded (or attached-to-grounded) points

    mmask[0,:] = 0 #Prevent issues due to circular boundaries
    mmask[:,0] = 0 

    nleft = 1e20 #Number of unclassified ice shelf points. Start with arbitrarily high number
    kmask = np.where(mmask<3,0,1) #Counter for unclassified shelf points
    counter = 0
    object.print2log("============ Starting removal of icebergs =================")
    while sum(sum(kmask))<nleft:
        counter += 1

        #Number of unclassified ice shelf points remaining
        nleft = sum(sum(kmask))
        object.print2log(f"{nleft} unclassified grid points remaining ...")

        #Get mask of grounded ice + ice shelf points classified as attached to grounded ice
        gmask = np.where(mmask==2,1,0)
        gmask = np.where(mmask==1,1,gmask)

        #Classify another strip of ice shelf points as attached to grounded ice
        gmask2 = np.where(kmask==1,np.minimum(1,np.roll(gmask,-1,axis=0)+np.roll(gmask,1,axis=0)+np.roll(gmask,-1,axis=1)+np.roll(gmask,1,axis=1)),0) 

        #Add newly classified points to total mask
        mmask = np.where(gmask2,2,mmask)

        #Determine unclassified points
        kmask = np.where(mmask<3,0,1)

        #Stop to prevent infinite loop
        if counter == 1000:
            print("WARNING: stopping routine 'remove_icebergs' early, this leads to removal of parts of the ice shelf. To prevent this, increase the maximum number of iterations ")
            break
    
    object.mask = xr.where(kmask,0,object.mask)
    object.print2log(f"============= Removed {nleft} grid points classified as iceberg ===========")
    object.print2log(f"===========================================================================")
    return 
