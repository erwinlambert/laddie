import numpy as np
import xarray as xr
import sys

def read_geom(object):
    #Read input file

    try:
        ds = xr.open_dataset(object.geomfile)

        #Check for time dimension
        if len(ds.dims) ==3:
            ds = ds.isel(t=object.geomyear)
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
        object.x    = ds.x.values
        object.y    = ds.y.values

        #Read mask and convert to BedMachine standard (0: ocean, 1 and/or 2: grounded, 3: ice shelf)
        if object.maskoption == "BM":
            object.mask = ds.mask.values
        elif object.maskoption == "UFEMISM":
            object.mask = ds.mask.values
            object.mask = np.where(object.mask==1,3,object.mask)
        elif object.maskoption == "ISOMIP":
            object.mask = ds.groundedMask.values
            object.mask = np.where(ds.floatingMask,3,object.mask)

        #Try to read draft
        gotdraft = False
        for v in ['draft','Hib','zb','lowerSurface']:
            if v in ds.variables:
                object.zb = ds[v].values
                gotdraft = True
                object.print2log(f"Got ice shelf draft from '{v}'")
       
        #If this failed, try to get draft from thickness and surface
        if gotdraft == False:
            gotthick = False
            gotsurf  = False
            #Get thickness
            for v in ['thickness','Hi']:
                if v in ds.variables:
                    object.H = ds[v].values
                    gotthick = True
            #Get thickness
            for v in ['surface','Hs']:
                if v in ds.variables:
                    object.zs = ds[v].values
                    gotsurf = True

            if gotthick and gotsurf:
                #Extract draft
                object.zb = object.zs-object.H
                gotdraft = True
                object.print2log(f"Got ice shelf draft from thickness and surface")
            else:
                print(f"INPUT ERROR: Could not find or extract ice shelf draft from input geometry. Check variable names.")
                print(f"Need either draft ('draft', 'Hb','zb',or 'lowerSurface') or thickness ('thickness' or 'Hi') and surface ('surface' or 'Hs')")
                sys.exit()

        #Try to read bed
        if object.save_B:
            gotbed = False
            for v in ['bed','Hb','bedrockTopography']:
                if v in ds.variables:
                    object.B = ds[v].values
                    gotbed = True
            if gotbed == False:
                object.save_B = False
                object.print2log("Warning: no Bed included in input file, so omitted from output")
    
        ds.close()

        #Apply calving threshold
        draftlim = -object.rhoi/object.rho0*object.calvthresh
        ncalv = sum(sum(np.logical_and(object.mask==3,object.zb>draftlim)))
        object.mask = np.where(np.logical_and(object.mask==3,object.zb>draftlim),0,object.mask)
        object.print2log(f"Removed {ncalv} grid points with thickness below {object.calvthresh} m")
       
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
