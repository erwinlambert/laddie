import numpy as np
import xarray as xr
import sys
import pyproj

from constants import ModelConstants

def read_geom(object):
    #Read input file

    try:
        ds = xr.open_dataset(object.geomfile)

        #Check for time dimension
        if len(ds.dims) ==3:
            #ds = ds.isel(t=object.geomyear)
            ds = ds.isel(Nisf=object.geomyear)
            object.print2log(f'selecting geometry time index {object.geomyear}')

        #Check order of x and y
        object.dx = ds.x[1]-ds.x[0]
        object.dy = ds.y[1]-ds.y[0]
        if object.dx<0:
            object.print2log('inverting x-coordinates')
            ds = ds.reindex(x=list(reversed(ds.x)))
            object.dx = -object.dx
        if object.dy<0:
            object.print2log('inverting y-coordinates')
            ds = ds.reindex(y=list(reversed(ds.y)))
            object.dy = -object.dy

        if object.coarsen>1:
            ds = apply_coarsen(ds,object.coarsen)
            object.print2log(f"Coarsened geometry by a factor of {object.coarsen}")

        if object.lonlat:
            ds = add_lonlat(ds,object.projection)
            object.lon = ds.lon
            object.lat = ds.lat
            object.print2log(f"Added longitude and latitude dimensions")

        #Read variables
        object.x    = ds.x
        object.y    = ds.y     
        object.mask = ds.mask
        object.H    = ds.thickness
        object.zs   = ds.surface

        object.readsavebed = True
        try:
            object.B    = ds.bed
        except:
            object.readsavebed = False
            object.print2log("Warning: no Bed included in input file, so omitted from output")

        ds.close()

        #Apply calving threshold
        ncalv = sum(sum(np.logical_and(object.mask==3,object.H<object.calvthresh))).values
        object.mask = xr.where(np.logical_and(object.mask==3,object.H<object.calvthresh),0,object.mask)
        object.print2log(f"Removed {ncalv} grid points with thickness below {object.calvthresh} m")

        #Remove icebergs
        if object.removebergs:
            remove_icebergs(object)

        #Set surface height and thickness to 0 at new ocean grid points
        object.zs = xr.where(object.mask==0,0,object.zs)
        object.H = xr.where(object.mask==0,0,object.H)

        #Get depth of ice shelf base
        object.zb = object.zs-object.H



    except:
        object.print2log(f"Error, cannot read geometry file {object.geomfile}. Check whether it exists and contains the correct variables")
        sys.exit()

    object.res = (object.x[1]-object.x[0]).values/1000
    object.print2log(f"Finished reading geometry {object.geomfile} at resolution {object.res} km. All good")

    return


def apply_coarsen(ds,N):
    """Coarsen grid resolution by a factor N"""
    ds['mask'] = xr.where(ds.mask==0,np.nan,ds.mask)
    ds['thickness'] = xr.where(np.isnan(ds.mask),np.nan,ds.thickness)
    ds['surface'] = xr.where(np.isnan(ds.mask),np.nan,ds.surface)
    ds['bed'] = xr.where(np.isnan(ds.mask),np.nan,ds.bed)
    ds = ds.coarsen(x=N,y=N,boundary='trim').median()
    ds['mask'] = np.round(ds.mask)
    ds['mask'] = xr.where(np.isnan(ds.mask),0,ds.mask)
    ds['thickness'] = xr.where(ds.mask==0,0,ds.thickness)
    ds['surface'] = xr.where(ds.mask==0,0,ds.surface)
    ds['thickness'] = xr.where(np.isnan(ds.thickness),0,ds.thickness)
    ds['surface'] = xr.where(np.isnan(ds.surface),0,ds.surface)
    return ds

def add_lonlat(ds,proj):
    project = pyproj.Proj(proj)
    xx, yy = np.meshgrid(ds.x, ds.y)
    lons, lats = project(xx, yy, inverse=True)
    dims = ['y','x']
    ds = ds.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})
    return ds

def remove_icebergs(object):
    mmask = object.mask.copy() #Copy of mask on which shelf points are overwritten as grounded points
    kmask = object.mask.copy() #Unclassified shelf points
    gmask = object.mask.copy() #Mask of grounded (or attached-to-grounded) points

    mmask[0,:] = 0 #Prevent issues due to circular boundaries
    mmask[:,0] = 0 

    nleft = 1e20 #Number of unclassified shelf points. Start with arbitrarily high number
    kmask = np.where(mmask<3,0,1) #Counter for unclassified shelf points
    for i in range(200):
    #while sum(sum(kmask))<nleft:
        nleft = sum(sum(kmask))
        #print(nleft)
        gmask = np.where(mmask==2,1,0)
        gmask2 = np.where(kmask==1,np.minimum(1,np.roll(gmask,-1,axis=0)+np.roll(gmask,1,axis=0)+np.roll(gmask,-1,axis=1)+np.roll(gmask,1,axis=1)),0)
        mmask = np.where(gmask2,2,mmask)
        kmask = np.where(mmask<3,0,1)
    
    object.mask = xr.where(kmask,0,object.mask)
    object.print2log(f"Removed {nleft} grid points classified as iceberg")
    return 
