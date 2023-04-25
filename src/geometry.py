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
            ds = ds.isel(t=object.geomyear)
            print(f'selecting year {object.geomyear}')

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

        object.print2log("test")

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
        object.B    = ds.bed
        object.zs   = ds.surface

        ds.close()

        object.zb = object.zs-object.H

    except:
        object.print2log(f"Error, cannot read geometry file {object.geomfile}. Check whether it exists and contains the correct variables")
        sys.exit()

    object.res = (object.x[1]-object.x[0]).values/1000
    object.print2log(f"Finished reading geometry {object.geomfile} at resolution {object.res} km. All good")

    return


def apply_coarsen(ds,N):
    """Coarsen grid resolution by a factor N"""
    print('start')
    ds['mask'] = xr.where(ds.mask==0,np.nan,ds.mask)
    ds['thickness'] = xr.where(np.isnan(ds.mask),np.nan,ds.thickness)
    ds['surface'] = xr.where(np.isnan(ds.mask),np.nan,ds.surface)
    ds['bed'] = xr.where(np.isnan(ds.mask),np.nan,ds.bed)
    ds = ds.coarsen(x=N,y=N,boundary='trim').mean()
    print('test1')
    ds['mask'] = np.round(ds.mask)
    ds['mask'] = xr.where(np.isnan(ds.mask),0,ds.mask)
    ds['thickness'] = xr.where(ds.mask==0,0,ds.thickness)
    ds['surface'] = xr.where(ds.mask==0,0,ds.surface)
    print('test2')
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
