import numpy as np
import xarray as xr
import sys
import pyproj

from constants import ModelConstants

def read_geom(object):
    #Read input file

    try:
        ds = xr.open_dataset(object.geomfile)

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

    object.print2log(f"Finished reading geometry {object.geomfile}. All good")

    return




class Geometry(ModelConstants):
    """Create geometry input from ISOMIP+ """
    def __init__(self,filename,year=0):
        self.ds = xr.open_dataset(filename)

        vargeom = False
        if len(self.ds.dims) ==3:
            vargeom = True
            self.ds = self.ds.isel(t=year)
            print(f'selecting year {year}')

        assert (self.ds.x[1]-self.ds.x[0]).values == (self.ds.y[1]-self.ds.y[0]).values

        self.ds['draft'] = self.ds.lowerSurface.astype('float64')
        self.ds['mask'] = 0.*self.ds.draft
        self.ds['mask'][:] = np.where(self.ds.floatingMask.values,3,0)
        self.ds['mask'][:] = np.where(self.ds.groundedMask.values,2,self.ds.mask)

        #Calving criterium
        self.ds['draft'][:] = np.where(np.logical_and(self.ds.mask==3,self.ds.draft>-90),0,self.ds.draft)
        self.ds['mask'][:] = np.where(np.logical_and(self.ds.mask==3,self.ds.draft>-90),0,self.ds.mask)

        self.ds['mask'][-1,:] = 2 #Prevent cyclic boundary conditions

        if vargeom:
            self.name = f'{filename[-26:-20]}_{year:02.0f}'
        else:
            self.name = filename[-26:-20]
        print(self.name)
        ModelConstants.__init__(self)
    
    def coarsen(self,N):
        """Coarsen grid resolution by a factor N"""
        self.ds['mask'] = xr.where(self.ds.mask==0,np.nan,self.ds.mask)
        self.ds['draft'] = xr.where(np.isnan(self.ds.mask),np.nan,self.ds.draft)
        self.ds = self.ds.coarsen(x=N,y=N,boundary='trim').mean()

        self.ds['mask'] = np.round(self.ds.mask)
        self.ds['mask'] = xr.where(np.isnan(self.ds.mask),0,self.ds.mask)
        self.ds['draft'] = xr.where(self.ds.mask==0,0,self.ds.draft)
        self.ds['draft'] = xr.where(np.isnan(self.ds.draft),0,self.ds.draft)

        self.res = (self.ds.x[1]-self.ds.x[0]).values/1000
        print(f'Resolution set to {self.res} km')
        
    def smoothen(self,N):
        """Smoothen geometry"""
        for n in range(0,N):
            self.ds.draft = .5*self.ds.draft + .125*(np.roll(self.ds.draft,-1,axis=0)+np.roll(self.ds.draft,1,axis=0)+np.roll(self.ds.draft,-1,axis=1)+np.roll(self.ds.draft,1,axis=1))

    def create(self):
        """Create geometry"""
        geom = self.ds[['mask','draft']]
        geom['name_geo'] = f'{self.name}_{self.res:1.1f}'
        print('Geometry',geom.name_geo.values,'created')
        
        #Add lon lat
        #project = pyproj.Proj("epsg:3031")
        #xx, yy = np.meshgrid(geom.x, geom.y)
        #lons, lats = project(xx, yy, inverse=True)
        #dims = ['y','x']
        #geom = geom.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})  
        return geom
