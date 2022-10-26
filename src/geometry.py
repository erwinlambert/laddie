import numpy as np
import xarray as xr
import pyproj

from constants import ModelConstants

class Geometry(ModelConstants):
    """Create geometry input"""
    def __init__(self,name):
        if name=='Thwaites_e':
            x0,x1,y0,y1 = 3460,3640,7425,7700
        elif name=='Thwaites':
            x0,x1,y0,y1 = 3460,3640,7425,7642
        elif name=='PineIsland':
            x0,x1,y0,y1 = 3290,3550,7170,7400
        elif name=='CrossDots':
            x0,x1,y0,y1 = 3445,3705,7730,8065
        elif name=='Dotson':
            x0,x1,y0,y1 = 3465,3705,7865,8065
        elif name=='Getz':
            x0,x1,y0,y1 = 3510,4330,8080,9050
        elif name=='Cosgrove':
            x0,x1,y0,y1 = 3070,3190,7210,7420  
        elif name=='TottenMU':
            x0,x1,y0,y1 = 10960,11315,8665,9420
        elif name=='Amery':
            x0,x1,y0,y1 = 10010,11160,4975,5450
        elif name=='FRIS':
            x0,x1,y0,y1 = 3600,5630,4560,6420
        elif name=='Ross':
            x0,x1,y0,y1 = 5470,7500,7500,9400
        elif name=='LCIS':
            x0,x1,y0,y1 = 1900,2700,4050,4720
    
        self.ds = xr.open_dataset('../data/BedMachineAntarctica_2020-07-15_v02.nc')
        self.ds = self.ds.isel(x=slice(x0,x1),y=slice(y0,y1))
        #self.mask = self.ds.mask
        self.ds.mask[:] = xr.where(self.ds.mask==1,2,self.ds.mask)
        self.ds['draft'] = (self.ds.surface-self.ds.thickness).astype('float64')
        self.name = name
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
        self.res *= N
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
        project = pyproj.Proj("epsg:3031")
        xx, yy = np.meshgrid(geom.x, geom.y)
        lons, lats = project(xx, yy, inverse=True)
        dims = ['y','x']
        geom = geom.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})  
        return geom

class GeometryAS(ModelConstants):
    """Create geometry input"""
    def __init__(self):
        x0,x1,y0,y1 = 900,2200,300,2000
        name = 'CrossDots'
        self.ds = xr.open_dataset('../../../data/annsofie/DotsonDraftAndMask_100m.nc')
        self.ds = self.ds.isel(x=slice(x0,x1),y=slice(y0,y1))
        #self.mask = self.ds.mask
        self.ds.mask[:] = xr.where(self.ds.mask==1,3,self.ds.mask)
        self.ds['draft'] = (self.ds.ice_draft).astype('float64')
        self.name = name
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
        self.res = 0.1*N
        print(f'Resolution set to {self.res} km')
        
    def create(self):
        """Create geometry"""
        geom = self.ds[['mask','draft']]
        geom['name_geo'] = f'{self.name}_{self.res:1.1f}'
        print('Geometry',geom.name_geo.values,'created')
        
        #Add lon lat
        project = pyproj.Proj("epsg:3031")
        xx, yy = np.meshgrid(geom.x, geom.y)
        lons, lats = project(xx, yy, inverse=True)
        dims = ['y','x']
        geom = geom.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})  
        return geom
