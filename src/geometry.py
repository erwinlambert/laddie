import numpy as np
import xarray as xr
import pyproj

class Geometry(object):
    """Create geometry input"""
    def __init__(self,name):
        if name=='Thwaites_e':
            x0,x1,y0,y1 = 3460,3640,7425,7700
        elif name=='PineIsland':
            x0,x1,y0,y1 = 3290,3550,7170,7400
        elif name=='CrossDots':
            x0,x1,y0,y1 = 3445,3700,7730,8060
        elif name=='Dotson':
            x0,x1,y0,y1 = 3465,3700,7870,8060
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
    
        self.ds = xr.open_dataset('../../data/BedMachineAntarctica_2020-07-15_v02.nc')
        self.ds = self.ds.isel(x=slice(x0,x1),y=slice(y0,y1))
        self.mask = self.ds.mask
        self.ds.mask[:] = xr.where(self.ds.mask==1,2,self.ds.mask)
        self.ds['draft'] = (self.ds.surface-self.ds.thickness).astype('float64')
        self.name = name
    
    def coarsen(self,N):
        """Coarsen grid resolution by a factor N"""
        self.ds.coarsen(x=N,y=N)
        
    def smoothen(self,N):
        """Smoothen geometry"""
        for n in range(0,N):
            self.ds.draft = .5*self.ds.draft + .125*(np.roll(self.ds.draft,-1,axis=0)+np.roll(self.ds.draft,1,axis=0)+np.roll(self.ds.draft,-1,axis=1)+np.roll(self.ds.draft,1,axis=1))

    def create(self):
        """Create geometry"""
        geom = self.ds[['mask','draft']]
        geom['name_geo'] = self.name
        
        #Add lon lat
        project = pyproj.Proj("epsg:3031")
        xx, yy = np.meshgrid(geom.x, geom.y)
        lons, lats = project(xx, yy, inverse=True)
        dims = ['y','x']
        geom = geom.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})  
        return geom
