import numpy as np
import pyproj

def add_lonlat(ds):
    project = pyproj.Proj("epsg:3031")
    xx, yy = np.meshgrid(ds.x, ds.y)
    lons, lats = project(xx, yy, inverse=True)
    dims = ['y','x']
    ds = ds.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})    
    return ds