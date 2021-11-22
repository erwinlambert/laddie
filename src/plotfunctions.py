import numpy as np
import xarray as xr
import pyproj
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean as cmo
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cartopy.crs as ccrs

def prettyplot(dsav,figsize=(10,10)):
 
    try:
        ds = xr.open_dataset(f"{dsav['filename'].values}.nc")
    except:
        print('No output saved yet, cannot plot')
        return

    fig,ax = plt.subplots(1,1,figsize=figsize)            
    ax.set_aspect('equal', adjustable='box') 
    
    x = ds['x'].values
    y = ds['y'].values
    melt  = ds['melt'].values
    mask  = ds['mask'].values
    
    xx,yy = np.meshgrid(x,y)
    
    dx = x[1]-x[0]
    dy = y[1]-y[0]
    x_ = np.append(x,x[-1]+dx)-dx/2
    y_ = np.append(y,y[-1]+dy)-dy/2

    ax.pcolormesh(x_,y_,mask,cmap=plt.get_cmap('ocean'),vmin=-4,vmax=3)
    
    cmap = plt.get_cmap('inferno')
    
    melt = np.where(melt>1,melt,1)
    levs = np.power(10, np.arange(0,np.log10(200),.01))
    #IM = ax.contourf(xx,yy,np.where(mask==3,melt,np.nan),levs,cmap=cmap,norm=mpl.colors.LogNorm())
    IM = ax.pcolormesh(xx,yy,np.where(mask==3,melt,np.nan),norm=mpl.colors.LogNorm(vmin=1,vmax=100),cmap=cmap,shading='nearest')

    the_divider = make_axes_locatable(ax)
    color_axis = the_divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(IM, cax=color_axis)
    cbar.set_ticks([1,10,100])
    cbar.set_ticklabels([1,10,100])
    cbar.ax.tick_params(labelsize=21)
    cbar.set_label('Melt [m/yr]', fontsize=21, labelpad=-2)
    
    U = ds['U'].values*ds['D'].values
    V = ds['V'].values*ds['D'].values

    spd = (U**2 + V**2)**.5
    lw = .04*spd
    strm = ax.streamplot(x,y,U,V,linewidth=lw,color='w',density=8,arrowsize=0)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    fname = f"../../results/figures/{ds['name_geo'].values}_{ds.attrs['name_forcing']}__{ds['tend'].values:.3f}"

    plt.savefig(f"{fname}.png")
    plt.show()
    
def add_lonlat(ds):
    project = pyproj.Proj("epsg:3031")
    xx, yy = np.meshgrid(ds.x, ds.y)
    lons, lats = project(xx, yy, inverse=True)
    dims = ['y','x']
    ds = ds.assign_coords({'lat':(dims,lats), 'lon':(dims,lons)})    
    return ds