import numpy as np
#import xarray as xr

from constants import ModelConstants
import preprocess as pp
import integrate as it
import savetools as st

class Laddie(ModelConstants):
    """ LADDIE model
    
        input:
        ds including:
            x      ..  [m]     x coordinate
            y      ..  [m]     y coordinate
            2D [y,x] fields:
            mask   ..  [bin]   mask identifying ocean (0), grounded ice (2), ice shelf (3)
            draft  ..  [m]     ice shelf draft
            
            1D [z] or 3D [z,y,x] fields:
            Tz     ..  [degC]  ambient temperature
            Sz     ..  [psu]   ambient salinity
        output:  [calling `.compute()`]
        ds  ..  xarray Dataset holding all quantities with their coordinates
    """
    
    def __init__(self, ds):
        #Read input
        self.dx = ds.x[1]-ds.x[0]
        self.dy = ds.y[1]-ds.y[0]
        if self.dx<0:
            print('inverting x-coordinates')
            ds = ds.reindex(x=list(reversed(ds.x)))
            self.dx = -self.dx
        if self.dy<0:
            print('inverting y-coordinates')
            ds = ds.reindex(y=list(reversed(ds.y)))
            self.dy = -self.dy
            
        #Inherit input values
        self.ds   = ds
        self.x    = ds.x
        self.y    = ds.y      
        self.z    = ds.z.values  
        self.mask = ds.mask

        self.zb   = ds.draft
        self.H    = ds.thickness
        self.B    = ds.bed
        self.zs   = ds.surface
        
        self.Tz   = ds.Tz.values
        self.Sz   = ds.Sz.values
        self.ind  = np.indices(self.zb.shape)

        self.Ussa = np.zeros((2,len(self.y),len(self.x)))
        self.Vssa = np.zeros((2,len(self.y),len(self.x)))

        #Physical parameters
        ModelConstants.__init__(self)

    def compute(self,days=12,restartfile=None):
        """Run the model"""
        self.restartfile = restartfile
        self.days = days
        
        #Preprocessing
        pp.create_mask(self)
        pp.create_grid(self)
        pp.initialize_vars(self)

        self.nt = int(self.days*24*3600/self.dt)+1    # Number of time steps
        self.tend = self.tstart+self.days
        self.time = np.linspace(self.tstart,self.tend,self.nt)  # Time in days
        
        #Integration
        for self.t in range(self.nt):
            it.updatevars(self)
            it.integrate(self)
            it.timefilter(self)
            it.cutforstability(self)       

            st.savefields(self)
            st.saverestart(self)
            st.printdiags(self)
            
        print('-----------------------------')
        print(f'Run completed')
        
        return self.ds
