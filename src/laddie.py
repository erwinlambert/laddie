#import numpy as np
#import xarray as xr
import sys
import preprocess as pp
import integrate as it
import savetools as st
import geometry as ge
import forcing as fo
import tomli

class Laddie():
    """ 
    LADDIE model
    """
    
    def __init__(self,configfile):
        """All preprocessing steps"""

        #Read config file
        with open(configfile,'rb') as f:
            self.config = tomli.load(f)

        #Create run directory
        pp.create_rundir(self)

        #Read config file
        pp.read_config(self)

        #Read input geometry
        ge.read_geom(self)

        #Read or create forcing
        fo.create_forcing(self)

        sys.exit()
        
        #Create grid
        pp.create_grid(self)

        #Create mask
        pp.create_mask(self)

        #Initialise variables
        pp.initialise_vars(self)

        #Prepare output files
        pp.prepare_output(self)

    def compute(self):
        """Run the model"""

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

    def print2log(self,text):
        with open(self.logfile,"a") as file:
            file.write(f"{text}\n")
        return