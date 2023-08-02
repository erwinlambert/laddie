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

        self.modelversion = '1.0'

        #Read config file
        with open(configfile,'rb') as f:
            self.config = tomli.load(f)

        #Create run directory
        pp.create_rundir(self,configfile)

        #Read config file
        pp.read_config(self)

        #Read input geometry
        ge.read_geom(self)

        #Read or create forcing
        if self.forcop == "file":
            fo.read_forcing(self)
        else:
            fo.create_forcing(self)
        
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

        self.print2log("Starting computation...")

        #Integration
        for self.t in range(self.nt):
            it.updatevars(self)
            it.integrate(self)
            it.timefilter(self)
            it.cutforstability(self)       

            st.printdiags(self)
            st.savefields(self)
            st.saverestart(self)


        self.print2log('-----------------------------')
        self.print2log(f'Run completed')
        
        return

    def print2log(self,text):
        with open(self.logfile,"a") as file:
            file.write(f"{text}\n")
        return