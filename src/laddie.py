import preprocess as pp
import integrate as it
import savetools as st
import geometry as ge
import forcing as fo
import tomli
from time import process_time

class Laddie():
    """ 
    The main script of the one-Layer Antarctic model for Dynamical Downscaling of Ice-ocean Exchanges (LADDIE)
    
    Main developer:
    dr. E. (Erwin) Lambert
    Affiliation: Royal Netherlands Meteorological Institute (KNMI)
    E-mail: erwin.lambert@knmi.nl
    """
    
    def __init__(self,configfile):
        """All preprocessing steps before time-integration"""

        self.startwalltime = process_time()

        self.modelversion = '1.1'

        #Load configuration file
        with open(configfile,'rb') as f:
            self.config = tomli.load(f)

        #Create directory to store output for this run
        pp.create_rundir(self,configfile)

        #Read config file and extract parameters
        pp.read_config(self)

        #Read input geometry from external file
        ge.read_geom(self)

        #Read or create forcing
        if self.forcop == "file":
            #Read forcing from external file
            fo.read_forcing(self)
        else:
            #Create forcing using an internal routine
            fo.create_forcing(self)
        
        #Create grid based on provided geometry
        pp.create_grid(self)

        #Create masks and extract ice shelf front and grounding line
        pp.create_mask(self)

        #Initialise variables used in simulation and apply first time step to get time derivatives
        pp.initialise_vars(self)

        #Prepare datasets for output and restart files
        pp.prepare_output(self)

    def compute(self):
        """Run the model"""

        print(f"Starting computation, follow progress here: {self.logfile}")
        self.print2log("Preprocessing succesful, starting computation")

        #Apply time-stepping loop
        for self.t in range(self.nt):
            #Move 1 timestep
            it.updatevars(self)

            #Integrate main variables
            it.integrate(self)

            #Limit velocities through threshold cutoff
            it.cutforstability(self)   

            #Apply time-filter 
            it.timefilter(self)

            #If required, compute and print diagnostics to log file
            st.printdiags(self)

            #Store snapshot variables into dataset for time-average output. If required, save output and renew dataset
            st.savefields(self)

            #If required, save snapshot variables and time-derivatives for restarting future run
            st.saverestart(self)

        self.print2log('-----------------------------')
        self.print2log(f'Run completed')
        
        #Compute time since start of run
        hours, rem = divmod(process_time()-self.startwalltime, 3600)
        minutes, seconds = divmod(rem, 60)
        #print(f"Run completed in [{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}]")

        return

    def print2log(self,text):
        """Function to print a line to the log file"""

        #Ignore if logfile is not defined yet
        if hasattr(self,'logfile'):

            #Compute time since start of run
            hours, rem = divmod(process_time()-self.startwalltime, 3600)
            minutes, seconds = divmod(rem, 60)

            with open(self.logfile,"a") as file:
                file.write(f"[{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}] {text}\n")
            return