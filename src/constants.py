class ModelConstants(object):
    """ Input parameters"""
    def __init__(self):
        #General physical parameters
        self.g         = 9.81      # [m/s^2]     gravitational acceleration
        self.L         = 3.34e5    # [J/kg]      Latent heat of fusion for ice
        self.cp        = 3.974e3   # [J/kg/degC] specific heat capacity of ocean
        self.ci        = 2009      # [J/kg/degC] specific heat capacity of ice
        #self.spy       = 86400*365 # [s/yr]      seconds per year
        self.alpha     =  3.733e-5   # [1/degC]    Thermal expansion coefficient
        self.beta      =  7.843e-4   # [1/psu]     Haline contraction coefficient
        self.l1        = -5.73e-2  # [degC/psu]  freezing point salinity coefficient
        self.l2        = 8.32e-2   # [degC]      freezing point offset
        self.l3        = 7.61e-4   # [degC/m]    freezing point depth coefficient
        
        #Model-specific constant parameters
        self.CG        = 5.9e-4    # []          effective thermal Stanton number
        self.f         = -1.37e-4  # [1/s]       Coriolis parameter
        self.cl        = 0.0245    #             Parameter for Holland entrainment
        self.utide     = 0.01      # [m/s]       RMS tidal velocity
        self.Pr        = 13.8      # []          Prandtl number
        self.Sc        = 2432.     # []          Schmidt number
        self.nu0       = 1.95e-6   # [m^2/s]     Molecular viscosity
        self.rhofw     = 1000.     # [kg/m^3]    Density of freshwater
        self.rho0      =  1028     # [kg/m^3]    Reference density of seawater
        #self.rhoi      =   910     # [kg/m^3]
        #self.rhow      =  1028     # [kg/m^3]  
        #self.E0        = 3.6e-2    # []          entrainment coefficient        
        
        #Run parameters
        self.nu        = .8        # []          Factor for Robert Asselin time filter
        self.slip      = 1         # []          Factor free slip: 0, no slip: 2, partial no slip: [0..2]  
        self.dt        = 40        # [s]         Time step
        self.boundop   = 1         # []          Option for boundary conditions D,T,S. [use 1 for isomip]
        self.minD      = .2        # [m]         Cutoff thickness
        self.maxD      = 3000.     # [m]         Cutoff maximum thickness
        self.mindrho   = .005       # [kg/m^3]    Minimum density difference with ambient water
        self.vcut      = 1.414     # [m/s]       Cutoff velocity U and V
        self.Dinit     = 10.       # [m]         Initial uniform thickness
        self.res       = 0.5       # [km]        Spatial resolution
        self.Ti        = -25       # [degC]      Ice shelf temperature
        
        #Tunable physical parameters
        self.Cd        = 2.5e-3    # []          drag coefficient
        self.Cdtop     = 1.1e-3    # []          Drag coefficient in Ustar
        self.Ah        = 6         # [m^2/s]     Laplacian viscosity
        self.Kh        = 1         # [m^2/s]     Diffusivity
        self.entpar    = 'Gaspar'  #             Entrainment parameterisation, either 'Holland' or 'Gaspar'
        self.mu        = 2.5       # []          Parameter in Gaspar entrainment. Gaspar: 0.5; Gladish: 2.5
        self.maxdetr   = .5        # [m/s]       Cutoff detrainment rate, only effective when O(1e-3)
        self.correctisf = False    #             Remove ice shelf grids sticking out and isolated points
        
        #Some parameters for saving and displaying output
        self.diagday   = .1        # [days]      Timestep at which to print diagnostics
        self.verbose   = True      # Bool        Whether to print diagnostics
        self.saveday   = 10        # [days]      Interval at which to save time-average fields
        self.restday   = 10        # [days]      Interval at which to save restart file
        
