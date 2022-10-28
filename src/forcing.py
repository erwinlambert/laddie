import numpy as np
import xarray as xr

from constants import ModelConstants

class Forcing(ModelConstants):
    """ """

    def __init__(self, geom):
        """ 
        input:
        geom     (xr.Dataset)  geometry dataset from one of the Geometry classes
        output:
        self.ds  (xr.Dataset)  original `geom` with additional fields:
            3D fields:
            Tz   (z,y,x)   [degC]  ambient temperature
            Sz   (z,y,x)   [psu]   ambient salinity
            
            or

            1D fields:
            Tz   (z)       [degC]  ambient temperature
            Sz   (z)       [psu]   ambient salinity            
        """
        assert 'draft' in geom
        self.ds = geom
        self.ds = self.ds.assign_coords({'z':np.arange(-5000.,0,1)})
        ModelConstants.__init__(self)
        return

    def tanh(self, ztcl, Tdeep, drhodz=.2/1000,z1=-200):
        """ creates 1D tanh thermocline forcing profile
        input:
        ztcl    ..  (float)  [m]       thermocline depth
        Tdeep   ..  (float)  [degC]    in situ temperature at depth
        drhodz  ..  (float)  [kg/m^4]  linear density stratification
        z1      ..  (float)  [m]       thermocline sharpness
        """
        if ztcl>0:
            print(f'z-coordinate is postive upwards; ztcl was {ztcl}, now set ztcl=-{ztcl}')
            ztcl = -ztcl
        S0 = 34.0                     # [psu]  reference surface salinity
        T0 = self.l1*S0+self.l2       # [degC] surface freezing temperature
        
        #drho = drhodz*np.abs(self.ds.z)
        drho = .01*np.abs(self.ds.z)**.5

        self.ds['Tz'] = Tdeep + (T0-Tdeep) * (1+np.tanh((self.ds.z-ztcl)/z1))/2
        self.ds['Sz'] = S0 + self.alpha*(self.ds.Tz-T0)/self.beta + drho/(self.beta*self.rho0)
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'tanh_Tdeep{Tdeep:.1f}_ztcl{ztcl}'
        return self.ds
    
    def linear(self, S1,T1,S0=33.8,T0=-1.9,z1=-720):
        """ creates 1D linear forcing profiles
        input:
        z1      ..  (float)  [m]       reference depth
        T0      ..  (float)  [degC]    temperature at the surface
        T1      ..  (float)  [degC]    temperature at depth
        S0      ..  (float)  [psu]     salinity at the surface
        S1      ..  (float)  [psu]     salinity at depth
        """
        if z1>0:
            print(f'z-coordinate is postive upwards; z1 was {z1}, now set z1=-{z1}')
            z1 = -z1
        
        self.ds['Tz'] = T0 + self.ds.z*(T1-T0)/z1 
        self.ds['Sz'] = S0 + self.ds.z*(S1-S0)/z1
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'linear_S1{S1:.1f}_T1{T1}'
        return self.ds
    
    def isomip(self,cond):
        """ creates ISOMIP+ warm or cold profiles
        input:
        cond    ..  (string)    'WARM' or 'COLD'
        """
        z1 = -720
        T0 = -1.9
        S0 = 33.8
        if cond=='WARM':
            T1 = 1.0
            S1 = 34.7
        elif cond=='COLD':
            T1 = -1.9
            S1 = 34.55
        else:
            print("Invalid input to isomip forcing, should be 'WARM' or 'COLD' ")
            return

        if z1>0:
            print(f'z-coordinate is postive upwards; z1 was {z1}, now set z1=-{z1}')
            z1 = -z1

        self.ds['Tz'] = T0 + self.ds.z*(T1-T0)/z1 
        self.ds['Sz'] = S0 + self.ds.z*(S1-S0)/z1
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'isomip_{cond}'
        return self.ds

    def calc_fields(self):
        """ adds Ta/Sa fields to dataset"""
        assert 'Tz' in self.ds
        assert 'Sz' in self.ds
        Sa = np.interp(self.ds.draft.values, self.ds.z.values, self.ds.Sz.values)
        Ta = np.interp(self.ds.draft.values, self.ds.z.values, self.ds.Tz.values)
        self.ds['Ta'] = (['y', 'x'], Ta)
        self.ds['Sa'] = (['y', 'x'], Sa)
        self.ds['Tf'] = self.l1*self.ds.Sa + self.l2 + self.l3*self.ds.draft  # l3 -> potential temperature
        self.ds.Ta.attrs = {'long_name':'ambient potential temperature' , 'units':'degC'}
        self.ds.Sa.attrs = {'long_name':'ambient salinity'              , 'units':'psu' }
        self.ds.Tf.attrs = {'long_name':'local potential freezing point', 'units':'degC'}  # from:Eq. 3 of Favier19
        return self.ds