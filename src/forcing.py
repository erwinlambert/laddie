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
            mitgcm:
            Tz   (x,y,z)   [degC]  ambient potential temperature
            Sz   (x,y,z)   [psu]   ambient salinity
            tanh:
            Tz   (z)       [degC]  ambient potential temperature
            Sz   (z)       [psu]   ambient salinity            
        """
        assert 'draft' in geom
        self.ds = geom
        self.ds = self.ds.assign_coords({'z':np.arange(-5000.,0,1)})
        ModelConstants.__init__(self)
        return

    def mitgcm(self,startyear,endyear,option='interp',kup=2,kdwn=1,Slimit=0,nsm=1):
        """Forcing from MITgcm output
        kup : number of upper layers to remove
        kdwn: number of lower layers to remove
        nsm : smoothing factor
        
        """
        assert startyear>=1956,'Need to choose a later startyear, 1956 earliest available'
        assert endyear<=2019,'Need to choose an earlier endyear, 2019 latest available'
        assert startyear<=endyear,'Startyear must be earlier than endyear'
        
        lon3 = self.ds.lon.values
        lat3 = self.ds.lat.values
        mask = self.ds.mask.values
        
        #Read MITgcm data for region corresponding to geometry
        timep= slice(f"{startyear}-1-1",f"{endyear}-12-31")
        ds = xr.open_dataset('../../../data/paulholland/PAS_851/stateTheta.nc')
        ds = ds.sel(LONGITUDE=slice(360+np.min(lon3),360+np.max(lon3)),LATITUDE=slice(np.min(lat3),np.max(lat3)),TIME=timep)
        ds = ds.mean(dim='TIME')
        lon   = (ds.LONGITUDE-360.).values
        lat   = (ds.LATITUDE-.05).values
        dep   = ds.DEPTH.values
        theta = ds.THETA.values
        ds.close()
        ds = xr.open_dataset('../../../data/paulholland/PAS_851/stateSalt.nc')
        ds = ds.sel(LONGITUDE=slice(360+np.min(lon3),360+np.max(lon3)),LATITUDE=slice(np.min(lat3),np.max(lat3)),TIME=timep)
        ds = ds.mean(dim='TIME')
        salt  = ds.SALT.values
        ds.close()
        
        #Extrapolate profiles to top and bottom
        llon,llat = np.meshgrid(lon,lat)
        Th = theta.copy()
        Sh = salt.copy()
        Sh = np.where(Sh>Slimit,Sh,0)
        weight = [.1,.8,.1]
        nvsm = 10
        for j,jj in enumerate(lat):
            for i,ii in enumerate(lon):
                knz = np.nonzero(Sh[:,j,i])[0]
                if len(knz) == 0:
                    llon[j,i] = 1e36
                    llat[j,i] = 1e36
                else:
                    if Sh[0,j,i] == 0:
                        k0 = knz[0]+kup
                        Th[:k0,j,i] = Th[k0,j,i]
                        Sh[:k0,j,i] = Sh[k0,j,i]
                    if Sh[-1,j,i] == 0:
                        k1 = knz[-1]-kdwn
                        Th[k1:,j,i] = Th[k1,j,i]
                        Sh[k1:,j,i] = Sh[k1,j,i]
                if sum(Sh[:,j,i]) == 0:
                    llon[j,i] = 1e36
                    llat[j,i] = 1e36
                else:
                    for n in range(nvsm):
                        Th[1:-1,j,i] = np.convolve(Th[:,j,i],weight,'valid')
                        Sh[1:-1,j,i] = np.convolve(Sh[:,j,i],weight,'valid')

        #Apply nearest neighbour onto model grid
        depth = np.arange(0,5000,100) #depth also used as index, so must be positive with steps of 1
        Tz = np.zeros((len(depth),mask.shape[0],mask.shape[1]))
        Sz = np.zeros((len(depth),mask.shape[0],mask.shape[1]))
        for j in range(mask.shape[0]):
            for i in range(mask.shape[1]):
                if mask[j,i] == 3:
                    #Get nearest indices at low end
                    i0 = np.argmax(lon>lon3[j,i])-1
                    j0 = np.argmax(lat>lat3[j,i])-1
                    if option=='interp':
                        #Distance squared
                        dist = np.cos(np.pi*lat3[j,i]/180.)*(np.pi*(lon3[j,i]-llon[j0-nsm:j0+2+nsm,i0-nsm:i0+2+nsm])/180.)**2+(np.pi*(lat3[j,i]-llat[j0-nsm:j0+2+nsm,i0-nsm:i0+2+nsm])/180.)**2
                        #epsilon = 5e-8
                        epsilon = 1e-6
                        weight = 1./(dist+epsilon)
                        TT = np.sum(Th[:,j0-nsm:j0+2+nsm,i0-nsm:i0+2+nsm]*weight,axis=(1,2))/np.sum(weight)
                        SS = np.sum(Sh[:,j0-nsm:j0+2+nsm,i0-nsm:i0+2+nsm]*weight,axis=(1,2))/np.sum(weight)
                    elif option=='nn':
                        #Direct nearest neighbor:
                        TT = Th[:,j0,i0]
                        SS = Sh[:,j0,i0]
                    
                    Tz[:,j,i] = np.interp(depth,dep,TT)
                    Sz[:,j,i] = np.interp(depth,dep,SS)
        del TT,SS,Th,Sh,theta,salt
        self.ds = self.ds.assign_coords({'z':depth})
        self.ds['Tz'] = (['z','y','x'],Tz)
        self.ds['Sz'] = (['z','y','x'],Sz)
        self.ds.attrs['name_forcing'] = f"mitgcm_{startyear}_{endyear}"
        return self.ds
        
    def tanh(self, ztcl, Tdeep, drhodz=.2/1000,z1=200):
        """ creates tanh thermocline forcing profile
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
    
    def linear(self, S1,T1,z1=2000):
        """ creates linear forcing profile
        input:
        ztcl    ..  (float)  [m]       thermocline depth
        Tdeep   ..  (float)  [degC]    in situ temperature at depth
        drhodz  ..  (float)  [kg/m^4]  linear density stratification
        """
        if z1>0:
            print(f'z-coordinate is postive upwards; z1 was {z1}, now set z1=-{z1}')
            z1 = -z1
        S0 = 34.5                     # [psu]  reference surface salinity
        T0 = self.l1*S0+self.l2       # [degC] surface freezing temperature
        
        self.ds['Tz'] = T0 + self.ds.z*(T1-T0)/z1 
        self.ds['Sz'] = S0 + self.ds.z*(S1-S0)/z1
        self.ds = self.calc_fields()
        self.ds.attrs['name_forcing'] = f'linear_S1{S1:.1f}_T1{T1}'
        return self.ds
    
    def calc_fields(self):
        """ adds Ta/Sa fields to geometry dataset: forcing  = frac*COLD + (1-frac)*WARM """
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