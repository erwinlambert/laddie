from tools import *
from physics import *
from preprocess import *

def savefields(object,filename=None):
    """Store time-average fields and save"""
    object.dsav['U'] += im(object.u[1,:,:])
    object.dsav['V'] += jm(object.v[1,:,:])
    object.dsav['D'] += object.D[1,:,:]
    object.dsav['T'] += object.T[1,:,:]
    object.dsav['S'] += object.S[1,:,:]
    object.dsav['melt'] += object.melt
    object.dsav['entr'] += object.entr
    object.dsav['ent2'] += object.ent2
    object.dsav['detr'] += object.detr    
    
    object.count += 1

    if object.t in np.arange(object.saveint,object.nt+object.saveint,object.saveint):
        """Output average fields"""
        object.dsav['U'] *= 1./object.count
        object.dsav['V'] *= 1./object.count
        object.dsav['D'] *= 1./object.count
        object.dsav['S'] *= 1./object.count
        object.dsav['T'] *= 1./object.count
        object.dsav['melt'] *= 3600*24*365.25/object.count
        object.dsav['entr'] *= 3600*24*365.25/object.count
        object.dsav['ent2'] *= 3600*24*365.25/object.count
        object.dsav['detr'] *= 3600*24*365.25/object.count        

        object.dsav['mav']  = 3600*24*365.25*(object.dsav.melt*object.dx*object.dy).sum()/(object.tmask*object.dx*object.dy).sum()
        object.dsav['mmax'] = 3600*24*365.25*object.dsav.melt.max()            

        object.dsav['tend'] = object.time[object.t]
        if filename == None:
            object.dsav['filename'] = f"../../results/{object.ds['name_geo'].values}_{object.ds.attrs['name_forcing']}_{object.dsav['tend'].values:03.0f}"
        else:
            object.dsav['filename'] = filename
        object.dsav.to_netcdf(f"{object.dsav['filename'].values}.nc")
        print(f'-------------------------------------------------------------------------------------')
        print(f"{object.time[object.t]:8.03f} days || Average fields saved as {object.dsav['filename'].values}.nc")
        print(f'-------------------------------------------------------------------------------------')
        
        """Set to zero"""
        object.count = 0
        object.dsav['U'] *= 0
        object.dsav['V'] *= 0
        object.dsav['D'] *= 0
        object.dsav['T'] *= 0
        object.dsav['S'] *= 0
        object.dsav['melt'] *= 0
        object.dsav['entr'] *= 0
        object.dsav['ent2'] *= 0
        object.dsav['detr'] *= 0        
        
        object.dsav['tstart'] = object.time[object.t]
        
def saverestart(object):
    if object.t in np.arange(object.restint,object.nt+object.restint,object.restint):
        """Output restart file"""
        object.dsre['u'] = (['n','y','x'], object.u)
        object.dsre['v'] = (['n','y','x'], object.v)
        object.dsre['D'] = (['n','y','x'], object.D)
        object.dsre['T'] = (['n','y','x'], object.T)
        object.dsre['S'] = (['n','y','x'], object.S)
        object.dsre['tend'] = object.time[object.t]
        object.restartfile = f"{object.ds.attrs['name_forcing']}_{object.dsre['tend'].values:03.0f}"
        object.dsre.to_netcdf(f"../../results/restart/{object.ds['name_geo'].values}_{object.restartfile}.nc")

        print(f'-------------------------------------------------------------------------------------')
        print(f"{object.time[object.t]:8.03f} days || Restart file saved")
        print(f'-------------------------------------------------------------------------------------')
        
        print(f"Restarting from {object.restartfile}")
        initialize_vars(object)
        print(f'Restarted')
        
        
def printdiags(object):
    if object.t in np.arange(object.diagint,object.nt+object.diagint,object.diagint):
        """Print diagnostics at given intervals as defined below"""
        #Maximum thickness
        d_Dmax = (object.D[1,:,:]*object.tmask).max()
        d_Dmin = (np.where(object.tmask,object.D[1,:,:],100)).min()
        #Average thickness [m]
        d_Dav = div0((object.D[1,:,:]*object.tmask*object.dx*object.dy).sum(),(object.tmask*object.dx*object.dy).sum())
        #Maximum melt rate [m/yr]
        d_Mmax = 3600*24*365.25*object.melt.max()
        #Average melt rate [m/yr]
        d_Mav = 3600*24*365.25*div0((object.melt*object.dx*object.dy).sum(),(object.tmask*object.dx*object.dy).sum())
        #Meltwater fraction [%]
        d_MWF = 100.*(object.melt*object.tmask*object.dx*object.dy).sum()/((object.melt+(object.entr+object.ent2-object.detr))*object.tmask*object.dx*object.dy).sum()
        #Integrated entrainment [Sv]
        d_Etot = 1e-6*(object.entr*object.tmask*object.dx*object.dy).sum()
        d_E2tot = 1e-6*(object.ent2*object.tmask*object.dx*object.dy).sum()
        #Integrated detrainment [Sv]
        d_DEtot = 1e-6*(object.detr*object.tmask*object.dx*object.dy).sum()
        #Integrated volume thickness convergence == net in/outflow [Sv]
        d_PSI = -1e-6*(convT(object,object.D[1,:,:])*object.tmask*object.dx*object.dy).sum()
        #Average temperature [degC]
        d_Tav = div0((object.D[1,:,:]*object.T[1,:,:]*object.tmask).sum(),(object.D[1,:,:]*object.tmask).sum())
        #Average salinity [psu]
        d_Sav = div0((object.D[1,:,:]*object.S[1,:,:]*object.tmask).sum(),(object.D[1,:,:]*object.tmask).sum())   
        #Average speed [m/s]
        #d_Vav = div0((object.D[1,:,:]*(im(object.u[1,:,:])**2 + jm(object.v[1,:,:])**2)**.5*object.tmask).sum(),(object.D[1,:,:]*object.tmask).sum())
        d_Vmax = ((im(object.u[1,:,:])**2 + jm(object.v[1,:,:])**2)**.5*object.tmask).max()
        #TKE
        d_TKE = 1e-9*((im(object.u[1,:,:])**2 + jm(object.v[1,:,:])**2)**.5*object.tmask*object.D[1,:,:]).sum()*object.dx*object.dy
        #drho
        d_drho = 1000*np.where(object.tmask,object.drho,100).min()
        
        print(f'{object.time[object.t]:8.03f} days || {d_Dav:5.01f}  [{d_Dmin:4.02f} {d_Dmax:4.0f}] m || {d_Mav: 5.02f} | {d_Mmax: 3.0f} m/yr || {d_MWF:5.02f} % || {d_Etot:5.03f} + {d_E2tot:5.03f} - {d_DEtot:5.03f} | {d_PSI: 5.03f} Sv || {d_Vmax: 3.02f} m/s || {d_drho: 5.02f} []')