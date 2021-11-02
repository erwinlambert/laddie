from tools import *
from physics import *

def savefields(object):
    """Store time-average fields and save"""
    object.dsav['U'] += im(object.u[1,:,:])
    object.dsav['V'] += jm(object.v[1,:,:])
    object.dsav['D'] += object.D[1,:,:]
    object.dsav['T'] += object.T[1,:,:]
    object.dsav['S'] += object.S[1,:,:]
    object.dsav['melt'] += object.melt
    object.dsav['entr'] += object.entr
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

        object.dsav['mav']  = 3600*24*365.25*(object.dsav.melt*object.dx*object.dy).sum()/(object.tmask*object.dx*object.dy).sum()
        object.dsav['mmax'] = 3600*24*365.25*object.dsav.melt.max()            

        object.dsav['tend'] = object.time[object.t]

        object.dsav['filename'] = f"../../results/{object.ds['name_geo'].values}_{object.ds.attrs['name_forcing']}_{object.dsav['tend'].values:.3f}"
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
        object.dsre.to_netcdf(f"../../results/restart/{object.ds['name_geo'].values}_{object.ds.attrs['name_forcing']}_{object.dsre['tend'].values:.3f}.nc")

        print(f'-------------------------------------------------------------------------------------')
        print(f"{object.time[object.t]:8.03f} days || Restart file saved")
        print(f'-------------------------------------------------------------------------------------')
        
def printdiags(object):
    if object.t in np.arange(object.diagint,object.nt+object.diagint,object.diagint):
        """Print diagnostics at given intervals as defined below"""
        #Maximum thickness
        d0 = (object.D[1,:,:]*object.tmask).max()
        #d0b = (np.where(object.tmask,object.D[1,:,:],100)).min()
        #Average thickness [m]
        d1 = div0((object.D[1,:,:]*object.tmask*object.dx*object.dy).sum(),(object.tmask*object.dx*object.dy).sum())
        #Maximum melt rate [m/yr]
        d2 = 3600*24*365.25*object.melt.max()
        #Average melt rate [m/yr]
        d3 = 3600*24*365.25*div0((object.melt*object.dx*object.dy).sum(),(object.tmask*object.dx*object.dy).sum())
        #Meltwater fraction [%]
        d4 = 100.*(object.melt*object.tmask*object.dx*object.dy).sum()/((object.melt+object.entr)*object.tmask*object.dx*object.dy).sum()
        #Integrated entrainment [Sv]
        d6 = 1e-6*(object.entr*object.tmask*object.dx*object.dy).sum()
        d6b = 1e-6*(object.ent2*object.tmask*object.dx*object.dy).sum()
        #Integrated volume thickness convergence == net in/outflow [Sv]
        d5 = -1e-6*(convT(object,object.D[1,:,:])*object.tmask*object.dx*object.dy).sum()
        #Average temperature [degC]
        d7 = div0((object.D[1,:,:]*object.T[1,:,:]*object.tmask).sum(),(object.D[1,:,:]*object.tmask).sum())
        #Average salinity [psu]
        d8 = div0((object.D[1,:,:]*object.S[1,:,:]*object.tmask).sum(),(object.D[1,:,:]*object.tmask).sum())   
        #Average speed [m/s]
        d9 = div0((object.D[1,:,:]*(im(object.u[1,:,:])**2 + jm(object.v[1,:,:])**2)**.5*object.tmask).sum(),(object.D[1,:,:]*object.tmask).sum())

        print(f'{object.time[object.t]:8.03f} days || {d1:7.03f} | {d0:8.03f} m || {d3: 7.03f} | {d2: 8.03f} m/yr || {d4:6.03f} % || {d6:6.03f} {d6b:6.03f} | {d5: 6.03f} Sv || {d9: 6.03f} m/s || {d7: 8.03f} C || {d8: 8.03f} psu')