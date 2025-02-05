import os
from tools import *
from physics import *
from preprocess import *

def savefields(object):
    """Store time-average fields and save"""

    #Accumulate value each timestep
    if object.save_Ut:
        object.Uav[object.jmin:object.jmax+1,object.imin:object.imax+1] += im(object.U[1,1:-1,1:-1]) #U on tgrid
    if object.save_Uu:
        object.Uuav[object.jmin:object.jmax+1,object.imin:object.imax+1] += object.U[1,1:-1,1:-1]    #U on ugrid
    if object.save_Vt:
        object.Vav[object.jmin:object.jmax+1,object.imin:object.imax+1] += jm(object.V[1,1:-1,1:-1]) #V on tgrid
    if object.save_Vv:
        object.Vvav[object.jmin:object.jmax+1,object.imin:object.imax+1] += object.V[1,1:-1,1:-1]    #V on vgrid
    if object.save_D:
        object.Dav[object.jmin:object.jmax+1,object.imin:object.imax+1] += object.D[1,1:-1,1:-1]
    if object.save_T:
        object.Tav[object.jmin:object.jmax+1,object.imin:object.imax+1] += object.T[1,1:-1,1:-1]
    if object.save_S:
        object.Sav[object.jmin:object.jmax+1,object.imin:object.imax+1] += object.S[1,1:-1,1:-1]
    if object.save_melt:
        object.meltav[object.jmin:object.jmax+1,object.imin:object.imax+1] += object.melt[1:-1,1:-1]
    if object.save_entr:
        object.entrav[object.jmin:object.jmax+1,object.imin:object.imax+1] += object.entr[1:-1,1:-1]
    if object.save_ent2:
        object.ent2av[object.jmin:object.jmax+1,object.imin:object.imax+1] += object.ent2[1:-1,1:-1]
    if object.save_detr:
        object.detrav[object.jmin:object.jmax+1,object.imin:object.imax+1] += object.detr[1:-1,1:-1]
    
    #Counter for the number of timesteps added
    object.count += 1

    if object.t in np.arange(object.saveint,object.nt+object.saveint,object.saveint):
        """Output average fields"""

        #Divide accumulated values by number of time steps and apply mask
        #Added to dsav data set
        if object.save_Ut:
            object.dsav['Ut'][:] = object.Uav/object.count * np.where(object.tmask_full,1,np.nan)
        if object.save_Uu:
            object.dsav['Uu'][:] = object.Uuav/object.count * np.where(object.umask_full,1,np.nan)
        if object.save_Vt:
            object.dsav['Vt'][:] = object.Vav/object.count * np.where(object.tmask_full,1,np.nan)
        if object.save_Vv:
            object.dsav['Vv'][:] = object.Vvav/object.count * np.where(object.vmask_full,1,np.nan)
        if object.save_D:
            object.dsav['D'][:] = object.Dav/object.count * np.where(object.tmask_full,1,np.nan)
        if object.save_T:
            object.dsav['T'][:] = object.Tav/object.count * np.where(object.tmask_full,1,np.nan)
        if object.save_S:
            object.dsav['S'][:] = object.Sav/object.count * np.where(object.tmask_full,1,np.nan)

        #Scale fluxes from m/s to m/yr
        if object.save_melt:
            object.dsav['melt'][:] = object.meltav * 3600*24*365.25/object.count * np.where(object.tmask_full,1,np.nan)
        if object.save_entr:
            object.dsav['entr'][:] = object.entrav * 3600*24*365.25/object.count * np.where(object.tmask_full,1,np.nan)
        if object.save_ent2:
            object.dsav['ent2'][:] = object.ent2av * 3600*24*365.25/object.count * np.where(object.tmask_full,1,np.nan)
        if object.save_detr:
            object.dsav['detr'][:] = object.detrav * 3600*24*365.25/object.count * np.where(object.tmask_full,1,np.nan)

        #Bulk values
        object.dsav['mav']  = 3600*24*365.25*(object.meltav*object.dx*object.dy).sum()/(object.tmask_full*object.dx*object.dy).sum()
        object.dsav['mmax'] = 3600*24*365.25*object.meltav.max()            

        object.dsav.attrs['time_end'] = object.time[object.t]

        #Create filename based on end time of run
        filename = os.path.join(object.rundir,f"output_{object.dsav.attrs['time_end']:06.0f}.nc")
        
        #Save time-average output
        object.dsav.to_netcdf(filename)

        object.print2log(f'-------------------------------------------------------------------------------------')
        object.print2log(f"{object.time[object.t]:8.0f} days || Average fields saved as {filename}")
        object.print2log(f'-------------------------------------------------------------------------------------')

        if object.save_BMB:

            #Convert to kg/m2/s
            BMBn = -object.rhofw / object.count * object.meltav[object.jmin:object.jmax+1,object.imin:object.imax+1]

            #Extrapolate below grounded ice
            BMBext = object.grd[1:-1,1:-1]*compute_average_NN_2D(BMBn,object.tmask[1:-1,1:-1])+object.tmask[1:-1,1:-1]*BMBn
            BMBext = np.where(np.isnan(BMBext),0,BMBext)

            object.dsbmb['BMB'][object.jmin+object.BMBborder:object.jmax+1+object.BMBborder,object.imin+object.BMBborder:object.imax+1+object.BMBborder] = BMBn
            object.dsbmb['BMBext'][object.jmin+object.BMBborder:object.jmax+1+object.BMBborder,object.imin+object.BMBborder:object.imax+1+object.BMBborder] = BMBext

            object.dsbmb.attrs['time_end'] = object.time[object.t]

            object.dsbmb.to_netcdf(os.path.join(object.rundir,object.BMBfilename))

            object.print2log(f"{object.time[object.t]:8.0f} days || Updated BMB in {object.BMBfilename}")
            object.print2log(f'-------------------------------------------------------------------------------------')
            
            #Check if you need to export a file with BMB and BMBext on the UFE main grid (needed when running laddie only for ROI)
            if object.save_BMB_to_full_UFE_domain == True:
                bmb_UFE_domain = object.dsbmb.interp(x=object.UFE_x, y=object.UFE_y, method="linear")
                bmb_UFE_domain.to_netcdf(os.path.join(object.rundir,object.BMB_filename_UFE_domain))

                object.print2log(f"{object.time[object.t]:8.0f} days || Updated BMB_UFE in {object.BMB_filename_UFE_domain}")
                object.print2log(f'-------------------------------------------------------------------------------------')
            

        #Set all fields used for accumulation back to zero
        object.count = 0
        if object.save_Ut:
            object.Uav *= 0
        if object.save_Uu:
            object.Uuav *= 0
        if object.save_Vt:
            object.Vav *= 0
        if object.save_Vv:
            object.Vvav *= 0
        if object.save_D:
            object.Dav *= 0
        if object.save_T:
            object.Tav *= 0
        if object.save_S:
            object.Sav *= 0
        if object.save_melt:
            object.meltav *= 0
        if object.save_entr:
            object.entrav *= 0
        if object.save_ent2:
            object.ent2av *= 0
        if object.save_detr:
            object.detrav *= 0        
        
        #Start time for next time-average 
        object.dsav.attrs['time_start'] = object.time[object.t]

        if object.save_BMB:
            object.dsbmb.attrs['time_start'] = object.time[object.t]
        
def saverestart(object):
    if object.t in np.arange(object.restint,object.nt+object.restint,object.restint):
        """Output restart file"""

        #Save full fields necessary to start a run from restart file
        object.dsre['U'][:,object.jmin:object.jmax+1,object.imin:object.imax+1] = object.U[:,1:-1,1:-1]
        object.dsre['V'][:,object.jmin:object.jmax+1,object.imin:object.imax+1] = object.V[:,1:-1,1:-1]
        object.dsre['D'][:,object.jmin:object.jmax+1,object.imin:object.imax+1] = object.D[:,1:-1,1:-1]
        object.dsre['T'][:,object.jmin:object.jmax+1,object.imin:object.imax+1] = object.T[:,1:-1,1:-1]
        object.dsre['S'][:,object.jmin:object.jmax+1,object.imin:object.imax+1] = object.S[:,1:-1,1:-1]
        object.dsre.attrs['time'] = object.time[object.t]

        #Name of the restartfile
        object.restartfile = os.path.join(object.rundir,f"restart_{object.dsre.attrs['time']:06.0f}.nc")

        #Save restartfile
        object.dsre.to_netcdf(object.restartfile)
        object.dsre.to_netcdf(os.path.join(object.rundir,f"restart_latest.nc"))

        object.print2log(f'-------------------------------------------------------------------------------------')
        object.print2log(f"{object.time[object.t]:8.0f} days || Restart file saved as {object.restartfile}")
        object.print2log(f'-------------------------------------------------------------------------------------')
        
        object.print2log(f"Restarting from {object.restartfile}")

        #Do an actual restart from restartfile 
        #initialise_vars(object)        
        
def printdiags(object):
    """Print diagnostics to log file"""

    #Check whether current time step overlaps with required interval for printing diagnostics
    if object.t in np.arange(object.diagint,object.nt+object.diagint,object.diagint):

        #Maximum thickness [m]
        d_Dmax = (object.D[1,:,:]*object.tmask).max()
        #Minimum thickness [m]
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
        #Integrated additional entrainment [Sv]
        d_E2tot = 1e-6*(object.ent2*object.tmask*object.dx*object.dy).sum()
        #Integrated detrainment [Sv]
        d_DEtot = 1e-6*(object.detr*object.tmask*object.dx*object.dy).sum()
        #Integrated volume thickness convergence == net in/outflow [Sv]
        d_PSI = -1e-6*(object.convD*object.tmask*object.dx*object.dy).sum()
        #Average temperature [degC]
        #d_Tav = div0((object.D[1,:,:]*object.T[1,:,:]*object.tmask).sum(),(object.D[1,:,:]*object.tmask).sum())
        #Average salinity [psu]
        #d_Sav = div0((object.D[1,:,:]*object.S[1,:,:]*object.tmask).sum(),(object.D[1,:,:]*object.tmask).sum())   
        #Average speed [m/s]
        #d_Vav = div0((object.D[1,:,:]*(im(object.u[1,:,:])**2 + jm(object.v[1,:,:])**2)**.5*object.tmask).sum(),(object.D[1,:,:]*object.tmask).sum())
        d_Vmax = ((im(object.U[1,:,:])**2 + jm(object.V[1,:,:])**2)**.5*object.tmask).max()
        #TKE
        #d_TKE = 1e-9*((im(object.U[1,:,:])**2 + jm(object.V[1,:,:])**2)**.5*object.tmask*object.D[1,:,:]).sum()*object.dx*object.dy
        #Minimum drho, for checking convective instability
        d_drho = 1000*np.where(object.tmask,object.drho,100).min()
        #Number of grid cells where convection is applied
        d_conv = object.convection.sum()
        
        object.print2log(f'{object.time[object.t]:8.03f} days || {d_Dav:5.01f}  [{d_Dmin:4.02f} {d_Dmax:4.0f}] m || {d_Mav: 5.02f} | {d_Mmax: 3.0f} m/yr || {d_MWF:5.02f} % || {d_Etot:5.03f} + {d_E2tot:5.03f} - {d_DEtot:5.03f} | {d_PSI: 5.03f} Sv || {d_Vmax: 3.02f} m/s || {d_drho: 5.05f} {d_conv: 3.0f} []')