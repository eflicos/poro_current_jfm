"""
Integrated gravity current equations. Line injection

From command line, call as: 
python3 poro_current.py --Q_inj <Q_inj> --tmax <tmax> --dx <dx_init> --perm_ampl <perm_ampl> --perm_mean <perm_mean> --perm_wl <perm_wl> --output_dir <output_dir>

From python, import and call as: 
poro_current.model(Q_inj=Q_inj,tmax=tmax,dx_init=dx_init,perm_ampl=perm_ampl,perm_mean=perm_mean,perm_wl=perm_wl,output_dir=output_dir)
"""
# Based on poro_nondimensional.f90
# Translated to python by ejf61

# Packages
import numpy as np
# from numba import njit,jit
import scipy.linalg as la
import logging
import yaml
import argparse
from time import time
import random
from pathlib import Path

# My other scripts
import generate_permeability as gp
import generate_plot_times as gt

# Input parameters

# Injection grid point
# I know this is set as a variable, but I've assumed it in so many places
Q_x = 1

itmax = 200000 # max number of iterations
errmax = 1e-5 # max error allowed in timestep check 
threshold = 1e-4 # height below which we take as 0

# Save full height and permeability arrays?
save_height = True

# The time parameters can probably be changed
dtinit = 1e-12 # initial timestep
dtmin = 1e-14 # min timestep         
t0 = 1e-6 # start time in years
hplot_freq = 1 # Save height every X saves

zeta_n = 1.48

input_default = {
    'Q': 1,
    'tmax': 1000,
    'tplotno': 50,
    'nx': 100,
    'L':4,
    'dx_init': 2**(-10),
    'perm_type': 'cos', # Types 'uniform', 'cos', 'gauss', 'channel'
    'perm_ampl': (0.2,),
    'perm_mean': (1,),
    'perm_wl': (1,),
    'output_dir': './Output/',
    'config_file':'./config.yml'
}

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--Q', type=float, help='Injection flux')
    parser.add_argument('--tmax', type=int, help='Maximum time')
    parser.add_argument('--tplotno', type=int, help='Number of saves')
    parser.add_argument('--nx', type=int, help='Number of grid points in x')
    parser.add_argument('--L', type=int, help='Size of domain in y')
    parser.add_argument('--dx_init', type=float, help='Intial grid spacing')
    parser.add_argument('--perm_type', type=str, help='Type of permeability field')
    parser.add_argument('--perm_ampl', type=tuple, help='Minimum permeability')
    parser.add_argument('--perm_mean', type=tuple, help='Mean permeability')
    parser.add_argument('--perm_wl', type=tuple, help='Distance between permeability maxima')
    parser.add_argument('--output_dir', type=str, help='Base directory for output files')
    parser.add_argument('--config_file', type=str, help='Location of configuration file') # Read in config file
    
    input_cmd = vars(parser.parse_args()) # turn inputs into a dictionary
    
    # Wrap in try statement in case no config file exists
    try:
        with open(input_cmd['config_file'], 'r') as file:
            input_file =  yaml.load(file.read(),Loader=yaml.Loader) # Parse config file, safeload doesn't parse tuples properly
    except:
        input_file = {}
    
    input_cmd_filtered = {k: v for k, v in input_cmd.items() if v is not None} # Remove all the Nones from command line
    
    input_tmp = {**input_default,**input_file} # config file overwrites defaults
    input_final = {**input_tmp, **input_cmd_filtered} # command line overwrites everything
    
    Q_inj = input_final['Q']
    tmax = input_final['tmax']
    tplotno = input_final['tplotno']
    
    nx = input_final['nx']
    # ny = input_final['ny']
    L = input_final['L']
    
    dx_init = input_final['dx_init']
    dy_init = input_final['dx_init']

    # Perm params
    perm_type = input_final['perm_type']
    perm_ampl = input_final['perm_ampl']
    perm_mean = input_final['perm_mean']
    perm_wl = input_final['perm_wl']
    
    output_dir = input_final['output_dir']
    # Run model
    model(Q_inj=Q_inj,tmax=tmax,tplotno=tplotno,nx=nx,L=L,dx_init=dx_init,perm_type=perm_type,perm_ampl=perm_ampl,perm_mean=perm_mean,perm_wl=perm_wl,output_dir=output_dir)

# Default list inputs must be tuples (), not proper lists [] or numpy arrays
# Because default inputs must be immutable otherwise bad errors
# Since my list inputs aren't changing and don't need to be dealt with like numpy arrays, tuples are fine
def model(Q_inj=input_default['Q'],tmax=input_default['tmax'],tplotno=input_default['tplotno'],nx=input_default['nx'],L=input_default['L'],dx_init=input_default['dx_init'],perm_type=input_default['perm_type'],perm_ampl=input_default['perm_ampl'],perm_mean=input_default['perm_mean'],perm_wl=input_default['perm_wl'],output_dir=input_default['output_dir']):
    # Logging setup
    logging.basicConfig(filename=output_dir + 'poro_current.log', filemode='w', level=logging.INFO, format='%(asctime)s [%(levelname)s] (%(funcName)s) %(message)s')
    
    logging.info(f'Starting flow simulation')

    # Generate plot times
    tplot = gt.log_dt(t0,tmax,tplotno)
    if len(perm_wl)>1: # len(perm_wl)==1 throws errors if perm_wl is int 
        # Splits tplotno into three
        # third before first est transition time
        # third after last est transition time
        # third in between
        tplot = gt.log_dt_multimodal(t0,tmax,tplotno,perm_wl)
    
    tplot_diff = np.zeros(tplotno)
    for i in range(tplotno-1):
        tplot_diff[i] = tplot[i+1] - tplot[i]
    logging.info(f'End time: {tplot[-1]}')
    logging.info(f'Number of saves: {tplotno}')    
    
    # Initialise variables
    dx = dx_init
    dx_max = 1 # maximum grid spacing
    
    # Not regridding in y, so set dy immediately to max
    pts_per_perm = 16 # number of grid pts per wavelength in y    
    perm_wl_min = min(perm_wl)
    dy = perm_wl_min/pts_per_perm
    
    ny = int(L/dy)+3 # +3 for first/last col and extra pt to reach final min
    
    # Create dx array. Initially set all to dx_init
    # dx = x[i+1]-x[i]
    dx = np.ones(nx-1)*dx_init
    dx_mid = np.zeros(nx)
    dx_mid[0] = dx[0]/2
    dx_mid[1:-1] = 1/2*(dx[1:]+dx[:-1])
    dx_mid[-1] = dx_mid[-2]/2
    dx = dx[:,None] # So that dx multiplies 2d arrays correctly
    dx_mid = dx_mid[:,None]

    # Record input values to file
    with open(output_dir + '/Other/perm_wavelength.txt','w') as f:
        f.write(str(perm_wl))
    with open(output_dir + '/Other/grid_size.txt','w') as f:
        f.write(str(nx) + '\n' + str(ny))
    with open(output_dir+'/Other/dy.txt','w') as f:
        f.write(str(dy))
    
    logging.info(f'Using grid size: nx={nx}, ny={ny}')
    logging.info(f'Initial grid spacing: dx={dx_init}, dy={dy}')  
    logging.info(f'Maximum grid spacing: dx={dx_max}')
    
    # Set up base arrays
    # Inputs 
    h_top = np.zeros((nx,ny)) # topography
    # These aren't included for now
    #Sr = np.zeros((nx,ny)) # residual saturation
    #r_trap = np.ones((nx,ny)) # initial residual saturation
    #phi = np.ones((nx,ny)) # porosity profile
    
    # Injection point
    Q = np.zeros((nx,ny))
    Q[1,1:-1] = 1
    np.savetxt(output_dir + 'Other/injection.txt',Q)
    
    # Permeability
    logging.info(f'Permeability type: {perm_type}')
    perm = gp.gen_perm(nx,ny,dx,dy,perm_type,perm_mean,perm_ampl,perm_wl)
    
    perm_min_calc = np.min(perm)
    perm_mean_calc = np.mean(perm)
    logging.info(f'Min perm is: {perm_min_calc:.2f} mean perm is: {perm_mean_calc:.2f}')
    np.savetxt(output_dir + './Permeability/perm.txt', perm)
    
    if perm_min_calc<=0 or perm_mean_calc<=0:
        logging.error('Minimum permeability is less than 0! Quitting...')
        exit()
    
    h = np.zeros((nx,ny)) # height 0 to start
    t = t0                # start time
    # dtmax = tplot[1] # initial dtmax for linear time spacing
    dtmax = tplot[0] + 0.1*tplot_diff[0] # initial dtmax for log time spacing
    dt = dtinit # initial dt
    plot = 0 # Data output counter  

    dt_check = 1 # number of iterations before checking timestep
    itlast = 0
    it = 1
    
    regrid_count = 0

    start_time = time()
    while t <= tmax and it < itmax:
        
        # Check domain - will regrid if too close to edge 
        # For line injection, only concerned with dx
        bool_regrid = check_regrid(h,t,dt,nx,ny,dx,output_dir)
        if h[-1,1]>=threshold or h[-1,-1]>=threshold:
            logging.error(f'ERROR: Current has reached edge of domain')
            np.savetxt(output_dir + './Current_Thickness/h_error.txt', h)
            break
        
        it_regrid=0
        while bool_regrid:
            # start_regrid = time()
            # Save the time
            save_var(t,output_dir+'./Other/regrid_times.txt',regrid_count)

            h,perm,dx,dx_mid,dy = regrid(h,perm,t,dt,nx,ny,dx,dx_mid,dy,dx_max,perm_ampl,perm_mean,perm_wl,output_dir,regrid_count)
            bool_regrid = check_regrid(h,t,dt,nx,ny,dx,output_dir)
            
            regrid_count+=1
            it_regrid+=1
            logging.debug(f'Regridding, dx now varies between {np.min(dx):.2g} and {np.max(dx):.2g}')
            if it_regrid%10==0:
                logging.warning(f'Done {it_regrid} regrids in a row at time {t}:.3g')
            #logging.info(f'Regridding, dx now varies between {np.min(dx):.2g} and {np.max(dx):.2g}')
            
        # Check if time to alter dt
        if it-itlast == dt_check:
            # Check profiles, see if can increase timestep
            itlast = it # update last iteration
            # Compare one timestep to two half timesteps
            htest2 = timestep(h,perm,h_top,Q,t,dt,nx,ny,dx,dx_mid,dy,output_dir)   
            htest1 = timestep(h,perm,h_top,Q,t,dt/2,nx,ny,dx,dx_mid,dy,output_dir) 
            htest1 = timestep(htest1,perm,h_top,Q,t+dt/2,dt/2,nx,ny,dx,dx_mid,dy,output_dir)
            toterr = error_size(htest1,htest2,nx,ny)
            
            # decrease timestep if error too big
            # keep decreasing until below threshold
            while toterr > 2.*errmax:
                dt_check = 1
                dt = dt/1.5
                logging.debug(f'dt DECREASED t={t}, dt={dt}')
                if (dt < dtmin):
                    logging.error('ERROR: stepsize too small')
                    np.savetxt(output_dir + './Current_Thickness/h1_error.txt', htest1)
                    np.savetxt(output_dir + './Current_Thickness/h2_error.txt', htest2)
                    exit()
                # Repeat timestep with new, smaller dt
                # Compare one timestep to two half timesteps
                htest2 = timestep(h,perm,h_top,Q,t,dt,nx,ny,dx,dx_mid,dy,output_dir)
                htest1 = timestep(h,perm,h_top,Q,t,dt/2,nx,ny,dx,dx_mid,dy,output_dir) 
                htest1 = timestep(htest1,perm,h_top,Q,t+dt/2,dt/2,nx,ny,dx,dx_mid,dy,output_dir)
                toterr = error_size(htest1,htest2,nx,ny) 
                       
            # Advance t and take htest1 as the new height
            t += dt
            h = htest1
                       
            # Increase timestep for next loop if error small enough
            dt, dtmax, dt_check = inc_dt(t,dt,dtmax,tplot,tplot_diff,plot,toterr,dt_check,dx)
            dt_check=1    
        else:
            # Not time to alter dt so do normal timestep
            # logging.info(f'Not checking whether can alter dt, t={t}')
            h = timestep(h,perm,h_top,Q,t,dt,nx,ny,dx,dx_mid,dy,output_dir)
            t += dt
        
        # Check if height has got too big or if any nans
        if np.isnan(np.sum(h)):
            logging.error(f'Found nan in {array_name} at time {t:.4g}. Saving as {save_name}')
            error_save_quit(h,t,dt,dx,output_dir)
        if np.nanmax(h) >= 10**5:
            logging.error(f'Found value greater than {big:.2g} in {array_name} at time {t:.4g}. Saving as {save_name}')
            error_save_quit(h,t,dt,dx,output_dir)
        
        # If looking at residual trapping, take max height here
        
        # Update iteration count 
        it += 1
        
        # Plotting/save loop
        if plot <= tplotno and t >= tplot[plot] - t0:
            logging.info(f'Save #{plot+1}, Current time={t}')
            logging.info(f'Overshot by {(t-tplot[plot]):.2g}, {(100*(t-tplot[plot])/tplot[plot]):.2g}%')
            
            # Calc total volume
            vol_calc = np.sum(h[1:-1,1:-1]*dx[1:]*dy)
            vol_exp = np.sum(Q)*dy * t
            # Calc residual trapped vol here
            logging.info(f'Difference between injected {vol_exp:.2g} and integrated {vol_calc:.2g} volumes is {(100*(vol_exp - vol_calc)/vol_exp):.2g}%')
            
            # Find and save the edge: full co-ords, mean and std
            find_edge_save(h,t,dt,nx,ny,dx,dx_mid,dy,output_dir,plot)
            
            # Save output h if flag set to True
            if save_height and plot%hplot_freq==0:
                np.savetxt(output_dir + './Current_Thickness/h'+'{0:02d}'.format(plot)+'.txt', h)
                save_var(t,output_dir+'./Other/plot_height_times.txt',plot)
            
            # Save actual time value and integrated volume to file
            save_var(t,output_dir+'./Other/plot_times.txt',plot)
            save_var(vol_calc,output_dir+'./Other/volume.txt',plot)
            #save_var(dx,output_dir+'./Other/dx.txt',plot)
            np.savetxt(output_dir + './Other/dx'+'{0:02d}'.format(plot)+'.txt', dx)
            
            # Save height values along injection line
            np.savetxt(output_dir + './Current_at_Injection/h_inj'+'{0:02d}'.format(plot)+'.txt', h[Q_x])
            
            # Update plot number
            plot +=1
            
            # If final plot time, stop
            if plot >= tplotno:
                end_time = time() - start_time
                logging.info(f'Time taken to run={end_time} seconds')
                print(f'Time taken to run={end_time} seconds')
                print(f'Number of iterations: {it}')
                #np.savetxt(output_dir + './Current_Thickness/h'+'{0:02d}'.format(plot)+'.txt', h)
                break
            # Otherwise, prepare for next loop
            else:
                logging.info(f'New plot time={tplot[plot]}')
            logging.info(f'Current iteration count: {it}')
        
    # Check max iterations
    if it == itmax:
        logging.info(f'Maximum number of iterations reached! Iterations={it}')

#######
# Functions and subroutines start here
#######

def timestep(h,perm,h_top,Q,t,dt,nx,ny,dx,dx_mid,dy,output_dir):
    """
    (start height, permeability, caprock topography, flux, time, dt, dx, dy)
    Advance from t to t+dt using predictor/corrector in x and y
    """
    logging.debug(f'Starting timestep for t={t}')
    dt_half = dt/2
    dt_quart = dt_half/2
    # start = time()
    # In x
    # Predictor step - t -> t+dt_quart
    cxp, cxm, cyp, cym = set_var_cur(h,perm,h_top,nx,ny,dx,dx_mid[1:],dy)
    h_x = set_eqnx_cur(h,cxp,cxm,cyp,cym,Q,t,dt_quart,nx,ny,dx[1:],dy,output_dir)
    
    # Corrector step - t -> t+dt_half
    cxp, cxm, cyp, cym = set_var_cur(h_x,perm,h_top,nx,ny,dx,dx_mid[1:],dy)
    h_x = set_eqnx_cur(h,cxp,cxm,cyp,cym,Q,t,dt_half,nx,ny,dx[1:],dy,output_dir)
        
    # In y
    # Predictor step - t+dt_half -> t+dt_half+dt_quart
    cxp, cxm, cyp, cym = set_var_cur(h_x,perm,h_top,nx,ny,dx,dx_mid[1:],dy)
    h_y = set_eqny_cur(h_x,cxp,cxm,cyp,cym,Q,t+dt_half,dt_quart,nx,ny,dx[1:],dy,output_dir)
    
    # Corrector step - t+dt_half -> t+dt
    cxp, cxm, cyp, cym = set_var_cur(h_y,perm,h_top,nx,ny,dx,dx_mid[1:],dy)
    h_y = set_eqny_cur(h_x,cxp,cxm,cyp,cym,Q,t+dt_half,dt_half,nx,ny,dx[1:],dy,output_dir)
    
    # end = time()-start
    # print(end)
    logging.debug(f'Ending timestep for t={t}')
    return h_y

def set_var_cur(h,perm,h_top,nx,ny,dx,dx_mid,dy):
    """
    (start height, permeability, caprock topography, dx, dy)
    Sets flux discretisation 
    dF/dx[i+1/2,j] = cxm[i+1/2,j]*h[i+1,j] - cxp[i+1/2,j]*h[i,j]
    """
    # Initialise variables
    cxp = np.zeros((nx,ny))
    cxm = np.zeros((nx,ny))
    cyp = np.zeros((nx,ny))
    cym = np.zeros((nx,ny))
    diffx= np.zeros((nx-1,ny-1))
    advx = np.zeros((nx-1,ny-1))
    qx = np.zeros((nx-1,ny-1))
    alphax = np.zeros((nx-1,ny-1))
    diffy = np.zeros((nx-1,ny-1))
    advy = np.zeros((nx-1,ny-1))
    qy = np.zeros((nx-1,ny-1))
    alphay = np.zeros((nx-1,ny-1))
    
    # Set variables for x
    # Diffusive term
    diffx[:-1] = (h[1:nx-1,0:ny-1]*perm[1:nx-1,0:ny-1]*dx[:-1] + h[0:nx-2,0:ny-1]*perm[0:nx-2,0:ny-1]*dx[1:])/(2*dx_mid[:-1])
    # Advective term
    advx[:] = - (perm[1:nx,0:ny-1] + perm[0:nx-1,0:ny-1])/2 * (h_top[1:nx,0:ny-1] - h_top[0:nx-1,0:ny-1])/dx_mid
    
    diffx[diffx<=0] = 1e-20
    
    qx = advx * dx_mid / (2*diffx)
    ind_qx_small = np.where(abs(qx)<=0.1)
    #qx[ind_qx_small] = qx[ind_qx_small]**2
    alphax[ind_qx_small] = 1/3. * qx[ind_qx_small] - 1/45. * qx[ind_qx_small]**3 + 2/945. * qx[ind_qx_small]**5 # Add **7?
    
    ind_qx_big = np.where(abs(qx)>0.1)
    alphax[ind_qx_big] = 1/np.tanh(qx[ind_qx_big]) - 1/qx[ind_qx_big]
    
    # Calculate the flux discretisation coeffs
    cxp[0:nx-1,0:ny-1] = diffx/dx_mid + advx/2. * (1 + alphax)
    cxm[0:nx-1,0:ny-1] = diffx/dx_mid - advx/2. * (1 - alphax)
    
    # Necessary??
    cxp[-1,:] = 0
    cxm[-1,:] = 0
    
    
    # Set variables for y    
    # Diffusive term
    diffy[:] = (h[0:nx-1,1:ny]*perm[0:nx-1,1:ny] + h[0:nx-1,0:ny-1]*perm[0:nx-1,0:ny-1])/2
    # Advective term
    advy[:] = - (perm[0:nx-1,1:ny] + perm[0:nx-1,0:ny-1])/2 * (h_top[0:nx-1,1:ny] - h_top[0:nx-1,0:ny-1])/dy
    
    diffy[diffy<=0]=1e-20
    
    qy = advy * dy / (2*diffy)
    ind_qy_small = np.where(abs(qy)<=0.1)
    #qy[ind_qy_small] = qy[ind_qy_small]**2
    alphay[ind_qy_small] = 1/3. * qy[ind_qy_small] - 1/45. * qy[ind_qy_small]**3 + 2/945. * qy[ind_qy_small]**5
    
    ind_qy_big = np.where(abs(qy)>0.1)
    alphay[ind_qy_big] = 1/np.tanh(qy[ind_qy_big]) - 1/qy[ind_qy_big]
    
    # Calculate the flux discretisation coeffs
    cyp[0:nx-1,0:ny-1] = diffy/dy + advy/2. * (1 + alphay)
    cym[0:nx-1,0:ny-1] = diffy/dy - advy/2. * (1 - alphay)
    
    cyp[:,-1] = 0
    cym[:,-1] = 0    
    return cxp, cxm, cyp, cym

def set_eqnx_cur(h,cxp,cxm,cyp,cym,Q,t,dt,nx,ny,dx,dy,output_dir): # output_dir only there for error_save_quit
    """
    (height, flux discretisation coeffs, injection flux, time, dt, dx, dy)
    Advances t to t+dt in x
    """
    # Initialise variables
    bx = np.zeros((nx,ny))
    lx = np.zeros((nx,ny))
    ddx = np.zeros((nx,ny))
    ux = np.zeros((nx,ny))
    h_out = np.zeros((nx,ny))
    abx = np.zeros((3,nx)) # 0 is upper, 1 diag, 2 is lower
    
    # Set up tridiagonal matrix
    # LnXn-1 + DnXn + UnXn+1 = Bn
    
    # BCs at i=0,nx-1
    # dh/dn = 0 for initial and far-field
    bx[0,:] = 0#h[0]+Q[0]*2*dt/dx#0
    lx[0,:] = 0
    ddx[0,:] = 1
    ux[0,:] = -1
    
    bx[-1,:] = 0
    lx[-1,:] = -1
    ddx[-1,:] = 1
    ux[-1,:] = 0
    
    # i=1,nx-2
    bx[1:nx-1,1:ny-1] = h[1:nx-1,1:ny-1] + \
                        dt/dy * (cym[1:nx-1,1:ny-1]*h[1:nx-1,2:ny] - \
                        (cyp[1:nx-1,1:ny-1] + cym[1:nx-1,0:ny-2])*h[1:nx-1,1:ny-1] + \
                        cyp[1:nx-1,0:ny-2]*h[1:nx-1,0:ny-2]) + \
                        dt*Q[1:nx-1,1:ny-1]/dx
    
    lx[1:nx-1,1:ny-1] = -cxp[0:nx-2,1:ny-1]*dt/dx
    ddx[1:nx-1,1:ny-1] = 1 + (cxp[1:nx-1,1:ny-1] + cxm[0:nx-2,1:ny-1])*dt/dx
    ux[1:nx-1,1:ny-1] = -cxm[1:nx-1,1:ny-1]*dt/dx
    
    for j in range(1, ny-1):
        # Tridiagonal solver 
        # 0 is upper, 1 diag, 2 is lower
        abx[0,1:] = ux[:-1,j] # Needs to start with a 0
        abx[1] = ddx[:,j]
        abx[2,:-1] = lx[1:,j] # Needs to end with a 0
        h_out[:,j]=la.solve_banded((1,1),abx,bx[:,j],overwrite_ab=True,overwrite_b=True, check_finite=False)
    
    # Remove any negative heights 
    try:
        h_out[h_out < 0] = 0
    except:
        logging.error(f'ERROR: something has gone when removing negative height values')
        error_save_quit(h_out,t,dt,dx,output_dir)
    return h_out

def set_eqny_cur(h,cxp,cxm,cyp,cym,Q,t,dt,nx,ny,dx,dy,output_dir):
    """
    (height, flux discretisation coeffs, injection flux, time, dt, dx, dy)
    Advances t to t+dt in y
    """
    # Initialise variables
    by = np.zeros((nx,ny))
    ly = np.zeros((nx,ny))
    ddy = np.zeros((nx,ny))
    uy = np.zeros((nx,ny))
    h_out = np.zeros((nx,ny))
    aby = np.zeros((3,ny)) # 0 is upper, 1 diag, 2 is lower
    
    # Set up tridiagonal matrix
    # LnXn-1 + DnXn + UnXn+1 = Bn
    
    # BCs at j=0,ny-1
    # dh/dn = 0 for initial and far-field
    by[:,0] = 0
    ly[:,0] = 0
    ddy[:,0] = 1
    uy[:,0] = -1
    
    by[:,-1] = 0
    ly[:,-1] = -1
    ddy[:,-1] = 1
    uy[:,-1] = 0

    # j=1,ny-2
    by[1:nx-1,1:ny-1] = h[1:nx-1,1:ny-1] + \
                        dt/dx*(cxm[1:nx-1,1:ny-1]*h[2:nx,1:ny-1] - \
                        (cxp[1:nx-1,1:ny-1] + cxm[0:nx-2,1:ny-1])*h[1:nx-1,1:ny-1] + \
                        cxp[0:nx-2,1:ny-1]*h[0:nx-2,1:ny-1]) + \
                        dt*Q[1:nx-1,1:ny-1]/dx
    ly[1:nx-1,1:ny-1] = -cyp[1:nx-1,0:ny-2]*dt/dy
    ddy[1:nx-1,1:ny-1] = 1 + (cyp[1:nx-1,1:ny-1] + cym[1:nx-1,0:ny-2])*dt/dy
    uy[1:nx-1,1:ny-1] = -cym[1:nx-1,1:ny-1]*dt/dy
    
    for i in range(1, nx-1):
        # Tridiagonal solver 
        # 0 is upper, 1 diag, 2 is lower
        aby[0,1:] = uy[i,:-1] # Needs to start with a 0
        aby[1] = ddy[i]
        aby[2,:-1] = ly[i,1:] # Needs to end with a 0
        h_out[i]=la.solve_banded((1,1),aby,by[i],overwrite_ab=True,overwrite_b=True, check_finite=False)
    #h_out[0]=0
    try:
        h_out[h_out < 0] = 0
    except:
        logging.error(f'ERROR: something has gone when removing negative height values')
        error_save_quit(h_out,t,dt,dx,output_dir)
    return h_out

def error_size(h1,h2,nx,ny):
    """
    Calculate error between two arrays, h1 and h2
    Uses L2 norm
    """
    err = np.linalg.norm( (h1 - h2) / np.amax(np.abs(h2)) ) / (nx*ny)
    
    return err


def inc_dt(t,dt,dtmax,tplot,tplot_diff,plot,toterr,dt_check,dx):
    """
    (time, dt, max dt, plot times, difference between plot times, plot counter, error, dt_check)
    Checks if dt can be increased.
    """
    if toterr < errmax:
        if t < tplot[plot]:
            dt_next = tplot[plot] - t # Time to next plot
            # Estimated distance it travels
            # based on x_n = zeta_n * t**(2/3)
            # *0.9 as a buffer
            dt_max_est = 0.9 * 3/2 * np.min(dx) * t**(1/3)/zeta_n 
            dtmax = min(dt_next, dt_max_est)
            dt = min(1.5*dt, dtmax)
        else:
            dt = 1.5*dt
        logging.debug(f'dt increased t={t}, dt={dt}, dtmax={dtmax}')
        
    return dt, dtmax, dt_check

def check_regrid(h,t,dt,nx,ny,dx,output_dir):
    """
    (height, nx, ny, dx, dx_max)
    Checks whether it is time to regrid
    """
    # Find max value in last 10% of rows
    num_row = int(nx/10)
    row_max = np.max(h[-num_row:,:])
    
    # Comparing with a value smaller than threshold for non-zero height
    # Small values will propagate ahead of the edge of the current - this is to catch those
    small = 1e-7
    if row_max > small:
        bool_regrid = True
    else:
        bool_regrid = False
    
    return bool_regrid

def regrid(h,perm,t,dt,nx,ny,dx,dx_mid,dy,dx_max,perm_ampl,perm_mean,perm_wl,output_dir,regrid_count):
    """
    (height time, perm, dx, dy)
    Doubles dx, dy to scale up grid size. Only run if current it close to the edge
    Averages h to create new h
    Regenerates permeability field with new dx/dy
    """
    
     # Initialise
    h_regrid = np.zeros((nx,ny))
    dx_regrid = np.zeros(nx-1)
    dx_mid_regrid = np.zeros(nx)
    dx_flat = dx.flatten()
    # Find edge indexes. Could be part of find_edge function, but that finds actual lengths
    edge_ind = find_edge_ind(h,t,dt,nx,ny,dx,output_dir)
    edge_ind_min = min(edge_ind)
    # Only regrid where dx<dx_max
    if np.max(dx)>=dx_max:
        old_ind_min = list(x<dx_max for x in dx).index(True)
    else:
        old_ind_min = 1
    
    # How many grid points will be regridded
    old_ind_max = int(min(edge_ind_min*0.9, edge_ind_min-2))
    # make sure even number of points. Since old_ind_min included in range, difference should be ODD
    if (old_ind_max-old_ind_min)%2==0:
        old_ind_max-=1
    # /2 to get the new range of regridded pts
    regrid_ind_min = old_ind_min
    regrid_ind_max = old_ind_min + int((old_ind_max-old_ind_min-1)/2)
        
    # Set index ranges
    if regrid_ind_min>1:
        ind_same_source = np.arange(1,regrid_ind_min)
    ind_regrid = np.arange(regrid_ind_min,regrid_ind_max) # The pts where dx --> 2*dx
    ind_same_nose = np.arange(regrid_ind_max,regrid_ind_max + (nx-old_ind_max) + 1) # The range where dx stays same; heights just need shifting in
    
    if len(ind_regrid)==0:
        logging.error(f'No points to regrid')
        error_save_quit(h,t,dt,dx,output_dir)
    # logging.info(f'Regridding {old_ind_max-old_ind_min} pts to {len(ind_regrid)} points')

    # Average the two heights in the 2i-regrid_ind_min,2i-regrid_ind_min+1 pair
    h_regrid[ind_regrid] = (h[2*ind_regrid-regrid_ind_min]*dx[2*ind_regrid-regrid_ind_min] + h[2*ind_regrid-regrid_ind_min+1]*dx[2*ind_regrid-regrid_ind_min+1])/(dx[2*ind_regrid-regrid_ind_min]+dx[2*ind_regrid-regrid_ind_min+1])
    # Shift the heights which aren't being averaged in
    h_regrid[ind_same_nose] = h[old_ind_max-1:]
    # If reached dx_max, preserve the heights which have dx>=dx_max
    if regrid_ind_min>1:
        h_regrid[ind_same_source] = h[ind_same_source]
    
    # Boundaries
    # Coming in, h[:,0] = h[:,1] and h[0] = 0, so preserving that
    h_regrid[0,:] = 0
    h_regrid[:,0] = h_regrid[:,1]
    h_regrid[:,-1] = h_regrid[:,-2]
    # Shift corners in
   # h_regrid[0,0] = h[0,0]
   # h_regrid[int(nx/2),0] = h[-1,0]
   # h_regrid[int(nx/2),-1] = h[-1,-1]
    
    # Change dx accordingly
    # Define new index arrays b/c dx is len nx-1
    if regrid_ind_min>1:
        ind_same_source_dx = np.arange(1,regrid_ind_min)
    ind_regrid_dx = np.arange(regrid_ind_min,regrid_ind_max)
    ind_same_nose_dx = np.arange(regrid_ind_max,regrid_ind_max + (nx-old_ind_max)) 
    
    dx_regrid[ind_regrid_dx] = dx_flat[2*ind_regrid_dx-regrid_ind_min]+dx_flat[2*ind_regrid_dx-regrid_ind_min+1]
    dx_regrid[ind_same_nose_dx] = dx_flat[old_ind_max-1:] 
    dx_regrid[ind_same_nose_dx[-1]+1:]=dx_flat[-1] # All newly created grid pts have smallest spacing
    if regrid_ind_min>1:
        dx_regrid[ind_same_source_dx] = dx_flat[ind_same_source_dx]
    dx_regrid[0]=dx_regrid[1] # dx0=dx1
    
    dx_mid_regrid[0] = dx_regrid[0]/2
    dx_mid_regrid[1:-1] = 1/2*(dx_regrid[1:]+dx_regrid[:-1])
    dx_mid_regrid[-1] = dx_mid_regrid[-2]/2
    dx_regrid = dx_regrid[:,None]
    dx_mid_regrid = dx_mid_regrid[:,None]
    
    # Recalculate permeability field -- not needed for line inj
    #perm = gp.sin_cheq(nx,ny,dx_regrid,dy_regrid,perm_mean,perm_ampl,perm_wl,perm_wl,inj_max)
    #perm = gp.chequerboard(nx,ny,dx_regrid,dy_regrid,perm_mean,perm_ampl,perm_wl,perm_wl)
    
    # Compare volume before and after regridding
    vol_old = np.sum(h[1:-1,1:-1]*dx[1:]*dy)
    vol_regrid = np.sum(h_regrid[1:-1,1:-1]*dx_regrid[1:]*dy)
    vol_diff = (vol_regrid - vol_old)/vol_old * 100
    logging.debug(f'Regridding volume check. Old volume: {vol_old:.3g}, new volume: {vol_regrid:.3g}, change: {vol_diff:.3g}%')
    if np.abs(vol_diff)>0.1:
        logging.warning(f'Regridding has caused a change in volume of >0.1%. You may want to check regridding thresholds')
    
    # Check a random line to see if nose is the same
    ind_check = random.randint(1,ny-1)
    h_slice_old = h[1:-1,ind_check]
    h_slice_regrid = h_regrid[1:-1,ind_check]
    
    ind_edge_old = int(np.where(h_slice_old<threshold)[0][0])    
    ind_edge_regrid = int(np.where(h_slice_regrid<threshold)[0][0])
    
    edge_old = np.sum(dx_mid[2:ind_edge_old-1])+np.float(dx_mid[1])/2
    edge_regrid = np.sum(dx_mid_regrid[2:ind_edge_regrid-1])+np.float(dx_mid_regrid[1])/2
    edge_diff = (edge_regrid - edge_old)/edge_old * 100
    
    logging.debug(f'Regridding edge check (line {ind_check}). Old edge: {edge_old:.3g}, new edge: {edge_regrid:.3g}, change: {edge_diff:.3g}%')
    if np.abs(edge_diff)>0.1:
        logging.warning(f'Regridding has caused a change in edge of >0.1% for a random line. You may want to check regridding thresholds')
    
    save_var([old_ind_min,old_ind_max],output_dir+'./Other/regrid_range.txt',regrid_count)
    return h_regrid,perm,dx_regrid,dx_mid_regrid,dy

def find_edge_save(h,t,dt,nx,ny,dx,dx_mid,dy,output_dir,plot):
    """
    Finds and saves the mean and std of edge of current
    """
    edge = np.zeros((ny-2,2))
    reach_edge = False
    
    # First construct x_array
    x_array = np.zeros(nx)
    for i in range(nx-1):
        x_array[i+1] = x_array[i] + dx_mid[i]
    x_array[:]-=dx[0] 
    
    for j in range(1,ny-1):
        if (h[:,j]==0).all():
            continue # Ignore all fully zero rows
        try:
            i = list(x<threshold for x in h[1:-1,j]).index(True)
            edge[j-1] = [x_array[i+1],(j-1)*dy] # i+1 b/c trimmed off first row in h but not x_array. j-1 b/c trimmed off first ind in edge
        except ValueError as e:
            reach_edge = True
            logging.error(f'ERROR: Current has reached edge of domain')
            logging.error('Full ValueError:' + str(e))
            error_save_quit(h,t,dt,dx,output_dir)
    if not reach_edge:
        # Mean and dev of x-coords
        edge_mean = np.mean(edge[:,0])
        edge_dev = np.std(edge[:,0])
        
        # Save
        save_var(edge_mean,output_dir+'./Other/edge_mean.txt',plot)
        save_var(edge_dev,output_dir+'./Other/edge_dev.txt',plot)
        np.savetxt(output_dir + './Current_Edge/edge'+'{0:02d}'.format(plot)+'.txt', edge)
    return

def find_edge_ind(h,t,dt,nx,ny,dx,output_dir):
    edge_ind_x = np.zeros(ny-2)
    
    for j in range(1,ny-1):
        if (h[:,j]==0).all():
            continue # Ignore all fully zero rows
        try:
            i = list(x<threshold for x in h[1:-1,j]).index(True)
            edge_ind_x[j-1] = i
        except ValueError as e:
            logging.error(f'ERROR: Current has reached edge of domain')
            logging.error('Full ValueError:' + str(e))
            error_save_quit(h,t,dt,dx,output_dir)
    
    return edge_ind_x

def save_var(var,file_name,plot):
    """
    (var,file_name,plot)
    Variable to save, file to save it in
    If plot=0, writes new file, otherwise appends to existing
    """
    if plot==0:
        # Open new file for first
        with open(file_name,'w') as f:
            f.write(str(var)+'\n')
    else:
        # Append to existing
        with open(file_name,'a') as f:
            f.write(str(var)+'\n')

def error_save_quit(h,t,dt,dx,output_dir):
    """
    Call when catching errors. Saves key variables for debugging before quitting
    """
    # Create directory for the save
    Path(output_dir+'Error/').mkdir(parents=True, exist_ok=True)
    
    # Save the variables
    np.savetxt(output_dir + 'Error/h.txt',h)
    np.savetxt(output_dir + 'Error/dx.txt',dx)
    with open(output_dir + 'Error/t.txt','w') as f:
        f.write(str(t))
    with open(output_dir + 'Error/dt.txt','w') as f:
        f.write(str(dt))
        
    exit()

if __name__ == "__main__":
    main()
