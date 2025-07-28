import os
import sys
import time
import trimesh
from numba import njit 
import numpy as np
import multiprocessing
from scipy.integrate import solve_ivp
from dataclasses import dataclass
sys.dont_write_bytecode = True
import constants as C
const  = C.constants()
target = C.apophis()
from equations import v_calc, Calc_Ham
######################################
# @dataclass
# class Task:
#     Time: np.ndarray
#     a0: np.ndarray
#     CM: np.ndarray
#     mu_I: np.ndarray
#     omega: float
#     Ham: float
     
# global y0, yf, dy, H0, Hf, dH
##################################################################
##################################################################
################### Simulation Settings <<<<<<<<<<<<<<<<<<<<<<<<<<
# Turn on/off the OCSER CPU count (1/0): uses half cpu count of PC if 0
OCSER_CPU = 0
# Asteroid name
aster    = 'Apophis'
# data path
datpth   = 'Databank/Smap_60days/'  
###
# Hill Sphere (km)
esc_lim = 34.0
# Rotation Rate (rev/hr)
T = 30.5
###
# Save Poincare & Traj on/off (1/0)
ps_svflg = 0
tr_svflg = 0
sm_svflg = 1
###
# Exclude unnecessary
# initial conditions 
srt = 0.00
end = 0.1
exclude_List = []
for i in np.arange(srt, end, step=0.01):
    exclude_List.append(np.round(i,2))
##
y0 = 0.5
yf = 10.0
dy = 0.01
###
H0 = 0.1e-9
Hf = 5.0e-9
dH = 0.01e-9
###########
str_t = 0.0
dt    = 1.0
days  = 60.0
########################
###################################################
# Create a directory to save the data
isExist = os.path.exists(datpth)
if not isExist:
    os.mkdir(datpth)
###########################################################
################################################ File Path 
if OCSER_CPU == 1:
    obj_Path = '/home/eablosser/Apophis/' + aster + '.obj' 
    CM_Path  = '/home/eablosser/Apophis/' + aster + '_CM.in' 
    mu_Path  = '/home/eablosser/Apophis/' + aster + '_mu.in'
else:
    obj_Path =  aster + '.obj' 
    CM_Path =   aster + '_CM.in' 
    mu_Path =   aster + '_mu.in'
#######################################################
################################ Load Model & Constants 
### Spin Rate 
omega = 2.0*np.pi/(T*3600.0)
### Time 
end_t   = days*const.day
dN = round((end_t - str_t )/dt)
Time = np.linspace(start=str_t, stop=end_t, num=dN)
### MASCONs (tetrahedron center of masses)
CM = np.loadtxt(CM_Path, delimiter=' ',dtype=float)
Terta_Count = len(CM)
### Gravitational Parameter of each MASCON
mu_I = np.loadtxt(mu_Path, delimiter=' ')
# Polyhedron Mesh
mesh = trimesh.load_mesh(obj_Path)
Gamma  = target.gamma
mesh.apply_scale(Gamma)
###########################################################
###########################################################
# Smap Files
###########################################################
Bound_File  = datpth +  "Smap_Bound_Events"  + '.dat'
Crash_File  = datpth +  "Smap_Crash_Events"  + '.dat'
Escape_File = datpth +  "Smap_Escape_Events" + '.dat'
################################################################
#################################################################
@njit
def EOM_MASCON(Time,a,CM,mu_I, omega, Ham):
    # print(f"Time = {Time}")
    x,y,z,vx,vy,vz = a
    dxdt = vx
    dydt = vy
    dzdt = vz
    #########
    Ux = np.zeros(1,dtype="float64")
    Uy = np.zeros(1,dtype="float64")
    Uz = np.zeros(1,dtype="float64")
    for it in range(len(CM)):
        ###
        r_x = a[0] - CM[it,0] 
        r_y = a[1] - CM[it,1]
        r_z = a[2] - CM[it,2]
        ###
        vec = np.array([r_x,r_y,r_z])
        r_i = np.linalg.norm(vec)
        ###
        Ux += - mu_I[it]*r_x/(r_i**3)
        Uy += - mu_I[it]*r_y/(r_i**3)
        Uz += - mu_I[it]*r_z/(r_i**3)
    ### End loop
    dvxdt = omega**2*x + 2*omega*vy + Ux[0]
    dvydt = omega**2*y - 2*omega*vx + Uy[0]
    dvzdt = Uz[0]
    ###
    dadt  = [dxdt,dydt,dzdt,dvxdt,dvydt,dvzdt]
    return dadt 

@njit
def bdy2aprx(Time,a,CM,mu_I, omega, Ham):
    # print(f"Time = {Time}")
    x,y,z,vx,vy,vz = a
    dxdt = vx
    dydt = vy
    dzdt = vz
    #########
    r = np.sqrt(x**2 + y**2 + z**2)
    #################
    mu = np.sum(mu_I)
    dvxdt = omega**2*x + 2*omega*vy - (mu*x)/r**3
    dvydt = omega**2*y - 2*omega*vx - (mu*y)/r**3
    dvzdt = - (mu*z)/r**3
    ###
    dadt  = [dxdt,dydt,dzdt,dvxdt,dvydt,dvzdt]
    return dadt 

################################################
def poincare(state,sv_file,Ham):
    nt = state.shape[1] -1
    ###
    x1 = np.zeros(6)
    xp = np.zeros(6)
    ###
    x1[0] = state[0,0]
    x1[1] = state[1,0]
    x1[2] = state[2,0]
    x1[3] = state[3,0]
    x1[4] = state[4,0]
    x1[5] = state[5,0]
    for itp in range(1, nt+1):
        ###
        # Check
        # right side 
        # if state[0,itp]*x1[0] < 0.0 and state[0,itp] > 0.0:
        # Both sides
        if state[0,itp]*x1[0] < 0.0 and state[0,itp] > 0.0:
            
            xp[0] = (state[0, itp] + x1[0])/2.0
            xp[1] = (state[1, itp] + x1[1])/2.0
            xp[2] = (state[2, itp] + x1[2])/2.0
            xp[3] = (state[3, itp] + x1[3])/2.0
            xp[4] = (state[4, itp] + x1[4])/2.0
            xp[5] = (state[5, itp] + x1[5])/2.0
            ###################################
            with open(sv_file, "a") as file_PS:
                np.savetxt(file_PS, xp, newline=' ')
                file_PS.write(str(round(state[1,0], 5)) + ' ' + str(round(state[3,0], 14)) + \
                    ' ' + str(round(Ham, 14)) + "\n")
            file_PS.close()
        ##############################
        ##############################            
        # Update the sate
        # for each sign check
        x1[0] = state[0, itp]
        x1[1] = state[1, itp]
        x1[2] = state[2, itp]
        x1[3] = state[3, itp]
        x1[4] = state[4, itp]
        x1[5] = state[5, itp]
        #####################
        
################################################
################# Events
###############################################
def collision(Time, a, CM,  mu_I, omega, Ham):
    global cond, critical 
    ###
    # Initialize
    cond = 0
    pnt_coord = np.array([a[0],a[1],a[2]])
    r_mag = np.linalg.norm(pnt_coord)
    ###
    clos_pt, dis, _ = mesh.nearest.on_surface([pnt_coord])
    ###
    # wihtin 0.2 m
    tol = 0.0002
    on_surf = dis[0] < tol
    ###
    if on_surf:
        cond = 1
    ###
    if cond != 0:
        critical = 0
    else:
        critical = 1
    return critical
collision.direction = -1
collision.terminal  = True
###
def escape(Time, a, CM,  mu_I, omega, Ham):
    global cond, critical
    ###
    # Initialize
    cond = 0
    r_mag = np.linalg.norm(a[0:3])
    if r_mag >= esc_lim:
        cond = 9
        
    if cond != 0:
        critical = 0
    else:
        critical = 1
    return critical
escape.direction = -1
escape.terminal  = True
###############################################
###############################################
###
def solve_orbit(task):
    Time, a0, CM,  mu_I, omega, Ham = task
    print(f"Solving orbit for y0 = {a0[1]} and Ham = {Ham}")
    mu = np.sum(mu_I)
    T_orb = round(2 * np.pi * np.sqrt(a0[1]**3/mu))
    print(f"Orbital Period = {T_orb/86400} days")
    #
    sol = solve_ivp(
            fun=EOM_MASCON,
            # fun=bdy2aprx,           
            t_span=[Time[0], Time[-1]],           
            y0=a0,          
            args=(CM,  mu_I, omega, Ham),
            events=[collision, escape],
            method='DOP853',     
            first_step=dt,
            rtol=1e-10,
            atol=1e-12,             
            t_eval=Time,  
            #min_step=dt/100,
            #max_step=dt*13 
            dense_output=True
    )
    ###
    
    ###
    state = sol.y 
    
    
    ################################ Checkpoint ##########################################
    ################################
    # print(sol)
    # print(sol.message)
    # input("Press Enter to continue...")
    sol_t = sol.t[-1]
    ####################
    # Survival Map 
    ########################
    if sm_svflg == 1:
        if sol.status == 0:
            with open(Bound_File, "a") as file_PS:
                file_PS.write(str(a0[1]) + ' ' + str(Ham)  + "\n")
                file_PS.close()
        ########################
        ########################
        if sol.status == 1:
            if sol.t_events[0].size > 0:
                with open(Crash_File, "a") as file_PS:
                    file_PS.write(str(a0[1]) + ' ' + str(Ham)  + "\n")
                    file_PS.close()
        ########################
        ########################
            if sol.t_events[1].size > 0:
                with open(Escape_File, "a") as file_PS:
                    file_PS.write(str(a0[1]) + ' ' + str(Ham)  + "\n")
                    file_PS.close()
    ######################################################################
    # print("There was a critical event -- excluding orbit from PS.\n")
    # print("Check PS_log.dat file for details.\n")
    # Set tags for the file name
    # and generate poincare section
    aux0 = str(sol.status)
    aux1 = str(Ham)
    aux2 = str(round(a0[1], 5))
    #############################
    # Poincare Section
    if ps_svflg == 1:
        file1 = datpth  + 'PY-S'+ aux0 + '-H' + aux1 + 'Yi' + aux2 + '.dat'
        # print (it, state[9, it], xa[1], state[9, it]*xa[1])
        # if y == y0:
        # Test if the file alreay exists.
        isExist = os.path.exists(file1)
        if isExist:
            print (file1, "already exists. Overwriting.")
            os.remove(file1)
        ############################
        #   #
        if sol.status == 0:
            # Save the poincare section
            poincare(state,file1,Ham)
    #################
    # Traj 
    if tr_svflg == 1:
        # file name
        file2 = datpth  + 'TR-S'+ aux0 + '-H' + aux1 + 'Yi' + aux2 + '.dat'
        # Save first part of the trajectory
        # T_orb = round(2 * np.pi * np.sqrt(a0[1]**3/mu))
        # aux_state = np.transpose(state[:, :T_orb])
        aux_state = np.transpose(state[:, :])
        # print(aux_state.shape)
        
        isExist = os.path.exists(file2)
        if isExist:
            print (file2, "already exists. Overwriting.")
            os.remove(file2)
        with open(file2, "a") as file_traj:
            np.savetxt(file_traj, aux_state)
            file_traj.close()
            # pause()
    #################
    # Delete the state variable
    del state 
    del sol 
    
################################################
########################################### Main 
# Multiprocessing
tasks = []

# As Thang said: we need to set this as SLURM cpu 
#   count 
#
if OCSER_CPU == 1:
    CPU_COUNT = int(os.getenv("SLURM_CPUS_PER_TASK"))
else:
    # CPU_COUNT = int(1)
    CPU_COUNT = int(multiprocessing.cpu_count() / 2 )
###############################################################################
########################### Parallel processing ###############################
if __name__ == "__main__":
    #######################################################
    
    #######################################################
    ##################### Begin Loops ##################### 
    Calc_Start_Time = time.time() 
    N1 =  round((yf-y0)/dy)
    # Loop through y0
    for ii in range (0, N1+1):
        y = y0 + float(ii)*dy
        ###################
        # Test this after velocity test 
        # use 6 for kilometers
        # to keep precision  
        y = round(y, 6)
        ################################
        if np.round(y,2) in exclude_List:
            continue
        # Same for energy, this one can be nested inside position
        N2 = round((Hf - H0) / dH)
        for jj in range(0, N2+1):
            Ham = H0 + float(jj)*dH
            # Calculate the initial velocity
            x_dot = v_calc(Ham,omega,mu_I,CM,y)
            if np.isnan(x_dot):
                print(f"Skipping y0 = {y} and Ham = {Ham} due to NaN velocity.")
                continue
            # Define initial conditions for this iteration
            #      x0  y0      x_dot  y_dot 
            a0 = [ 0.0, y, 0.0, x_dot, 0.0,  0.0]
            # a0 = np.array([ 0.0, y, 0.0, x_dot, 0.0,  0.0], dtype="float64")
            
            ###############################################################
            tasks.append((Time, a0, CM,  mu_I, omega, Ham))
            
            # Task.Time = Time
            # Task.a0 = np.array(a0)
            # Task.CM = CM
            # Task.mu_I = mu_I
            # Task.omega = omega
            # Task.Ham = Ham
            
    #################################################################################
    ###################### Solve orbits using multiprocessing #######################
    #
    Start_Time = time.time()
    
    with multiprocessing.Pool(processes=CPU_COUNT) as pool:
        results = pool.map(solve_orbit, tasks)
        
        
    End_Time = time.time()
    Calculated_In = End_Time - Start_Time   
    print(f"Elapsed Time: {Calculated_In} seconds")
    Time_File_Name = datpth + "_execution_time_" + '.dat'
    with open(Time_File_Name, mode='w') as file:
        file.write(str(Calculated_In) + ' (sec)')


