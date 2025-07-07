import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import sys
import trimesh
from scipy import stats
###
# Asteroid Name
asteroid = 'Apophis'
Spin = 30.5
# 
omega = (2*np.pi)/(Spin * 3600)

folder   = "Databank/OG_may27/" 
########
xi = 1.0
xf = 1.0
dx = 0.01
nx = round((xf - xi)/dx)
########################
Hi = 1.6e-9
Hf = 1.6e-9
dH = 0.1e-9
nH = round((Hf - Hi) / dH)
########################
def Calc_Ham(state,omega,mu_I,CM):
    U = np.zeros(state.shape[1], dtype="float64")
    for it in range(len(CM)):
        x = state[0,:] - CM[it,0]
        y = state[1,:] - CM[it,1]
        z = state[2,:] - CM[it,2]
        r = np.sqrt(x**2 + y**2 + z**2) 
        U += mu_I[it]/r
    ############################
    X = state[0,:]
    Y = state[1,:]
    Z = state[2,:]
    VX = state[3,:]
    VY = state[4,:]
    VZ = state[5,:]
    V_mag = np.sqrt(VX**2 + VY**2 + VZ**2)
    Energy = 0.5*(VX**2 + VY**2 + VZ**2) - 0.5*omega**2* (X**2 + Y**2 + Z**2) - U[0]
    # print(Energy)
    return Energy
###
def Poincare (state):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # print(f"| Generating Poincare Section...")
    nt = state.shape[1] - 1
    # global CJ, x0, vy0
    xa = np.zeros(6)
    # xm = np.zeros(6)
    xm = np.zeros((6,nt))
    #####################
    xa[0] = state[0, 0]
    xa[1] = state[1, 0]
    xa[2] = state[2, 0]
    xa[3] = state[3, 0]
    xa[4] = state[4, 0]
    xa[5] = state[5, 0]
    for it in range(1, nt + 1):
        ##########################################
        # Check x2*x1 < 0 & x2 > 0 for sign change
        neg_si = state[0, it]*xa[0]
        pos_si = state[0, it]
        #
        if neg_si < 0.0 and pos_si > 0.0:
            # print(f"Negative Sign Check: {neg_si:.6f}")
            # print(f"Positive Sign Check: {pos_si:.6f}")
            # print(f"X Position: {state[0, it]:.6f}")
            # print(f"Y Position: {state[1, it]:.6f}")
            # print(f"Z Position: {state[2, it]:.6f}")
            # print(f"X Velocity: {state[3, it]:.6f}")
            # print(f"Y Velocity: {state[4, it]:.6f}")
            # print(f"Z Velocity: {state[5, it]:.6f}")
            # input('| Press Enter to continue...')
            # print('----------------------------')
            #####
            # Fill matrix of points 
            # xm[0,it] = (state[0, it] + xa[0])/2.0
            # xm[1,it] = (state[1, it] + xa[1])/2.0
            # xm[2,it] = (state[2, it] + xa[2])/2.0
            # xm[3,it] = (state[3, it] + xa[3])/2.0
            # xm[4,it] = (state[4, it] + xa[4])/2.0
            # xm[5,it] = (state[5, it] + xa[5])/2.0
            #####
            # Single Point
            xm[0] = (state[0, it] + xa[0])/2.0
            xm[1] = (state[1, it] + xa[1])/2.0
            xm[2] = (state[2, it] + xa[2])/2.0
            xm[3] = (state[3, it] + xa[3])/2.0
            xm[4] = (state[4, it] + xa[4])/2.0
            xm[5] = (state[5, it] + xa[5])/2.0
            ax.plot(xm[1], xm[4], 'r.')
        ##############################
        ##############################            
        # Update the sate
        # for each sign check
        xa[0] = state[0, it]
        xa[1] = state[1, it]
        xa[2] = state[2, it]
        xa[3] = state[3, it]
        xa[4] = state[4, it]
        xa[5] = state[5, it]
    ########################
    # print(xm)
    # ##### Estimate the phase space density
    # points = np.vstack((xm[0], xm[4]))
    # kde = stats.gaussian_kde(points)
    # density = kde.evaluate(points)
    # scatter = ax.scatter(xm[1], xm[4], c=density, cmap='viridis')
    # plt.colorbar(scatter, label='Phase Space Density')
#####


########################################################
#################################### Personal Packages #    
sys.dont_write_bytecode = True
import constants as C
const    = C.constants()
target = C.apophis()
###########################################################
################################################ Load files
# object file
obj_Path =  asteroid + '.obj' 
# MASCONs (tetrahedron center of masses)
CM_Path =  asteroid + '_CM.in' 
CM = np.loadtxt(CM_Path, delimiter=' ',dtype=float)
Terta_Count = len(CM)
########################################
# Gravitational Parameter of each MASCON
mu_Path =   asteroid + '_mu.in'
mu_I = np.loadtxt(mu_Path, delimiter=' ')
mu = np.sum(mu_I)
print(f"Total Gravitational Parameter: {mu}")
#######################################################
# Polyhedron Center of Mass
mesh = trimesh.load_mesh(obj_Path)
Poly_CM = [target.Poly_x, target.Poly_y, target.Poly_z]
#######################################################
#######################################################
# plt.rcParams["figure.figsize"] = [6.5, 6.5]
plt.rcParams["figure.autolayout"] = True
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

# plt.axvline(x = 4.45e-3, color = 'r')
# plt.axvline(x = (1 - 4.45e-3), color = 'b')


########################################################
########################################################
all_z = []
scatter_plots = []
mesh = trimesh.load_mesh('Apophis.obj')
gamma = 0.285 
v = mesh.vertices*gamma
f = mesh.faces - 1
print(f"Vertices: {v.shape}")
print(f"Faces: {f.shape}") 
# mesh = Poly3DCollection([v[ii] for ii in f], 
#                 edgecolor='black',
#                 facecolors="white",
#                 linewidth=0.75,
#                 alpha=0.0)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.add_collection3d(mesh)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

col_ls = ['r', 'b', 'g', 'y', 'm', 'c', 'k']

fig1 = plt.figure()
ax1  = fig1.add_subplot(111, projection='3d')
ax1.set_title('Trajectory')
ax1.set_xlabel('X (km)')
ax1.set_ylabel('Y (km)')
ax1.set_zlabel('Z (km)')

fig2 = plt.figure()
ax2  = fig2.add_subplot(111)
ax2.set_title('Hamiltonian Energy')
ax2.set_xlabel('time (sec)')
ax2.set_ylabel('Hamiltonian (km^2/s^2)')

fig3 = plt.figure()
ax3  = fig3.add_subplot(111)
ax3.set_title(r'Velocity')
ax3.set_xlabel('time (sec)')
ax3.set_ylabel(r'Velocity $\frac{km}{s}$')


fig4 = plt.figure()
ax4  = fig4.add_subplot(111)
ax4.set_title(r'Position')
ax4.set_xlabel('time (sec)')
ax4.set_ylabel(r'Position (km)')

fig5 = plt.figure()
ax5  = fig5.add_subplot(111)
ax5.set_title('Inertial Frame Velocity')
ax5.set_xlabel('time (sec)')
ax5.set_ylabel(r'Velocity $\frac{km}{s}$')

fig6 = plt.figure()
ax6  = fig6.add_subplot(111)
ax6.set_title('Inertial Frame Position')
ax6.set_xlabel('time (sec)')
ax6.set_ylabel(r'Position (km)')

fig7 = plt.figure()
ax7  = fig7.add_subplot(111, projection='3d')
ax7.set_title('Inertial Frame Trajectory')  
ax7.set_xlabel('X (km)')
ax7.set_ylabel('Y (km)')
ax7.set_zlabel('Z (km)')

########################################################
####################

for ii in range(0, nx + 1):
    x0 = xi + float(ii)*dx
    for jj in range(0, nH + 1):
        H0 = Hi + float(jj)*dH
    
        aux1 = f"{H0:.1e}"
        #aux1 = f"{H0}"
        #aux1 = '0.0' 
        aux2 = str(round(x0, 5))
        
        #file = folder + '/' + 'PY-C' + aux1 + 'Xi' + aux2 + '.dat'
        
        #file = folder + '/' + 'PY-C' + aux1 + 'Yi' + aux2 + '.dat'
        
                
        file = folder + '/' + 'TR-S0' +'-H' + aux1 + 'Yi' + aux2 + '.dat'
    
    
    
        # Skip missing files
        if os.path.isfile(file) == False:
            continue 
        # Print loaded files after skipping
        print(file)
        ps = np.loadtxt(file, dtype=float)
 
        mesh = Poly3DCollection([v[ii] for ii in f], 
                        edgecolor='black',
                        facecolors="white",
                        linewidth=0.75,
                        alpha=0.0)


        x = list(ps[:, 0])
        y = list(ps[:, 1])
        z = list(ps[:, 2])
        all_z.extend(z) 
        #####################
        print(f"y0 = {y[0]}")
        #
        #
        col = col_ls[(ii * (nH + 1) + jj) % len(col_ls)]
        ax1.plot(x, y, z ,alpha=1, color="#b2396b")
        ax1.add_collection3d(mesh)
        ax1.set_aspect('equal', 'box') 
        #####################################
        enr = Calc_Ham(ps.T, omega, mu_I, CM)
        t = np.linspace(0,ps.shape[0],ps.shape[0])
        # print(enr)
        # print(t)
        # print('####################')
        #####################
        #
        #
        col = col_ls[(ii * (nH + 1) + jj) % len(col_ls)]
        ax2.plot(t,enr, alpha=1, color=col)
        ###
        vel = np.sqrt(ps[:, 3]**2 + ps[:, 4]**2 + ps[:, 5]**2)
        ax3.plot(t,vel, alpha=1, color='blue',label='Rotating')
        r = np.sqrt(ps[:, 0]**2 + ps[:, 1]**2 + ps[:, 2]**2)
        ax4.plot(t, r, alpha=1, color='blue',label='Rotating')
        ###################################
        r = np.sqrt(ps[:, 0]**2 + ps[:, 1]**2 + ps[:, 2]**2)
        r_max = np.max(r)
        r_min = np.min(r)
        r_bar = (r_max - r_min)/y[0]
        if np.isclose(r_bar, 0):
            e = "Circular"
        elif r_bar > 1.0:
            e = "Hyperbolic"
        else:
            e = "Elliptical"
        analysis = f"""
    {"-"*42}
    | r_max = {r_max}
    | r_min = {r_min}
    | r_bar = {r_bar}
    | e_apr = {e}
    |--- Velocity ---
    | max   = {ps[:,4].max()}  
    | min   = {ps[:,4].min()}
    | mean  = {np.mean(ps[:,4])}  +/- {np.std(ps[:,4])}
    |----------------
    |
    {"-"*42}
        """
        print(analysis)

        state = ps
        ###################################
        # Translate to inertial frame 
        #
        # fixing rot2fixed frame
        #
        def rot2fix(xr, t):
            print(xr[0, 0])
            # input('| Press Enter to continue...')
            ct = np.cos(t)
            st = np.sin(t)

            xf = np.zeros_like(xr)
            
            
            xf[:, 0] = xr[:, 0]*ct - xr[:, 1]*st
            xf[:, 1] = xr[:, 1]*ct + xr[:, 0]*st
            xf[:, 2] = xr[:, 2]
            xf[:, 3] =-xr[:, 0]*st - xr[:, 4]*st + xr[:, 3]*ct - xr[:, 1]*ct
            xf[:, 4] = xr[:, 3]*st - xr[:, 1]*st + xr[:, 0]*ct + xr[:, 4]*ct
            xf[:, 5] = xr[:, 5]

            return xf
        
        inS = rot2fix(state, omega*t)
        print(inS)
        vel_in = np.sqrt(inS[:,3]**2 + inS[:,4]**2 + inS[:,5]**2)
        r_in = np.sqrt(inS[:,0]**2 + inS[:,1]**2 + inS[:,2]**2)
        print(vel.shape)
        print(r.shape)
        ax5.plot(t, vel_in, alpha=1, color='red', label='Inertial')
        ax6.plot(t, r_in, alpha=1, color='red', label='Inertial')
        ####
        ax7.plot(inS[:,0], inS[:,1], inS[:,2], alpha=0.5, color='red')

        ###########
        ###
        Poincare(ps.T)

        plt.show()





# fig.set_facecolor('#000000')
# ax.set_facecolor('#000000')
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')

# plt.show()
# plt.savefig(folder + 'Poincare_Sec.png', dpi=300)