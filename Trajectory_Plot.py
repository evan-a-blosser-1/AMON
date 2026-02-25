import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import sys
import trimesh
from numba import njit
sys.dont_write_bytecode = True
import constants as C
const    = C.constants()
day = const.day
############################################################
############################################# Main Settings
# Asteroid Name
asteroid = '1950DA_Prograde'
target = C.DA1950()
# 
folder   = "Databank/1950DA_res/Landing_trj/"
####
# Set:
# 0: Bounded motion
# 1: Collision Or Escape 
typ_flg = '0'
########
xi = 3.0
xf = 3.0
dx = 0.01
nx = round((xf - xi)/dx)
########################
Hi = 2.0e-8
Hf = 2.0e-8
dH = 0.1e-8
nH = round((Hf - Hi) / dH)
#############################################################################
################################################ angular velocity calculation
Spin = target.spin
omega = (2*np.pi)/(Spin * 3600)
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
########################################################
#################################### Load Shape File 
all_z = []
scatter_plots = []
mesh = trimesh.load_mesh(obj_Path)
gamma = target.gamma
mesh = mesh.apply_scale(gamma)
v = mesh.vertices
f = mesh.faces 
print(f"Vertices: {v.shape}")
print(f"Faces: {f.shape}") 
#######################################################
##################################### Plot Settings <<<
# plt.rcParams["figure.figsize"] = [6.5, 6.5]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
# plt.axvline(x = 4.45e-3, color = 'r')
# plt.axvline(x = (1 - 4.45e-3), color = 'b')

col_ls = ['r', 'b', 'g', 'y', 'm', 'c', 'k']

fig1 = plt.figure(figsize=(10, 10))
ax1  = fig1.add_subplot(111, projection='3d')
ax1.set_xlabel(r'$X$ $(km)$', fontsize=25, labelpad=25)
ax1.set_ylabel(r'$Y$ $(km)$', fontsize=25, labelpad=25)
ax1.set_zlabel(r'$Z$ $(km)$', fontsize=25, labelpad=25)
ax1.tick_params(axis='both', which='major', labelsize=25)
####
# # Dark Mode: uncomment to enable dark mode
# fig1.set_facecolor('#000000')
# ax1.set_facecolor('#000000')
# ax1.tick_params(axis='x', colors='white')
# ax1.tick_params(axis='y', colors='white')
########################################################
####################
for ii in range(0, nx + 1):
    x0 = xi + float(ii)*dx
    for jj in range(0, nH + 1):
        H0 = Hi + float(jj)*dH
    
        aux1 = f"{H0:.0e}"
        #aux1 = f"{H0}"
        #aux1 = '0.0' 
        aux2 = str(round(x0, 5))
        
                
        file = folder + '/' + 'TR-S' + typ_flg + '-H' + aux1 + 'Yi' + aux2 + '.dat'
    
    
        print(file)
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


        # Get last 1000 points
        # [-1000:, 2]
        x = list(ps[-10000:, 0])
        y = list(ps[-10000:, 1])
        z = list(ps[-10000:, 2])
        all_z.extend(z) 
        #####################
        print(f"y0 = {y[0]}")
        #
        #
        col = col_ls[(ii * (nH + 1) + jj) % len(col_ls)]
        ax1.plot(x, y, z ,alpha=1, color="#0FAD1E")
        ax1.add_collection3d(mesh)
        ax1.set_aspect('equal', 'box') 
        #####################################
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
        
        # inS = rot2fix(state, omega*t)
        # print(inS)
        # vel_in = np.sqrt(inS[:,3]**2 + inS[:,4]**2 + inS[:,5]**2)
        # r_in = np.sqrt(inS[:,0]**2 + inS[:,1]**2 + inS[:,2]**2)
        # print(vel.shape)
        # print(r.shape)
        # ####
        # ax7.plot(inS[:,0], inS[:,1], inS[:,2], alpha=0.5, color='red')

        ###########
        plt.show()

# plt.savefig(folder + 'Poincare_Sec.png', dpi=300)