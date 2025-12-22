import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import numpy as np
import trimesh
####
sys.dont_write_bytecode = True
import constants as C
###
# Asteroid Name
asteroid = '1950DA_Prograde'
target = C.DA1950()
folder   = "Databank/OG_3.7km/" 
aux1 = "1.6e-09"
aux2 = "3.7"
# file1 = folder + '/' + 'TR-S0' +'-H' + aux1 + 'Yi' + aux2 + '.dat'
# data = np.loadtxt(file1, dtype=str)
###########################################################
################################################ Load files
# object file
obj_Path =  asteroid + '.obj' 
# MASCONs (tetrahedron center of masses)
CM_Path =  asteroid + '_CM.in' 
CM = np.loadtxt(CM_Path, delimiter=' ',dtype=float)
Terta_Count = len(CM)
print(f"Number of Tetrahedrons (MASCONs): {Terta_Count}")
#######################################################
#######################################################
mesh = trimesh.load_mesh(asteroid + '.obj')
gamma = target.gamma
mesh = mesh.apply_scale(gamma)
v = mesh.vertices
f = mesh.faces 
print(f"Number of vertices: {len(v)}")
print(f"Number of faces: {len(f)}")
###
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mesh = Poly3DCollection([v[ii] for ii in f], 
                edgecolor='black',
                facecolors="white",
                linewidth=0.05,
                alpha=0.0)

ax.scatter(CM[:,0], CM[:,1], CM[:,2], color='green', s=10)
           

ax.add_collection3d(mesh)

# ax.plot(data[:, 0], data[:, 1], data[:, 2],color='yellow', linewidth=1.5, label='Trajectory')
ax.set_aspect('equal', 'box') 


#####
# Dark mode 
# Background = "#000000"
# # Hide Grid 
# #Grid_Color = "#000000"
# #plt.rcParams['grid.color'] = Grid_Color
# # Display gird 
# Grid_Color = 'white'

#####
# Light mode 
Background = '#FFFFFF'
Grid_Color = "#000000"
plt.rcParams["figure.autolayout"] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

fig.set_facecolor(Background)
ax.set_facecolor(Background)
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.tick_params(axis='x', colors=Grid_Color)
ax.tick_params(axis='y', colors=Grid_Color)
ax.tick_params(axis='z', colors=Grid_Color)
ax.yaxis.label.set_color(Grid_Color)
ax.xaxis.label.set_color(Grid_Color)
ax.zaxis.label.set_color(Grid_Color)
ax.xaxis.line.set_color(Grid_Color)
ax.yaxis.line.set_color(Grid_Color)
ax.zaxis.line.set_color(Grid_Color)

ax.grid(False)
ax.set_xlabel('X (km)', fontsize=24,labelpad=15)
ax.set_ylabel('Y (km)', fontsize=24,labelpad=15)
ax.set_zlabel('Z (km)', fontsize=24,labelpad=15)
ax.tick_params(axis='both', labelsize=24)
ax.set_aspect('equal', 'box') 
plt.show()