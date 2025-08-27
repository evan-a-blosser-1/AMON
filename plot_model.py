import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import numpy as np
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
#######################################################
#######################################################
def OBJ_2_VertFace(Asteroid_file):
    """Reads an OBJ file in and 
        converts the vertice and
        face data into arrays.
    Args:
        Asteroid_file (file): .obj file
    Returns:
        numpy array: 2 arrays of Vertice and Face data
    """
    ##############################################################
    OBJ_Data = np.loadtxt(Asteroid_file, delimiter=' ', dtype=str) 
    ######################
    ## Process OBJ File ##
    ###############################################################
    # Set Vertex/Faces denotaed as v or f in .obj format to array #
    vertex_faces = OBJ_Data[:,0]                                  #
    # Get Length of the Vertex/Faces array for range counting     #
    V_F_Range = vertex_faces.size                                 #
    # Define varibale for number of vertices & faces              #
    numb_vert = 0                                                 #
    numb_face = 0                                                 #
    # Scan Data for v & f and count the numbers of each.          #
    #  Used for sorting x, y, & z as vertices                     #
    for i in range(0,V_F_Range):                                  #
        if vertex_faces[i] == 'v':                                #
            numb_vert += 1                                        #
        else:                                                     #
            numb_face += 1                                        #
    ###############################################################
    #########################
    # Assigning Vertex Data #
    #########################
    # Vertex data assigned to x, y, & z
    #  then cpnverts to float type
    ########################################
    # Assign 2nd row of .txt as x input    
    x_input = OBJ_Data[range(0,numb_vert),1]   
    # Assign 3rd row of .txt as y input    
    y_input = OBJ_Data[range(0,numb_vert),2]   
    # Assign 4th row of .txt as z input    
    z_input = OBJ_Data[range(0,numb_vert),3] 
    ########################################  
    # Convert Vertices data to float type  #
    x_0 = x_input.astype(float)            #
    y_0 = y_input.astype(float)            #
    z_0 = z_input.astype(float)            #
    ########################################
    #
    ##############################################
    # Fill zero indecies with dummy values       #
    #  to allow faces to call vertices 1 to 1014 #
    x = np.append(0,x_0)  ########################
    y = np.append(0,y_0)  #
    z = np.append(0,z_0)  #
    #######################
    #
    #######################
    # Assigning Face Data #
    #######################
    # Face data assigned to fx, fy, & fz
    #  then cpnverts to float type
    #############################################
    # Range count for face data                 
    row_tot = numb_face + numb_vert             
    # Assign 2nd row of .txt as x input         
    fx_input = OBJ_Data[range(numb_vert,row_tot),1] 
    # Assign 3rd row of .txt as y input         
    fy_input = OBJ_Data[range(numb_vert,row_tot),2] 
    # Assign 4th row of .txt as z input         
    fz_input = OBJ_Data[range(numb_vert,row_tot),3] 
    #######################################
    # Convert Vertices data to float type #
    fx = fx_input.astype(int)             #
    fy = fy_input.astype(int)             #
    fz = fz_input.astype(int)             #
    #######################################
    #
    ##########################
    # Creating Output Arrays #
    ##########################
    #    Number of Vertex is (N-1)             
    #     numb_vert += 1
    #########################################
    # Number of Vertex set to array         #
    numb_vert_array = []                    #
    numb_vert_array.append(numb_vert)       #
    # Number of Faces set to array          #
    numb_face_array = []                    #
    numb_face_array.append(numb_face)       #
    # Stacking Columns of Vertex Data       #################
    Vert_Data_Out_0 = np.column_stack((x, y))               #
    Vert_Data       = np.column_stack((Vert_Data_Out_0, z)) #
    # Stacking Columns of Face Data                         #
    Face_Data_Out_0 = np.column_stack((fx,fy))              #
    Face_Data       = np.column_stack((Face_Data_Out_0,fz)) #
    #########################################################
    return Vert_Data , Face_Data

v,f = OBJ_2_VertFace(obj_Path)
gamma = target.gamma
v = v*gamma
f = f-1 
###
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

mesh = Poly3DCollection([v[ii] for ii in f], 
                edgecolor='black',
                facecolors="white",
                linewidth=0.05,
                alpha=0.0)

ax.scatter(CM[:,0], CM[:,1], CM[:,2], color='green', s=10)
           

# ax.add_collection3d(mesh)

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
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_aspect('equal', 'box') 
plt.show()