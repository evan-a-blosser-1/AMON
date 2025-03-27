""" Asteroid Package 

    This handles backend stuff 

"""
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 

#%% File Handling 
def Loading():
    # Define the frames of the animation
    frames = ['-', '\\', '|', '/']

    # Loop to create the animation
    while True:
        for frame in frames:
            # Print the frame and flush the output
            sys.stdout.write('\r' + frame)
            sys.stdout.flush()
            # Delay for a short period
            time.sleep(0.1)


def print_progress(current_time, total_time, bar_length=11):
    fraction = current_time / total_time
    block = int(round(bar_length * fraction))
    bar = '\u25A1' * block + '-' * (bar_length - block)
    sys.stdout.write(f'\r[{bar}] {fraction:.2%}')
    sys.stdout.flush()
           
            
            

def OBJ_Read (OBJ_File):
    ##############################################################
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex and face data
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    
    return vertices, faces

def OBJ_2_volInt(OBJ_File):
    """Reads an OBJ file and converts the vertex and face data into arrays.
    
    Args:
        OBJ_File (file): .obj file
    
    Returns:
        tuple: Two numpy arrays containing vertex and face data
    """
    # Read the OBJ file data
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex data and prepend an empty row of zeros
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    vertices = np.vstack(([0, 0, 0], vertices))
    
    # Extract face data and prepend a 3 to each row
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    faces = np.hstack((np.full((faces.shape[0], 1), 3), faces))
    
    return vertices, faces

#%% Plotting

def Plot_State(state,OBJ_File,gamma,Mesh_color,M_line=0.5,M_alpha=0.05):
    ##############################################################
    OBJ_Data = np.loadtxt(OBJ_File, delimiter=' ', dtype=str)
    
    # Extract vertex and face data
    vertices = np.array([line[1:].astype(float) for line in OBJ_Data if line[0] == 'v'])
    faces = np.array([line[1:].astype(int) for line in OBJ_Data if line[0] == 'f'])
    # Scale vertices
    vertices = vertices * gamma
    # Set faces to start at index 0 instead of 1
    faces = faces - 1
    # Create mesh
    mesh = Poly3DCollection([vertices[ii] for ii in faces], 
                            edgecolor=Mesh_color,
                            facecolors="white",
                            linewidth=M_line,
                            alpha=M_alpha)
    X = state[0,:]
    Y = state[1,:]  
    Z = state[2,:]
    #######################################################
    #######################################################
    figure3D = plt.figure()
    ax = figure3D.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)
    ax.plot(X, Y, Z, label='Trajectory',color='purple')
    ax.set_aspect('equal', 'box') 
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    return 


def Plot_Ham(state,Time,omega,mu_I,CM):
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.plot(V_mag)
    ax1.set_title('Velocity Magnitude')
    ax1.set_xlabel('Time')
    ax1.set_ylabel(r'Velocity $\frac{km}{s}$')
    ax2.plot(Time, Energy)
    ax2.set_title('Hamiltonian')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy')
    plt.tight_layout()
    return



def plot_4by4(SimData):
    # Axis limit 
    # axes[1, 0].set_ylim([-0.05, 0.05])
    X = SimData['Position Vec. X']
    Y = SimData['Position Vec. Y']
    Z = SimData['Position Vec. Z']
    V_mag = np.sqrt(SimData['Velocity Vec. X']**2 + SimData['Velocity Vec. Y']**2 + SimData['Velocity Vec. Z']**2)
    # Corrected subplot creation
    fig, axes = plt.subplots(2, 2, figsize=(20, 20), subplot_kw={'projection': '3d'})
    axis0 = axes[0, 0]
    line_color ='#332288'
    ############################################################################
    # 3D Plot with V_mag as color
    scatter = axis0.scatter(X, Y, Z, c=V_mag, cmap='cool',s=0.1)
    cbar = fig.colorbar(scatter, ax=axis0, shrink=0.5, aspect=20)  
    cbar.set_label(r'$V (km/s)$', fontsize=25,labelpad=20) 
    axis0.view_init(elev=30, azim=45)   
    axis0.set_xlabel(r'$X (km)$', fontsize=25,labelpad=20)
    axis0.set_ylabel(r'$Y (km)$', fontsize=25,labelpad=20)
    axis0.set_zlabel(r'$Z (km)$', fontsize=25,labelpad=25)
    axis0.tick_params(axis='x', labelsize=20) 
    axis0.tick_params(axis='y', labelsize=20) 
    axis0.tick_params(axis='z', labelsize=20) 
    ############################################################################
    # Convert the rest of the axes back to 2D for the remaining plots
    for ax in axes.flat[1:]:
        ax.remove()
    axes[0, 1] = fig.add_subplot(2, 2, 2)
    axes[1, 0] = fig.add_subplot(2, 2, 3)
    axes[1, 1] = fig.add_subplot(2, 2, 4)
    # X-Y Plot
    axes[0, 1].plot(X, Y, color=line_color,linewidth=0.05)
    axes[0, 1].set_title("X-Y", fontsize=25)
    axes[0, 1].set_xlabel(r'$X (km)$', fontsize=25,labelpad=20)
    axes[0, 1].set_ylabel(r'$Y (km)$', fontsize=25,labelpad=20)
    axes[0, 1].tick_params(axis='x', labelsize=20) 
    axes[0, 1].tick_params(axis='y', labelsize=20) 
    ############################################################################
    # Y-Z Plot
    axes[1, 0].plot(Y, Z, color=line_color,linewidth=0.05)
    axes[1, 0].set_title("Y-Z", fontsize=25)
    axes[1, 0].set_xlabel(r'$Y (km)$', fontsize=25,labelpad=20)
    axes[1, 0].set_ylabel(r'$Z (km)$', fontsize=25,labelpad=20)
    axes[1, 0].tick_params(axis='x', labelsize=20) 
    axes[1, 0].tick_params(axis='y', labelsize=20) 
    ############################################################################
    # X-Z Plot
    axes[1, 1].plot(X, Z, color=line_color,linewidth=0.05)
    axes[1, 1].set_title("X-Z", fontsize=25)
    axes[1, 1].set_xlabel(r'$X (km)$', fontsize=25,labelpad=20)
    axes[1, 1].set_ylabel(r'$Z (km)$', fontsize=25,labelpad=20)
    axes[1, 1].tick_params(axis='x', labelsize=20) 
    axes[1, 1].tick_params(axis='y', labelsize=20) 
    ############################################################################
    plt.tight_layout()

#%% Tetra_Volume 


def Tetra_Volume(Verts,Faces,MASCON_Div,total_vol=0):
    """Tetrahedron Volume Calculation 
            - for polyhedron shape models. 
    Args:
        Verts (array): Polyhedron Vertices
        Faces (array): Polyhedron Face Data
        MASCON_Div (fracton): Divides the tetrahedron, set = 1 for total
        total_vol (bool): 0/1 for total volume calculations, or  just tetra calculations
    """
    tetra_count=np.shape(Faces)[0]
    ###################    
    # Make list
    Volume_Tetra_Array  = np.zeros(tetra_count, dtype="float64")

    Tetra_U = np.zeros((tetra_count, 3), dtype="float64")
    Tetra_V = np.zeros((tetra_count, 3), dtype="float64")
    Tetra_W = np.zeros((tetra_count, 3), dtype="float64")
    ############################### Analytical Method:
    for it in range(0,tetra_count):
    ##### Center of mass to vertex vectors
        U_vec = np.array([Verts[Faces[it,0],0]*MASCON_Div,
                          Verts[Faces[it,0],1]*MASCON_Div,
                          Verts[Faces[it,0],2]*MASCON_Div
                        ]) 
        V_vec = np.array([Verts[Faces[it,1],0]*MASCON_Div,
                          Verts[Faces[it,1],1]*MASCON_Div,
                          Verts[Faces[it,1],2]*MASCON_Div
                        ]) 
        W_vec = np.array([Verts[Faces[it,2],0]*MASCON_Div,
                          Verts[Faces[it,2],1]*MASCON_Div,
                          Verts[Faces[it,2],2]*MASCON_Div
                        ])
        Tetra_U[it] = U_vec
        Tetra_V[it] = V_vec 
        Tetra_W[it] = W_vec 
        ######################################################
        ############### Triple Scalar Product ################
        Vol_Full_Sum =  U_vec[0]*(V_vec[1]*W_vec[2] - V_vec[2]*W_vec[1]) -\
                        U_vec[1]*(V_vec[0]*W_vec[2] - V_vec[2]*W_vec[0]) +\
                        U_vec[2]*(V_vec[0]*W_vec[1] - V_vec[1]*W_vec[0]) 
        #
        Vol_tetra_full = (1/6) * abs((Vol_Full_Sum))
        Volume_Tetra_Array[it] = Vol_tetra_full
    ###############################################
    ######### Sum Tetra Volumes for Check #########
    Total_Volume_Out = np.sum(Volume_Tetra_Array)
    #######################################################
    ######### Output ######################################
    if total_vol == 0:
        Volume_Tetra_Array_Message = f"""
{'-'*42}
|
|  Total Volume Calculated as:
|    V = {Total_Volume_Out}
|
{'-'*42}
"""
        # print(Volume_Tetra_Array_Message )
        return Volume_Tetra_Array,Total_Volume_Out
    elif total_vol == 1:
        return Volume_Tetra_Array



#%%





