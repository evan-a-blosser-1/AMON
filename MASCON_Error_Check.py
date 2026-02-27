
import numpy as np
import matplotlib.pyplot as plt
import sys
########################################################
# Tsuolis Grav Model
from polyhedral_gravity import (Polyhedron, GravityEvaluable, 
                                evaluate, PolyhedronIntegrity, 
                                NormalOrientation)
#################################################
####### Enter Asteroid Name & Update Here #######
Asteroid_Name = 'Apophis'

# Big G in km^3/kg.s^2
G = 6.6741e-20

# Asteroid Density kg/m^3
Den = 1.75e3 

# scaling factor (m)
gamma = 285.0

##################################################
########### Load File ############################
Aster_File_CM   = Asteroid_Name + "_CM.in"
Aster_File_OBJ  = Asteroid_Name + ".obj"
Aster_CM  = np.loadtxt(Aster_File_CM, delimiter=' ')
mu_I = np.loadtxt(Asteroid_Name + '_mu.in', delimiter=' ')
##########################################################
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

############################################################
# Load OBJ file 
print(f'| Asteroid File: {Aster_File_OBJ} |')
vertices, faces = OBJ_2_VertFace(Aster_File_OBJ)
vertices = vertices * gamma 
vertices = vertices[1:]
print(f'| Vertices: {vertices} |')
faces = faces - 1
print(f'| Faces: {faces} |')



###################
# Tsoulis Model
#
# Create the model 
polyhedron = Polyhedron(
   polyhedral_source=(vertices, faces),
   density=Den,
   integrity_check=PolyhedronIntegrity.DISABLE,   # VERIFY (default), DISABLE or HEAL
)




# Point Mass approximation
def Approx_2Body(mu_i, x_in):
    mu = G * 5.31e10
    x = x_in
    y = 0 
    z = 0
    r = np.sqrt(x**2 + y**2 + z**2) 
    U = mu/r
    return U

# MASCON Model
def MASCON_U(CM,mu_i, x_in):
    U = np.zeros(len(x_in))
    for it in range(len(CM)):
        x = x_in - CM[it,0]
        y = 0    - CM[it,1]
        z = 0    - CM[it,2]
        r = np.sqrt(x**2 + y**2 + z**2) 
        U += mu_i[it]/r
    print(f'U: {U}')
    return U
    
    
    
####
# Solve for the potential
#
# Point & MASCON Models 
x = np.linspace(0.01, 2.0, 10000)    

B2 = Approx_2Body(mu_I, x)
M1 = MASCON_U(Aster_CM, mu_I, x)
#################################


##############
# Solve Tsuolis Model
#
# take to meters 
VALUES = np.arange(10.0, 2000.0, (2000.0 - 10.0) / 10000)
X = VALUES
#
computation_points = np.array(np.meshgrid(X, [0], [0])).T.reshape(-1, 3)
print(f'| Computation Points: {computation_points} |')
#
evaluable = GravityEvaluable(polyhedron=polyhedron)
results = evaluable(
  computation_points=computation_points,
  parallel=True,
)
#
potentials =  np.array([i[0] for i in results])
TP = potentials.reshape((len(VALUES)))
#########
# Convert from km^2 to m^2
TP = TP * (1/1e6)
print(TP)


################ Plot Results ##################
################################################
fig, ax = plt.subplots(figsize=(10, 5))
plt.rcParams["figure.autolayout"] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
########
Rel_Error = ((B2-M1)/B2)
plt.plot(x, Rel_Error,label='2-Body Approximation Vs. MASCON', color='blue')
########
Rel_Error_T_M = ((TP-M1)/TP)
plt.plot(x, Rel_Error_T_M,label='MASCON Vs. Tsuolis',color='red')
########
Rel_Error = ((B2-TP)/B2)
plt.plot(x, Rel_Error,label='2-Body Approximation Vs. Tsuolis', color='green')
#########
ax.set_xlabel('X (km)', fontsize=24)
ax.tick_params(axis='x', labelsize=24)
ax.set_ylabel('Relative Error', fontsize=24)
ax.tick_params(axis='y', labelsize=24)
# ax.set_title('Error in Potential', fontsize=14, fontweight='bold')
plt.legend(fontsize=18)
############
plt.show()




# fig2, ax2 = plt.subplots()
# ax2.set_xlabel('X (km)', fontsize=12, fontweight='bold')
# ax2.set_ylabel('Relative Error', fontsize=12, fontweight='bold')
# ax2.set_title('2-Body Approximation Vs. MASCON', fontsize=14, fontweight='bold')

# data = {
#     'Tsuolis': TP,
#     '2-Body': B2,
#     'MASCON': M1
# }
# df = pd.DataFrame(data)

# # Save the DataFrame to a CSV file
# df.to_csv('Potentials.csv', index=False)
