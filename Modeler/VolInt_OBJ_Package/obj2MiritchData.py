"""
Program:   obj2Miritch Data.py

Description: This takes either an .obj or .txt      
                ployhedron data file and creates a   
                data file that can be used with      
                Brian Mirtich's volInt.c program. 

MIT License

Copyright (c) [2023] [Evan Blosser]

"""
import numpy as np                  
#################
## User Input  ##
###############################################################
# Ask for Desired File Name                                   #
Asteroid_file_in = input('|Please input desired file name: ') #
# Add .obj extension to file name                             #
Asteroid_file = (Asteroid_file_in +'.obj')
#######################
# Load ,txt data file ###################################### 
data = np.loadtxt(Asteroid_file, delimiter=' ', dtype=str) #
############################################################
from icecream import ic
ic(data)

################################
# Determine where Vertices End # 
################################
# This section allows for any .txt in the .OBJ format file to be loaded
#  and used for analysis 
###############################################################
# Set Vertex/Faces denotaed as v or f in .obj format to array #
vertex_faces = data[:,0]                                      #
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
#
#########################
# Assigning Vertex Data #
#########################
# Vertex data assigned to x, y, & z
#  then cpnverts to float type
########################################
# Assign 2nd row of .txt as x input    #
x_input = data[range(0,numb_vert),1]   #
# Assign 3rd row of .txt as y input    #
y_input = data[range(0,numb_vert),2]   #
# Assign 4th row of .txt as z input    #
z_input = data[range(0,numb_vert),3]   #
# Convert Vertices data to float type  #
x = x_input.astype(float)             #
y = y_input.astype(float)            #
z = z_input.astype(float)           #
####################################
#
#######################
# Assigning Face Data #
#######################
# Face data assigned to fx, fy, & fz
#  then cpnverts to float type
#############################################
# Range count for face data                 #
row_tot = numb_face + numb_vert             #
# Assign 2nd row of .txt as x input         #
fx_input = data[range(numb_vert,row_tot),1] #
# Assign 3rd row of .txt as y input         #
fy_input = data[range(numb_vert,row_tot),2] #
# Assign 4th row of .txt as z input         #
fz_input = data[range(numb_vert,row_tot),3] #
# Convert Vertices data to float type       #
fx = fx_input.astype(str)                   #
fy = fy_input.astype(str)                  #
fz = fz_input.astype(str)                 #
# Define: number of vertices on ith face #
#  - used for .C program                 # 
ith_face = []                            #
for j in range(numb_vert,row_tot):       #
    ith_face.append(3)                   #
ith_array = np.array(ith_face)           #
##########################################
#
##########################
# Creating Output Arrays #
##########################
#    Number of Vertex is (N-1)             
#     numb_vert += 1
# Removed:
#
#########################################
# Number of Vertex set to array         #
numb_vert_array = []                    #
numb_vert_array.append(numb_vert)       #
# Number of Faces set to array          #
numb_face_array = []                    #
numb_face_array.append(numb_face)       #
# Stacking Columns of Vertex Data       #################
Vert_Data_Out_0 = np.column_stack((x, y))               #
Vert_Data_Out   = np.column_stack((Vert_Data_Out_0, z)) #
# Stacking Columns of Face Data                         #
Face_Data_Out_0 = np.column_stack((ith_array, fx))      #
Face_Data_Out_Y = np.column_stack((Face_Data_Out_0,fy)) #
Face_Data_Out   = np.column_stack((Face_Data_Out_Y,fz)) #
#########################################################
#
#########################################
# Write Data File for Mirtich's Program #
#########################################################################
with open(Asteroid_file_in+".in","w") as Poly_Data_file:                #
    np.savetxt(Poly_Data_file,numb_vert_array,fmt='%s',delimiter='\t'); #
    np.savetxt(Poly_Data_file,Vert_Data_Out,fmt='%.8f',delimiter='\t'); #
    np.savetxt(Poly_Data_file,numb_face_array,fmt='%s',delimiter=' ');  #
    np.savetxt(Poly_Data_file,Face_Data_Out,fmt='%s',delimiter=' ');    #
#########################################################################
    