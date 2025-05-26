import numpy as np
from icecream import ic
#######################################
File_name = "Sylvia_Convex"
File = File_name + ".obj"
# Read in .obj file & define as data  #
with open(File, 'r') as obj_file: 
    obj_data = obj_file.readlines()                          
#######################################
from icecream import ic
ic(obj_data)
################ Empty Array  
Vert_Data = []        
Face_Data = []        
################ Count Variables
numb_vert = 0
numb_face = 0
####################################
################# Scan Lines in data  
for line in obj_data: 
###################################       
######## For lines starting with v
    if line.startswith('v '):
        ################## Count
        numb_vert += 1
        ################## Strip & Assign     
        _, x, y, z = line.strip().split()
        # Append to array
        Vert_Data.append(f" {x} {y} {z}\n")
####################################
######## For lines starting with f
    if line.startswith('f '):
        ################## Count
        numb_face += 1
        ################## Strip & Assign 
        _, x, y, z = line.strip().split()
        # Append to array
        Face_Data.append(f"3 {x} {y} {z}\n")
##################################################
Vert_Data = np.array([Vert_Data])
Face_Data = np.array([Face_Data])
numb_vert = np.array([numb_vert])
numb_face = np.array([numb_face])

with open(File_name + ".in","w") as Poly_Data_file:                     
    np.savetxt(Poly_Data_file,numb_vert,fmt='%s',delimiter=' '); 
    np.savetxt(Poly_Data_file,Vert_Data,fmt='%s',delimiter=' '); 
    np.savetxt(Poly_Data_file,numb_face,fmt='%s',delimiter=' ');  
    np.savetxt(Poly_Data_file,Face_Data,fmt='%s',delimiter=' ');    
#     ##############################################