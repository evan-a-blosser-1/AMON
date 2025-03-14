"""
Program: Asteroid_MASCON_Modeler.py
Description: Calculates the Center of mass for selected
                Models:
                        - MASCON I
                        - MASCON III
                        - MASCON VIII
            

MIT License

Copyright (c) [2024] [Evan Blosser]

"""
#############################
import os
import sys
from sys import exit
# Mathmatical!!                     
import numpy as np                  
import pandas as pd
import subprocess
# Plot 
import matplotlib.pyplot as plt
########################################################
#################################### Personal Packages #    
sys.dont_write_bytecode = True
import ASTER_Pack as AP
import Asteroid_Package as ASTER
#################################################################################
#################### Settings 
# Big G in km^3/kg.s^2
G = 6.67410e-20 


# Scaling Settings
rel_tol = 1e-09
abs_tol = 1e-09
Max_Iter = 1000
Gamma_Coeff = 1

# Mu calc settings 
Max_Iter2 = 10000
rtol = 1e-14
atol = 1e-14

asteroid = 'Apophis'

# Apophis 
Vol_Accepted = 0.03034285         # km^3
Mass_Accepted = 5.31e10                    # kg
Density = 1.75e12                 # kg/km^3
###################################
#################################
# OBJ File                        
Asteroid_file = (asteroid)
OBJ_Data = np.loadtxt(Asteroid_file, delimiter=' ', dtype=str) 
#################################################################
################################################### Constants ###
mu_const      = G*Mass_Accepted

###################################### Assign Vertices & Faces ##
Verts_IN, Faces =  ASTER.OBJ_2_VertFace(Asteroid_file)


######################################################################
###################### volInt.c Calculation ##########################
Aster_volInt_File =  asteroid + '.in'   
Asteroid_volInt_PATH_Name = asteroid
ASTER.volInt_Input_FileMake(Asteroid_file, Asteroid_volInt_PATH_Name)
# Define the input file and density value
input_file = Aster_volInt_File
density = str(Density)
# Assuming the compiled C program is named 'volInt'
result = subprocess.run(['./volInt', input_file, density], capture_output=True, text=True)


# Capture the output
output = result.stdout.strip()
output_values = [float(value.strip()) for value in output.split(',')]


#############################################################
######################################### Assign Constants ##                 
# CM Calculated by vloInt.c            
CM= np.array([output_values[0],  
              output_values[1],  
              output_values[2] ])
# Volume from volInt.c
Volume_Mirtich = output_values[3]
# Set Face Data as tetrahedron count
tetra_count=np.shape(Faces)[0]

# Use the output in Python
print(f"Poly CM: {CM}")
############################################################


#%% Scaling 

####################################################################
# Look at decimal function
# x = np.zeros(3, dtpye="float64")
# or np.array definition for more accurate value 
#
#
Del_Gamma = 0.01
for it_1 in range(Max_Iter):
    # Check % and decrease del_gamma?
    #
    Verts = Verts_IN*Gamma_Coeff
    #
    tetra_vol, total_vol = ASTER.Tetra_Volume(Verts, Faces, MASCON_Div=1,total_vol=0)
    ##############################
    Perr_Diff = (total_vol - Vol_Accepted)/Vol_Accepted
    if Perr_Diff < 1.0:
        Del_Gamma = 0.001
    if Perr_Diff < 0.5:
        Del_Gamma = 0.0001
    if Perr_Diff < 0.01:
        Del_Gamma = 0.00001
    if Perr_Diff < 0.001:
        Del_Gamma = 0.000001
    if Perr_Diff < 0.0001:
        Del_Gamma = 0.0000001
    if Perr_Diff < 0.00001:
        Del_Gamma = 0.00000001
    if Perr_Diff < 0.000001:
        Del_Gamma = 0.000000001
    if Perr_Diff < 0.0000001:
        Del_Gamma = 0.0000000001
    ##############################
    if total_vol > Vol_Accepted:
        Gamma_Coeff -= Del_Gamma
        # print(f"Gamma Coeff: {Gamma_Coeff}")
    ##############################
    elif total_vol < Vol_Accepted:
        Gamma_Coeff += Del_Gamma
        # print(f"Gamma Coeff: {Gamma_Coeff}")
    ##############################
    if np.isclose(total_vol,Vol_Accepted, rtol=rel_tol, atol=abs_tol):
        Converge_Message = f""" 
|  Accepted Volume: {Vol_Accepted}
| 
| Calculated Volume: {total_vol}
|
| Gamma Coefficient: {Gamma_Coeff}
|
| Converged at Iteration: {it_1}        
        """
        print(Converge_Message)    
        break
    ##############################
else:
    Out_Message = f""" 
|  Accepted Volume: {Vol_Accepted}
| 
| Calculated Volume: {total_vol}
|
| Gamma Coefficient: {Gamma_Coeff}
|
| Max Iterations Reached: {it_1 +1}        
    """
    print(Out_Message) 
#############################################



#%% Volume Function Call
#
# Note: All tetra and Prism numbers are from outer layer 
#       to inner from 1 to number 'n' respectively  
#   
#       i.e. 1 is outer, 2 is second layer, 3 is inner tetra in MIII
####################################################################
########################################################### MASCON I
Full_Volume_Tetra_Array,Total_Volume_Out = ASTER.Tetra_Volume(Verts, Faces,MASCON_Div=1,total_vol=0)
####################################################################
######################################################### MASCON III
############# Tetrahedron Volumes ###############
Volume_T_I   = ASTER.Tetra_Volume(Verts, Faces,MASCON_Div=1,total_vol=1)
Volume_M3_T_II  = ASTER.Tetra_Volume(Verts, Faces,MASCON_Div=(2/3),total_vol=1)
Volume_M3_T_III = ASTER.Tetra_Volume(Verts, Faces,MASCON_Div=(1/3),total_vol=1)
############## Prism Volumes ####################
Vol_M3_Prism_I  =  Volume_T_I  - Volume_M3_T_II
Vol_M3_Prism_II =  Volume_M3_T_II - Volume_M3_T_III
# Check
M1_Check = np.sum(Volume_T_I)
M3_Check = np.sum(Volume_M3_T_III) + np.sum(Vol_M3_Prism_I) + np.sum(Vol_M3_Prism_II)
####################################################################
######################################################## MASCON VIII
Volume_T_II   = ASTER.Tetra_Volume(Verts, Faces, MASCON_Div=(7/8),total_vol=1)
Volume_T_III  = ASTER.Tetra_Volume(Verts, Faces, MASCON_Div=(6/8),total_vol=1)
Volume_T_IV   = ASTER.Tetra_Volume(Verts, Faces, MASCON_Div=(5/8),total_vol=1)
Volume_T_V    = ASTER.Tetra_Volume(Verts, Faces, MASCON_Div=(4/8),total_vol=1)
Volume_T_VI   = ASTER.Tetra_Volume(Verts, Faces, MASCON_Div=(3/8),total_vol=1)
Volume_T_VII  = ASTER.Tetra_Volume(Verts, Faces, MASCON_Div=(2/8),total_vol=1)
Volume_T_VIII = ASTER.Tetra_Volume(Verts, Faces, MASCON_Div=(1/8),total_vol=1)
#################################################
######################### Prism Volumes #########
Vol_Prism_VII =  Volume_T_VII  - Volume_T_VIII
Vol_Prism_VI  =  Volume_T_VI - Volume_T_VII 
Vol_Prism_V   =  Volume_T_V  - Volume_T_VI
Vol_Prism_IV  =  Volume_T_IV - Volume_T_V 
Vol_Prism_III =  Volume_T_III  - Volume_T_IV
Vol_Prism_II  =  Volume_T_II - Volume_T_III 
Vol_Prism_I   =  Volume_T_I  - Volume_T_II
# Check
M8_Check = np.sum(Volume_T_VIII) + np.sum(Vol_Prism_VII) +\
            np.sum(Vol_Prism_VI) + np.sum(Vol_Prism_V) +\
            np.sum(Vol_Prism_IV) + np.sum(Vol_Prism_III) +\
            np.sum(Vol_Prism_II) + np.sum(Vol_Prism_I)
############################################################
Volume_Check_Message = f"""
{'-'*42}
| Volume Check
{'-'*42}
|-----SUMS-----
| MASCON I:    {M1_Check}
| MASCON III:  {M3_Check}
| MASCON VIII: {M8_Check}
|
| Total Volume: {Total_Volume_Out}
{'-'*42}
"""
print(Volume_Check_Message)





# Debug
    # if tt in [0,1,2,3,4,5]:
    #     print(f'Tetra: {Tetra_U[tt]}')
    #     print(U_vec)
    #     print(V_vec)
    #     print(W_vec)
    #     print(Vol_Full_Sum)
    #     print(A_Matrix)
########
# x_coords_U = [tetra[0] for tetra in Tetra_U]
# y_coords_U = [tetra[1] for tetra in Tetra_U]
# z_coords_U = [tetra[2] for tetra in Tetra_U]
# x_coords_V = [tetra[0] for tetra in Tetra_V]
# y_coords_V = [tetra[1] for tetra in Tetra_V]
# z_coords_V = [tetra[2] for tetra in Tetra_V]
# x_coords_W = [tetra[0] for tetra in Tetra_W]
# y_coords_W = [tetra[1] for tetra in Tetra_W]
# z_coords_W = [tetra[2] for tetra in Tetra_W]
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x_coords_U, y_coords_U, z_coords_U, c='r', marker='o', label='Tetra_U')
# ax.scatter(x_coords_V, y_coords_V, z_coords_V, c='g', marker='^', label='Tetra_V')
# ax.scatter(x_coords_W, y_coords_W, z_coords_W, c='b', marker='s', label='Tetra_W')
# plt.show()
######################################################################################

###############################################
#%% Grav Parameters 

####################################################################
####### Calculate Each Grav. Param. ################################
#
#
Den = Density
Del_Den = 0.01e12
for it_2 in range(Max_Iter2):
    mu_i =  np.zeros(tetra_count, dtype="float64")
    Mass =  np.zeros(tetra_count, dtype="float64")
    mu_MIII = np.zeros((tetra_count, 3), dtype="float64")
    mu_MVIII = np.zeros((tetra_count, 8), dtype="float64")
    for i in range(tetra_count):
        ####################################################################
        ########################################################### MASCON I
        Mass[i] =     tetra_vol[i] * Den 
        mu_i[i] = G * tetra_vol[i] * Den
        ####################################################################
        ######################################################### MASCON III
        mu_MIII[i,0]  = G * Volume_M3_T_III[i] * Den
        mu_MIII[i,1]  = G * Vol_M3_Prism_II[i] * Den
        mu_MIII[i,2]  = G * Vol_M3_Prism_I[i]  * Den
        ####################################################################
        ######################################################## MASCON VIII
        mu_MVIII[i,0] = G * Volume_T_VIII[i] * Den
        mu_MVIII[i,1] = G * Vol_Prism_VII[i] * Den
        mu_MVIII[i,2] = G * Vol_Prism_VI[i]  * Den
        mu_MVIII[i,3] = G * Vol_Prism_V[i]   * Den
        mu_MVIII[i,4] = G * Vol_Prism_IV[i]  * Den
        mu_MVIII[i,5] = G * Vol_Prism_III[i] * Den
        mu_MVIII[i,6] = G * Vol_Prism_II[i]  * Den
        mu_MVIII[i,7] = G * Vol_Prism_I[i]   * Den
    ###############################################
    ######################################### Check
    mu = np.sum(mu_i)
    mu_3 = np.sum(mu_MIII)
    mu_8 = np.sum(mu_MVIII)    
    Mass_Check = np.sum(Mass)
    ##############################
    Perr_Diff_Mu = (mu - mu_const)/mu_const
    if Perr_Diff_Mu < 1.0:
        Del_Den = 0.001e12
    if Perr_Diff_Mu < 0.5:
        Del_Den = 0.0001e12
    if Perr_Diff_Mu < 0.01:
        Del_Den = 0.00001e12
    if Perr_Diff_Mu < 0.001:
        Del_Den = 0.000001e12
    if Perr_Diff_Mu < 0.0001:
        Del_Den = 0.0000001e12
    if Perr_Diff_Mu < 0.00001:
        Del_Den = 0.00000001e12
    if Perr_Diff_Mu < 0.000001:
        Del_Den = 0.000000001e12
    ##############################
    if mu > mu_const:
        Den -= Del_Den
        # print(f"Current Density: {Den}")
    ##############################
    elif mu < mu_const:
        Den += Del_Den
        # print(f"Current Density: {Den}")
    ##############################
    if np.isclose(mu, mu_const, rtol=rtol, atol=atol):
        mu_converge_message = f"""
{'-'*42}
|Gravitational Param
| MI:      {mu:.5e}
| MIII:    {mu_3:.5e}
| MVIII:   {mu_8:.5e}
| True:    {mu_const:.5e}
|Accepted Density: {Density:.5e}
|The total Grav Parameter converged
| in: {it_2} iterations
| at:      {Den:.5e}
| Density: {Density:.5e}
{'-'*42}
{'-'*42}
|Accepted Volume: {Vol_Accepted}
|Calculated Volume: {total_vol:.7f}
{'-'*42}
|Accepted Mass: {Mass_Accepted:.5e}
|Calculated Mass: {Mass_Check:.5e}
{'-'*42}
        """
        print(mu_converge_message)
        break
else:
    Vol_Mass_Den_Out_Message = f"""
{'-'*42}
| MAX Iterations Reached !
{'-'*42}
|Gravitational Param
| MI:      {mu:.5e}
| MIII:    {mu_3:.5e}
| MVIII:   {mu_8:.5e}
| True:    {mu_const:.5e}
|The total Grav Parameter calculated
| at: {Den:.5e}
|Accepted Density: {Density:.5e}
{'-'*42}
{'-'*42}
|Accepted Volume: {Vol_Accepted}
|Calculated Volume: {total_vol:.7f}
{'-'*42}
|Accepted Mass: {Mass_Accepted:.5e}
|Calculated Mass: {Mass_Check:.5e}
{'-'*42}
    """ 
    print(Vol_Mass_Den_Out_Message)

#################################################


#%% Tetra CM Calculation

# Arrays 
####################################################################
########################################################### MASCON I
Output_Array_M1 =  np.zeros((tetra_count, 3), dtype="float64")
####################################################################
######################################################### MASCON III
CM_tetra_L1_M3 = np.zeros((tetra_count, 3), dtype="float64")
CM_tetra_L2_M3 = np.zeros((tetra_count, 3), dtype="float64")
CM_tetra_L3_M3 = np.zeros((tetra_count, 3), dtype="float64")
Output_Array_MIII = np.zeros((tetra_count, 9), dtype="float64")
####################################################################
######################################################## MASCON VIII
CM_tetra_M1 = np.empty((tetra_count,3))
CM_tetra_M2 = np.empty((tetra_count,3))
CM_tetra_M3 = np.empty((tetra_count,3))
CM_tetra_M4 = np.empty((tetra_count,3))
CM_tetra_M5 = np.empty((tetra_count,3))
CM_tetra_M6 = np.empty((tetra_count,3))
CM_tetra_M7 = np.empty((tetra_count,3))
CM_tetra_M8 = np.empty((tetra_count,3))
Output_Array_MVIII = np.zeros((tetra_count, 24), dtype="float64")
#####################################################################
############## Center of Mass Calculations ##########################
for it in range(0,tetra_count):
####################################################################
########################################################### MASCON I
    Center_mass_calc_x = (Verts[Faces[it,0],0] + Verts[Faces[it,1],0] + Verts[Faces[it,2],0] + CM[0])/4
    Center_mass_calc_y = (Verts[Faces[it,0],1] + Verts[Faces[it,1],1] + Verts[Faces[it,2],1] + CM[1])/4
    Center_mass_calc_z = (Verts[Faces[it,0],2] + Verts[Faces[it,1],2] + Verts[Faces[it,2],2] + CM[2])/4
    # Fill array
    Output_Array_M1[it] = (Center_mass_calc_x,Center_mass_calc_y,Center_mass_calc_z)
####################################################################
######################################################### MASCON III
    ############################################################################################### Tetra Center of Masses
    #######################################################################################################################
    ###################################################################################################### Center Layer ###
    CM_M3_calc_x3 = (Verts[Faces[it,0],0]*(1/3) + Verts[Faces[it,1],0]*(1/3) + Verts[Faces[it,2],0]*(1/3) + CM[0])/4
    CM_M3_calc_y3 = (Verts[Faces[it,0],1]*(1/3) + Verts[Faces[it,1],1]*(1/3) + Verts[Faces[it,2],1]*(1/3) + CM[1])/4
    CM_M3_calc_z3 = (Verts[Faces[it,0],2]*(1/3) + Verts[Faces[it,1],2]*(1/3) + Verts[Faces[it,2],2]*(1/3) + CM[2])/4
    # Fill array
    CM_tetra_L3_M3[it] =  (CM_M3_calc_x3,CM_M3_calc_y3,CM_M3_calc_z3)
    ######################################################################################################################
    #################################################################################################### Middle Layer ###
    # Tetrahedron CM
    CM_M3_Tetra_x2 = (Verts[Faces[it,0],0]*(2/3) + Verts[Faces[it,1],0]*(2/3) + Verts[Faces[it,2],0]*(2/3) + CM[0])/4
    CM_M3_Tetra_y2 = (Verts[Faces[it,0],1]*(2/3) + Verts[Faces[it,1],1]*(2/3) + Verts[Faces[it,2],1]*(2/3) + CM[1])/4
    CM_M3_Tetra_z2 = (Verts[Faces[it,0],2]*(2/3) + Verts[Faces[it,1],2]*(2/3) + Verts[Faces[it,2],2]*(2/3) + CM[2])/4
    # Fill array
    CM_tetra_L2_M3[it] =  (CM_M3_Tetra_x2,CM_M3_Tetra_y2,CM_M3_Tetra_z2)
    #####################################################################################################################
    ##################################################################################################### Outer Layer ###
    # Tetrahedron CM
    CM_M3_Tetra_x1 = (Verts[Faces[it,0],0] + Verts[Faces[it,1],0] + Verts[Faces[it,2],0] + CM[0])/4
    CM_M3_Tetra_y1 = (Verts[Faces[it,0],1] + Verts[Faces[it,1],1] + Verts[Faces[it,2],1] + CM[1])/4
    CM_M3_Tetra_z1 = (Verts[Faces[it,0],2] + Verts[Faces[it,1],2] + Verts[Faces[it,2],2] + CM[2])/4
    # Fill array
    CM_tetra_L1_M3[it] =  (CM_M3_Tetra_x1, CM_M3_Tetra_y1,CM_M3_Tetra_z1)
    ############################################################################################### Prism Center of Masses
    ######################################################################################################################
    #################################################################################################### Middle Layer ###
    # Prism CM
    CM_M3_Prism2_x2 = ( CM_tetra_L2_M3[it,0]*Volume_M3_T_II[it] - CM_tetra_L3_M3[it,0]*Volume_M3_T_III[it] )/Vol_M3_Prism_II[it]
    CM_M3_Prism2_y2 = ( CM_tetra_L2_M3[it,1]*Volume_M3_T_II[it] - CM_tetra_L3_M3[it,1]*Volume_M3_T_III[it] )/Vol_M3_Prism_II[it]
    CM_M3_Prism2_z2 = ( CM_tetra_L2_M3[it,2]*Volume_M3_T_II[it] - CM_tetra_L3_M3[it,2]*Volume_M3_T_III[it] )/Vol_M3_Prism_II[it]
    #####################################################################################################################
    ##################################################################################################### Outer Layer ###
    # Prism CM
    CM_M3_Prism1_x1 = ( CM_tetra_L1_M3[it,0]*Volume_T_I[it] - CM_tetra_L2_M3[it,0]*Volume_M3_T_II[it] )/Vol_M3_Prism_I[it]
    CM_M3_Prism1_y1 = ( CM_tetra_L1_M3[it,1]*Volume_T_I[it] - CM_tetra_L2_M3[it,1]*Volume_M3_T_II[it] )/Vol_M3_Prism_I[it]
    CM_M3_Prism1_z1 = ( CM_tetra_L1_M3[it,2]*Volume_T_I[it] - CM_tetra_L2_M3[it,2]*Volume_M3_T_II[it] )/Vol_M3_Prism_I[it]
    #################################################
    ### Fill Data Array 
    Output_Array_MIII[it] = (CM_M3_calc_x3,CM_M3_calc_y3,CM_M3_calc_z3,
                             CM_M3_Prism2_x2,CM_M3_Prism2_y2,CM_M3_Prism2_z2,
                             CM_M3_Prism1_x1,CM_M3_Prism1_y1,CM_M3_Prism1_z1) 
#############################################################################################################################
######################################################## MASCON VIII ########################################################
    ###############################################################################################################
    # Tetrahedron VIII ############################################################################################
    CM_M8_Tetra_x = (Verts[Faces[it,0],0]*(1/8) + Verts[Faces[it,1],0]*(1/8) + Verts[Faces[it,2],0]*(1/8) + CM[0])/4
    CM_M8_Tetra_y = (Verts[Faces[it,0],1]*(1/8) + Verts[Faces[it,1],1]*(1/8) + Verts[Faces[it,2],1]*(1/8) + CM[1])/4
    CM_M8_Tetra_z = (Verts[Faces[it,0],2]*(1/8) + Verts[Faces[it,1],2]*(1/8) + Verts[Faces[it,2],2]*(1/8) + CM[2])/4
    # Fill array
    CM_tetra_M8[it] =  (CM_M8_Tetra_x,CM_M8_Tetra_y,CM_M8_Tetra_z)
    ###############################################################################################################
    # Tetrahedron VII #############################################################################################
    CM_Tetra_x7 = (Verts[Faces[it,0],0]*(2/8) + Verts[Faces[it,1],0]*(2/8) + Verts[Faces[it,2],0]*(2/8) + CM[0])/4
    CM_Tetra_y7 = (Verts[Faces[it,0],1]*(2/8) + Verts[Faces[it,1],1]*(2/8) + Verts[Faces[it,2],1]*(2/8) + CM[1])/4
    CM_Tetra_z7 = (Verts[Faces[it,0],2]*(2/8) + Verts[Faces[it,1],2]*(2/8) + Verts[Faces[it,2],2]*(2/8) + CM[2])/4
    # Fill array
    CM_tetra_M7[it] =  (CM_Tetra_x7,CM_Tetra_y7,CM_Tetra_z7)
    ###############################################################################################################
    # Tetrahedron VI  #############################################################################################
    CM_Tetra_x6 = (Verts[Faces[it,0],0]*(3/8) + Verts[Faces[it,1],0]*(3/8) + Verts[Faces[it,2],0]*(3/8) + CM[0])/4
    CM_Tetra_y6 = (Verts[Faces[it,0],1]*(3/8) + Verts[Faces[it,1],1]*(3/8) + Verts[Faces[it,2],1]*(3/8) + CM[1])/4
    CM_Tetra_z6 = (Verts[Faces[it,0],2]*(3/8) + Verts[Faces[it,1],2]*(3/8) + Verts[Faces[it,2],2]*(3/8) + CM[2])/4
    # Fill array
    CM_tetra_M6[it] =  (CM_Tetra_x6,CM_Tetra_y6,CM_Tetra_z6)
    ###############################################################################################################
    # Tetrahedron V  ##############################################################################################
    CM_Tetra_x5 = (Verts[Faces[it,0],0]*(4/8) + Verts[Faces[it,1],0]*(4/8) + Verts[Faces[it,2],0]*(4/8) + CM[0])/4
    CM_Tetra_y5 = (Verts[Faces[it,0],1]*(4/8) + Verts[Faces[it,1],1]*(4/8) + Verts[Faces[it,2],1]*(4/8) + CM[1])/4
    CM_Tetra_z5 = (Verts[Faces[it,0],2]*(4/8) + Verts[Faces[it,1],2]*(4/8) + Verts[Faces[it,2],2]*(4/8) + CM[2])/4
    # Fill array
    CM_tetra_M5[it] =  (CM_Tetra_x5,CM_Tetra_y5,CM_Tetra_z5)
    ###############################################################################################################
    # Tetrahedron IV  #############################################################################################
    CM_Tetra_x4 = (Verts[Faces[it,0],0]*(5/8) + Verts[Faces[it,1],0]*(5/8) + Verts[Faces[it,2],0]*(5/8) + CM[0])/4
    CM_Tetra_y4 = (Verts[Faces[it,0],1]*(5/8) + Verts[Faces[it,1],1]*(5/8) + Verts[Faces[it,2],1]*(5/8) + CM[1])/4
    CM_Tetra_z4 = (Verts[Faces[it,0],2]*(5/8) + Verts[Faces[it,1],2]*(5/8) + Verts[Faces[it,2],2]*(5/8) + CM[2])/4
    # Fill array
    CM_tetra_M4[it] =  (CM_Tetra_x4,CM_Tetra_y4,CM_Tetra_z4)
    ###############################################################################################################
    # Tetrahedron III #############################################################################################
    CM_Tetra_x3 = (Verts[Faces[it,0],0]*(6/8) + Verts[Faces[it,1],0]*(6/8) + Verts[Faces[it,2],0]*(6/8) + CM[0])/4
    CM_Tetra_y3 = (Verts[Faces[it,0],1]*(6/8) + Verts[Faces[it,1],1]*(6/8) + Verts[Faces[it,2],1]*(6/8) + CM[1])/4
    CM_Tetra_z3 = (Verts[Faces[it,0],2]*(6/8) + Verts[Faces[it,1],2]*(6/8) + Verts[Faces[it,2],2]*(6/8) + CM[2])/4
    # Fill array
    CM_tetra_M3[it] =  (CM_Tetra_x3,CM_Tetra_y3,CM_Tetra_z3)
    ###############################################################################################################
    # Tetrahedron it ##############################################################################################
    CM_Tetra_x2 = (Verts[Faces[it,0],0]*(7/8) + Verts[Faces[it,1],0]*(7/8) + Verts[Faces[it,2],0]*(7/8) + CM[0])/4
    CM_Tetra_y2 = (Verts[Faces[it,0],1]*(7/8) + Verts[Faces[it,1],1]*(7/8) + Verts[Faces[it,2],1]*(7/8) + CM[1])/4
    CM_Tetra_z2 = (Verts[Faces[it,0],2]*(7/8) + Verts[Faces[it,1],2]*(7/8) + Verts[Faces[it,2],2]*(7/8) + CM[2])/4
    # Fill array
    CM_tetra_M2[it] =  (CM_Tetra_x2,CM_Tetra_y2,CM_Tetra_z2)
    #############################################################################################
    # Total Tetrahedron #########################################################################
    CM_Tetra_x1 = (Verts[Faces[it,0],0] + Verts[Faces[it,1],0] + Verts[Faces[it,2],0] + CM[0])/4
    CM_Tetra_y1 = (Verts[Faces[it,0],1] + Verts[Faces[it,1],1] + Verts[Faces[it,2],1] + CM[1])/4
    CM_Tetra_z1 = (Verts[Faces[it,0],2] + Verts[Faces[it,1],2] + Verts[Faces[it,2],2] + CM[2])/4
    # Fill array
    CM_tetra_M1[it] =  (CM_Tetra_x1,CM_Tetra_y1,CM_Tetra_z1)
    #########################################################################################
    ################################# Prism Center of Masses ################################
    ######################################################################################################################
    # Prism CM VII Layer #################################################################################################
    CM_calc_x7 = ( CM_tetra_M7[it,0]*Volume_T_VII[it] - CM_tetra_M8[it,0]*Volume_T_VIII[it] )/Vol_Prism_VII[it]
    CM_calc_y7 = ( CM_tetra_M7[it,1]*Volume_T_VII[it] - CM_tetra_M8[it,1]*Volume_T_VIII[it] )/Vol_Prism_VII[it]
    CM_calc_z7 = ( CM_tetra_M7[it,2]*Volume_T_VII[it] - CM_tetra_M8[it,2]*Volume_T_VIII[it] )/Vol_Prism_VII[it]
    ######################################################################################################################
    # Prism CM VI Layer #################################################################################################
    CM_calc_x6 = ( CM_tetra_M6[it,0]*Volume_T_VI[it] - CM_tetra_M7[it,0]*Volume_T_VII[it] )/Vol_Prism_VI[it]
    CM_calc_y6 = ( CM_tetra_M6[it,1]*Volume_T_VI[it] - CM_tetra_M7[it,1]*Volume_T_VII[it] )/Vol_Prism_VI[it]
    CM_calc_z6 = ( CM_tetra_M6[it,2]*Volume_T_VI[it] - CM_tetra_M7[it,2]*Volume_T_VII[it] )/Vol_Prism_VI[it]
    ####################################################################################################################
    # Prism CM V Layer #################################################################################################
    CM_calc_x5 = ( CM_tetra_M5[it,0]*Volume_T_V[it] - CM_tetra_M6[it,0]*Volume_T_VI[it] )/Vol_Prism_V[it]
    CM_calc_y5 = ( CM_tetra_M5[it,1]*Volume_T_V[it] - CM_tetra_M6[it,1]*Volume_T_VI[it] )/Vol_Prism_V[it]
    CM_calc_z5 = ( CM_tetra_M5[it,2]*Volume_T_V[it] - CM_tetra_M6[it,2]*Volume_T_VI[it] )/Vol_Prism_V[it] 
    #####################################################################################################################
    # Prism CM IV Layer #################################################################################################
    CM_calc_x4 = ( CM_tetra_M4[it,0]*Volume_T_IV[it] - CM_tetra_M5[it,0]*Volume_T_V[it] )/Vol_Prism_IV[it]
    CM_calc_y4 = ( CM_tetra_M4[it,1]*Volume_T_IV[it] - CM_tetra_M5[it,1]*Volume_T_V[it] )/Vol_Prism_IV[it]
    CM_calc_z4 = ( CM_tetra_M4[it,2]*Volume_T_IV[it] - CM_tetra_M5[it,2]*Volume_T_V[it] )/Vol_Prism_IV[it]  
    ######################################################################################################################
    # Prism CM III Layer #################################################################################################
    CM_calc_x3 = ( CM_tetra_M3[it,0]*Volume_T_III[it] - CM_tetra_M4[it,0]*Volume_T_IV[it] )/Vol_Prism_III[it]
    CM_calc_y3 = ( CM_tetra_M3[it,1]*Volume_T_III[it] - CM_tetra_M4[it,1]*Volume_T_IV[it] )/Vol_Prism_III[it]
    CM_calc_z3 = ( CM_tetra_M3[it,2]*Volume_T_III[it] - CM_tetra_M4[it,2]*Volume_T_IV[it] )/Vol_Prism_III[it]   
    ######################################################################################################################
    # Prism CM it Layer  #################################################################################################
    CM_calc_x2 = ( CM_tetra_M2[it,0]*Volume_T_II[it] - CM_tetra_M3[it,0]*Volume_T_III[it] )/Vol_Prism_II[it]
    CM_calc_y2 = ( CM_tetra_M2[it,1]*Volume_T_II[it] - CM_tetra_M3[it,1]*Volume_T_III[it] )/Vol_Prism_II[it]
    CM_calc_z2 = ( CM_tetra_M2[it,2]*Volume_T_II[it] - CM_tetra_M3[it,2]*Volume_T_III[it] )/Vol_Prism_II[it]
    #####################################################################################################################
    # Prism CM Top Layer ################################################################################################
    CM_calc_x1 = ( CM_tetra_M1[it,0]*Volume_T_I[it] - CM_tetra_M2[it,0]*Volume_T_II[it] )/Vol_Prism_I[it]
    CM_calc_y1 = ( CM_tetra_M1[it,1]*Volume_T_I[it] - CM_tetra_M2[it,1]*Volume_T_II[it] )/Vol_Prism_I[it]
    CM_calc_z1 = ( CM_tetra_M1[it,2]*Volume_T_I[it] - CM_tetra_M2[it,2]*Volume_T_II[it] )/Vol_Prism_I[it]
################################################# 
#################################################
    Output_Array_MVIII[it] = (CM_M8_Tetra_x,CM_M8_Tetra_y,CM_M8_Tetra_z,
                                CM_calc_x7,CM_calc_y7,CM_calc_z7,
                                CM_calc_x6,CM_calc_y6,CM_calc_z6,
                                CM_calc_x5,CM_calc_y5,CM_calc_z5,
                                CM_calc_x4,CM_calc_y4,CM_calc_z4,
                                CM_calc_x3,CM_calc_y3,CM_calc_z3,
                                CM_calc_x2,CM_calc_y2,CM_calc_z2,
                                CM_calc_x1,CM_calc_y1,CM_calc_z1)  
###################################### fin ####################################
###############################################################################
#%% File Save
####################################################################
########################################################### CM Save
np.savetxt( asteroid +"_CM.in", Output_Array_M1,delimiter=' ')
# np.savetxt( asteroid +"_M3CM.in", Output_Array_MIII,delimiter=' ')
# np.savetxt( asteroid +"_M8CM.in", Output_Array_MVIII,delimiter=' ')

#####################################################################################
######################### Gravitational Parameter Save ##############################
np.savetxt( asteroid +"_mu.in", mu_i,delimiter=' ')
# np.savetxt( asteroid +"_M3mu.in", mu_MIII,delimiter=' ')
# np.savetxt( asteroid +"_M8mu.in", mu_MVIII,delimiter=' ')

####################################################################
######################################################### Volume 
# Vol_DF_Out = pd.DataFrame({'Vol Tetra':Full_Volume_Tetra_Array[:],})
# Vol_DF_Out_M3 = pd.DataFrame({
# 'Vol P I'   :Vol_Prism_I[:],
# 'Vol P II'  :Vol_Prism_II[:],
# 'Vol T III' :Volume_T_III[:] 
# })
# Vol_DF_Out_M8 = pd.DataFrame({
# 'Vol P I'   :Vol_Prism_I[:],
# 'Vol P II'  :Vol_Prism_II[:] ,
# 'Vol P III' :Vol_Prism_III[:] ,
# 'Vol P IV'  :Vol_Prism_IV[:] ,
# 'Vol P V'   :Vol_Prism_V[:] ,
# 'Vol P VI'  :Vol_Prism_VI[:] ,
# 'Vol P VII' :Vol_Prism_VII[:] ,
# 'Vol T VIII':Volume_T_VIII[:] 
# })
################################################################
########################################## Tetra/Prism Volume ##
# Vol_DF_Out.to_csv(Aster_Vol_PATH + asteroid + '_VolM1.csv' ,sep=' ' ,index=False )
# Vol_DF_Out_M3.to_csv(Aster_Vol_PATH_MIII + asteroid + '_VolM3.csv' ,sep=' ' ,index=False )
# Vol_DF_Out_M8.to_csv(Aster_Vol_PATH_MVIII + asteroid + '_VolM8.csv' ,sep=' ' ,index=False )
###############################################################
######## Total Volume Save ###################
# Append and save Constants file
# Data_append = {'Volume Calc': Total_Volume_Out,
#                'Volume Mirt': Volume_Mirtich,
#                'Scaling':     Gamma_Coeff,
#                }
# # Append data frame
# Asteroid_Const = Asteroid_Const.assign(**Data_append)
# print(Asteroid_Const.head())
# # Save data frame
# Asteroid_Const.to_csv(Aster_Const_PATH + asteroid  + "_const.in", sep=' ' ,index=False)
###################################################################
Data_message = f"""
{'-'*42}
| Data ready, See respective directories
{'-'*42}
"""
print(Data_message)
###########################


#%% Plot
########
# Plot #
########
Plot_Prompt_Decision = input('| Would you like to plot these real quick? (Y/N): ')
if Plot_Prompt_Decision in ['Y','y','Yes','yes','YES']:
    Plot_message = f"""
{'-'*42}
| Plotting Center of Masses...
{'-'*42}
"""
    print(Plot_message)
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')
    #
    # FACE MESH
    Mesh_Color = 'black'
    Aster_File_OBJ_Dir = asteroid + '.obj'
    Asteroid_Mesh = ASTER.OBJ_2_Mesh(Aster_File_OBJ_Dir,Gamma_Coeff,Mesh_Color)
    Asteroid_Mesh2 = ASTER.OBJ_2_Mesh(Aster_File_OBJ_Dir,Gamma_Coeff,Mesh_Color)
    Asteroid_Mesh3 = ASTER.OBJ_2_Mesh(Aster_File_OBJ_Dir,Gamma_Coeff,Mesh_Color)
    ax1.add_collection3d(Asteroid_Mesh)
    ax2.add_collection3d(Asteroid_Mesh2)
    ax3.add_collection3d(Asteroid_Mesh3)
    ####################################################################
    ########################################################### MASCON I
    xpM1 = Output_Array_M1[:,0]
    ypM1 = Output_Array_M1[:,1]
    zpM1 = Output_Array_M1[:,2]
    ax1.scatter3D(xpM1,ypM1,zpM1,
                marker='.',
                color='red')
    ####################################################################
    ######################################################### MASCON III
    xpM3  = Output_Array_MIII[:,0]
    ypM3  = Output_Array_MIII[:,1]
    zpM3  = Output_Array_MIII[:,2]
    ax2.scatter3D(xpM3 ,ypM3 ,zpM3 ,
                marker='.',
                s=1,
                color='#0000E6')
    ##########
    # M2
    xp2M3  = Output_Array_MIII[:,3]
    yp2M3  = Output_Array_MIII[:,4]
    zp2M3  = Output_Array_MIII[:,5]
    ax2.scatter3D(xp2M3 ,yp2M3 ,zp2M3 ,
                marker='.',
                s=1,
                color='#E67300')
    ##########
    # M3
    xp3M3  = Output_Array_MIII[:,6]
    yp3M3  = Output_Array_MIII[:,7]
    zp3M3  = Output_Array_MIII[:,8]
    ax2.scatter3D(xp3M3 ,yp3M3 ,zp3M3 ,
                marker='.',
                s=1,
                color='#00E673')
    ####################################################################
    ######################################################## MASCON VIII
    xpM8 = Output_Array_MVIII[:,0]
    ypM8 = Output_Array_MVIII[:,1]
    zpM8 = Output_Array_MVIII[:,2]
    ax3.scatter3D(xpM8,ypM8,zpM8,
                marker='.',
                s=0.9,
                color='#00F5FF')
    ##########
    # M2
    xp2M8 = Output_Array_MVIII[:,3]
    yp2M8 = Output_Array_MVIII[:,4]
    zp2M8 = Output_Array_MVIII[:,5]
    ax3.scatter3D(xp2M8,yp2M8,zp2M8,
                marker='.',
                s=0.9,
                color='#00B0FF')
    ##########
    # M3
    xp3M8 = Output_Array_MVIII[:,6]
    yp3M8 = Output_Array_MVIII[:,7]
    zp3M8 = Output_Array_MVIII[:,8]
    ax3.scatter3D(xp3M8,yp3M8,zp3M8,
                marker='.',
                s=0.9,
                color='#0083FF')
    ##########
    # M4
    xp4M8 = Output_Array_MVIII[:,9]
    yp4M8 = Output_Array_MVIII[:,10]
    zp4M8 = Output_Array_MVIII[:,11]
    ax3.scatter3D(xp4M8,yp4M8,zp4M8,
                marker='.',
                s=0.9,
                color='#0057FF')
    ##########
    # M5
    xp5M8 = Output_Array_MVIII[:,12]
    yp5M8 = Output_Array_MVIII[:,13]
    zp5M8 = Output_Array_MVIII[:,14]
    ax3.scatter3D(xp5M8,yp5M8,zp5M8,
                marker='.',
                s=0.9,
                color='#000BFF')
    ##########
    # M6
    xp6M8 = Output_Array_MVIII[:,15]
    yp6M8 = Output_Array_MVIII[:,16]
    zp6M8 = Output_Array_MVIII[:,17]
    ax3.scatter3D(xp6M8,yp6M8,zp6M8,
                marker='.',
                s=0.9,
                color='#0009CD')
    ##########
    # M7
    xp7M8 = Output_Array_MVIII[:,18]
    yp7M8 = Output_Array_MVIII[:,19]
    zp7M8 = Output_Array_MVIII[:,20]
    ax3.scatter3D(xp7M8,yp7M8,zp7M8,
                marker='.',
                s=0.9,
                color='#0108AF')
    ##########
    # M8
    xp8M8 = Output_Array_MVIII[:,21]
    yp8M8 = Output_Array_MVIII[:,22]
    zp8M8 = Output_Array_MVIII[:,23]
    ax3.scatter3D(xp8M8,yp8M8,zp8M8,
                marker='.',
                s=0.9,
                color='#00068D')
    ####################################################################
    #################################################### Plot Settings #
    # Set Aspect
    ax1.set_aspect('equal', 'box') 
    ax2.set_aspect('equal', 'box') 
    ax3.set_aspect('equal', 'box') 
    # Title
    ax1.set_title("MASCON I",fontsize=14, fontweight='bold',color='Black')                     
    ax2.set_title("MASCON III",fontsize=14, fontweight='bold',color='Black')      
    ax3.set_title("MASCON VIII",fontsize=14, fontweight='bold',color='Black')      
    #
    plt.show()
    ###############################################################
else:
    Exit_Message = f"""
{'-'*42}
| Plot not selected, Exiting...
{'-'*42}
    """
    print(Exit_Message)