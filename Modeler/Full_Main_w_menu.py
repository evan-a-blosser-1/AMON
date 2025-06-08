"""
    Author: Evan A. Blosser                              
    Date:   Thu Sep 28 02:03:18 2023                     
    Email:  evan.a.blosser-1@ou.edu/galactikhan@gmail.com                                                                 
    Program Name: Asteroid Grav. Field Simulation 3.0  
    Description:  
                 A fixed frame orbital simulation for asteroid
                 shape models in Mass Concentration (MASCON).    
"""
#%% Imported Packages & Global Variable Def 
######################
# Operating System & Time
import os
import time  
import sys
from sys import exit
# Mathematical!!                     
import numpy as np                  
import pandas as pd
# Debug Print                      
from icecream import ic                   
# ODE solver                        
from scipy.integrate import odeint  
# Plotting & Animation              
import matplotlib.pyplot as plt                         
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import matplotlib.ticker as ticker
########################################################
#################################### Personal Packages # 
# sys.dont_write_bytecode = True   
# import Asteroid_Package as ASTER 
###########################
################################
# Global Constants Declaration #
################################
# Big G in km^3/kg.s^2
G = 6.67430e-20                                                   
################################
##################
# Path Settings  #
#########################################
######################### Data Loaded ###
# MASCON I
Aster_M1CM_PATH  = 'Asteroid_Database/Asteroid_CM/MASCON1/'
Aster_VolM1_PATH = 'Asteroid_Database/Asteroid_CM/MASCON1/Tetra_Vol/'
# MASCON III
Aster_M3CM_PATH  = 'Asteroid_Database/Asteroid_CM/MASCON3/'
Aster_VolM3_PATH = 'Asteroid_Database/Asteroid_CM/MASCON3/Tetra_Pris_Vol/'
# MASCON VIII
Aster_M8CM_PATH  = 'Asteroid_Database/Asteroid_CM/MASCON8/'
Aster_VolM8_PATH = 'Asteroid_Database/Asteroid_CM/MASCON8/Tetra_Pris_Vol/'
# OBJ & Constant
Aster_OBJ_PATH   = 'Asteroid_Database/OBJ_Files/'
Aster_Const_PATH = 'Asteroid_Database/Asteroid_Constants/' 
# Saving in Databank
Aster_Sim_Data_PATH = 'Databank/RotFrame_Sim_Data/'
########################################
#%% User Menu & Selection Menu
############################
## User Welcome And Input ##
############################
#
###############################
# Print Asteroid in Directory #
###############################
Aster_OBJ_Directory = os.listdir(Aster_OBJ_PATH)
################################################
# for loop to remove file extensions
for ii, File_Name in enumerate(Aster_OBJ_Directory):
    # Take File Extension off
    Aster_OBJ_Directory[ii] = os.path.splitext(File_Name)[0]
# Create Data Frame
Asteroid_List_df = pd.DataFrame({'File Names': Aster_OBJ_Directory})
#####
# Create the formatted output string
Welcome_Message_Out = f"""
{'-'*42}
|{'-'*13} Welcome User {'-'*13}|
{'-'*42}
| Program: ASim_Rot_1.0                  |
|   - simulates orbits around asteroids  |
|     in a Rotating frame for the        |
|     following shape models:            |
|          1) MASCON I                   |
|          2) MASCON III                 |
|          3) MASCON VIII                |
|                                        |
|  The following are the available       |
|    asteroids:                          |
|            - 1950DA Pro & Ret          |
|            = Kleopatra                 |
|                                        |
|         All others need spin!!         |
{'-'*42}
        {Asteroid_List_df}
"""
print(Welcome_Message_Out)
######################################
# Select Asteroid
while True:
    print(f"|{'-'*40}|") 
    ####################################################################
    ############################################# DataFrame Num Input ##
    Aster_File_Number = int(input("| Enter Asteroid's Number: "))
    ###########################################
    ############################### Error Check
    # Df column # -1 for header
    A_Err_Inpt = Asteroid_List_df.shape[0] - 1 
    ##########################################
    ######################### Check If present
    if Aster_File_Number <= A_Err_Inpt and 0 <= Aster_File_Number:
    ### Assign ASteroid
        Aster_File_Name = Asteroid_List_df['File Names'][Aster_File_Number]
    ### EXIT DREADED WHILE LOOP!!!
        break
    else:
        Initial_input_error = f"""
{'-'*42}
|{'-'*13} WARNING!!!!! {'-'*13}|
{'-'*42}
| The following is an 
|  invalid input: {Aster_File_Number}      
{'-'*42}   
        """
        print(Initial_input_error)
        continue
#################
#%% File Path assign
#########################################
# Set File Names
Aster_File_OBJ = Aster_File_Name + '.obj'
Aster_File_Const =  Aster_File_Name + '_const.in'
# Set File Paths
Aster_File_OBJ_Dir = Aster_OBJ_PATH  + Aster_File_OBJ
Aster_File_Const_Dir = Aster_Const_PATH + Aster_File_Const
##########################################################
#
####################################
#  Already Used this with file name
OBJ_Check_Complete_message = f"""
{'-'*42}
| Loading: {Aster_File_OBJ}
{'-'*42}
"""
print(OBJ_Check_Complete_message)
#################################
#%% Load Constants & Check Directory
###################################################
############# Checking Directory ##################
Aster_Const_Directory = os.listdir(Aster_Const_PATH)
######################
# Constant File Check
if Aster_File_Const in Aster_Const_Directory:
    ######################################
    # Load In Asteroid Constants         #
    Asteroid_Const = pd.read_csv(Aster_File_Const_Dir, delimiter=' ')
    # Update User 
    Loading_Const_File_Message = f"""
{'-'*42}
|  Loading: {Aster_File_Const}
{'-'*42}
    """
    print(Loading_Const_File_Message)
    print(Asteroid_Const.head())
else:
    Error_NO_Const_File = f"""
{'-'*42}  
| {Aster_File_Const} 
| Not in: 
| {Aster_File_Const_Dir}
{'-'*42} 
    """
    print(Error_NO_Const_File)
    input('| Press any Enter to exit...')
    exit()
#########################################
#%% Assign Var./Volume & Mass Calculations
####################
# Assign Constants #
#########################################
# Rad/sec ((np.pi*2)/3600)
omega = Asteroid_Const['omega (rot/hr)'][0]*((np.pi*2)/3600)
# Correct asteroid distance units scale
scale = Asteroid_Const['Scaling'][0]
# Asteroid Mean Radius                 
R_body = Asteroid_Const['Mean Radius (km)'][0]                       
# CM Calculated by vloInt.c            
Center_of_mass = np.array([
    Asteroid_Const['Poly CM X'][0] ,  
    Asteroid_Const['Poly CM Y'][0] ,  
    Asteroid_Const['Poly CM Z'][0] ])
# Assign Asteroid Mass 
Asteroid_Accept_Mass = Asteroid_Const['Mass (kg)'][0] 
# Scaled: Volume calculation by volInt.c 
Volume_mirtich = Asteroid_Const['Volume Mirt'][0]*scale  
Volume_calc    = Asteroid_Const['Volume Approx'][0]*scale 
#######################################################
Data_Loaded_output = f"""
{'-'*42}
| The following was calculated
|  for {Aster_File_Name} from 
|  the database files
{'-'*42}
{'-'*42}
| Scaling Factor:
|  Gamma = {scale:.4f}
{'-'*42}
"""
print(Data_Loaded_output)
##############################
Volume_Choice_Prompt = f"""
{'-'*42}
|{'-'*11} Volume Selection {'-'*11}|
{'-'*42}
| The volume calculated by volInt.c
| is:
|       {Volume_mirtich:.8f} km^3
{'-'*42}
| The volume calculated analytically 
| is:
|       {Volume_calc:.8f} km^3
{'-'*42}
| Would you like to use the 
|    analytical volume?
"""
# Prompt user to select volume
print(Volume_Choice_Prompt)
print(f"|{'-'*40}|")
Volume_Choice = input('| Enter (Y/N): ')
# Choice loop
if Volume_Choice in ['Y','y','yes','YES','Yes','YEs','YeS']:
    Volume_Poly = Volume_calc
    Volume_Setting_Message = f"""
{'-'*42}
|{'-'*12} Volume Setting {'-'*12}|
{'-'*42}
| Analytical: {Volume_Poly:.6f} km^3 
{'-'*42}
    """
    print(Volume_Setting_Message)
else:
    Volume_Poly = Volume_mirtich 
    Volume_Setting_Message = f"""
{'-'*42}
|{'-'*12} Volume Setting {'-'*12}|
{'-'*42}
| volInt.c: {Volume_Poly:.6f} km^3
{'-'*42} 
    """
    print(Volume_Setting_Message) 
##############################################################
#
####################
# Mass Calculation #
##############################################################
# Assign Constants Converted to kg/km^3                                          
Density_asteroid = Asteroid_Const['Density (g/cm^3)'][0] *1e12
Density_Uncert =   Asteroid_Const['Density Uncert'][0] *1e12
##############################################################             
Asteroid_Calc_mass = Volume_Poly * Density_asteroid
#############################################################
mass_data_output = f"""
{'-'*42}
|{'-'*11} Mass Calculation {'-'*11}|
{'-'*42}
| The Density is:
|    {Density_asteroid:.3e} +/- {Density_Uncert:.3e} kg/km^3
| The calculated masses is:
|   {Asteroid_Calc_mass:.3e} +/- {Density_Uncert:.3e} kg
{'-'*42}
|  Compared to the Accepted Mass
|   {Asteroid_Accept_Mass:.3e} kg
{'-'*42}
"""
print(mass_data_output)
##################################################
#%% MASCON Choice & CM File Load 
################################
######## MASCON Choice #########
MASCON_Choice_message = f"""
{'-'*42}
|{'-'*11} MASCON Selection {'-'*11}|
{'-'*42}
| Would you like the solution
|  for MACON I, III, or VIII
{'-'*42}
"""
print(MASCON_Choice_message)
print(f"|{'-'*40}|") 
while True:
    MASCON_Choice = input('| Please enter 1, 3, or 8: ')
    if MASCON_Choice == '1':
        # Set MASCON I Path & File Name
        Aster_File_CM = Aster_File_Name + '_CM.dat'
        Aster_File_CM_Dir = Aster_M1CM_PATH + Aster_File_CM
        # Set Tetrahedron division
        Mascon_Div = 1
        # Set Desired Path
        Mascon_choice_PATH = Aster_M1CM_PATH
        break
    elif MASCON_Choice == '3':
        # Set MASCON III Path & File Name
        Aster_File_CM = Aster_File_Name + '_M3CM.dat'
        Aster_File_CM_Dir = Aster_M3CM_PATH + Aster_File_CM
        # Set Tetrahedron division
        Mascon_Div = 3
        # Set Desired Path
        Mascon_choice_PATH = Aster_M3CM_PATH
        break
    elif MASCON_Choice == '8':
        # Set MASCON VIII Path & File Name
        Aster_File_CM = Aster_File_Name + '_M8CM.dat'
        Aster_File_CM_Dir = Aster_M8CM_PATH + Aster_File_CM 
        # Set Tetrahedron division
        Mascon_Div = 8
        # Set Desired Path
        Mascon_choice_PATH = Aster_M8CM_PATH
        break
    elif MASCON_Choice in ['Exit','exit','EXIT']:
        exit() 
        break   
    else:
        Error_MASCON_Choice_Input = f"""
{'-'*42}
| Error!!:
|    '{MASCON_Choice}' Is not a valid
|           input. Please try again.
{'-'*42}
|  Enter: exit
|    to exit the program
{'-'*42}
        """
        print(Error_MASCON_Choice_Input)
        continue
################################################
########### Checking Directory #################
Aster_CM_Directory = os.listdir(Mascon_choice_PATH)
    # Center of Mass File Check
if Aster_File_CM in Aster_CM_Directory:
    ####################
    # READ IN CM File  #
    Initial_Conditions = np.loadtxt(Aster_File_CM_Dir, delimiter=' ',dtype=float)
    #################################
    # Get Tetrahedron/CM Count
    Tetra_Count = np.shape(Initial_Conditions)[0] * Mascon_Div
    Loading_CM_File_Message = f"""
{'-'*42}
|  Loading: {Aster_File_CM}
|
| Reading in: {Tetra_Count}
| (Center of Masses for Tetrahedrons) 
{'-'*42}
    """
    print(Loading_CM_File_Message)
##################################
########### Error ################
else:
    Error_NO_CM_File = f"""
{'-'*42}   
| Error: File NOT FOUND!
| {Aster_File_CM_Dir} 
{'-'*42} 
    """
    print(Error_NO_CM_File)
    input('| Press any Enter to exit...')
    exit()
################
#%% Grav Parameter: MASCON III & VIII Tetra/Prism Vol
###############################################
############################# MASCON I BASIC ##
if MASCON_Choice == '1':
    ###################################################################
    ######################################################### File Load
    Aster_Vol_File = Aster_VolM1_PATH + Aster_File_Name  + '_VolM1.csv'
    Vol_Tetra = pd.read_csv(Aster_Vol_File, delimiter=' ')
    ####### Calculate Each Grav. Param. ################################
    mu_I = []
    for i in range(len(Initial_Conditions)):
        mu_1 = G * Vol_Tetra['Vol Tetra'][i]*scale * Density_asteroid
        ############## Append ##########################################
        mu_I.append(mu_1)
    #######################
    mu = np.sum(mu_I)
###############################################
############################# MASCON III ######
elif MASCON_Choice == '3':
    ###################################################################
    ######################################################### File Load
    Aster_Vol_File = Aster_VolM3_PATH + Aster_File_Name  + '_VolM3.csv'
    Vol_Tetra_Prism_M3 = pd.read_csv(Aster_Vol_File, delimiter=' ')
    ####### Calculate Each Grav. Param. ################################
    mu_I, mu_II, mu_III = [],[],[]
    for i in range(len(Initial_Conditions)):
        mu_1 = G * Vol_Tetra_Prism_M3['Vol P I'][i]*scale * Density_asteroid
        mu_2 = G * Vol_Tetra_Prism_M3['Vol P II'][i]*scale * Density_asteroid
        mu_3 = G * Vol_Tetra_Prism_M3['Vol T III'][i]*scale * Density_asteroid
        ############## Append ##########################################
        mu_I.append(mu_1)
        mu_II.append(mu_2)
        mu_III.append(mu_3)
    #######################
    mu = np.sum(mu_I) + np.sum(mu_II) +np.sum(mu_III) 
#########################################################
###################################### MASCON VIII ######
elif MASCON_Choice == '8':
    ###################################################################
    ######################################################### File Load
    Aster_Vol_File = Aster_VolM8_PATH + Aster_File_Name  + '_VolM8.csv'
    Vol_Tetra_Prism_M8 = pd.read_csv(Aster_Vol_File, delimiter=' ')
    ####### Calculate Each Grav. Param. ################################
    mu_I, mu_II, mu_III, mu_IV, mu_V, mu_VI,mu_VII, mu_VIII  = [],[],[],[],[],[],[],[]
    for i in range(len(Initial_Conditions)):
        mu_1 = G * Vol_Tetra_Prism_M8['Vol P I'][i]*scale * Density_asteroid
        mu_2 = G * Vol_Tetra_Prism_M8['Vol P II'][i]*scale * Density_asteroid
        mu_3 = G * Vol_Tetra_Prism_M8['Vol P III'][i]*scale * Density_asteroid
        mu_4 = G * Vol_Tetra_Prism_M8['Vol P IV'][i]*scale * Density_asteroid
        mu_5 = G * Vol_Tetra_Prism_M8['Vol P V'][i]*scale * Density_asteroid
        mu_6 = G * Vol_Tetra_Prism_M8['Vol P VI'][i]*scale * Density_asteroid
        mu_7 = G * Vol_Tetra_Prism_M8['Vol P VII'][i]*scale * Density_asteroid
        mu_8 = G * Vol_Tetra_Prism_M8['Vol T VIII'][i]*scale * Density_asteroid      
        ############## Append ##########################################
        mu_I.append(mu_1)
        mu_II.append(mu_2)
        mu_III.append(mu_3)
        mu_IV.append(mu_4)
        mu_V.append(mu_5)
        mu_VI.append(mu_6)
        mu_VII.append(mu_7)
        mu_VIII.append(mu_8)
    #######################
    mu = np.sum(mu_I) + np.sum(mu_II) +np.sum(mu_III) \
        + np.sum(mu_IV) + np.sum(mu_V) +np.sum(mu_VI) \
        + np.sum(mu_VII) + np.sum(mu_VIII) 
#########################################################
#########################################################
#%% Orbit Parameter Input
################
# Orbit Inputs #
######################
## Debug Inputs
# r_p_a = 0.35
# e     = 0.05
# i_in  = 0.001
# w_in  = 0.0
# nu_in = 0
# Omega_in = 0
# t_inp = 1     
# # 3600 for hours  
# Graph_time = 3600 
######################
Loading_Sim_Message_Out = f"""
{'-'*42}
|{'-'*5} Data is ready, loading Sim...{'-'*5}|
{'-'*42}
| Please Input the desired parameters    |
| for this simulation:                   |
{'-'*42}
"""
print(Loading_Sim_Message_Out)
#################
## User Inputs ##
#####################################################
# COE
print(f"{'-'*42}")  
print(f"|{'-'*10} State Vector Input {'-'*10}|")
State_X  = float(input('| Enter the x-component: '))             
State_Y  = float(input('| Enter the y-component: ')) 
State_Z  = float(input('| Enter the z-component: '))                 
print(f"{'-'*42}")  
# Prompt Time input
print(f"|{'-'*10} Time Unit & Length {'-'*10}|")
# Time input is WORKING!!!                                         
Graph_input = input('| Enter H -hours or D -days: ')              
t_inp      = int(input('| Enter desired Sim. length: '))   
print(f"{'-'*42}")         
####################################################################
#
###############
# Time inputs #
###############
##############################################
# Setting for Hours                          #
if Graph_input == 'h' or Graph_input == 'H': #
# In seconds (60*60) from hours              #
    Graph_time = 3600                        #
    Graph_Time_Unit = 'Hours'                #
# Days                                        #
elif Graph_input == 'd' or Graph_input == 'D': # 
# In seconds (60*60*24) from days              #
    Graph_time = 86400                         #
    Graph_Time_Unit = 'Days'                   #                      
else:                                          ###############
    exit('ERROR: input for time unit is incorrect, exit...') #
##############################################################
##########################################
# Time step, t_inp times the chosen unit #
# Sets n & T span to seconds             #
n     = t_inp*Graph_time                 #
# Graph points, set to the chosen unit   # 
points = t_inp*Graph_time                #
##########################################
##############################################################
# Time and Points                                            #
t_span  = np.linspace(0,n,points)                            #
##############################################################
#%% MASCON I ODE Definition
#####################
## ODE Calculation ##
#####################
# Cpu Clock         
ODE_Calc_Start_Time = time.time() 
#####################
# Define ODE System #
#####################
# Define ODE System #
def Orbit_M1(a,t):
    ##########################################
    rx  = a[0] # Define Position Vector
    ry  = a[1] #   x, y, z component
    rz  = a[2] #
    vx  = a[3] # Define Velocity Vector
    vy  = a[4] #   x, y, z component
    vz  = a[5] #
    ############
    ######################
    # Asteroid Potential #############
    U_x_sum,U_y_sum,U_z_sum = [],[],[]
    ###########################################
    # For Each Tetrahedron CM
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,0] - Center_of_mass[0]
        R_y = Initial_Conditions[i,1] - Center_of_mass[1]
        R_z = Initial_Conditions[i,2] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        # Magnitude
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        # Potential in the x,y,z-direction
        U_x = - (mu_I[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_I[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_I[i]*r_tetra_z)/(r_tetra**3)
        # Store value for each Tetra
        U_x_sum.append(U_x)
        U_y_sum.append(U_y)
        U_z_sum.append(U_z)
    ####EXIT LOOP ############
    # Summation  
    U_pot_x = np.sum(U_x_sum) 
    U_pot_y = np.sum(U_y_sum) 
    U_pot_z = np.sum(U_z_sum) 
    ##################################################
    # Derivatives/ 6 Equations of Motion #############
    drxdt = vx 
    drydt = vy 
    drzdt = vz 
    dvxdt = (omega**2)*rx + 2*omega*vy + U_pot_x
    dvydt = (omega**2)*ry - 2*omega*vx + U_pot_y
    dvzdt = U_pot_z
    dadt  = [drxdt,drydt,drzdt,dvxdt,dvydt,dvzdt]
    return dadt
# End ODE Definition ###################################                                                    
######################
#%% MASCON III ODE Definition
#####################
# Define ODE System #
#####################
# Define ODE System #
def Orbit_M3(a,t):
    ##########################################
    rx  = a[0] # Define Position Vector
    ry  = a[1] #   x, y, z component
    rz  = a[2] #
    vx  = a[3] # Define Velocity Vector
    vy  = a[4] #   x, y, z component
    vz  = a[5] #
    ############
    ######################
    # Asteroid Potential #
    ######################
    # STARTING FROM LAYER I - OUTER  LAYER
    ##############################################
    ######### MASCON III - Layer I ###############
    U_x_sum_1,U_y_sum_1,U_z_sum_1 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,0] - Center_of_mass[0]
        R_y = Initial_Conditions[i,1] - Center_of_mass[1]
        R_z = Initial_Conditions[i,2] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_I[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_I[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_I[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_1.append(U_x)
        U_y_sum_1.append(U_y)
        U_z_sum_1.append(U_z)
    ####  
    U_x_1 = np.sum(U_x_sum_1) 
    U_y_1 = np.sum(U_y_sum_1) 
    U_z_1 = np.sum(U_z_sum_1) 
    ##############################################
    ######### MASCON III - Layer II ##############
    U_x_sum_2,U_y_sum_2,U_z_sum_2 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,3] - Center_of_mass[0]
        R_y = Initial_Conditions[i,4] - Center_of_mass[1]
        R_z = Initial_Conditions[i,5] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_II[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_II[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_II[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_2.append(U_x)
        U_y_sum_2.append(U_y)
        U_z_sum_2.append(U_z)
    ####  
    U_x_2 = np.sum(U_x_sum_2) 
    U_y_2 = np.sum(U_y_sum_2) 
    U_z_2 = np.sum(U_z_sum_2) 
    ##############################################
    ######### MASCON III - Layer III #############
    U_x_sum_3,U_y_sum_3,U_z_sum_3 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,6] - Center_of_mass[0]
        R_y = Initial_Conditions[i,7] - Center_of_mass[1]
        R_z = Initial_Conditions[i,8] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_III[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_III[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_III[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_3.append(U_x)
        U_y_sum_3.append(U_y)
        U_z_sum_3.append(U_z)
    ####  
    U_x_3 = np.sum(U_x_sum_3) 
    U_y_3 = np.sum(U_y_sum_3) 
    U_z_3 = np.sum(U_z_sum_3) 
    #############################
    # Sum Each Layer ############
    U_x = U_x_1 + U_x_2 + U_x_3
    U_y = U_y_1 + U_y_2 + U_y_3
    U_z = U_z_1 + U_z_2 + U_z_3
    ##################################################
    # Derivatives/ 6 Equations of Motion #############
    drxdt = vx 
    drydt = vy 
    drzdt = vz 
    dvxdt = omega**2 *rx + 2*omega*vy + U_x
    dvydt = omega**2 *ry - 2*omega*vx + U_y
    dvzdt = U_z
    dadt  = [drxdt,drydt,drzdt,dvxdt,dvydt,dvzdt]
    return dadt
# End ODE Definition ###################################                                                    
######################
#%% MASCON VIII ODE Definition
#####################
# Define ODE System #
#####################
# Define ODE System #
def Orbit_M8(a,t):
    ##########################################
    rx  = a[0] # Define Position Vector
    ry  = a[1] #   x, y, z component
    rz  = a[2] #
    vx  = a[3] # Define Velocity Vector
    vy  = a[4] #   x, y, z component
    vz  = a[5] #
    ############
    ######################
    # Asteroid Potential #
    ######################
    # STARTING FROM LAYER I - OUTER  LAYER
    ###############################################
    ######### MASCON VIII - Layer I ###############
    U_x_sum_1,U_y_sum_1,U_z_sum_1 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,0] - Center_of_mass[0]
        R_y = Initial_Conditions[i,1] - Center_of_mass[1]
        R_z = Initial_Conditions[i,2] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_I[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_I[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_I[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_1.append(U_x)
        U_y_sum_1.append(U_y)
        U_z_sum_1.append(U_z)
    ####  
    U_x_1 = np.sum(U_x_sum_1) 
    U_y_1 = np.sum(U_y_sum_1) 
    U_z_1 = np.sum(U_z_sum_1) 
    ###############################################
    ######### MASCON VIII - Layer II ##############
    U_x_sum_2,U_y_sum_2,U_z_sum_2 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,3] - Center_of_mass[0]
        R_y = Initial_Conditions[i,4] - Center_of_mass[1]
        R_z = Initial_Conditions[i,5] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_II[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_II[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_II[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_2.append(U_x)
        U_y_sum_2.append(U_y)
        U_z_sum_2.append(U_z)
    ####  
    U_x_2 = np.sum(U_x_sum_2) 
    U_y_2 = np.sum(U_y_sum_2) 
    U_z_2 = np.sum(U_z_sum_2) 
    ###############################################
    ######### MASCON VIII - Layer III #############
    U_x_sum_3,U_y_sum_3,U_z_sum_3 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,6] - Center_of_mass[0]
        R_y = Initial_Conditions[i,7] - Center_of_mass[1]
        R_z = Initial_Conditions[i,8] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_III[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_III[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_III[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_3.append(U_x)
        U_y_sum_3.append(U_y)
        U_z_sum_3.append(U_z)
    ####  
    U_x_3 = np.sum(U_x_sum_3) 
    U_y_3 = np.sum(U_y_sum_3) 
    U_z_3 = np.sum(U_z_sum_3) 
    #############################################
    ######### MASCON VIII - Layer IV ###############
    U_x_sum_4,U_y_sum_4,U_z_sum_4 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,9] - Center_of_mass[0]
        R_y = Initial_Conditions[i,10] - Center_of_mass[1]
        R_z = Initial_Conditions[i,11] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_IV[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_IV[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_IV[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_4.append(U_x)
        U_y_sum_4.append(U_y)
        U_z_sum_4.append(U_z)
    ####  
    U_x_4 = np.sum(U_x_sum_4) 
    U_y_4 = np.sum(U_y_sum_4) 
    U_z_4 = np.sum(U_z_sum_4) 
    ##############################################
    ######### MASCON VIII - Layer V ###############
    U_x_sum_5,U_y_sum_5,U_z_sum_5 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,12] - Center_of_mass[0]
        R_y = Initial_Conditions[i,13] - Center_of_mass[1]
        R_z = Initial_Conditions[i,14] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_V[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_V[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_V[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_5.append(U_x)
        U_y_sum_5.append(U_y)
        U_z_sum_5.append(U_z)
    ####  
    U_x_5 = np.sum(U_x_sum_5) 
    U_y_5 = np.sum(U_y_sum_5) 
    U_z_5 = np.sum(U_z_sum_5) 
    ###############################################
    ######### MASCON VIII - Layer VI ##############
    U_x_sum_6,U_y_sum_6,U_z_sum_6 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,15] - Center_of_mass[0]
        R_y = Initial_Conditions[i,16] - Center_of_mass[1]
        R_z = Initial_Conditions[i,17] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_VI[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_VI[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_VI[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_6.append(U_x)
        U_y_sum_6.append(U_y)
        U_z_sum_6.append(U_z)
    ####  
    U_x_6 = np.sum(U_x_sum_6) 
    U_y_6 = np.sum(U_y_sum_6) 
    U_z_6 = np.sum(U_z_sum_6) 
    #################################################
    ######### MASCON VIII - Layer VII ###############
    U_x_sum_7,U_y_sum_7,U_z_sum_7 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,18] - Center_of_mass[0]
        R_y = Initial_Conditions[i,19] - Center_of_mass[1]
        R_z = Initial_Conditions[i,20] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_VII[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_VII[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_VII[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_7.append(U_x)
        U_y_sum_7.append(U_y)
        U_z_sum_7.append(U_z)
    ####  
    U_x_7 = np.sum(U_x_sum_7) 
    U_y_7 = np.sum(U_y_sum_7) 
    U_z_7 = np.sum(U_z_sum_7) 
    ##################################################
    ######### MASCON VIII - Layer VIII ###############
    U_x_sum_8,U_y_sum_8,U_z_sum_8 = [],[],[]
    ###########################################
    for i in range(len(Initial_Conditions)):
        R_x = Initial_Conditions[i,21] - Center_of_mass[0]
        R_y = Initial_Conditions[i,22] - Center_of_mass[1]
        R_z = Initial_Conditions[i,23] - Center_of_mass[2]
        ###############################
        # Particle w.r.t. Tetra CM
        r_tetra_x = a[0] - R_x
        r_tetra_y = a[1] - R_y
        r_tetra_z = a[2] - R_z
        r_tetra   = np.sqrt( r_tetra_x**2 + r_tetra_y**2 + r_tetra_z**2 )
        U_x = - (mu_VIII[i]*r_tetra_x)/(r_tetra**3)
        U_y = - (mu_VIII[i]*r_tetra_y)/(r_tetra**3)
        U_z = - (mu_VIII[i]*r_tetra_z)/(r_tetra**3)
        U_x_sum_8.append(U_x)
        U_y_sum_8.append(U_y)
        U_z_sum_8.append(U_z)
    ####  
    U_x_8 = np.sum(U_x_sum_8) 
    U_y_8 = np.sum(U_y_sum_8) 
    U_z_8 = np.sum(U_z_sum_8) 
    #############################
    # Sum Each Layer ############
    U_x = U_x_1 + U_x_2 + U_x_3 + U_x_4 + U_x_5 + U_x_6 + U_x_7 + U_x_8
    U_y = U_y_1 + U_y_2 + U_y_3 + U_y_4 + U_y_5 + U_y_6 + U_y_7 + U_y_8
    U_z = U_z_1 + U_z_2 + U_z_3 + U_z_4 + U_z_5 + U_z_6 + U_z_7 + U_z_8
    ##################################################
    # Derivatives/ 6 Equations of Motion #############
    drxdt = vx 
    drydt = vy 
    drzdt = vz 
    dvxdt = omega**2 *rx + 2*omega*vy + U_x
    dvydt = omega**2 *ry - 2*omega*vx + U_y
    dvzdt = U_z
    dadt  = [drxdt,drydt,drzdt,dvxdt,dvydt,dvzdt]
    return dadt
# End ODE Definition ###################################                                                    
######################

#%% Calling ODE & Solving for COE
##################################
# Call function for state vector #
# Initial Conditions            
a0 = [State_X,State_Y,State_Z,
               0.0,0.0,0.0]

#################################
# Choose EOM Definition to use  #
#################################
if MASCON_Choice == '1':
    ODE_Calculating_message = f"""
{'-'*42}
|{'-'*9} Simulation Initiated {'-'*9}|
{'-'*42}
| Plotting orbit around {Aster_File_Name} 
|   using MASCON I, please wait...
{'-'*42}
"""
    print(ODE_Calculating_message)
    #########################
    # ODE MASCON I Solution ########
    a = odeint(Orbit_M1,a0,t_span) #
    ################################
elif MASCON_Choice == '3':
    ODE_Calculating_message = f"""
{'-'*42}
|{'-'*9} Simulation Initiated {'-'*9}|
{'-'*42}
| Plotting orbit around {Aster_File_Name} 
|   using MASCON III, please wait...
{'-'*42}
"""
    print(ODE_Calculating_message)
    ###########################
    # ODE MASCON III Solution ######
    a = odeint(Orbit_M3,a0,t_span) #
    ################################
else:
    ODE_Calculating_message = f"""
{'-'*42}
|{'-'*9} Simulation Initiated {'-'*9}|
{'-'*42}
| Plotting orbit around {Aster_File_Name} 
|   using MASCON VIII, please wait...
{'-'*42}
"""
    print(ODE_Calculating_message)
    ############################
    # ODE MASCON VIII Solution #####
    a = odeint(Orbit_M8,a0,t_span) #
    ################################
#####
#############
# END Timer #
ODE_Calc_End_Time = time.time() 
# ODE Calculations Execution Time 
ODE_Calc_Execution_Sec = ODE_Calc_End_Time - ODE_Calc_Start_Time
################################
# Convert to Minutes if to long
ode_calc_t_unit = 'Seconds'
ODE_Calc_Execution = ODE_Calc_Execution_Sec 
# Minutes
if ODE_Calc_Execution_Sec > 120:
    ODE_Calc_Execution_Min = ODE_Calc_Execution_Sec/60
    ODE_Calc_Execution = ODE_Calc_Execution_Min 
    ode_calc_t_unit = 'Minutes'
    # Hours
    if  ODE_Calc_Execution_Min > 120:
        ODE_Calc_Execution_Hr = ODE_Calc_Execution_Min/60
        ODE_Calc_Execution = ODE_Calc_Execution_Hr
        ode_calc_t_unit = 'Hours'
##############################################################
ODE_Solved_message = f"""
{'-'*42}
|{'-'*9} Simulation Complete! {'-'*9}|
{'-'*42}
| ODE Solved in: {ODE_Calc_Execution:.3f} {ode_calc_t_unit}
{'-'*42}
"""
print(ODE_Solved_message)
#################################################
#%% Saving Data
###########################
Save_Data_Prompt = f"""
{'-'*42}
|{'-'*14} Data Save: {'-'*14}|
{'-'*42}
"""
print(Save_Data_Prompt)
print(f"|{'-'*40}|") 
Save_Data_Question = input('| Save the orbit data?  (Y/N): ')
#######################################################################
if Save_Data_Question in ['y','yes','Y','Yes','YEs','YeS','YES','yES']:
    #####################
    # Tag File !!
    Out_File_Tag = input('| Enter Data File Tag: ')
    ##############################################
    # Save Classical Orbital Elements
    COE_Data_Out = pd.DataFrame({'Position Vec. X': a[:,0],
                                'Position Vec. Y': a[:,1],
                                'Position Vec. Z': a[:,2],
                                'Velocity Vec. X': a[:,3],
                                'Velocity Vec. Y': a[:,4],
                                'Velocity Vec. Z': a[:,5],
                                'Angular Momentum': Orbital_Elements[0],
                                'Semi-Major Axis': Orbital_Elements[1],
                                'Eccentricity': Orbital_Elements[2],
                                'Inclination': Orbital_Elements[3],
                                'Longitude of Asc. Node': Orbital_Elements[4],
                                'Arg of Periapsis': Orbital_Elements[5],
                                'True Anomaly': Orbital_Elements[6],})
    ##################################################################
    # Save Data @ Path
    isExist = os.path.exists(Aster_Sim_Data_PATH)
    if not isExist:
        os.mkdir(Aster_Sim_Data_PATH)
    COE_Data_Out_File_Name = Aster_Sim_Data_PATH + Out_File_Tag + '_' + Aster_File_Name + '.csv'
    COE_Data_Out.to_csv(COE_Data_Out_File_Name, sep=' ' ,index=False)
    # Shape of output file
    COE_DATA_SHAPE = np.shape(COE_Data_Out)
    # Data Print Out
    print(f"|{'-'*40}|")
    print(f"| Data File: \n| {COE_Data_Out_File_Name} ")
    print(f"{'-'*42}") 
    print(COE_Data_Out.head())
    Data_Ready_message = f"""
{'-'*42}
|{'-'*15} Data set {'-'*15}|
{'-'*42}
| Variable Count:  
|       {COE_DATA_SHAPE[1]}
| Time points: 
|       {COE_DATA_SHAPE[0]} Seconds
{'-'*42}
| Data ready, See program directory
{'-'*42}
    """
    print(Data_Ready_message)
else:
    print("| Data NOT Saved...")
###########################################
#%% Plots
Plot_message = f"""
{'-'*42}
| Plotting data, Please wait...
{'-'*42}
"""
print(Plot_message)
############################################
######################## Orbit Gauge Cluster
# ASTER.Orbit_Gauge_Cluster(Orbital_Elements,
#                           a, 
#                           mu,R_body, 
#                           t_inp,
#                           Graph_time,
#                           Graph_Time_Unit,
#                           Aster_File_Name)
# ############################################
# ################################## 3D Orbit
# ASTER.MASCON_Orbit_3D(Initial_Conditions,
#                       MASCON_Choice, 
#                       Center_of_mass, 
#                       a, 
#                       Aster_File_OBJ_Dir,
#                       scale)
#############################################
################################## Show Plots
plt.show() 




