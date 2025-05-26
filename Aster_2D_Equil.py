import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scipy.optimize import fsolve
from icecream import ic
from scipy.linalg import inv 
import os
import datetime
########################################
################################### PATH
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
########################################
# Saving in Databank
Time_Stamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"{Time_Stamp}"
Data_PATH = 'Databank/Eq_Pts_' + folder_name + '/'
################################
# Big G in km^3/kg.s^2
G = 6.67430e-20
################################
##################
Spin_Rate = 2.1216
omega = ((2*np.pi)/Spin_Rate)*(1/3600)

Contour_Size = 100

Font_Size = 14
#################################################
####### Enter Asteroid Name & Update Here #######
Asteroid_Name = '1950DA_Prograde'
##################################################
########### Load File ############################
Aster_File_CM   = Aster_M1CM_PATH + Asteroid_Name + "_CM.dat"
Aster_File_OBJ  = Aster_OBJ_PATH  + Asteroid_Name + ".obj"

Aster_CM  = np.loadtxt(Aster_File_CM, delimiter=' ')


Aster_File_Const = Aster_Const_PATH + Asteroid_Name + '_const.in'
Asteroid_Const = pd.read_csv(Aster_File_Const, delimiter=' ')

Asteroid_Const.head()
#############################################################
R_eff = Asteroid_Const['Mean Radius (km)'][0]
Aster_Vol_File = Aster_VolM1_PATH + Asteroid_Name  + '_VolM1.csv'
Vol_Tetra = pd.read_csv(Aster_Vol_File, delimiter=' ')
scale = Asteroid_Const['Scaling'][0]
CM_MI = (Aster_CM*scale)/R_eff
Poly_CM_X = (Asteroid_Const['Poly CM X'][0]*scale)/R_eff
Poly_CM_Y = (Asteroid_Const['Poly CM Y'][0]*scale)/R_eff
Poly_CM_Z = (Asteroid_Const['Poly CM Z'][0]*scale)/R_eff
##################################################
##################################################
mu_I = []
for i in range(len(CM_MI)):
    mu = 1/len(CM_MI)
    mu_I.append(mu)
#################################################################
#%% Potential Contours
####################
def Poten_XY(x,y,CM_MI,mu_I,omega):
    U = np.zeros_like(x)
    for i in range(len(CM_MI)):
        R_x = x - CM_MI[i,0]
        R_y = y - CM_MI[i,1]
        R_z = 0 - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        U  += - (mu_I[i])/R
    #######################
    return - (1/2)*(omega**2)*(X**2 + Y**2) + U
####################
def Poten_XZ(x,y,CM_MI,mu_I,omega):
    U = np.zeros_like(x)
    for i in range(len(CM_MI)):
        R_x = x - CM_MI[i,0]
        R_y = 0 - CM_MI[i,1]
        R_z = y - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        U  += - (mu_I[i])/R
    #######################
    return - (1/2)*(omega**2)*(X**2 + Y**2) + U
####################
def Poten_YZ(x,y,CM_MI,mu_I,omega):
    U = np.zeros_like(x)
    for i in range(len(CM_MI)):
        R_x = 0 - CM_MI[i,0]
        R_y = x - CM_MI[i,1]
        R_z = y - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        U  +=  -(mu_I[i])/R
    #######################
    return - (1/2)*(omega**2)*(X**2 + Y**2) + U

########################
LowBound = -R_eff*2
UpBound  =  R_eff*2
x = np.linspace(LowBound,UpBound,Contour_Size)
y = np.linspace(LowBound,UpBound,Contour_Size)
X,Y = np.meshgrid(x,y)
#######
PE_XY = Poten_XY(X,Y,CM_MI,mu_I,omega)
PE_XZ = Poten_XZ(X,Y,CM_MI,mu_I,omega)
PE_YZ = Poten_YZ(X,Y,CM_MI,mu_I,omega)
# Set subplot  
Contour_Levels = 350                                                                                                 
fig, axis = plt.subplots( 1, 3,  figsize=(20, 8))                            
############################################################################
sc1 = axis[0].contour(X ,Y , PE_XY, Contour_Levels, cmap='viridis')
cbar1 = fig.colorbar(sc1, ax=axis[0])
cbar1.set_label('Gradient')
cbar1.set_label(r'$DU^2/TU^2$')
axis[0].set_title("X-Y")
axis[0].set_xlabel(r'$Y (DU)$', fontsize=Font_Size )
axis[0].set_ylabel(r'$Y (DU)$', fontsize=Font_Size )
############################################################################
sc2 = axis[1].contour(X ,Y , PE_XZ, Contour_Levels, cmap='viridis')
cbar2 = fig.colorbar(sc2, ax=axis[1])
cbar2.set_label(r'$DU^2/TU^2$')
axis[1].set_title("X-Z")
axis[1].set_xlabel(r'$Y (DU)$', fontsize=Font_Size )
axis[1].set_ylabel(r'$Z (DU)$', fontsize=Font_Size )
############################################################################
sc3 = axis[2].contour(X ,Y , PE_YZ, Contour_Levels, cmap='viridis')
cbar3 = fig.colorbar(sc3, ax=axis[2])
cbar3.set_label(r'$DU^2/TU^2$')
axis[2].set_title("Y-Z")
axis[2].set_xlabel(r'$Y (DU)$', fontsize=Font_Size )
axis[2].set_ylabel(r'$Z (DU)$', fontsize=Font_Size )
############################################################################
plt.tight_layout()
plt.show()



#%% 2D Newton Raphson XY
##########################################
################################# Jacobian
def Hx(vars,CM_MI,mu_I):
    x,y = vars
    Ux = np.zeros_like(x,dtype=np.float64)
    Uy = np.zeros_like(x,dtype=np.float64)
    Uz = np.zeros_like(x,dtype=np.float64)
    for i in range(len(CM_MI)):
        R_x = x - CM_MI[i,0]
        R_y = y - CM_MI[i,1]
        R_z = 0 - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        Ux += - (mu_I[i]*R_x)/R**3
        Uy += - (mu_I[i]*R_y)/R**3
        Uz += - (mu_I[i]*R_z)/R**3
    Hx  =  x  + Ux
    Hy  =  y  + Uy
    Output = np.stack((Hx,Hy))
    return Output

###########################################
###################### Hessian 2017 Aljbaae
def Hxx(vars,CM_MI,mu_I):
    x,y = vars
    Uxx = np.zeros_like(x,dtype=np.float64)
    Uyy = np.zeros_like(x,dtype=np.float64)
    Uzz = np.zeros_like(x,dtype=np.float64)
    Uxy = np.zeros_like(x,dtype=np.float64)
    Uxz = np.zeros_like(x,dtype=np.float64)
    Uyx = np.zeros_like(x,dtype=np.float64)
    Uyz = np.zeros_like(x,dtype=np.float64)
    Uzx = np.zeros_like(x,dtype=np.float64)
    Uzy = np.zeros_like(x,dtype=np.float64)
    for i in range(len(CM_MI)):
        R_x = x - CM_MI[i,0]
        R_y = y - CM_MI[i,1]
        R_z = 0 - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        Uxx += (3*mu_I[i]*R_x**2)/R**5
        Uyy += (3*mu_I[i]*R_y**2)/R**5
        Uzz += (3*mu_I[i]*R_z**2)/R**5
        ##############################  
        Uxy += (3*mu_I[i]*R_x*R_y)/R**5
        Uxz += (3*mu_I[i]*R_x*R_z)/R**5
        Uyx += (3*mu_I[i]*R_y*R_x)/R**5
        Uyz += (3*mu_I[i]*R_y*R_z)/R**5
        Uzx += (3*mu_I[i]*R_z*R_x)/R**5
        Uzy += (3*mu_I[i]*R_z*R_z)/R**5
    Top_Row = np.column_stack((Uxx,Uxy))
    Bot_Row = np.column_stack((Uyx,Uyy))
    Output  = np.concatenate((Top_Row,Bot_Row), axis =0)
    return Output


def New_Raph_2D_XY(x0, tol, max_iter):
    x = x0

    for i in range(max_iter):
        hx  = Hx(x,CM_MI,mu_I)
        hxx = Hxx(x,CM_MI,mu_I)
        
        
        # Debug
        # Check = np.dot(hxx,inv(hxx))
        # if hx[0] == 0.0 or hx[1] == 0.0:
        #     return x
        try:
            delta_x = np.matmul(-inv(hxx), hx)
            # delta_x = np.linalg.solve(hxx, -hx) 
                
                
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping...")
            return x
        
        x = x + delta_x
        if np.linalg.norm(delta_x) <tol:
            print(f"| Converged to equilibrium point after {i+1} iterations")
            return x
        
    print("| Did not converge within max iterations.")
    return x 
####################################################
####################################################
# x0 = np.array([R_eff*2.5, - R_eff*2.5])
# Meshgrid based on Asteroid Shape
LB = -R_eff*2.5
UB =  R_eff*2.5
size = 2
x_range = np.linspace(LB, UB, size)  
y_range = np.linspace(LB, UB, size) 
X, Y = np.meshgrid(x_range, y_range)
# Flatten the meshgrid to create a list of initial guesses
# initial_guesses = np.column_stack([X.ravel(), Y.ravel()])
#########################################
# Step 1: Define column vectors
vec1 = np.array([[1.0], [-0.5]])
vec2 = np.array([[-1.0], [-0.5]])
# vec3 = np.array([[0.0], [-1.0]])
# vec4 = np.array([[1.0], [0.0]])
vec3 = np.array([[0.2], [1.0]])
vec4 = np.array([[0.0], [-1.0]])
vec5 = np.array([[1.0], [0.0]])

# Step 2: Store vectors in an array
initial_guesses = [vec1, vec2, vec3, vec4, vec5]


##########################################
Equi_Pts_XY = []
for i, guess in enumerate(initial_guesses):
#guess = np.array([[-1.0],[-0.5]])
    Equilibrium = New_Raph_2D_XY(guess,tol=1e-6,max_iter=10000)
    Equi_Pts_XY.append(Equilibrium)
    Out_Message = f"""
{'-'*42}
| From initial Guess:     x = {guess[0,0]:.6f}, y = {guess[1,0]:.6f} 
| Equilibrium Reached at: x = {Equilibrium[0,0]:.6f}, y = {Equilibrium[1,0]:.6f}
{'-'*42}
    """
    print(Out_Message)
    
print('|---Done---')
##################################################



#%% 2D Newton Raphson YZ
##########################################
################################# YZ
def Hx_YZ(vars,CM_MI,mu_I):
    y,z = vars
    Uy = np.zeros_like(y,dtype=np.float64)
    Uz = np.zeros_like(z,dtype=np.float64)
    for i in range(len(CM_MI)):
        R_x = 0 
        R_y = y - CM_MI[i,1]
        R_z = z - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        Uy += - (mu_I[i]*R_y)/R**3
        Uz += - (mu_I[i]*R_z)/R**3
    Hy  =  y  + Uy
    Hz  = Uz
    Output = np.stack((Hy,Hz))
    return Output

###########################################
###################### Hessian 2017 Aljbaae
def Hxx_YZ(vars,CM_MI,mu_I):
    y,z = vars
    Uyy = np.zeros_like(y,dtype=np.float64)
    Uzz = np.zeros_like(z,dtype=np.float64)
    Uyz = np.zeros_like(y,dtype=np.float64)
    Uzy = np.zeros_like(z,dtype=np.float64)
    for i in range(len(CM_MI)):
        R_x = 0 
        R_y = y - CM_MI[i,1]
        R_z = z - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        Uyy += (3*mu_I[i]*R_y**2)/R**5
        Uzz += (3*mu_I[i]*R_z**2)/R**5
        ##############################  
        Uyz += (3*mu_I[i]*R_y*R_z)/R**5
        Uzy += (3*mu_I[i]*R_z*R_y)/R**5
    Top_Row = np.column_stack((Uyy,Uyz))
    Bot_Row = np.column_stack((Uzy,Uzz))
    Output  = np.concatenate((Top_Row,Bot_Row), axis =0)
    # ic(Output)
    return Output


def New_Raph_2D_YZ(x0, tol, max_iter, lambda_reg, step_limit):
    x = x0
    for i in range(max_iter):
        hx  = Hx_YZ(x,CM_MI,mu_I)
        hxx = Hxx_YZ(x,CM_MI,mu_I)
        
        # Regularization to handle singular/ill-conditioned Hessian
        hxx_reg = hxx + lambda_reg * np.eye(len(x))
        
        # Debug
        # Check = np.dot(hxx,inv(hxx))
        # if hx[0] == 0.0 or hx[1] == 0.0:
        #     return x
        try:
            delta_x = np.matmul(-inv(hxx_reg), hx)
            # delta_x = np.linalg.solve(hxx_reg, -hx) 
            
 
                
                
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping...")
            break
        
        
        # Limit the step size
        if np.linalg.norm(delta_x) > step_limit:
            delta_x = delta_x * (step_limit / np.linalg.norm(delta_x))
        
        
        x += delta_x
        
        
        if np.linalg.norm(delta_x) <tol:
            print(f"| Converged to equilibrium point after {i+1} iterations")
            return x
        
    
        
        
    print("| Did not converge within max iterations.")
    return x 
####################################################
####################################################
vec1 = np.array([[0.25], [1.0]])
vec2 = np.array([[-0.6], [0.8]])
vec3 = np.array([[0.9], [0.4]])
vec4 = np.array([[0.6], [-0.9]])
vec5 = np.array([[-0.9], [-0.5]])
#
initial_guesses = [vec1, vec2, vec3, vec4, vec5]
##########################################
Equi_Pts_YZ = []
# for i, guess in enumerate(initial_guesses):
    
#     Equilibrium = New_Raph_2D_YZ(guess,tol=1e-6,max_iter=10000, lambda_reg=1e-8, step_limit=1.0e-2)
    
#     Equi_Pts_YZ.append(Equilibrium)
#     Out_Message = f"""
# {'-'*42}
# | From initial Guess:     y = {guess[0,0]:.6f}, z = {guess[1,0]:.6f} 
# | Equilibrium Reached at: y = {Equilibrium[0,0]:.6f}, z = {Equilibrium[1,0]:.6f}
# {'-'*42}
#     """
#     print(Out_Message)
    
# print('|---Done---')



#%% 2D Newton Raphson XZ
##########################################
################################# XZ
def Hx_XZ(vars,CM_MI,mu_I):
    x,y = vars
    Ux = np.zeros_like(x,dtype=np.float64)
    Uz = np.zeros_like(x,dtype=np.float64)
    for i in range(len(CM_MI)):
        R_x = x - CM_MI[i,0]
        R_y = 0 - CM_MI[i,1]
        R_z = y - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        Ux += - (mu_I[i]*R_x)/R**3
        Uz += - (mu_I[i]*R_z)/R**3
    Hx  =  x  + Ux
    Hz  = Uz
    Output = np.stack((Hx,Hz))
    return Output

###########################################
###################### Hessian 2017 Aljbaae
def Hxx_XZ(vars,CM_MI,mu_I):
    x,y = vars
    Uxx = np.zeros_like(x,dtype=np.float64)
    Uzz = np.zeros_like(x,dtype=np.float64)
    Uxz = np.zeros_like(x,dtype=np.float64)
    Uzx = np.zeros_like(x,dtype=np.float64)
    for i in range(len(CM_MI)):
        R_x = x - CM_MI[i,0]
        R_y = 0 - CM_MI[i,1]
        R_z = y - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        Uxx += (3*mu_I[i]*R_x**2)/R**5
        Uzz += (3*mu_I[i]*R_z**2)/R**5
        ##############################  
        Uxz += (3*mu_I[i]*R_x*R_z)/R**5
        Uzx += (3*mu_I[i]*R_z*R_x)/R**5
        ##############################  
    Top_Row = np.column_stack((Uxx,Uxz))
    Bot_Row = np.column_stack((Uzx,Uzz))
    Output  = np.concatenate((Top_Row,Bot_Row), axis =0)
    return Output


def New_Raph_2D_XZ(x0, tol, max_iter, lambda_reg, step_limit):
    x = x0

    for i in range(max_iter):
        hx  = Hx_XZ(x,CM_MI,mu_I)
        hxx = Hxx_XZ(x,CM_MI,mu_I)
        
        # if np.linalg.det(hxx) == 0:
        #     print('| Determinate of hxx is Zero, stopping...')
        #     break
        
        # Regularization to handle singular/ill-conditioned Hessian
        hxx_reg = hxx + lambda_reg * np.eye(len(x))
        
        # Debug
        # Check = np.dot(hxx,inv(hxx))
        # if hx[0] == 0.0 or hx[1] == 0.0:
        #     return x
        try:
            delta_x = np.matmul(-inv(hxx_reg), hx)
            # delta_x = np.linalg.solve(hxx_reg, -hx)   
                
        except np.linalg.LinAlgError:
            print("Hessian is singular, stopping...")
            break
        
        # Limit the step size
        if np.linalg.norm(delta_x) > step_limit:
            delta_x = delta_x * (step_limit / np.linalg.norm(delta_x))
        
        x = x + delta_x
        if np.linalg.norm(delta_x) <tol:
            print(f"| Converged to equilibrium point after {i+1} iterations")
            return x
        
    print("| Did not converge within max iterations.")
    return x 
####################################################
####################################################
# x0 = np.array([R_eff*2.5, - R_eff*2.5])
# Meshgrid based on Asteroid Shape
LB = -R_eff*2.5
UB =  R_eff*2.5
size = 2
x_range = np.linspace(LB, UB, size)  
y_range = np.linspace(LB, UB, size) 
X, Y = np.meshgrid(x_range, y_range)
# Flatten the meshgrid to create a list of initial guesses
# initial_guesses = np.column_stack([X.ravel(), Y.ravel()])
#########################################
# Step 1: Define column vectors
vec1 = np.array([[0.5], [-1.0]])
vec2 = np.array([[-0.5], [-1.0]])
# vec3 = np.array([[0.0], [-1.0]])
# vec4 = np.array([[1.0], [0.0]])

# Step 2: Store vectors in an array
initial_guesses = [vec1, vec2]

##########################################
Equi_Pts_XZ = []
# for i, guess in enumerate(initial_guesses):
# #guess = np.array([[-1.0],[-0.5]])
#     Equilibrium = New_Raph_2D_XZ(guess,tol=1e-6,max_iter=10000, lambda_reg=1e-8, step_limit=1.0e-2)
#     Equi_Pts_XZ.append(Equilibrium)
#     Out_Message = f"""
# {'-'*42}
# | From initial Guess:     x = {guess[0,0]:.6f}, z = {guess[1,0]:.6f} 
# | Equilibrium Reached at: x = {Equilibrium[0,0]:.6f}, z = {Equilibrium[1,0]:.6f}
# {'-'*42}
#     """
#     print(Out_Message)
    
# print('|---Done---')


#%% Main 
####################################################
########################################### Settings 
Tolerance  = 1e-6
Mx_it = 10000
####################################################
####################################################
# x0 = np.array([R_eff*2.5, - R_eff*2.5])
# Meshgrid based on Asteroid Shape
# LB = -R_eff*1.1
# UB =  R_eff*1.1
# size = 2
# x_range = np.linspace(LB, UB, size)  
# y_range = np.linspace(LB, UB, size) 
# X, Y = np.meshgrid(x_range, y_range)
# # Flatten the meshgrid to create a list of initial guesses
# initial_guesses = np.column_stack([X.ravel(), Y.ravel()])
# #########################################
# Equi_XY_Pts, Equi_XZ_Pts, Equi_YZ_Pts = [], [], []
# print("Starting...")
# for guess in initial_guesses:
#     #############################################################
#     EquilibriumXY = New_Raph_2D_XY(guess,tol=Tolerance,max_iter=Mx_it)
#     EquilibriumXZ = New_Raph_2D_XZ(guess,tol=Tolerance,max_iter=Mx_it)  
#     EquilibriumYZ = New_Raph_2D_YZ(guess,tol=Tolerance,max_iter=Mx_it)
#     #############################################################
#     Equi_XY_Pts.append(EquilibriumXY)
#     # Equi_XZ_Pts.append(EquilibriumXZ)
#     #Equi_YZ_Pts.append(EquilibriumYZ)



#%% Plot Equilibrium
#################################################
#################################################
# Remove duplicate roots
Equi_XY = np.unique(Equi_Pts_XY, axis=0)
Equi_XZ = np.unique(Equi_Pts_XZ, axis=0)
Equi_YZ = np.unique(Equi_Pts_YZ, axis=0)
########################################
EQXY_text, EQXZ_text, EQYZ_text = [] , [] , []
for ii in range(len(Equi_XY)):
    EQXY_text.append(f"E{ii}")
for ii in range(len(Equi_XZ)):
    EQXZ_text.append(f"E{ii}")
for ii in range(len(Equi_YZ)):
    EQYZ_text.append(f"E{ii}")
# print(EQXY_text)
#######
LowBound = -R_eff*2
UpBound  =  R_eff*2
Size = 100
x = np.linspace(LowBound,UpBound,Size)
y = np.linspace(LowBound,UpBound,Size)
X,Y = np.meshgrid(x,y)
######################
#######
PE_XY = Poten_XY(X,Y,CM_MI,mu_I,omega)
PE_XZ = Poten_XZ(X,Y,CM_MI,mu_I,omega)
PE_YZ = Poten_YZ(X,Y,CM_MI,mu_I,omega)
Contour_Levels = 350
# Set subplot                                                  
fig, axis = plt.subplots( 1, 1,  figsize=(10, 10))                            
############################################################################
sc1 = axis.contour(X ,Y , PE_XY, Contour_Levels, cmap='viridis')
cbar1 = fig.colorbar(sc1, ax=axis)
cbar1.set_label('Gradient')
cbar1.set_label(r'$DU^2/TU^2$')
axis.scatter(Equi_XY[:,0] ,Equi_XY[:,1],s=20,color='Purple')
for i in range(len(EQXY_text)):
    axis.annotate( EQXY_text[i], xy=(Equi_XY[i,0] ,Equi_XY[i,1] ),color='Blue')
axis.set_title("X-Y")
axis.set_xlabel(r'$Y (DU)$', fontsize=Font_Size )
axis.set_ylabel(r'$Y (DU)$', fontsize=Font_Size )
############################################################################
# sc2 = axis[1].contour(X ,Y , PE_XZ, Contour_Levels, cmap='viridis')
# cbar2 = fig.colorbar(sc2, ax=axis[1])
# cbar2.set_label(r'$DU^2/TU^2$')
# axis[1].scatter(Equi_XZ[:,0] ,Equi_XZ[:,1],color='Purple')
# for i in range(len(EQXZ_text)):
#     axis[1].annotate( EQXZ_text[i], xy=(Equi_XZ[i,0] ,Equi_XZ[i,1]),color='Blue')
# axis[1].set_title("X-Z")
# axis[1].set_xlabel(r'$Y (DU)$', fontsize=Font_Size )
# axis[1].set_ylabel(r'$Z (DU)$', fontsize=Font_Size )
# ############################################################################
# sc3 = axis[2].contour(X ,Y , PE_YZ, Contour_Levels, cmap='viridis')
# cbar3 = fig.colorbar(sc3, ax=axis[2])
# cbar3.set_label(r'$DU^2/TU^2$')
# axis[2].scatter(Equi_YZ[:,0] ,Equi_YZ[:,1],color='Purple')
# for i in range(len(EQYZ_text)):
#     axis[2].annotate( EQYZ_text[i], xy=(Equi_YZ[i,0] ,Equi_YZ[i,1]),color='Blue')
# axis[2].set_title("Y-Z")
# axis[2].set_xlabel(r'$Y (DU)$', fontsize=Font_Size )
# axis[2].set_ylabel(r'$Z (DU)$', fontsize=Font_Size )
############################################################################
plt.tight_layout()

# Save Data @ Path
isExist = os.path.exists(Data_PATH)
if not isExist:
    os.mkdir(Data_PATH)
File_Name = Data_PATH + 'EqPts_'+ Asteroid_Name +'.jpg'

plt.savefig(File_Name)
plt.show()


XY_Data_Out = pd.DataFrame({'x': Equi_XY[:,0].ravel(),
                            'y': Equi_XY[:,1].ravel(),
                            })

# YZ_Data_Out = pd.DataFrame({'y': Equi_YZ[:,0].ravel(),
#                             'Z': Equi_YZ[:,1].ravel(),
#                             })

# XZ_Data_Out = pd.DataFrame({'x': Equi_XZ[:,0].ravel(),
#                             'Z': Equi_XZ[:,1].ravel(),
#                             })

##################################################################
XY_Out_File_Name = Data_PATH + 'XY_EqPts_' + Asteroid_Name   + '.csv'
XY_Data_Out.to_csv(XY_Out_File_Name, mode='w', header=True)


# YZ_Out_File_Name = Data_PATH + 'YZ_EqPts_' + Asteroid_Name   + '.csv'
# YZ_Data_Out.to_csv(YZ_Out_File_Name, mode='w', header=True)


# XZ_Out_File_Name = Data_PATH + 'XZ_EqPts_' + Asteroid_Name   + '.csv'
# XZ_Data_Out.to_csv(XZ_Out_File_Name, mode='w', header=True)









