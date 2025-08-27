import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import inv 
import os
import trimesh
import datetime
from numba import njit
import sys
sys.dont_write_bytecode = True
import constants as C
const    = C.constants()
import Asteroid_Package as AP
#################################################
####### Enter Asteroid Name & Update Here #######
Asteroid_Name = 'Apophis'
target = C.apophis()
##########################################
# Newton-Raphson settings
Tol = 1e-3
# max = 1000000 
Miter = 1000000
# Step limit for each iteration
Step = 3.0
# initial guess array size 
size = 3
# Multiples of effective radius
# for the initial guess
Guess_lim_Rad = 1.1
###################
# Graph settings
Contour_Size = 100
Contour_Levels = 350 
Font_Size = 14
########################################
########################################
# Saving in Databank
Time_Stamp  = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
folder_name = f"{Time_Stamp}"
Data_PATH   = 'Databank/Eq_Pts_' + folder_name + '/'
################################
################################
##################
Spin_Rate = target.spin
omega = 2.0*np.pi/(Spin_Rate*3600.0)
scale = target.gamma
Mesh_color = 'Black'
# R_eff = target.Re
# print(f"Effective Radius: {R_eff} km")
############################################
# Load the polyhedron mesh from a file (replace 'polyhedron.obj' with your mesh file)
mesh = trimesh.load_mesh(Asteroid_Name + '.obj')
# Scale the mesh
mesh.apply_scale(scale)
# Calculate the volume of the polyhedron
polyhedron_volume = mesh.volume
R_eff = (3 * polyhedron_volume / (4 * np.pi)) ** (1/3)
print(f"Volume: {polyhedron_volume} km^3")
print(f"Effective Radius: {R_eff} km")
Guess_Lim = R_eff*Guess_lim_Rad
###########################################
##################################################
########### Load File ############################
Aster_File_CM   =  Asteroid_Name + "_CM.in"
Aster_File_OBJ  =  Asteroid_Name + ".obj"
CM_MI = np.loadtxt(Aster_File_CM, delimiter=' ')
mu_I = np.loadtxt(Asteroid_Name + '_mu.in', delimiter=' ')
mu = np.sum(mu_I)
print(f"Gravitational Parameter: {mu} km^3/s^2")
##################################################
#################################################################
#################################################################
#%% 3D Newton Raphson
@njit
def Hx_3D(vars,CM_MI,mu_I):
    x,y,z = vars
    Ux = np.zeros_like(x,dtype=np.float64)
    Uy = np.zeros_like(x,dtype=np.float64)
    Uz = np.zeros_like(x,dtype=np.float64)
    for i in range(len(CM_MI)):
        R_x = x - CM_MI[i,0]
        R_y = y - CM_MI[i,1]
        R_z = z - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ##############################
        Ux += - (mu_I[i]*R_x)/R**3
        Uy += - (mu_I[i]*R_y)/R**3
        Uz += - (mu_I[i]*R_z)/R**3
    Hx  =  x  + Ux
    Hy  =  y  + Uy
    Hz  =   Uz
    Output = np.stack((Hx,Hy,Hz))
    return Output

###########################################
@njit
def Hxx_3D(vars,CM_MI,mu_I):
    x,y,z = vars
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
        R_z = z - CM_MI[i,2]
        R = np.sqrt(R_x**2 + R_y**2 + R_z**2)
        ############################## correction: - mu_I[i]/R**3 +
        Uxx += -mu_I[i]/R**3 + (3*mu_I[i]*R_x**2)/R**5
        Uyy += -mu_I[i]/R**3 + (3*mu_I[i]*R_y**2)/R**5
        Uzz += -mu_I[i]/R**3 + (3*mu_I[i]*R_z**2)/R**5
        ##############################  
        Uxy += (3*mu_I[i]*R_x*R_y)/R**5
        Uxz += (3*mu_I[i]*R_x*R_z)/R**5
        Uyx += (3*mu_I[i]*R_y*R_x)/R**5
        Uyz += (3*mu_I[i]*R_y*R_z)/R**5
        Uzx += (3*mu_I[i]*R_z*R_x)/R**5
        Uzy += (3*mu_I[i]*R_z*R_z)/R**5
    Top_Row = np.column_stack((Uxx,Uxy,Uxz))
    Mid_Row = np.column_stack((Uyx,Uyy,Uyz))
    Bot_Row = np.column_stack((Uzx,Uzy,Uzz))
    Output  = np.concatenate((Top_Row,Mid_Row,Bot_Row), axis =0)
    return Output
###########################################
def New_Raph_3D(x, tol, max_iter, step_limit):
    #######
    for i in range(max_iter):
        ########################
        AP.print_progress(i, max_iter, bar_length=50)
        ########################
        hx  = Hx_3D(x,CM_MI,mu_I)
        hxx = Hxx_3D(x,CM_MI,mu_I)
        
        # Debug
        # Check = np.dot(hxx,inv(hxx))
        # if hx[0] == 0.0 or hx[1] == 0.0:
        #     return x
        try:
            inv_hxx = - np.linalg.inv(hxx)
            delta_x = np.dot(inv_hxx, hx)

            
            
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
    print("\n")
    print("| Did not converge within max iterations.")
    DNC = 1
    return x, DNC
####################################################
####################################################
LB = -Guess_Lim
UB =  Guess_Lim

x_range = np.linspace(LB, UB, size)
y_range = np.linspace(LB, UB, size)
z_range = np.linspace(LB, UB, size)
X, Y, Z = np.meshgrid(x_range, y_range, z_range)
# Flatten the meshgrid to create a list of initial guesses
initial_guesses = [np.array([[x], [y], [z]]) for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel())]
print(f"Total initial guesses: {len(initial_guesses)}")
print(f"Initial guess range: {LB} to {UB} km")
####################################################
Equi_Pts_3D = []
for i, guess in enumerate(initial_guesses):
    
    guess_copy = guess.copy()
    print(f"| Initial Guess:     x = {guess[0,0]:.6f}, y = {guess[1,0]:.6f}, z = {guess[2,0]:.6f}")
    
    #########
    # Call Newton-Raphson
    EqPt, DNC = New_Raph_3D(guess_copy,tol=Tol,max_iter=Miter, step_limit=Step)
    print('\n')
    ###
    if DNC == 1:
        print(f"|End Position: : x = {EqPt[0,0]:.6f}, y = {EqPt[1,0]:.6f}, z = {EqPt[2,0]:.6f}")
    #### 
    else: 
        Equi_Pts_3D.append(EqPt)
        print(f"| Equilibrium Reached at: x = {EqPt[0,0]:.6f}, y = {EqPt[1,0]:.6f}, z = {EqPt[2,0]:.6f}")
    
print('|---Done Computing---')
if len(Equi_Pts_3D) == 0:
    print("| No Equilibrium Points Found")
    sys.exit(0)
else:
    print(f"| Found {len(Equi_Pts_3D)} Equilibrium Points")
    print("| Testing Stability of Equilibrium Points...")
#%% Equilibrium Points Stability Analysis
#################################################
#################################################
# Remove duplicate roots
#
#   Use trimesh to chekc the values for being in the astoerid 
#    and remove them if they are!
#
# find a better way to analyze duplicates
#  and calculate the standard deviation of each
#
# Equi_3D = []
# for CC in range(len(Equi_Pts_3D)):
Equi_3D = np.unique(np.round(Equi_Pts_3D,4), axis=0)



########################################
STABLE_EQ = []
for EE in range(len(Equi_3D)):
    vars = Equi_3D[EE]
    print('-----------------------------------')
    print(f'Equilibrium Point: x={vars[0]}, y={vars[1]}, z={vars[2]}')
    Output_A = Hxx_3D(vars,CM_MI,mu_I)
    print(Output_A)


    U_xx = Hxx_3D(vars,CM_MI,mu_I)[0,0]
    U_xy = Hxx_3D(vars,CM_MI,mu_I)[0,1]
    U_xz = Hxx_3D(vars,CM_MI,mu_I)[0,2]
    U_yx = Hxx_3D(vars,CM_MI,mu_I)[1,0]
    U_yy = Hxx_3D(vars,CM_MI,mu_I)[1,1]
    U_yz = Hxx_3D(vars,CM_MI,mu_I)[1,2]
    U_zx = Hxx_3D(vars,CM_MI,mu_I)[2,0]
    U_zy = Hxx_3D(vars,CM_MI,mu_I)[2,1]
    U_zz = Hxx_3D(vars,CM_MI,mu_I)[2,2]
    # print(U_xx)
    # print(U_xy)
    ###################################
    #
    # Characteristic coefficients 

    Alpha = U_xx + U_yy + U_zz + 4*omega**2


    Beta = U_xx*U_yy + U_yy*U_zz + U_zz*U_xx \
        - U_xy**2 - U_yz**2 - U_zx**2 \
            -U_zz*U_xy**2


    Gamma = U_xx*U_yy*U_zz + 2*U_xy*U_yz*U_zx \
                - U_xx*U_yz**2 - U_yy*U_zx**2 - U_zz*U_xy**2


    def Charac_Lam(Alpha, Beta, Gamma):
        
        # Coefficients of the polynomial Lambda**6 + Alpha*Lambda**4 + Beta*Lambda**2 + Gamma = 0
        coefficients = [1, 0, Alpha, 0, Beta, 0, Gamma]
        
        # Find the roots of the polynomial
        eigenvalues = np.roots(coefficients)
        
        return eigenvalues

    print(Charac_Lam(Alpha, Beta, Gamma))

    ################################
    # Theorem 1 


    def is_positive_definite(matrix):
        """ Test eigenvalues of a matrix to 
            determine if it is positive definite.
            
        """
        eigenvalues = np.linalg.eigvals(matrix)
        return np.all(eigenvalues > 0)

    # Example usage
    matrix = np.array([[2, -1], [-1, 2]])

    print(is_positive_definite(Output_A)) 



    def Chol_positive_definite(matrix):
        """
        check for positive definite matrix
        using cholesky decomposition.
        
        This will only retunr true if the matrix is 
        positive definite 
        
        AND!!!
        
        symmetric, this if the eigenvalues retunr true,
        yet this returns false. Ten the matrix is 
        not symmetric, but positive definite. 


        """
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    # Example usage

    print(Chol_positive_definite(Output_A))  # Output: True




    ##########################################################
    ## Theorem 2
    #
    TH2_1 = U_xx + U_yy + U_zz  + 4*omega**2
    #
    #
    TH2_2 = U_xx*U_yy + U_yy*U_zz + U_zz*U_xx + 4 *omega*U_zz
    #
    TH2_2RS = U_xy**2 + U_yz**2 + U_xz**2 
    #
    #
    TH2_3 = U_xx*U_yy*U_zz + 2*U_xy*U_yz*U_xz 
    #
    TH2_3RS = U_xx*U_yz**2 + U_yy*U_xz**2 + U_zz*U_xy**2
    #
    #
    TH2_4 = Alpha**2 + 18 *Alpha*Beta*Gamma 
    #
    TH2_4RS = 4*Alpha**3*Gamma + 4 * Beta**3 + 27*Gamma**2
    #
    #
    #
    def Stability_Th1(TH2_1, TH2_2, TH2_2RS, TH2_3, TH2_3RS, TH2_4, TH2_4RS):
        
        if TH2_1 > 0:
            print("Theorem 1: 1st condition met")

            if TH2_2 > TH2_2RS:
                print("Theorem 1: 2nd condition met")
                if TH2_3 > TH2_3RS:
                    print("Theorem 1: 3rd condition met")
                    if TH2_4 > TH2_4RS:
                        print("Theorem 1: 4th condition met. The equilibrium point is stable")
                        
                        ###############
                        x = vars[0][0]
                        y = vars[1][0]
                        z = vars[2][0]
                        print(x)
                        print(y)
                        print(z)
                        STABLE_EQ.append([x,y,z ])
                    
                    
                    
                    else:
                        print("Theorem 1: 4th condition NOT met. The equilibrium point is unstable!! ")

    print('Stability:')
    Stability_Th1(TH2_1, TH2_2, TH2_2RS, TH2_3, TH2_3RS, TH2_4, TH2_4RS)
    print('-----------------------------------')

print(STABLE_EQ)

STABLE_EQ = np.array(STABLE_EQ)

print(STABLE_EQ[:,0])
print(STABLE_EQ[:,1])
print(STABLE_EQ[:,2])


#%% 3D plot
#######################################################
################################################# Plot 
EQ3D_text = []
for ii in range(len(Equi_3D)):
    EQ3D_text.append(f"E{ii}")
    print(f'{EQ3D_text[ii]}: {Equi_3D[ii]}')
######################
#
#######
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(STABLE_EQ[:,0] , STABLE_EQ[:,1], STABLE_EQ[:,2],s=20,color='Purple')

Asteroid_Mesh = AP.OBJ_2_Mesh(Aster_File_OBJ,scale,Mesh_color)
ax.add_collection3d(Asteroid_Mesh)


# for i in range(len(EQ3D_text)):
#     ax.text(Equi_3D[i, 0], 
#             Equi_3D[i, 1], 
#             Equi_3D[i, 2], 
#             EQ3D_text[i], 
#             color='Blue'
#             )
    
    
ax.set_title("3D Equilibrium Points")
ax.set_xlabel(r'$X (km)$', fontsize=Font_Size )
ax.set_ylabel(r'$Y (km)$', fontsize=Font_Size )
ax.set_zlabel(r'$Z (km)$', fontsize=Font_Size )
ax.set_aspect('equal', 'box') 
plt.tight_layout()
plt.show()
#################################################
#################################################



# # Save Data @ Path
# isExist = os.path.exists(Data_PATH)
# if not isExist:
#     os.mkdir(Data_PATH)
# File_Name = Data_PATH + 'EqPts_3D_'+ Asteroid_Name +'.jpg'







