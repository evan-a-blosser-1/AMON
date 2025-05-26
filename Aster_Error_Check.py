
import os
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import sys
from icecream import ic
from scipy.optimize import fsolve
from scipy.linalg import inv
from tqdm import tqdm
########################################################
#################################### Personal Packages #    
sys.dont_write_bytecode = True
import Asteroid_Package as ASTER
#################################################
####### Enter Asteroid Name & Update Here #######
Asteroid_Name = 'Apophis'
##################################################
########### Load File ############################
Aster_File_CM   = Asteroid_Name + "_CM.in"
Aster_File_OBJ  = Asteroid_Name + ".obj"
Aster_CM  = np.loadtxt(Aster_File_CM, delimiter=' ')
mu_I = np.loadtxt(Asteroid_Name + '_mu.in', delimiter=' ')


# Big G in km^3/kg.s^2
G = 6.6741e-20

Den = 1.75e3 # kg/m^3


# Tsuolis Grav Model
from polyhedral_gravity import Polyhedron, GravityEvaluable, evaluate, PolyhedronIntegrity, NormalOrientation

print(f'| Asteroid File: {Aster_File_OBJ} |')
vertices, faces = ASTER.OBJ_2_VertFace(Aster_File_OBJ)

vertices = vertices * 285.0 # m 

vertices = vertices[1:]
print(f'| Vertices: {vertices} |')

faces = faces - 1
print(f'| Faces: {faces} |')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2], 'o')

plt.show()



 
# Create the model 
polyhedron = Polyhedron(
   polyhedral_source=(vertices, faces),
   density=Den,
   integrity_check=PolyhedronIntegrity.DISABLE,   # VERIFY (default), DISABLE or HEAL
)







def Approx_2Body(mu_i, x_in):
    mu = G * 5.31e10
    x = x_in
    y = 0 
    z = 0
    r = np.sqrt(x**2 + y**2 + z**2) 
    U = mu/r
    return U

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
    
    
    
    
x = np.linspace(0.01, 2.0, 10000)    

B2 = Approx_2Body(mu_I, x)
M1 = MASCON_U(Aster_CM, mu_I, x)



# take to meters 
VALUES = np.arange(10.0, 2000.0, (2000.0 - 10.0) / 10000)
X = VALUES



computation_points = np.array(np.meshgrid(X, [0], [0])).T.reshape(-1, 3)
print(f'| Computation Points: {computation_points} |')



evaluable = GravityEvaluable(polyhedron=polyhedron)
results = evaluable(
  computation_points=computation_points,
  parallel=True,
)



potentials =  np.array([i[0] for i in results])
TP = potentials.reshape((len(VALUES)))

# Convert from km^2 to m^2
TP = TP * (1/1e6)

print(TP)

fig, ax = plt.subplots(figsize=(10, 4))
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
ax.set_xlabel('X (km)', fontsize=12, fontweight='bold')
ax.set_ylabel('Relative Error', fontsize=12, fontweight='bold')
ax.set_title('Error in Potential', fontsize=14, fontweight='bold')
plt.legend()
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
