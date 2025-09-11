import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

folder   = "Databank/Apophis/Tr_Check" 
########
yi = 1.0
yf = 1.0
dy = 0.01
nx = round((yf - yi)/dy)
########################
Hi = 1.6e-9
Hf = 1.6e-9
dH = 0.1e-9
nH = round((Hf - Hi) / dH)
########################
col_ls = ['r',  'g', 'y', 'm', 'c', 'k']

fig1 = plt.figure()
ax1  = fig1.add_subplot(111, projection='3d')
ax1.set_title('Trajectory')
ax1.set_xlabel('X (km)')
ax1.set_ylabel('Y (km)')
ax1.set_zlabel('Z (km)')
###########################
xplane = np.linspace(0.95,1.05,100)
zplane = np.linspace(-.025,.025,100)
X, Z = np.meshgrid(xplane, zplane)
Y = np.zeros_like(X) 

###########################
for ii in range(0, nx + 1):
    x0 = yi + float(ii)*dy
    for jj in range(0, nH + 1):
        H0 = Hi + float(jj)*dH
    
        aux1 = f"{H0:.1e}"
        #aux1 = f"{H0}"
        #aux1 = '0.0' 
        aux2 = str(round(x0, 5))
        
        #file = folder + '/' + 'PY-C' + aux1 + 'Xi' + aux2 + '.dat'
        
        #file = folder + '/' + 'PY-C' + aux1 + 'Yi' + aux2 + '.dat'
        
                
        file = folder + '/' + 'TR-S0' +'-H' + aux1 + 'Yi' + aux2 + '.dat'
    
    
    
        # Skip missing files
        if os.path.isfile(file) == False:
            continue 
        # Print loaded files after skipping
        print(file)
        ps = np.loadtxt(file, dtype=float)

        x = list(ps[:, 0])
        y = list(ps[:, 1])
        z = list(ps[:, 2])

        #####################
        col = col_ls[(ii * (nH + 1) + jj) % len(col_ls)]
        #################################################
        ax1.plot_surface(X, Y, Z, alpha=0.25, color='gray')

        ax1.plot(x, y, z ,alpha=1, color=col)
        ax1.set_aspect('equal', 'box') 
        
plt.show()