import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import sys
########################################################
#################################### Personal Packages #    

# plt.rcParams["figure.figsize"] = [6.5, 6.5]
plt.rcParams["figure.autolayout"] = True
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

# plt.axvline(x = 4.45e-3, color = 'r')
# plt.axvline(x = (1 - 4.45e-3), color = 'b')



folder   = "Databank/Test_CorEQ/" 

srt = 0.01
end = 0.03


exclude_List = []
for i in np.arange(srt, end, step=0.01):
    exclude_List.append(np.round(i,2))
# print(exclude_List)
######## 1.10
xi = 0.5
xf = 4.0
dx = 0.01
nx = round((xf - xi)/dx)
########################
Hi = 1.6e-9
Hf = 1.6e-9
dH = 0.1e-9
nH = round((Hf - Hi) / dH)
########################
all_z = []
scatter_plots = []

fig, ax = plt.subplots()
for ii in range(0, nx + 1):
    x0 = xi + float(ii)*dx
    for jj in range(0, nH + 1):
        ######################
        if np.round(x0,2) in exclude_List:
            continue
        ######################
        H0 = Hi + float(jj)*dH
    
        aux1 = f"{H0:.1e}"
        #aux1 = f"{H0}"
        #aux1 = '0.0' 
        aux2 = str(round(x0, 5))
        
        #file = folder + '/' + 'PY-C' + aux1 + 'Xi' + aux2 + '.dat'
        
        #file = folder + '/' + 'PY-C' + aux1 + 'Yi' + aux2 + '.dat'
        
                
        file = folder + '/' + 'PY-S0' +'-H' + aux1 + 'Yi' + aux2 + '.dat'
    
    
        # DEBUGG
        print(file)
    
    
    
        # Skip missing files
        if os.path.isfile(file) == False:
            continue 
        # Print loaded files after skipping
        print(file)
        ps = np.loadtxt(file, dtype=float)
        x = list(ps[:, 1])
        y = list(ps[:, 4])
        z = list(ps[:, 6])
        all_z.extend(z) 
        print(z[-1])
        POINCARE_Sec = plt.scatter(x, y, c=z, s=0.5, cmap='viridis',alpha=1)
        scatter_plots.append(POINCARE_Sec)
        
norm = plt.Normalize(min(all_z), max(all_z))

for scatter in scatter_plots:
    scatter.set_norm(norm)
    
    
cbar = plt.colorbar(POINCARE_Sec, 
                    orientation='vertical', 
                    pad=0.01,
                    aspect=20)

#cbar.set_label(r'$H (\frac{km^2}{s^2})$', fontsize=25, labelpad=10)

cbar.set_label(r'$y_0 (km)$', fontsize=25, labelpad=10)

cbar.ax.tick_params(labelsize=20) 
      
plt.xlabel(r'$y$ (km)', fontsize=25)
plt.xticks(fontsize=20)
# plt.xlim(-10, 10)

plt.ylabel(r'$\dot{y}$ (km/s)', fontsize=25)
plt.yticks(fontsize=20)


# fig.set_facecolor('#000000')
# ax.set_facecolor('#000000')
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')


plt.show()
# plt.savefig(folder + 'Poincare_Sec.png', dpi=300)
