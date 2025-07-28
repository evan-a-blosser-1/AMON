"""Plot Survival Map data

"""
import numpy as np
import matplotlib.pyplot as plt  
################################
# Path to Data
Dat_Path = "Databank/Smap_60days/"
##############################################
##############################################
BND_File = Dat_Path + 'Smap_Bound_Events.dat'
ESC_File = Dat_Path + 'Smap_Escape_Events.dat'
COL_File = Dat_Path + 'Smap_Crash_Events.dat'
# 
Bound     = np.loadtxt(BND_File, delimiter=' ')
Escape    = np.loadtxt(ESC_File, delimiter=' ')
Collision = np.loadtxt(COL_File, delimiter=' ')
print(Escape)
fig = plt.figure(figsize=(7,7))
plt.scatter(Bound[:,0],Bound[:,1],s=1,color='blue',label='Bounded')
plt.scatter(Collision[:,0],Collision[:,1],s=1,color='RED',label='Collision')
plt.scatter(Escape[:,0],Escape[:,1],s=1,color='GREY',label='Escape')

#plt.xlabel(r'Semi-Major Axis (km)$')
#plt.ylabel(r'$Eccentricity$')
plt.xlabel(r'$y_0$ (km)')
plt.ylabel(r'$H_0$ $(\frac{km^2}{s^2})$')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)


plt.show()
