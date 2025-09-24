"""Plot Survival Map data

"""
import numpy as np
import matplotlib.pyplot as plt  
################################
# Path to Data
Dat_Path = "Databank/1950DA/Smap_max/"
##############################################
##############################################
BND_File = Dat_Path + 'Smap_Bound_Events.dat'
ESC_File = Dat_Path + 'Smap_Escape_Events.dat'
COL_File = Dat_Path + 'Smap_Crash_Events.dat'
# 
Bound     = np.loadtxt(BND_File, delimiter=' ')
Escape    = np.loadtxt(ESC_File, delimiter=' ')
Collision = np.loadtxt(COL_File, delimiter=' ')
#
num_sims = Bound.shape[0] + Escape.shape[0] + Collision.shape[0]
print(f"Total number of simulations: {num_sims}")

plt.rcParams["figure.autolayout"] = True
plt.rcParams['mathtext.fontset'] = 'cm'

fig = plt.figure(figsize=(7,7))
plt.scatter(Bound[:,0],Bound[:,1],s=5,color='blue',label='Bounded')
plt.scatter(Collision[:,0],Collision[:,1],s=5,color='RED',label='Collision')
plt.scatter(Escape[:,0],Escape[:,1],s=5,color='GREY',label='Escape')

#plt.xlabel(r'Semi-Major Axis (km)$')
#plt.ylabel(r'$Eccentricity$')
plt.xlabel(r'$y_0$ $(km)$', fontsize=25)
plt.ylabel(r'$H_0$ $(\frac{km^2}{s^2})$', fontsize=25)
plt.tick_params(axis='y', labelsize=14) 
plt.tick_params(axis='x', labelsize=14)
# Increase font size of scientific notation on y-axis
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
plt.gca().yaxis.get_offset_text().set_fontsize(14)  # Adjust font size (default is usually 10-12)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, markerscale=3)

plt.show()
