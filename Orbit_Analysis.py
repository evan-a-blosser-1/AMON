import os
import sys
import numpy as np
from scipy import fft
from scipy.signal import windows, find_peaks
import trimesh
from numba import njit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
###
sys.dont_write_bytecode = True
import constants as C
const    = C.constants()
day = const.day
############################################################
############################################# Main Settings
# Asteroid Name
asteroid = '1950DA_Prograde'
target = C.DA1950()
# 
folder   = "Databank/1950DA_REDO/periodic/"
####
# Set:
# 0: Bounded motion
# 1: Collision Or Escape 
typ_flg = '0'
########
xi = 1.3812
xf = 1.3812
dx = 0.01
nx = round((xf - xi)/dx)
########################
Hi = 3.0e-7
Hf = 3.0e-7
dH = 0.1e-7
nH = round((Hf - Hi) / dH)
#############################################################################
################################################ angular velocity calculation
Spin = target.spin
print(Spin)
omega = (2*np.pi)/(Spin * 3600)
################################## Define Equations 
@njit
def Calc_Ham(state,omega,mu_I,CM):
    U = np.zeros(state.shape[1], dtype="float64")
    for it in range(len(CM)):
        x = state[0,:] - CM[it,0]
        y = state[1,:] - CM[it,1]
        z = state[2,:] - CM[it,2]
        r = np.sqrt(x**2 + y**2 + z**2) 
        U += mu_I[it]/r
    ############################
    X = state[0,:]
    Y = state[1,:]
    Z = state[2,:]
    VX = state[3,:]
    VY = state[4,:]
    VZ = state[5,:]
    V_mag = np.sqrt(VX**2 + VY**2 + VZ**2)
    Energy = 0.5*(VX**2 + VY**2 + VZ**2) - 0.5*omega**2* (X**2 + Y**2 + Z**2) - U[0]
    # print(Energy)
    return Energy
###
def Poincare (state):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Poincare Section')
    ax.set_xlabel(r'$y$ $(km)$')
    ax.set_ylabel(r'$y_0 (km)$')
    # print(f"| Generating Poincare Section...")
    nt = state.shape[1] - 1
    # global CJ, x0, vy0
    xa = np.zeros(6)
    # xm = np.zeros(6)
    xm = np.zeros((6,nt))
    #####################
    xa[0] = state[0, 0]
    xa[1] = state[1, 0]
    xa[2] = state[2, 0]
    xa[3] = state[3, 0]
    xa[4] = state[4, 0]
    xa[5] = state[5, 0]
    for it in range(1, nt + 1):
        ##########################################
        # Check x2*x1 < 0 & x2 > 0 for sign change
        #
        if state[0,it]*xa[0] < 0.0 and state[0,it] > 0.0:
            # print(f"Negative Sign Check: {neg_si:.6f}")
            # print(f"Positive Sign Check: {pos_si:.6f}")
            # print(f"X Position: {state[0, it]:.6f}")
            # print(f"Y Position: {state[1, it]:.6f}")
            # print(f"Z Position: {state[2, it]:.6f}")
            # print(f"X Velocity: {state[3, it]:.6f}")
            # print(f"Y Velocity: {state[4, it]:.6f}")
            # print(f"Z Velocity: {state[5, it]:.6f}")
            # input('| Press Enter to continue...')
            # print('----------------------------')
            #####
            # Fill matrix of points 
            # xm[0,it] = (state[0, it] + xa[0])/2.0
            # xm[1,it] = (state[1, it] + xa[1])/2.0
            # xm[2,it] = (state[2, it] + xa[2])/2.0
            # xm[3,it] = (state[3, it] + xa[3])/2.0
            # xm[4,it] = (state[4, it] + xa[4])/2.0
            # xm[5,it] = (state[5, it] + xa[5])/2.0
            #####
            # Single Point
            xm[0] = (state[0, it] + xa[0])/2.0
            xm[1] = (state[1, it] + xa[1])/2.0
            xm[2] = (state[2, it] + xa[2])/2.0
            xm[3] = (state[3, it] + xa[3])/2.0
            xm[4] = (state[4, it] + xa[4])/2.0
            xm[5] = (state[5, it] + xa[5])/2.0
            ax.scatter(xm[1], xm[4], color="#03bc4d")
        ##############################
        ##############################            
        # Update the sate
        # for each sign check
        xa[0] = state[0, it]
        xa[1] = state[1, it]
        xa[2] = state[2, it]
        xa[3] = state[3, it]
        xa[4] = state[4, it]
        xa[5] = state[5, it]
    ########################
    # print(xm)
    # ##### Estimate the phase space density
    # points = np.vstack((xm[0], xm[4]))
    # kde = stats.gaussian_kde(points)
    # density = kde.evaluate(points)
    # scatter = ax.scatter(xm[1], xm[4], c=density, cmap='viridis')
    # plt.colorbar(scatter, label='Phase Space Density')
#####
###########################################################
################################################ Load files
# object file
obj_Path =  asteroid + '.obj' 
# MASCONs (tetrahedron center of masses)
CM_Path =  asteroid + '_CM.in' 
CM = np.loadtxt(CM_Path, delimiter=' ',dtype=float)
Terta_Count = len(CM)
########################################
# Gravitational Parameter of each MASCON
mu_Path =   asteroid + '_mu.in'
mu_I = np.loadtxt(mu_Path, delimiter=' ')
mu = np.sum(mu_I)
print(f"Total Gravitational Parameter: {mu}")
#######################################################
#######################################################
# plt.rcParams["figure.figsize"] = [6.5, 6.5]
plt.rcParams["figure.autolayout"] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

# plt.axvline(x = 4.45e-3, color = 'r')
# plt.axvline(x = (1 - 4.45e-3), color = 'b')


########################################################
########################################################
all_z = []
scatter_plots = []
mesh = trimesh.load_mesh(obj_Path)
gamma = target.gamma
mesh = mesh.apply_scale(gamma)
v = mesh.vertices
f = mesh.faces 
print(f"Vertices: {v.shape}")
print(f"Faces: {f.shape}") 

col_ls = ['r', 'b', 'g', 'y', 'm', 'c', 'k']

fig1 = plt.figure(figsize=(10, 10))
ax1  = fig1.add_subplot(111, projection='3d')
ax1.set_xlabel(r'$X$ $(km)$', fontsize=25, labelpad=25)
ax1.set_ylabel(r'$Y$ $(km)$', fontsize=25, labelpad=25)
ax1.set_zlabel(r'$Z$ $(km)$', fontsize=25, labelpad=25)
ax1.tick_params(axis='both', which='major', labelsize=25)
# fig1.set_facecolor('#000000')
# ax1.set_facecolor('#000000')
# ax1.tick_params(axis='x', colors='white')
# ax1.tick_params(axis='y', colors='white')
#######################
# fig2 = plt.figure()
# ax2  = fig2.add_subplot(111)
# #ax2.set_title('Hamiltonian Energy')
# ax2.set_xlabel('Time (days)', fontsize=20)
# ax2.set_ylabel(r'Energy $(\frac{km^2}{s^2})$', fontsize=20)
# ax2.tick_params(axis='x', labelsize=24)
# ax2.tick_params(axis='y', labelsize=24)
# # Increase font size of scientific notation on y-axis
# ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
# ax2.yaxis.get_offset_text().set_fontsize(20)  # Fixed: removed gca() and increased font size
# ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, markerscale=3)


#######################
fig5 = plt.figure(figsize=(12, 8))
ax5  = fig5.add_subplot(111)
ax5.set_xlabel('Frequency (Hz)', fontsize=20)
ax5.set_ylabel('Power Spectrum', fontsize=20)
ax5.tick_params(axis='both', which='major', labelsize=18)
ax5.grid(True, alpha=0.3)

#######################
# fig3 = plt.figure()
# ax3  = fig3.add_subplot(111)
# ax3.set_title(r'Velocity')
# ax3.set_xlabel('time (sec)')
# ax3.set_ylabel(r'Velocity $\frac{km}{s}$')


#######################
# fig4 = plt.figure()
# ax4  = fig4.add_subplot(111)
# ax4.set_title(r'Position')
# ax4.set_xlabel('time (sec)')
# ax4.set_ylabel(r'Position (km)')

#######################
# fig7 = plt.figure()
# ax7  = fig7.add_subplot(111, projection='3d')
# ax7.set_title('Inertial Frame Trajectory')  
# ax7.set_xlabel('X (km)')
# ax7.set_ylabel('Y (km)')
# ax7.set_zlabel('Z (km)')
########################################################
########################################################

for ii in range(0, nx + 1):
    x0 = xi + float(ii)*dx
    for jj in range(0, nH + 1):
        H0 = Hi + float(jj)*dH
    
        aux1 = f"{H0:.0e}"
        #aux1 = f"{H0}"
        #aux1 = '0.0' 
        aux2 = str(round(x0, 5))
        
        #file = folder + '/' + 'PY-C' + aux1 + 'Xi' + aux2 + '.dat'
        
        #file = folder + '/' + 'PY-C' + aux1 + 'Yi' + aux2 + '.dat'
        
                
        file = folder + '/' + 'TR-S' + typ_flg + '-H' + aux1 + 'Yi' + aux2 + '.dat'
    
    
        print(file)
        # Skip missing files
        if os.path.isfile(file) == False:
            continue 
        # Print loaded files after skipping
        print(file)
        ps = np.loadtxt(file, dtype=float)
 
        mesh = Poly3DCollection([v[ii] for ii in f], 
                        edgecolor='black',
                        facecolors="white",
                        linewidth=0.75,
                        alpha=0.0)


        # Get last 1000 points
        # [-1000:, 2]
        x = list(ps[:, 0])
        y = list(ps[:, 1])
        z = list(ps[:, 2])
        all_z.extend(z) 
        #####################
        print(f"y0 = {y[0]}")
        ##################### 
        # Plot 3D trajectory with Shape Model
        col = col_ls[(ii * (nH + 1) + jj) % len(col_ls)]
        ax1.plot(x, y, z ,alpha=1, color="#0FAD1E")
        ax1.add_collection3d(mesh)
        ax1.set_aspect('equal', 'box') 
        ##############################
        # Frequency Analysis 
        n = len(ps[:, 0])
        dt = 1.0  # time step in seconds (adjust if your data has different dt)
        
        # Apply Hann window to reduce spectral leakage
        window = windows.hann(n)
        fft_x = fft.fft(ps[:, 0] * window)
        fft_y = fft.fft(ps[:, 1] * window)
        fft_z = fft.fft(ps[:, 2] * window)
        
        # Frequency array
        freqs = fft.fftfreq(n, dt)
        
        # Power spectrum (magnitude squared)
        power_x = np.abs(fft_x)**2
        power_y = np.abs(fft_y)**2
        power_z = np.abs(fft_z)**2
        power_total = power_x 
        
        # Only positive frequencies
        pos_mask = freqs > 0
        
        # Expected orbital frequency
        orbital_freq = omega / (2 * np.pi)  # Hz
        
        # Find peaks in the spectrum
        if np.max(power_total[pos_mask]) > 0:
            peaks, _ = find_peaks(power_total[pos_mask], height=np.max(power_total[pos_mask])*0.1)
            peak_freqs = freqs[pos_mask][peaks]
            print(f"Peak frequencies: {peak_freqs}")
            print(f"Expected orbital frequency: {orbital_freq} Hz")
            print(f"Spin period: {Spin/3600:.2f} hours")
            
            # Plot FFT
            col = col_ls[(ii * (nH + 1) + jj) % len(col_ls)]
            ax5.loglog(freqs[pos_mask], power_total[pos_mask], alpha=0.7, color=col, linewidth=1.5)
            
            # Mark the orbital frequency
            ax5.axvline(x=orbital_freq, color='red', linestyle='--', linewidth=2, label=f'Spin freq ({orbital_freq:.2e} Hz)')
            
            # Mark detected peaks
            if len(peaks) > 0:
                ax5.plot(peak_freqs, power_total[pos_mask][peaks], 'ro', markersize=8, label='Detected peaks')
        
        #####################################
        # enr = Calc_Ham(ps.T, omega, mu_I, CM)
        #
        # # print(enr)
        # #####################
        # #
        # #
        # col = col_ls[(ii * (nH + 1) + jj) % len(col_ls)]
        # ax2.plot(t/day,enr, alpha=1, color=col)
        # ###
        # vel = np.sqrt(ps[:, 3]**2 + ps[:, 4]**2 + ps[:, 5]**2)
        # ax3.plot(t,vel, alpha=1, color='blue',label='Rotating')
        # ax4.plot(t, r, alpha=1, color='blue',label='Rotating')
        ###################################
    #     r = np.sqrt(ps[:, 0]**2 + ps[:, 1]**2 + ps[:, 2]**2)
    #     r_max = np.max(r)
    #     r_min = np.min(r)
    #     r_bar = (r_max - r_min)/y[0]
    #     if np.isclose(r_bar, 0):
    #         e = "Circular"
    #     elif r_bar > 1.0:
    #         e = "Hyperbolic"
    #     else:
    #         e = "Elliptical"
    #     analysis = f"""
    # {"-"*42}
    # | r_max = {r_max}
    # | r_min = {r_min}
    # | r_bar = {r_bar}
    # | e_apr = {e}
    # |--- Velocity ---
    # | max   = {ps[:,4].max()}  
    # | min   = {ps[:,4].min()}
    # | mean  = {np.mean(ps[:,4])}  +/- {np.std(ps[:,4])}
    # |----------------
    # |
    # {"-"*42}
    #     """
    #     print(analysis)

    #     # CHAOS ANALYSIS - NEW ADDITION
    #     print(f"\n{'-'*42}")
    #     print("CHAOS ANALYSIS")
    #     print(f"{'-'*42}")
        
    #     chaos_summary = LP.analyze_trajectory_for_chaos(ps, 1.0)  # dt = 1.0 second
        

    #     state = ps
        ###################################
        # Translate to inertial frame 
        #
        # fixing rot2fixed frame
        #
        def rot2fix(xr, t):
            print(xr[0, 0])
            # input('| Press Enter to continue...')
            ct = np.cos(t)
            st = np.sin(t)

            xf = np.zeros_like(xr)
            
            
            xf[:, 0] = xr[:, 0]*ct - xr[:, 1]*st
            xf[:, 1] = xr[:, 1]*ct + xr[:, 0]*st
            xf[:, 2] = xr[:, 2]
            xf[:, 3] =-xr[:, 0]*st - xr[:, 4]*st + xr[:, 3]*ct - xr[:, 1]*ct
            xf[:, 4] = xr[:, 3]*st - xr[:, 1]*st + xr[:, 0]*ct + xr[:, 4]*ct
            xf[:, 5] = xr[:, 5]

            return xf
        
        # inS = rot2fix(state, omega*t)
        # print(inS)
        # vel_in = np.sqrt(inS[:,3]**2 + inS[:,4]**2 + inS[:,5]**2)
        # r_in = np.sqrt(inS[:,0]**2 + inS[:,1]**2 + inS[:,2]**2)
        # print(vel.shape)
        # print(r.shape)
        # ####
        # ax7.plot(inS[:,0], inS[:,1], inS[:,2], alpha=0.5, color='red')

        ###########
        ###
        # Poincare(ps.T)






# Add legend to FFT plot (remove duplicate labels)
handles, labels = ax5.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax5.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=14)

plt.show()

# plt.savefig(folder + 'Poincare_Sec.png', dpi=300)