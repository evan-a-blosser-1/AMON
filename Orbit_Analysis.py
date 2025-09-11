"""
Lyapunov Exponent Analysis for Orbital Dynamics - Orbit_Analysis.py

ACADEMIC CITATIONS AND SOURCES:
===============================

Primary Algorithm Source:
- Wolf, A., Swift, J.B., Swinney, H.L., & Vastano, J.A. (1985). 
  "Determining Lyapunov exponents from a time series." 
  Physica D: Nonlinear Phenomena, 16(3), 285-317.
  DOI: 10.1016/0167-2789(85)90011-9

Practical Implementation:
- Rosenstein, M.T., Collins, J.J., & De Luca, C.J. (1993). 
  "A practical method for calculating largest Lyapunov exponents from 
  small data sets." Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
  DOI: 10.1016/0167-2789(93)90009-P

Orbital Mechanics Theory:
- Scheeres, D.J. (2012). "Orbital Motion in Strongly Perturbed 
  Environments: Applications to Asteroid, Comet and Planetary 
  Satellite Orbiters." Springer-Praxis.

Spectral Analysis:
- Welch, P. (1967). "The use of fast Fourier transform for the estimation 
  of power spectra: a method based on time averaging over short, modified 
  periodograms." IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.

Mathematical Foundation:
- Oseledec, V.I. (1968). "A multiplicative ergodic theorem. Lyapunov 
  characteristic numbers for dynamical systems." Transactions of the Moscow 
  Mathematical Society, 19, 197-231.

See references_and_citations.py for complete bibliography.
"""

import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection 
import matplotlib.colors as mcolors
from matplotlib import cm
import numpy as np
import sys
import trimesh
from scipy import stats
from scipy import signal
###
# Asteroid Name
asteroid = 'Apophis'
Spin = 30.5
# 
omega = (2*np.pi)/(Spin * 3600)

folder   = "Databank/1950DA/tr_1e-7_6.7km/"

def estimate_lyapunov_exponent(trajectory_data, dt):
    """
    Estimate the largest Lyapunov exponent from trajectory data
    
    Mathematical Method: Nearby Trajectory Separation
    ================================================
    
    CITATION: This implementation is based on the algorithm described in:
    Wolf, A., Swift, J.B., Swinney, H.L., & Vastano, J.A. (1985). 
    "Determining Lyapunov exponents from a time series." 
    Physica D: Nonlinear Phenomena, 16(3), 285-317.
    
    With practical improvements from:
    Rosenstein, M.T., Collins, J.J., & De Luca, C.J. (1993). 
    "A practical method for calculating largest Lyapunov exponents from 
    small data sets." Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
    
    Theory:
    -------
    The largest Lyapunov exponent λ₁ is defined as:
    λ₁ = lim[t→∞] lim[δx₀→0] (1/t) ln(||δx(t)||/||δx₀||)
    
    This mathematical definition was originally formulated by Lyapunov (1892)
    and rigorously developed by Oseledec (1968).
    
    Implementation:
    --------------
    1. For reference point x_ref(t_r), find nearby point x_near(t_n)
    2. Track separation: d(t) = ||x_ref(t_r + t) - x_near(t_n + t)||
    3. Calculate local exponent: λ_local = d/dt[ln(d(t))] ≈ slope(ln(d) vs t)
    4. Average over multiple trajectory pairs: λ₁ = ⟨λ_local⟩
    
    Parameters:
    -----------
    trajectory_data : array
        Trajectory data (n_points, 6) - [x, y, z, vx, vy, vz]
    dt : float
        Time step in seconds
        
    Returns:
    --------
    lyap_estimate : float
        Estimated largest Lyapunov exponent (1/s)
        
    Mathematical Notes:
    ------------------
    • Positive λ₁: Chaotic (exponential divergence)
    • Zero λ₁: Neutral/marginal stability  
    • Negative λ₁: Stable (exponential convergence)
    • Lyapunov time: τ = 1/|λ₁| (predictability horizon)
    
    References:
    ----------
    See references_and_citations.py for complete bibliography.
    """
    n_points = len(trajectory_data)
    lyap_estimates = []
    
    # Mathematical Algorithm: Nearby Trajectory Method
    # ===============================================
    # CITATION: Algorithm based on Wolf et al. (1985) with improvements from
    # Rosenstein et al. (1993) for practical implementation from time series data.
    
    # Step 1: Sample reference points for computational efficiency
    # Theoretical basis: Statistical sampling improves robustness (Kantz, 1994)
    n_refs = min(50, n_points // 20)
    ref_indices = np.linspace(0, n_points - 500, n_refs, dtype=int)
    
    for ref_idx in ref_indices:
        ref_point = trajectory_data[ref_idx]  # Reference trajectory point x_ref(t_r)
        
        # Step 2: Find nearby point in 6D phase space
        # Mathematical constraint: ||x_ref - x_near|| = min{distances} > δ_min
        search_range = min(n_points // 10, 1000)
        end_idx = min(ref_idx + search_range, n_points)
        
        # Euclidean distance in 6D phase space: d = √(Σᵢ(xᵢ - x'ᵢ)²)
        distances = np.sqrt(np.sum((trajectory_data[ref_idx:end_idx] - ref_point)**2, axis=1))
        
        # Exclude self-distance (= 0) and enforce minimum separation threshold
        if len(distances) < 2:
            continue
            
        valid_distances = distances[1:]  # Exclude self
        if len(valid_distances) == 0 or np.min(valid_distances) < 1e-8:  # δ_min = 10⁻⁸
            continue
            
        nearest_local_idx = np.argmin(valid_distances) + 1
        nearest_idx = ref_idx + nearest_local_idx
        
        if nearest_idx >= n_points - 100:
            continue
        
        # Step 3: Track separation evolution d(t) = ||x_ref(t_r + t) - x_near(t_n + t)||
        # Mathematical model: d(t) ≈ d₀ * exp(λ₁ * t) for chaotic systems
        separations = []  # ln(d(t))
        times = []        # t
        max_evolution = min(100, n_points - max(ref_idx, nearest_idx))
        
        for i in range(max_evolution):
            if ref_idx + i < n_points and nearest_idx + i < n_points:
                # Calculate separation at time t = i * dt
                sep = np.linalg.norm(trajectory_data[ref_idx + i] - trajectory_data[nearest_idx + i])
                if sep > 1e-10:  # Avoid ln(0)
                    separations.append(np.log(sep))  # ln(d(t))
                    times.append(i * dt)             # t
        
        # Step 4: Linear regression to estimate λ₁
        # Mathematical model: ln(d(t)) = ln(d₀) + λ₁ * t
        # Slope of ln(d) vs t gives λ₁
        if len(separations) > 10:
            try:
                slope, _ = np.polyfit(times, separations, 1)  # λ₁ = slope
                lyap_estimates.append(slope)
            except:
                continue
    
    # Step 5: Statistical averaging for robust estimate
    # λ₁ = ⟨λ_local⟩ (ensemble average over multiple trajectory pairs)
    if len(lyap_estimates) > 0:
        return np.mean(lyap_estimates)
    else:
        return 0.0

def analyze_trajectory_for_chaos(trajectory_data, dt):
    """
    Quick chaos analysis for a trajectory
    
    Mathematical Framework: Multi-Modal Chaos Detection
    ==================================================
    
    This function combines multiple mathematical approaches to detect chaotic
    behavior in orbital trajectories:
    
    1. Spectral Analysis (Frequency Domain)
    2. Lyapunov Exponent Estimation (Phase Space)
    3. Statistical Characterization
    
    Parameters:
    -----------
    trajectory_data : array
        Trajectory data (n_points, 6)
    dt : float
        Time step in seconds
        
    Returns:
    --------
    chaos_summary : dict
        Summary of chaos indicators
        
    Mathematical Methods:
    --------------------
    
    A. Power Spectral Density Analysis:
       S(f) = |X(f)|²/T where X(f) = ∫ x(t)e^(-i2πft) dt
       
    B. Spectral Entropy:
       H_s = -Σᵢ pᵢ ln(pᵢ) where pᵢ = S(fᵢ)/Σⱼ S(fⱼ)
       
       Physical Meaning:
       • Low H_s: Regular/periodic motion (narrow spectrum)
       • High H_s: Chaotic motion (broad spectrum)
       
    C. Lyapunov Exponent:
       λ₁ = ⟨(1/t) ln(||δx(t)||/||δx₀||)⟩
       
       Physical Meaning:
       • λ₁ > 0: Chaotic (exponential divergence)
       • λ₁ ≈ 0: Critical/marginal behavior
       • λ₁ < 0: Stable (exponential convergence)
    """
    # Mathematical Implementation
    # ==========================
    
    # Step 1: Calculate trajectory statistics
    # Physical quantities: position and velocity magnitudes
    pos_mag = np.sqrt(np.sum(trajectory_data[:, :3]**2, axis=1))      # ||r(t)||
    vel_mag = np.sqrt(np.sum(trajectory_data[:, 3:]**2, axis=1))      # ||v(t)||
    
    # Step 2: Spectral analysis using Welch's method
    # Mathematical basis: Fourier analysis for frequency content
    freqs, psd = signal.periodogram(pos_mag, fs=1/dt)
    
    # Spectral entropy calculation
    # H_s = -Σᵢ pᵢ ln(pᵢ) where pᵢ = normalized power
    psd_normalized = psd / np.sum(psd)  # Normalize to probability distribution
    # Add small epsilon to avoid log(0)
    spectral_entropy = -np.sum(psd_normalized * np.log(psd_normalized + 1e-12))
    
    # Step 3: Lyapunov exponent estimation
    # Uses nearby trajectory method (see function documentation above)
    lyap_exp = estimate_lyapunov_exponent(trajectory_data, dt)
    
    # Step 4: Physical interpretation and classification
    # Mathematical thresholds based on orbital dynamics theory
    if lyap_exp > 1e-8:
        chaos_level = "CHAOTIC"
        lyap_time_days = 1.0 / lyap_exp / 86400    # τ = 1/λ₁ (Lyapunov time)
        predictability = f"{lyap_time_days:.1f} days"
    elif lyap_exp > -1e-8:
        chaos_level = "MARGINAL"
        predictability = "Uncertain"
    else:
        chaos_level = "STABLE"
        predictability = "Long-term"
    
    return {
        'lyapunov_exponent': lyap_exp,
        'chaos_level': chaos_level,
        'predictability_horizon': predictability,
        'spectral_entropy': spectral_entropy,
        'trajectory_duration_days': len(trajectory_data) * dt / 86400
    } 
########
xi = 1.0
xf = 7.0
dx = 0.01
nx = round((xf - xi)/dx)
########################
Hi = 1.0e-7
Hf = 1.0e-7
dH = 0.1e-6
nH = round((Hf - Hi) / dH)
########################
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
        neg_si = state[0, it]*xa[0]
        pos_si = state[0, it]
        #
        if neg_si < 0.0 and pos_si > 0.0:
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
            ax.plot(xm[1], xm[4], 'r.')
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


########################################################
#################################### Personal Packages #    
sys.dont_write_bytecode = True
import constants as C
const    = C.constants()
target = C.apophis()
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
# Polyhedron Center of Mass
mesh = trimesh.load_mesh(obj_Path)
Poly_CM = [target.Poly_x, target.Poly_y, target.Poly_z]
#######################################################
#######################################################
# plt.rcParams["figure.figsize"] = [6.5, 6.5]
plt.rcParams["figure.autolayout"] = True
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'

# plt.axvline(x = 4.45e-3, color = 'r')
# plt.axvline(x = (1 - 4.45e-3), color = 'b')


########################################################
########################################################
all_z = []
scatter_plots = []
mesh = trimesh.load_mesh('Apophis.obj')
gamma = 0.285 
mesh = mesh.apply_scale(gamma)
v = mesh.vertices
f = mesh.faces 
print(f"Vertices: {v.shape}")
print(f"Faces: {f.shape}") 
# mesh = Poly3DCollection([v[ii] for ii in f], 
#                 edgecolor='black',
#                 facecolors="white",
#                 linewidth=0.75,
#                 alpha=0.0)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.add_collection3d(mesh)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

col_ls = ['r', 'b', 'g', 'y', 'm', 'c', 'k']

fig1 = plt.figure()
ax1  = fig1.add_subplot(111, projection='3d')
ax1.set_title('Trajectory')
ax1.set_xlabel('X (km)')
ax1.set_ylabel('Y (km)')
ax1.set_zlabel('Z (km)')

fig2 = plt.figure()
ax2  = fig2.add_subplot(111)
ax2.set_title('Hamiltonian Energy')
ax2.set_xlabel('time (sec)')
ax2.set_ylabel('Hamiltonian (km^2/s^2)')

fig3 = plt.figure()
ax3  = fig3.add_subplot(111)
ax3.set_title(r'Velocity')
ax3.set_xlabel('time (sec)')
ax3.set_ylabel(r'Velocity $\frac{km}{s}$')


fig4 = plt.figure()
ax4  = fig4.add_subplot(111)
ax4.set_title(r'Position')
ax4.set_xlabel('time (sec)')
ax4.set_ylabel(r'Position (km)')

fig5 = plt.figure()
ax5  = fig5.add_subplot(111)
ax5.set_title('Inertial Frame Velocity')
ax5.set_xlabel('time (sec)')
ax5.set_ylabel(r'Velocity $\frac{km}{s}$')

fig6 = plt.figure()
ax6  = fig6.add_subplot(111)
ax6.set_title('Inertial Frame Position')
ax6.set_xlabel('time (sec)')
ax6.set_ylabel(r'Position (km)')

fig7 = plt.figure()
ax7  = fig7.add_subplot(111, projection='3d')
ax7.set_title('Inertial Frame Trajectory')  
ax7.set_xlabel('X (km)')
ax7.set_ylabel('Y (km)')
ax7.set_zlabel('Z (km)')

########################################################
####################

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
        
                
        file = folder + '/' + 'TR-S0' +'-H' + aux1 + 'Yi' + aux2 + '.dat'
    
    
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


        x = list(ps[:, 0])
        y = list(ps[:, 1])
        z = list(ps[:, 2])
        all_z.extend(z) 
        #####################
        print(f"y0 = {y[0]}")
        #
        #
        col = col_ls[(ii * (nH + 1) + jj) % len(col_ls)]
        ax1.plot(x, y, z ,alpha=1, color="#0ddd22")
        ax1.add_collection3d(mesh)
        ax1.set_aspect('equal', 'box') 
        #####################################
        enr = Calc_Ham(ps.T, omega, mu_I, CM)
        t = np.linspace(0,ps.shape[0],ps.shape[0])
        # print(enr)
        # print(t)
        # print('####################')
        #####################
        #
        #
        col = col_ls[(ii * (nH + 1) + jj) % len(col_ls)]
        ax2.plot(t,enr, alpha=1, color=col)
        ###
        vel = np.sqrt(ps[:, 3]**2 + ps[:, 4]**2 + ps[:, 5]**2)
        ax3.plot(t,vel, alpha=1, color='blue',label='Rotating')
        r = np.sqrt(ps[:, 0]**2 + ps[:, 1]**2 + ps[:, 2]**2)
        ax4.plot(t, r, alpha=1, color='blue',label='Rotating')
        ###################################
        r = np.sqrt(ps[:, 0]**2 + ps[:, 1]**2 + ps[:, 2]**2)
        r_max = np.max(r)
        r_min = np.min(r)
        r_bar = (r_max - r_min)/y[0]
        if np.isclose(r_bar, 0):
            e = "Circular"
        elif r_bar > 1.0:
            e = "Hyperbolic"
        else:
            e = "Elliptical"
        analysis = f"""
    {"-"*42}
    | r_max = {r_max}
    | r_min = {r_min}
    | r_bar = {r_bar}
    | e_apr = {e}
    |--- Velocity ---
    | max   = {ps[:,4].max()}  
    | min   = {ps[:,4].min()}
    | mean  = {np.mean(ps[:,4])}  +/- {np.std(ps[:,4])}
    |----------------
    |
    {"-"*42}
        """
        print(analysis)

        # CHAOS ANALYSIS - NEW ADDITION
        print(f"\n{'-'*42}")
        print("CHAOS ANALYSIS")
        print(f"{'-'*42}")
        
        chaos_summary = analyze_trajectory_for_chaos(ps, 1.0)  # dt = 1.0 second
        
        print(f"Lyapunov exponent: {chaos_summary['lyapunov_exponent']:.2e} /s")
        print(f"Chaos level: {chaos_summary['chaos_level']}")
        print(f"Predictability: {chaos_summary['predictability_horizon']}")
        print(f"Spectral entropy: {chaos_summary['spectral_entropy']:.3f}")
        
        if chaos_summary['chaos_level'] == "CHAOTIC":
            print("🔴 WARNING: Chaotic trajectory detected!")
            print("   Long-term predictions unreliable")
        elif chaos_summary['chaos_level'] == "MARGINAL":
            print("🟡 CAUTION: Marginally stable trajectory")
        else:
            print("🟢 STABLE: Predictable trajectory")

        state = ps
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
        
        inS = rot2fix(state, omega*t)
        print(inS)
        vel_in = np.sqrt(inS[:,3]**2 + inS[:,4]**2 + inS[:,5]**2)
        r_in = np.sqrt(inS[:,0]**2 + inS[:,1]**2 + inS[:,2]**2)
        print(vel.shape)
        print(r.shape)
        ax5.plot(t, vel_in, alpha=1, color='red', label='Inertial')
        ax6.plot(t, r_in, alpha=1, color='red', label='Inertial')
        ####
        ax7.plot(inS[:,0], inS[:,1], inS[:,2], alpha=0.5, color='red')

        ###########
        ###
        #Poincare(ps.T)

        plt.show()





# fig.set_facecolor('#000000')
# ax.set_facecolor('#000000')
# ax.tick_params(axis='x', colors='white')
# ax.tick_params(axis='y', colors='white')

# plt.show()
# plt.savefig(folder + 'Poincare_Sec.png', dpi=300)